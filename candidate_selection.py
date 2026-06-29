import argparse
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_evolution_effect import build_boundary_dedup_signature


RISK_SCORE = {"low": 0, "medium": -5, "high": -25}


def load_json_or_jsonl(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON input must be an array")
        return data
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str, *, append: bool = False) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    meta_info = item.get("meta_info")
    if not isinstance(meta_info, dict):
        return {}
    metadata = meta_info.get("question_evolution_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _effect_analysis(item: Dict[str, Any]) -> Dict[str, Any]:
    effect = item.get("effect_analysis")
    return effect if isinstance(effect, dict) else {}


def candidate_group_id(item: Dict[str, Any]) -> str:
    for field in ("candidate_group_id", "sample_id", "index"):
        value = item.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return _clean_text(item.get("prompt"))


def candidate_id(item: Dict[str, Any], fallback_index: int) -> str:
    value = item.get("candidate_id")
    if value is not None and str(value).strip():
        return str(value).strip()
    return f"{candidate_group_id(item)}::cand_{fallback_index}"


def candidate_operator(item: Dict[str, Any]) -> str:
    for field in ("candidate_operator", "operator_used"):
        value = _clean_text(item.get(field))
        if value:
            return value
    return _clean_text(_metadata(item).get("operator_used"))


def validation_result(item: Dict[str, Any]) -> Dict[str, Any]:
    result = item.get("validation_result")
    return result if isinstance(result, dict) else {"passed": False, "reject_reason": "缺少 validation_result"}


def boundary_selection_flags(item: Dict[str, Any]) -> Dict[str, Any]:
    effect = _effect_analysis(item)
    dedup_signature = _clean_text(effect.get("dedup_signature")) or build_boundary_dedup_signature(item)
    duplicate = bool(effect.get("duplicate_boundary_for_sample") or effect.get("discard_as_duplicate"))
    is_new_boundary = bool(effect.get("is_new_boundary_for_sample"))
    effect_label = _clean_text(effect.get("effect_label"))
    hit_confidence = _clean_text(effect.get("hit_confidence"))

    selected_as_boundary_leaf = is_new_boundary
    selected_into_mainline = not duplicate
    if (
        selected_as_boundary_leaf
        and effect_label == "effective_boundary_probe"
        and hit_confidence in {"medium", "high"}
    ):
        selected_into_mainline = False

    return {
        "selected_into_mainline": selected_into_mainline,
        "selected_as_boundary_leaf": selected_as_boundary_leaf,
        "discard_as_duplicate": duplicate,
        "dedup_signature": dedup_signature,
        "boundary_axis_detected": effect.get("boundary_axis_detected"),
        "target_boundary_axis": effect.get("target_boundary_axis"),
        "boundary_candidate_status": effect.get("boundary_candidate_status"),
    }


def attach_boundary_selection_flags(item: Dict[str, Any], flags: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(item)
    for field in (
        "selected_into_mainline",
        "selected_as_boundary_leaf",
        "discard_as_duplicate",
        "dedup_signature",
    ):
        result[field] = flags[field]
    return result


def passthrough_selection_flags(item: Dict[str, Any]) -> Dict[str, Any]:
    flags = boundary_selection_flags(item)
    flags["selected_into_mainline"] = True
    flags["selected_as_boundary_leaf"] = False
    flags["discard_as_duplicate"] = False
    flags["boundary_candidate_status"] = None
    return flags


def _risk_penalty(validation: Dict[str, Any], field: str) -> int:
    return RISK_SCORE.get(_clean_text(validation.get(field)), -10)


def score_candidate(item: Dict[str, Any]) -> Tuple[int, List[str]]:
    validation = validation_result(item)
    reasons: List[str] = []
    if not validation.get("passed"):
        return -10_000, [_clean_text(validation.get("reject_reason")) or "未通过复杂度校验"]
    flags = boundary_selection_flags(item)
    if flags["discard_as_duplicate"]:
        return -10_000, ["与同一样本已沉淀的能力边界重复"]

    score = 100
    main_axis_count = int(validation.get("main_axis_count", 1) or 0)
    prompt_chars = int(validation.get("estimated_prompt_chars", len(_clean_text(item.get("prompt")))) or 0)
    output_tasks = int(validation.get("output_tasks_count", 1) or 0)
    candidate_options = int(validation.get("candidate_options_count", 0) or 0)
    counterfactuals = int(validation.get("counterfactual_count", 0) or 0)

    if main_axis_count == 1:
        score += 15
        reasons.append("主轴唯一")
    if 120 <= prompt_chars <= 900:
        score += 10
        reasons.append("题长处于可控区间")
    elif prompt_chars <= 1200:
        score += 3
        reasons.append("题长未超过硬上限")
    if output_tasks <= 1:
        score += 8
        reasons.append("输出任务单一")
    if candidate_options <= 2:
        score += 4
        reasons.append("候选项数量克制")
    if counterfactuals == 0:
        score += 3
        reasons.append("未引入反事实复杂度")

    score += _risk_penalty(validation, "external_knowledge_risk")
    score += _risk_penalty(validation, "format_difficulty_risk")
    score += _risk_penalty(validation, "repeat_pattern_risk")

    generation = item.get("candidate_generation")
    if isinstance(generation, dict) and generation.get("operator_source") == "primary":
        score += 2
        reasons.append("来自 router primary operator")

    focus = _metadata(item).get("expected_evaluation_focus")
    if isinstance(focus, list) and focus:
        score += 3
        reasons.append("保留 expected_evaluation_focus 元数据")

    effect = _effect_analysis(item)
    if effect.get("is_new_boundary_for_sample"):
        score += 20
        reasons.append("形成新的能力边界候选")
    elif effect.get("structural_boundary_signal"):
        score += 6
        reasons.append("存在结构性边界信号，需保留审计")

    return score, reasons


def build_rejected_candidate(item: Dict[str, Any], fallback_index: int, *, forced_reason: Optional[str] = None) -> Dict[str, Any]:
    validation = validation_result(item)
    reason = forced_reason or _clean_text(validation.get("reject_reason")) or "未被选中"
    flags = boundary_selection_flags(item)
    return {
        "candidate_id": candidate_id(item, fallback_index),
        "operator_used": candidate_operator(item),
        "reject_reason": reason,
        "validation_result": validation,
        "discard_as_duplicate": flags["discard_as_duplicate"],
        "selected_as_boundary_leaf": False,
        "dedup_signature": flags["dedup_signature"],
    }


def build_invalid_case(item: Dict[str, Any], fallback_index: int, *, reason: Optional[str] = None) -> Dict[str, Any]:
    validation = validation_result(item)
    metadata = _metadata(item)
    return {
        "sample_id": item.get("sample_id", item.get("index", "")),
        "round": item.get("round"),
        "candidate_id": candidate_id(item, fallback_index),
        "operator_used": candidate_operator(item),
        "invalid_type": validation.get("invalid_type") or "not_selected",
        "reason": reason or _clean_text(validation.get("reject_reason")) or "candidate was not selected",
        "suggested_operator": metadata.get("operator_used") or item.get("candidate_operator"),
    }


def _strip_candidate_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(item)
    result.pop("candidate_generation", None)
    return result


def _restore_original_passthrough(item: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(item)
    meta_info = result.get("meta_info")
    meta_info = meta_info if isinstance(meta_info, dict) else {}
    old_prompt = meta_info.get("prompt_old")
    if isinstance(old_prompt, str) and old_prompt.strip():
        result["prompt"] = old_prompt.strip()
    stale_fields = {
        "rubric": "stale_rubric",
        "score_prompt": "stale_score_prompt",
        "scoring_result": "stale_scoring_result",
    }
    for field, stale_field in stale_fields.items():
        if field not in result and stale_field in meta_info:
            result[field] = meta_info.get(stale_field)
    result["question_evolved"] = False
    return result


def select_group(records: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not records:
        raise ValueError("candidate group is empty")

    if len(records) == 1 and records[0].get("question_evolved") is False:
        selected = _strip_candidate_fields(records[0])
        cid = candidate_id(records[0], 1)
        flags = passthrough_selection_flags(records[0])
        selected = attach_boundary_selection_flags(selected, flags)
        selected["candidate_selection"] = {
            "selected_candidate_id": cid,
            "selected_operator": "",
            "selection_reason": "透传样本不参与候选选择。",
            "rejected_candidates": [],
            **flags,
        }
        return selected, []

    scored: List[Tuple[int, List[str], int, Dict[str, Any]]] = []
    for index, record in enumerate(records, start=1):
        score, reasons = score_candidate(record)
        scored.append((score, reasons, index, record))
    scored.sort(key=lambda item: (item[0], -item[2]), reverse=True)

    best_score, best_reasons, best_index, best_record = scored[0]
    invalid_cases: List[Dict[str, Any]] = []
    rejected_candidates: List[Dict[str, Any]] = []

    if best_score < 0:
        selected = _strip_candidate_fields(_restore_original_passthrough(records[0]))
        flags = passthrough_selection_flags(records[0])
        selected = attach_boundary_selection_flags(selected, flags)
        selected["candidate_selection"] = {
            "selected_candidate_id": candidate_id(records[0], 1),
            "selected_operator": "",
            "selection_reason": "所有候选均未通过复杂度校验，回退为原题透传。",
            "rejected_candidates": [
                build_rejected_candidate(record, index)
                for index, record in enumerate(records, start=1)
            ],
            **flags,
        }
        for index, record in enumerate(records, start=1):
            invalid_cases.append(build_invalid_case(record, index))
        return selected, invalid_cases

    selected = _strip_candidate_fields(best_record)
    selected_id = candidate_id(best_record, best_index)
    selected_operator = candidate_operator(best_record)
    selected_flags = boundary_selection_flags(best_record)
    selected = attach_boundary_selection_flags(selected, selected_flags)

    for _, _, index, record in scored[1:]:
        reason = "低于入选候选的综合选择分"
        record_flags = boundary_selection_flags(record)
        if record_flags["discard_as_duplicate"]:
            reason = "与同一样本已沉淀的能力边界重复"
        elif not validation_result(record).get("passed"):
            reason = _clean_text(validation_result(record).get("reject_reason")) or reason
            invalid_cases.append(build_invalid_case(record, index, reason=reason))
        rejected_candidates.append(build_rejected_candidate(record, index, forced_reason=reason))

    selected["candidate_selection"] = {
        "selected_candidate_id": selected_id,
        "selected_operator": selected_operator,
        "selection_reason": "；".join(best_reasons) if best_reasons else "通过复杂度校验且综合分最高。",
        "rejected_candidates": rejected_candidates,
        **selected_flags,
    }
    return selected, invalid_cases


def select_candidates(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[candidate_group_id(record)].append(record)

    selected_records: List[Dict[str, Any]] = []
    invalid_cases: List[Dict[str, Any]] = []
    for group_records in groups.values():
        selected, group_invalid = select_group(group_records)
        selected_records.append(selected)
        invalid_cases.extend(group_invalid)
    return selected_records, invalid_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one validated evolved-question candidate per sample.")
    parser.add_argument("--input", required=True, help="Input validated candidate JSON/JSONL path.")
    parser.add_argument("--output", required=True, help="Output selected evolved JSONL path.")
    parser.add_argument(
        "--invalid-output",
        default=os.path.join("memory", "invalid_generation_cases.jsonl"),
        help="Append rejected invalid candidate cases to this JSONL path.",
    )
    parser.add_argument(
        "--no-invalid-output",
        action="store_true",
        help="Do not write invalid generation cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_or_jsonl(args.input)
    selected, invalid_cases = select_candidates(records)
    write_jsonl(selected, args.output)
    if invalid_cases and not args.no_invalid_output:
        write_jsonl(invalid_cases, args.invalid_output, append=True)


if __name__ == "__main__":
    main()
