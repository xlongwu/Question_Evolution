import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from analyze_evolution_effect import (
    get_metadata,
    get_operator_used,
    is_question_evolved,
    load_json_or_jsonl,
)


FAILURE_EFFECT_LABELS = {
    "full_score_no_drop",
    "no_clear_effect",
    "score_increased",
    "repeated_pattern",
}

TERMINAL_STOP_STATUSES = {
    "effective_boundary_sample",
    "stable_high_score_stop",
    "validated_high_score_sample",
    "invalid_complexity_sample",
    "unanswerable_or_trap_sample",
}

OPERATOR_AVOID_METHODS = {
    "O1_gap_choice": [
        "继续问最少还缺什么",
        "继续问最小前提",
        "继续问最小跳步",
    ],
    "O2_subclaim_localization": ["继续只定位同一子判断"],
    "O4_near_level_ranking": ["继续只做判据内外二分"],
    "O8_double_threshold_claim": ["继续只比较显眼动作层"],
    "O9_abnormal_clue_mainline_switch": ["继续只问找车还是找人"],
}

NEXT_OPERATOR_HINTS = {
    "O1_gap_choice": ["O2_subclaim_localization", "O4_near_level_ranking", "O8_double_threshold_claim"],
    "O2_subclaim_localization": ["O4_near_level_ranking", "O8_double_threshold_claim"],
    "O3_step_jump": ["O4_near_level_ranking"],
    "O4_near_level_ranking": ["O5_extra_premise_detection", "O6_single_variable_counterfactual"],
    "O5_extra_premise_detection": ["O4_near_level_ranking", "O7_fact_binding_constraint"],
    "O6_single_variable_counterfactual": ["O9_abnormal_clue_mainline_switch", "O4_near_level_ranking"],
    "O7_fact_binding_constraint": ["O2_subclaim_localization", "O4_near_level_ranking"],
    "O8_double_threshold_claim": ["O2_subclaim_localization", "O4_near_level_ranking"],
    "O9_abnormal_clue_mainline_switch": ["O6_single_variable_counterfactual", "O4_near_level_ranking"],
}


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


def _sample_id(item: Dict[str, Any]) -> Any:
    return item.get("sample_id", item.get("index", ""))


def _round_value(item: Dict[str, Any], previous_state: Dict[str, Any]) -> int:
    for value in (item.get("round"), previous_state.get("round")):
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            return number
    return 0


def _effect(item: Dict[str, Any]) -> Dict[str, Any]:
    effect = item.get("effect_analysis")
    if not isinstance(effect, dict):
        raise ValueError("record missing effect_analysis; run analyze_evolution_effect.py first")
    return effect


def _validation(item: Dict[str, Any]) -> Dict[str, Any]:
    validation = item.get("validation_result")
    return validation if isinstance(validation, dict) else {}


def _previous_state(item: Dict[str, Any]) -> Dict[str, Any]:
    state = item.get("evolution_state")
    return dict(state) if isinstance(state, dict) else {}


def _append_unique(items: List[str], values: Sequence[str]) -> List[str]:
    for value in values:
        text = _clean_text(value)
        if text and text not in items:
            items.append(text)
    return items


def sample_signature(item: Dict[str, Any]) -> Dict[str, str]:
    profile = item.get("sample_profile")
    diagnosis = item.get("overscore_diagnosis")
    profile = profile if isinstance(profile, dict) else {}
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    return {
        "core_capability": _clean_text(profile.get("core_capability")),
        "claim_level": _clean_text(profile.get("claim_level")),
        "problem_shape": _clean_text(profile.get("problem_shape")),
        "candidate_overscore_cause": _clean_text(diagnosis.get("candidate_overscore_cause")),
    }


def _expected_failure_mode(item: Dict[str, Any]) -> str:
    metadata = get_metadata(item)
    expected = _clean_text(metadata.get("expected_qwen_failure"))
    if expected:
        return expected
    diagnosis = item.get("overscore_diagnosis")
    if isinstance(diagnosis, dict):
        return _clean_text(diagnosis.get("target_failure_mode") or diagnosis.get("candidate_overscore_cause"))
    return ""


def _stop_status(
    item: Dict[str, Any],
    full_score_count: int,
    same_operator_count: int,
    operator_switched_after_full_score: bool,
) -> str:
    effect = _effect(item)
    label = _clean_text(effect.get("effect_label"))
    previous_stop = _clean_text(_previous_state(item).get("stop_status"))
    previous_recommended = list(_previous_state(item).get("recommended_next_methods") or [])

    if label == "effective_boundary_probe":
        return "effective_boundary_sample"
    if label == "invalid_complexity":
        invalid_type = _clean_text(_validation(item).get("invalid_type"))
        if invalid_type in {"external_knowledge_required", "empty_prompt"}:
            return "unanswerable_or_trap_sample"
        return "invalid_complexity_sample"
    if label == "pass_through":
        return previous_stop or "continue"
    if label == "score_increased":
        return "validated_high_score_sample"
    if label == "full_score_no_drop":
        if full_score_count >= 2 and operator_switched_after_full_score:
            return "stable_high_score_stop"
        if previous_recommended:
            return "continue_with_new_operator"
        return "local_tree_search_needed" if full_score_count >= 2 else "continue_with_new_operator"
    if label == "repeated_pattern":
        return "stable_high_score_stop" if same_operator_count >= 2 else "continue_with_new_operator"
    if label in {"needs_manual_review", "no_clear_effect", "score_increased"}:
        return "continue_with_new_operator"
    return previous_stop or "continue"


def _recommended_next_methods(operator_used: str, label: str, full_score_count: int) -> List[str]:
    if label == "effective_boundary_probe":
        return []
    if label == "score_increased":
        return []
    hints = list(NEXT_OPERATOR_HINTS.get(operator_used, []))
    if full_score_count >= 2 and "O4_near_level_ranking" not in hints:
        hints.append("O4_near_level_ranking")
    return hints


def build_next_state(item: Dict[str, Any]) -> Dict[str, Any]:
    effect = _effect(item)
    previous_state = _previous_state(item)
    operator_used = _clean_text(effect.get("operator_used")) or get_operator_used(item)
    previous_operator = _clean_text(previous_state.get("previous_operator"))
    previous_same_count = int(previous_state.get("consecutive_same_operator_count", 0) or 0)
    previous_full_count = int(previous_state.get("consecutive_full_score_count", 0) or 0)
    current_full = bool(effect.get("is_full_score"))
    full_score_count = previous_full_count + 1 if current_full else 0
    same_operator_count = previous_same_count + 1 if operator_used and operator_used == previous_operator else (1 if operator_used else 0)
    operator_switched_after_full_score = (
        current_full
        and previous_full_count >= 1
        and bool(operator_used)
        and bool(previous_operator)
        and operator_used != previous_operator
    )
    label = _clean_text(effect.get("effect_label"))

    avoid_methods = list(previous_state.get("avoid_methods") or [])
    if label in FAILURE_EFFECT_LABELS or label == "needs_manual_review":
        _append_unique(avoid_methods, OPERATOR_AVOID_METHODS.get(operator_used, []))

    recommended = _recommended_next_methods(operator_used, label, full_score_count)
    if not recommended and label not in {"effective_boundary_probe", "score_increased"}:
        recommended = list(previous_state.get("recommended_next_methods") or [])

    stop_status = _stop_status(
        item,
        full_score_count,
        same_operator_count,
        operator_switched_after_full_score,
    )
    if stop_status in TERMINAL_STOP_STATUSES:
        recommended = []

    return {
        "round": _round_value(item, previous_state),
        "previous_operator": operator_used or None,
        "previous_score_rate": effect.get("score_rate_after"),
        "previous_effect_status": label or None,
        "previous_failure_mode": _expected_failure_mode(item) or None,
        "consecutive_full_score_count": full_score_count,
        "consecutive_same_operator_count": same_operator_count,
        "avoid_methods": avoid_methods,
        "recommended_next_methods": recommended,
        "stop_status": stop_status,
    }


def attach_next_state(item: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(item)
    result["evolution_state"] = build_next_state(item)
    return result


def build_operator_memory_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    effect = _effect(item)
    metadata = get_metadata(item)
    confidence = _clean_text(effect.get("hit_confidence")) or "low"
    reuse_note = "自动轻量命中，进入下一轮路由前建议人工复核。"
    if confidence == "low":
        reuse_note = "低置信命中，仅供人工复核和后续对照，不应沉淀为强成功经验。"
    return {
        "sample_id": _sample_id(item),
        "round": _round_value(item, _previous_state(item)),
        "sample_signature": sample_signature(item),
        "operator_used": _clean_text(effect.get("operator_used")),
        "expected_qwen_failure": _clean_text(metadata.get("expected_qwen_failure")),
        "score_rate_before": effect.get("score_rate_before"),
        "score_rate_after": effect.get("score_rate_after"),
        "delta_score_rate": effect.get("delta_score_rate"),
        "question_length": effect.get("question_length"),
        "validation_passed": bool(effect.get("complexity_passed")),
        "hit_confidence": confidence,
        "needs_manual_review": bool(effect.get("needs_manual_review", True)),
        "effect_label": _clean_text(effect.get("effect_label")),
        "reuse_note": reuse_note,
    }


def build_failure_memory_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    effect = _effect(item)
    operator_used = _clean_text(effect.get("operator_used"))
    recommended = _recommended_next_methods(
        operator_used,
        _clean_text(effect.get("effect_label")),
        int(build_next_state(item).get("consecutive_full_score_count", 0) or 0),
    )
    return {
        "sample_id": _sample_id(item),
        "round": _round_value(item, _previous_state(item)),
        "sample_signature": sample_signature(item),
        "operator_used": operator_used,
        "score_rate_before": effect.get("score_rate_before"),
        "score_rate_after": effect.get("score_rate_after"),
        "failure_type": _clean_text(effect.get("effect_label")) or "operator_ineffective",
        "failure_reason": _clean_text(effect.get("lightweight_hit_reason")) or "未形成清晰降分。",
        "avoid_note": "建议切换到：" + "、".join(recommended) if recommended else "建议避免重复当前问法。",
    }


def build_invalid_generation_case(item: Dict[str, Any]) -> Dict[str, Any]:
    effect = _effect(item)
    validation = _validation(item)
    state = build_next_state(item)
    suggested = ""
    recommended = state.get("recommended_next_methods")
    if isinstance(recommended, list) and recommended:
        suggested = _clean_text(recommended[0])
    return {
        "sample_id": _sample_id(item),
        "round": _round_value(item, _previous_state(item)),
        "operator_used": _clean_text(effect.get("operator_used")),
        "invalid_type": _clean_text(validation.get("invalid_type")) or "invalid_complexity",
        "reason": _clean_text(validation.get("reject_reason")) or _clean_text(effect.get("lightweight_hit_reason")),
        "suggested_operator": suggested,
    }


def classify_memory_entries(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    operator_entries: List[Dict[str, Any]] = []
    failure_entries: List[Dict[str, Any]] = []
    invalid_entries: List[Dict[str, Any]] = []

    for record in records:
        effect = _effect(record)
        label = _clean_text(effect.get("effect_label"))
        if effect.get("lightweight_boundary_hit") and effect.get("complexity_passed") and is_question_evolved(record):
            operator_entries.append(build_operator_memory_entry(record))
        if label in FAILURE_EFFECT_LABELS and effect.get("complexity_passed") and is_question_evolved(record):
            failure_entries.append(build_failure_memory_entry(record))
        if label == "invalid_complexity" or effect.get("complexity_passed") is False:
            invalid_entries.append(build_invalid_generation_case(record))

    return operator_entries, failure_entries, invalid_entries


def update_records(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    updated = [attach_next_state(record) for record in records]
    operator_entries, failure_entries, invalid_entries = classify_memory_entries(records)
    return updated, operator_entries, failure_entries, invalid_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update evolution_state and append Stage 5 memory-bank entries.")
    parser.add_argument("--input", required=True, help="Input analyzed JSON/JSONL path.")
    parser.add_argument("--output", required=True, help="Output state-updated JSONL path.")
    parser.add_argument("--memory-dir", default="memory", help="Directory containing memory bank JSONL files.")
    parser.add_argument("--operator-memory", default=None, help="Override operator memory output path.")
    parser.add_argument("--failure-memory", default=None, help="Override failure memory output path.")
    parser.add_argument("--invalid-output", default=None, help="Override invalid generation case output path.")
    parser.add_argument("--no-memory-output", action="store_true", help="Do not append memory-bank entries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_or_jsonl(args.input)
    updated, operator_entries, failure_entries, invalid_entries = update_records(records)
    write_jsonl(updated, args.output)

    if args.no_memory_output:
        return

    operator_memory = args.operator_memory or os.path.join(args.memory_dir, "operator_memory_bank.jsonl")
    failure_memory = args.failure_memory or os.path.join(args.memory_dir, "failure_memory_bank.jsonl")
    invalid_output = args.invalid_output or os.path.join(args.memory_dir, "invalid_generation_cases.jsonl")
    if operator_entries:
        write_jsonl(operator_entries, operator_memory, append=True)
    if failure_entries:
        write_jsonl(failure_entries, failure_memory, append=True)
    if invalid_entries:
        write_jsonl(invalid_entries, invalid_output, append=True)


if __name__ == "__main__":
    main()
