import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_FULL_SCORE_THRESHOLD = 0.99
DEFAULT_SCORE_DROP_THRESHOLD = 0.15
DEFAULT_REVIEW_DROP_THRESHOLD = 0.05
DEFAULT_SCORE_INCREASE_THRESHOLD = 0.05
FOCUS_STOPWORDS = {
    "是否",
    "判断",
    "说明",
    "指出",
    "真正",
    "当前",
    "仍然",
    "没有",
    "不能",
    "可以",
    "需要",
    "一个",
    "什么",
    "为什么",
}


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


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_score_rate(value: Any) -> Optional[float]:
    try:
        score_rate = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= score_rate <= 1:
        return score_rate
    return None


def record_key(item: Dict[str, Any]) -> str:
    for field in ("sample_id", "index"):
        value = item.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return _clean_text(item.get("prompt"))


def records_by_key(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {record_key(record): record for record in records if record_key(record)}


def get_score_rate(item: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    top_level_score_rate = _coerce_score_rate(item.get("score_rate"))
    if top_level_score_rate is not None:
        return top_level_score_rate

    scoring_result = item.get("scoring_result")
    if not isinstance(scoring_result, dict):
        return None
    try:
        awarded = float(scoring_result.get("total_awarded", 0) or 0)
        possible = float(scoring_result.get("total_possible", 0) or 0)
    except (TypeError, ValueError):
        return None
    if possible <= 0:
        return None
    return max(0.0, min(1.0, awarded / possible))


def get_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    meta_info = item.get("meta_info")
    if not isinstance(meta_info, dict):
        return {}
    metadata = meta_info.get("question_evolution_metadata")
    return metadata if isinstance(metadata, dict) else {}


def get_validation_result(item: Dict[str, Any]) -> Dict[str, Any]:
    validation = item.get("validation_result")
    return validation if isinstance(validation, dict) else {}


def is_question_evolved(item: Dict[str, Any]) -> bool:
    if item.get("question_evolved") is True:
        return True
    if item.get("question_evolved") is False:
        return False
    return bool(get_metadata(item).get("question_evolved"))


def get_score_rate_before(item: Dict[str, Any], previous_item: Optional[Dict[str, Any]]) -> Optional[float]:
    metadata = get_metadata(item)
    trigger_score_rate = _coerce_score_rate(metadata.get("trigger_score_rate"))
    if trigger_score_rate is not None:
        return trigger_score_rate

    previous_score_rate = get_score_rate(previous_item)
    if previous_score_rate is not None:
        return previous_score_rate

    meta_info = item.get("meta_info")
    if isinstance(meta_info, dict):
        stale_scoring = meta_info.get("stale_scoring_result")
        if isinstance(stale_scoring, dict):
            stale_score_rate = get_score_rate({"scoring_result": stale_scoring})
            if stale_score_rate is not None:
                return stale_score_rate

    state = item.get("evolution_state")
    if isinstance(state, dict):
        state_score_rate = _coerce_score_rate(state.get("previous_score_rate"))
        if state_score_rate is not None:
            return state_score_rate

    return get_score_rate(item)


def get_operator_used(item: Dict[str, Any]) -> str:
    metadata = get_metadata(item)
    for value in (
        metadata.get("operator_used"),
        item.get("candidate_operator"),
        item.get("operator_used"),
    ):
        text = _clean_text(value)
        if text:
            return text

    selection = item.get("candidate_selection")
    if isinstance(selection, dict):
        return _clean_text(selection.get("selected_operator"))
    return ""


def get_expected_focus(item: Dict[str, Any]) -> List[str]:
    focus = get_metadata(item).get("expected_evaluation_focus")
    if isinstance(focus, list):
        return [_clean_text(value) for value in focus if _clean_text(value)]
    if isinstance(focus, str) and focus.strip():
        return [focus.strip()]
    return []


def get_candidate_answer(item: Dict[str, Any]) -> str:
    scoring_result = item.get("scoring_result")
    if isinstance(scoring_result, dict):
        return _clean_text(scoring_result.get("candidate_answer"))
    return _clean_text(item.get("candidate_answer"))


def has_candidate_answer(item: Dict[str, Any]) -> bool:
    return bool(get_candidate_answer(item))


def _focus_terms(texts: Sequence[str]) -> List[str]:
    terms: List[str] = []
    for text in texts:
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text):
            token = token.strip()
            if token and token not in FOCUS_STOPWORDS and token not in terms:
                terms.append(token)
            if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
                for size in (4, 3, 2):
                    for start in range(0, len(token) - size + 1):
                        gram = token[start:start + size]
                        if gram and gram not in FOCUS_STOPWORDS and gram not in terms:
                            terms.append(gram)
    return terms


def analyze_focus_answer_alignment(focus: Sequence[str], candidate_answer: str) -> Dict[str, Any]:
    focus_terms = _focus_terms(focus)
    answer_terms = set(_focus_terms([candidate_answer]))
    matched_terms = [term for term in focus_terms if term in candidate_answer or term in answer_terms]
    if not focus_terms:
        return {
            "matches": False,
            "confidence": "low",
            "matched_terms": [],
            "reason": "缺少 expected_evaluation_focus，无法确认候选答案错误方向。",
        }
    if not candidate_answer.strip():
        return {
            "matches": False,
            "confidence": "low",
            "matched_terms": [],
            "reason": "缺少 candidate_answer，无法确认错误方向。",
        }

    coverage = len(matched_terms) / max(1, len(focus_terms))
    if len(matched_terms) >= 2 or coverage >= 0.35:
        confidence = "medium" if coverage < 0.65 else "high"
        return {
            "matches": True,
            "confidence": confidence,
            "matched_terms": matched_terms,
            "reason": "candidate_answer 的错误表述与 expected_evaluation_focus 存在关键词和语义方向重合。",
        }

    return {
        "matches": False,
        "confidence": "low",
        "matched_terms": matched_terms,
        "reason": "candidate_answer 的主要错误方向未能匹配 expected_evaluation_focus。",
    }


def validation_passed_for_effect(item: Dict[str, Any]) -> bool:
    if not is_question_evolved(item):
        return True
    validation = get_validation_result(item)
    if not validation:
        return False
    return validation.get("passed") is True


def is_repeated_pattern(item: Dict[str, Any]) -> bool:
    validation = get_validation_result(item)
    return validation.get("repeat_pattern_risk") == "high" or bool(
        validation.get("repeated_pattern_with_previous_round")
    )


def _hit_confidence(
    score_drop: float,
    *,
    focus_matches: bool,
    answer_present: bool,
    focus_alignment_confidence: str,
    score_drop_threshold: float,
) -> str:
    if (
        score_drop >= max(0.30, score_drop_threshold * 2)
        and focus_matches
        and answer_present
        and focus_alignment_confidence in {"medium", "high"}
    ):
        return "high"
    if score_drop >= score_drop_threshold and focus_matches:
        return "medium"
    return "low"


def build_effect_analysis(
    item: Dict[str, Any],
    previous_item: Optional[Dict[str, Any]] = None,
    *,
    full_score_threshold: float = DEFAULT_FULL_SCORE_THRESHOLD,
    score_drop_threshold: float = DEFAULT_SCORE_DROP_THRESHOLD,
    review_drop_threshold: float = DEFAULT_REVIEW_DROP_THRESHOLD,
    score_increase_threshold: float = DEFAULT_SCORE_INCREASE_THRESHOLD,
) -> Dict[str, Any]:
    score_rate_after = get_score_rate(item)
    score_rate_before = get_score_rate_before(item, previous_item)
    if score_rate_after is None:
        raise ValueError(f"record {record_key(item)!r} missing score_rate_after")
    if score_rate_before is None:
        raise ValueError(f"record {record_key(item)!r} missing score_rate_before")

    delta_score_rate = score_rate_after - score_rate_before
    score_drop = score_rate_before - score_rate_after
    evolved = is_question_evolved(item)
    complexity_passed = validation_passed_for_effect(item)
    repeated = is_repeated_pattern(item)
    focus = get_expected_focus(item)
    focus_present = bool(focus)
    candidate_answer = get_candidate_answer(item)
    answer_present = bool(candidate_answer)
    focus_alignment = analyze_focus_answer_alignment(focus, candidate_answer)
    focus_matches = bool(focus_alignment.get("matches"))
    is_full_score = score_rate_after >= full_score_threshold
    full_score_broken = score_rate_before >= full_score_threshold and score_rate_after < full_score_threshold
    strong_drop = score_drop >= score_drop_threshold
    review_drop = score_drop >= review_drop_threshold

    lightweight_boundary_hit = (
        evolved
        and complexity_passed
        and not repeated
        and focus_present
        and focus_matches
        and (strong_drop or full_score_broken or review_drop)
    )
    confidence = (
        _hit_confidence(
            score_drop,
            focus_matches=focus_matches,
            answer_present=answer_present,
            focus_alignment_confidence=_clean_text(focus_alignment.get("confidence")),
            score_drop_threshold=score_drop_threshold,
        )
        if lightweight_boundary_hit
        else "low"
    )

    if evolved and not complexity_passed:
        effect_label = "invalid_complexity"
        reason = "候选题未通过复杂度或可回答性校验。"
    elif not evolved:
        effect_label = "pass_through"
        reason = "透传样本未进入 question evolution。"
    elif repeated:
        effect_label = "repeated_pattern"
        reason = "题型与上一轮重复，不应作为有效边界命中沉淀。"
    elif lightweight_boundary_hit and confidence in {"medium", "high"}:
        effect_label = "effective_boundary_probe"
        reason = "题目通过复杂度校验、分数下降，且 candidate_answer 错误方向与 expected_evaluation_focus 基本一致；仍需人工复核。"
    elif lightweight_boundary_hit:
        effect_label = "needs_manual_review"
        reason = "题目通过复杂度校验但分数下降幅度较小，仅能作为低置信命中候选。"
    elif evolved and complexity_passed and (strong_drop or full_score_broken or review_drop) and (not focus_present or not focus_matches):
        effect_label = "needs_manual_review"
        reason = _clean_text(focus_alignment.get("reason")) or "分数下降但无法确认错误方向压中预期 focus。"
    elif is_full_score:
        effect_label = "full_score_no_drop"
        reason = "新一轮评分仍为满分，当前算子未形成有效压测。"
    elif delta_score_rate > score_increase_threshold:
        effect_label = "score_increased"
        reason = "新一轮得分率升高，当前改写未带来更清晰边界。"
    else:
        effect_label = "no_clear_effect"
        reason = "未观察到足够清晰的得分变化。"

    return {
        "score_rate_before": score_rate_before,
        "score_rate_after": score_rate_after,
        "delta_score_rate": delta_score_rate,
        "operator_used": get_operator_used(item),
        "question_length": len(_clean_text(item.get("prompt"))),
        "is_full_score": is_full_score,
        "complexity_passed": complexity_passed,
        "repeated_pattern_with_previous_round": repeated,
        "lightweight_boundary_hit": lightweight_boundary_hit,
        "hit_confidence": confidence,
        "needs_manual_review": bool(lightweight_boundary_hit) or effect_label == "needs_manual_review",
        "focus_answer_alignment": focus_alignment,
        "lightweight_hit_reason": reason,
        "effect_label": effect_label,
    }


def attach_effect_analysis(
    item: Dict[str, Any],
    previous_item: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    result = dict(item)
    result["effect_analysis"] = build_effect_analysis(item, previous_item, **kwargs)
    return result


def analyze_records(
    records: Sequence[Dict[str, Any]],
    *,
    previous_records: Optional[Sequence[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    previous_by_key = records_by_key(previous_records or [])
    analyzed: List[Dict[str, Any]] = []
    for record in records:
        analyzed.append(
            attach_effect_analysis(
                record,
                previous_by_key.get(record_key(record)),
                **kwargs,
            )
        )
    return analyzed


def _signature_field(item: Dict[str, Any], source: str, field: str) -> str:
    value = item.get(source)
    if not isinstance(value, dict):
        return ""
    return _clean_text(value.get(field))


def build_effect_matrix(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: DefaultDict[Tuple[str, str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "sample_count": 0,
            "delta_score_rate_sum": 0.0,
            "lightweight_boundary_hit_count": 0,
            "full_score_count": 0,
            "invalid_complexity_count": 0,
            "repeated_pattern_count": 0,
        }
    )
    for record in records:
        effect = record.get("effect_analysis")
        if not isinstance(effect, dict):
            continue
        key = (
            _signature_field(record, "sample_profile", "core_capability"),
            _signature_field(record, "overscore_diagnosis", "candidate_overscore_cause"),
            _signature_field(record, "overscore_diagnosis", "target_failure_mode"),
            _clean_text(effect.get("operator_used")),
        )
        bucket = grouped[key]
        bucket["sample_count"] += 1
        bucket["delta_score_rate_sum"] += float(effect.get("delta_score_rate", 0) or 0)
        if effect.get("lightweight_boundary_hit"):
            bucket["lightweight_boundary_hit_count"] += 1
        if effect.get("is_full_score"):
            bucket["full_score_count"] += 1
        if not effect.get("complexity_passed"):
            bucket["invalid_complexity_count"] += 1
        if effect.get("repeated_pattern_with_previous_round"):
            bucket["repeated_pattern_count"] += 1

    matrix: List[Dict[str, Any]] = []
    for (
        core_capability,
        candidate_overscore_cause,
        target_failure_mode,
        operator_used,
    ), bucket in sorted(grouped.items()):
        sample_count = bucket["sample_count"]
        hit_count = bucket["lightweight_boundary_hit_count"]
        matrix.append(
            {
                "core_capability": core_capability,
                "candidate_overscore_cause": candidate_overscore_cause,
                "target_failure_mode": target_failure_mode,
                "operator_used": operator_used,
                "sample_count": sample_count,
                "avg_delta_score_rate": bucket["delta_score_rate_sum"] / sample_count if sample_count else 0,
                "lightweight_boundary_hit_count": hit_count,
                "lightweight_boundary_hit_rate": hit_count / sample_count if sample_count else 0,
                "full_score_count": bucket["full_score_count"],
                "invalid_complexity_count": bucket["invalid_complexity_count"],
                "repeated_pattern_count": bucket["repeated_pattern_count"],
            }
        )
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze lightweight question-evolution effects after scoring.")
    parser.add_argument("--input", required=True, help="Input current scored JSON/JSONL path.")
    parser.add_argument("--output", required=True, help="Output analyzed JSONL path.")
    parser.add_argument("--before", default=None, help="Optional previous scored JSON/JSONL path for score_rate_before.")
    parser.add_argument("--matrix-output", default=None, help="Optional Sample Type x Operator matrix JSONL path.")
    parser.add_argument("--full-score-threshold", type=float, default=DEFAULT_FULL_SCORE_THRESHOLD)
    parser.add_argument("--score-drop-threshold", type=float, default=DEFAULT_SCORE_DROP_THRESHOLD)
    parser.add_argument("--review-drop-threshold", type=float, default=DEFAULT_REVIEW_DROP_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_or_jsonl(args.input)
    previous_records = load_json_or_jsonl(args.before) if args.before else None
    analyzed = analyze_records(
        records,
        previous_records=previous_records,
        full_score_threshold=args.full_score_threshold,
        score_drop_threshold=args.score_drop_threshold,
        review_drop_threshold=args.review_drop_threshold,
    )
    write_jsonl(analyzed, args.output)
    if args.matrix_output:
        write_jsonl(build_effect_matrix(analyzed), args.matrix_output)


if __name__ == "__main__":
    main()
