import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


EVOLVE_HIGH_SCORE_OVERSCORE = "evolve_high_score_overscore"
RECONSTRUCT_LOW_SCORE_BOUNDARY = "reconstruct_low_score_boundary"
PASS_THROUGH_OR_SCORING_NOISE = "pass_through_or_scoring_noise"
STOP_EVOLUTION = "stop_evolution"

EVOLUTION_ACTIONS = {
    EVOLVE_HIGH_SCORE_OVERSCORE,
    RECONSTRUCT_LOW_SCORE_BOUNDARY,
    PASS_THROUGH_OR_SCORING_NOISE,
    STOP_EVOLUTION,
}

LOW_SCORE_BOUNDARY_TERMS = (
    "低分真实边界",
    "真实边界",
    "主线抓偏",
    "主线切换",
    "反常线索",
    "抓偏",
    "边界重构",
)
SCORING_NOISE_TERMS = (
    "评分噪声",
    "打分噪声",
    "rubric噪声",
    "rubric 噪声",
    "关键词",
    "格式",
    "负向项",
)
STOP_TERMS = (
    "停止",
    "无需进化",
    "基础边界判断过稳",
    "稳定满分",
    "已稳定",
)
STOP_STATUSES = {
    "stable_high_score_stop",
    "effective_boundary_sample",
    "invalid_complexity_sample",
    "unanswerable_or_trap_sample",
    "stop_evolution",
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


def coerce_score_rate(value: Any) -> Optional[float]:
    try:
        score_rate = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= score_rate <= 1:
        return score_rate
    return None


def get_score_rate(item: Dict[str, Any]) -> Optional[float]:
    top_level = coerce_score_rate(item.get("score_rate"))
    if top_level is not None:
        return top_level

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
    return awarded / possible


def _joined_diagnosis_text(item: Dict[str, Any]) -> str:
    diagnosis = item.get("overscore_diagnosis")
    if not isinstance(diagnosis, dict):
        return ""
    return " ".join(
        str(diagnosis.get(field, ""))
        for field in (
            "candidate_overscore_cause",
            "target_failure_mode",
            "why_high_score_is_suspicious",
        )
    )


def _has_any_term(text: str, terms: Tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _stop_status(item: Dict[str, Any]) -> str:
    state = item.get("evolution_state")
    if not isinstance(state, dict):
        return ""
    return str(state.get("stop_status", "")).strip()


def validate_profiled_record(item: Dict[str, Any]) -> None:
    if not isinstance(item.get("sample_profile"), dict):
        raise ValueError("record missing sample_profile; run profile_samples.py first")
    if not isinstance(item.get("overscore_diagnosis"), dict):
        raise ValueError("record missing overscore_diagnosis; run profile_samples.py first")


def decide_evolution_action(
    item: Dict[str, Any],
    *,
    high_score_threshold: float = 0.8,
    low_score_threshold: float = 0.6,
) -> Tuple[str, str]:
    validate_profiled_record(item)

    diagnosis = item["overscore_diagnosis"]
    worth_evolving = bool(diagnosis.get("is_worth_evolving"))
    score_rate = get_score_rate(item)
    diagnosis_text = _joined_diagnosis_text(item)
    stop_status = _stop_status(item)

    if stop_status in STOP_STATUSES:
        return STOP_EVOLUTION, f"evolution_state.stop_status={stop_status} indicates a terminal state."

    if _has_any_term(diagnosis_text, STOP_TERMS) and not worth_evolving:
        return STOP_EVOLUTION, "diagnosis says the sample is stable or should stop."

    if score_rate is None:
        return PASS_THROUGH_OR_SCORING_NOISE, "score_rate is missing or invalid."

    if score_rate >= high_score_threshold:
        if worth_evolving:
            return EVOLVE_HIGH_SCORE_OVERSCORE, (
                f"score_rate={score_rate:.4f} is high and diagnosis marks the score as worth evolving."
            )
        return PASS_THROUGH_OR_SCORING_NOISE, (
            f"score_rate={score_rate:.4f} is high but diagnosis does not mark a useful overscore."
        )

    if score_rate <= low_score_threshold:
        if worth_evolving and _has_any_term(diagnosis_text, LOW_SCORE_BOUNDARY_TERMS):
            return RECONSTRUCT_LOW_SCORE_BOUNDARY, (
                f"score_rate={score_rate:.4f} is low and diagnosis indicates a real boundary signal."
            )
        if _has_any_term(diagnosis_text, SCORING_NOISE_TERMS):
            return PASS_THROUGH_OR_SCORING_NOISE, "low score appears tied to scoring noise or formatting."
        return PASS_THROUGH_OR_SCORING_NOISE, (
            f"score_rate={score_rate:.4f} is low but diagnosis does not justify boundary reconstruction."
        )

    if worth_evolving:
        return PASS_THROUGH_OR_SCORING_NOISE, (
            f"score_rate={score_rate:.4f} is neither high overscore nor low boundary reconstruction."
        )
    return PASS_THROUGH_OR_SCORING_NOISE, "diagnosis does not mark this sample as worth evolving."


def select_record(
    item: Dict[str, Any],
    *,
    high_score_threshold: float = 0.8,
    low_score_threshold: float = 0.6,
) -> Dict[str, Any]:
    action, reason = decide_evolution_action(
        item,
        high_score_threshold=high_score_threshold,
        low_score_threshold=low_score_threshold,
    )
    result = dict(item)
    result["evolution_action"] = action
    result["evolution_action_reason"] = reason
    return result


def process_records(
    records: List[Dict[str, Any]],
    *,
    high_score_threshold: float = 0.8,
    low_score_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    return [
        select_record(
            record,
            high_score_threshold=high_score_threshold,
            low_score_threshold=low_score_threshold,
        )
        for record in records
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign evolution_action for profiled samples.")
    parser.add_argument("--input", required=True, help="Input profiled JSON/JSONL path.")
    parser.add_argument("--output", required=True, help="Output profiled_candidates JSONL path.")
    parser.add_argument(
        "--high-score-threshold",
        type=float,
        default=0.8,
        help="Minimum score_rate for high-score overscore evolution.",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=0.6,
        help="Maximum score_rate for low-score boundary reconstruction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_or_jsonl(args.input)
    selected = process_records(
        records,
        high_score_threshold=args.high_score_threshold,
        low_score_threshold=args.low_score_threshold,
    )
    write_jsonl(selected, args.output)


if __name__ == "__main__":
    main()
