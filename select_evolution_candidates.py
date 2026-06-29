import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from search_state_contract import FRONTIER_ACTION_TYPES, normalize_search_state


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
EXPAND_CURRENT_BRANCH = "expand_current_branch"
FORK_FROM_ROOT = "fork_from_root"
FORK_FROM_PARENT = "fork_from_parent"
STOP_BRANCH = "stop_branch"
STOP_SAMPLE = "stop_sample"
TREE_SEARCH_ACTIONS = FRONTIER_ACTION_TYPES

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
    "validated_high_score_sample",
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


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_axis_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    axes: List[str] = []
    for item in value:
        text = _clean_text(item)
        if text and text not in axes:
            axes.append(text)
    return axes


def _next_best_axes(item: Dict[str, Any]) -> List[str]:
    profile = item.get("sample_profile")
    if isinstance(profile, dict):
        axes = _as_axis_list(profile.get("next_best_axes"))
        if "next_best_axes" in profile:
            return axes
        if axes:
            return axes
        axes = _as_axis_list(profile.get("boundary_axis_candidates"))
        if axes:
            explored = set(_as_axis_list(profile.get("already_explored_axes")))
            return [axis for axis in axes if axis not in explored]

    state = normalize_search_state(item)
    return _as_axis_list(state.get("recommended_next_axes"))


def _target_boundary_axis(item: Dict[str, Any]) -> Optional[str]:
    axes = _next_best_axes(item)
    if axes:
        return axes[0]
    state = normalize_search_state(item)
    return _clean_text(state.get("boundary_axis")) or None


def _source_node_type_for(action_type: str) -> str:
    if action_type == FORK_FROM_ROOT:
        return "root"
    if action_type == FORK_FROM_PARENT:
        return "parent"
    return "current"


def _decision(
    action_type: str,
    reason: str,
    *,
    target_boundary_axis: Optional[str] = None,
    stop_reason: Optional[str] = None,
) -> Dict[str, Any]:
    if action_type not in TREE_SEARCH_ACTIONS:
        raise ValueError(f"unsupported tree search action: {action_type}")
    decision: Dict[str, Any] = {
        "action_type": action_type,
        "branch_intent": action_type,
        "source_node_type": _source_node_type_for(action_type),
        "target_boundary_axis": target_boundary_axis,
        "reason": reason,
    }
    if stop_reason:
        decision["stop_reason"] = stop_reason
    return decision


def decide_tree_search_action(
    item: Dict[str, Any],
    evolution_action: str,
) -> Dict[str, Any]:
    state = normalize_search_state(item)
    next_axes = _next_best_axes(item)
    target_axis = _target_boundary_axis(item)
    branch_budget = int(state.get("branch_budget_remaining", 0) or 0)
    sample_budget = int(state.get("sample_budget_remaining", 0) or 0)
    search_depth = int(state.get("search_depth", 0) or 0)
    max_depth = int(state.get("max_search_depth", 0) or 0)
    branch_status = _clean_text(state.get("branch_status"))
    stop_status = _clean_text(state.get("stop_status"))
    current_axis = _clean_text(state.get("boundary_axis"))
    profile = item.get("sample_profile")
    explored_axes = set(_as_axis_list(state.get("already_explored_axes")))
    if isinstance(profile, dict):
        explored_axes.update(_as_axis_list(profile.get("already_explored_axes")))

    if stop_status in STOP_STATUSES or evolution_action == STOP_EVOLUTION:
        return _decision(
            STOP_SAMPLE,
            "legacy stop status or selector action indicates a terminal sample.",
            target_boundary_axis=target_axis,
            stop_reason=stop_status or evolution_action,
        )

    if sample_budget <= 0:
        return _decision(
            STOP_SAMPLE,
            "sample_budget_remaining is exhausted.",
            target_boundary_axis=target_axis,
            stop_reason="sample_budget_exhausted",
        )

    if not next_axes:
        return _decision(
            STOP_SAMPLE,
            "no unexplored boundary axes are available.",
            target_boundary_axis=target_axis,
            stop_reason="no_unexplored_boundary_axis",
        )

    if evolution_action == PASS_THROUGH_OR_SCORING_NOISE:
        return _decision(
            STOP_BRANCH,
            "current branch is classified as pass-through or scoring noise.",
            target_boundary_axis=target_axis,
            stop_reason="non_evolvable_current_branch",
        )

    if branch_status == "boundary_hit":
        if state.get("parent_node_id") and search_depth > 1:
            return _decision(
                FORK_FROM_PARENT,
                "current branch already hit a boundary; backtrack to parent for sibling branch.",
                target_boundary_axis=target_axis,
            )
        return _decision(
            FORK_FROM_ROOT,
            "current branch already hit a boundary; fork a new root branch for another axis.",
            target_boundary_axis=target_axis,
        )

    if branch_status in {"exhausted", "duplicate", "invalid"}:
        return _decision(
            FORK_FROM_PARENT if state.get("parent_node_id") else FORK_FROM_ROOT,
            f"current branch_status={branch_status} should not keep expanding.",
            target_boundary_axis=target_axis,
        )

    if branch_budget <= 0 or (max_depth > 0 and search_depth >= max_depth):
        return _decision(
            FORK_FROM_PARENT if state.get("parent_node_id") else FORK_FROM_ROOT,
            "current branch depth or branch budget is exhausted; open another branch.",
            target_boundary_axis=target_axis,
        )

    if explored_axes and target_axis and target_axis != current_axis:
        return _decision(
            FORK_FROM_PARENT if state.get("parent_node_id") else FORK_FROM_ROOT,
            "another axis has already been explored; open a separate branch for the next axis.",
            target_boundary_axis=target_axis,
        )

    return _decision(
        EXPAND_CURRENT_BRANCH,
        "current branch is evolvable and still has branch budget.",
        target_boundary_axis=target_axis,
    )


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
    decision = item.get("tree_search_decision")
    decision = decision if isinstance(decision, dict) else {}
    frontier_action = _clean_text(decision.get("action_type"))

    if stop_status in STOP_STATUSES:
        return STOP_EVOLUTION, f"evolution_state.stop_status={stop_status} indicates a terminal state."

    if frontier_action in {EXPAND_CURRENT_BRANCH, FORK_FROM_ROOT, FORK_FROM_PARENT}:
        return EVOLVE_HIGH_SCORE_OVERSCORE, (
            f"tree_search_decision.action_type={frontier_action} requests frontier expansion."
        )

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
    result["tree_search_decision"] = decide_tree_search_action(result, action)
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
