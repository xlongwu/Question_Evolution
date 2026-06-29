import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from select_evolution_candidates import (
    EVOLVE_HIGH_SCORE_OVERSCORE,
    PASS_THROUGH_OR_SCORING_NOISE,
    RECONSTRUCT_LOW_SCORE_BOUNDARY,
    STOP_EVOLUTION,
    get_score_rate,
)


O1_GAP_CHOICE = "O1_gap_choice"
O2_SUBCLAIM_LOCALIZATION = "O2_subclaim_localization"
O3_STEP_JUMP = "O3_step_jump"
O4_NEAR_LEVEL_RANKING = "O4_near_level_ranking"
O5_EXTRA_PREMISE_DETECTION = "O5_extra_premise_detection"
O6_SINGLE_VARIABLE_COUNTERFACTUAL = "O6_single_variable_counterfactual"
O7_FACT_BINDING_CONSTRAINT = "O7_fact_binding_constraint"
O8_DOUBLE_THRESHOLD_CLAIM = "O8_double_threshold_claim"
O9_ABNORMAL_CLUE_MAINLINE_SWITCH = "O9_abnormal_clue_mainline_switch"

OPERATOR_IDS = {
    O1_GAP_CHOICE,
    O2_SUBCLAIM_LOCALIZATION,
    O3_STEP_JUMP,
    O4_NEAR_LEVEL_RANKING,
    O5_EXTRA_PREMISE_DETECTION,
    O6_SINGLE_VARIABLE_COUNTERFACTUAL,
    O7_FACT_BINDING_CONSTRAINT,
    O8_DOUBLE_THRESHOLD_CLAIM,
    O9_ABNORMAL_CLUE_MAINLINE_SWITCH,
}

EVOLUTION_REQUIRED_ACTIONS = {
    EVOLVE_HIGH_SCORE_OVERSCORE,
    RECONSTRUCT_LOW_SCORE_BOUNDARY,
}

NON_EVOLUTION_ACTIONS = {
    PASS_THROUGH_OR_SCORING_NOISE,
    STOP_EVOLUTION,
}

SIGNATURE_FIELDS = (
    "core_capability",
    "claim_level",
    "problem_shape",
    "candidate_overscore_cause",
)


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


def load_jsonl_if_exists(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _has_any(text: str, terms: Sequence[str]) -> bool:
    return any(term and term in text for term in terms)


def _append_unique(items: List[str], values: Sequence[Optional[str]]) -> None:
    for value in values:
        if value and value not in items:
            items.append(value)


def _remove_values(items: Sequence[str], blocked: Sequence[str]) -> List[str]:
    blocked_set = set(blocked)
    return [item for item in items if item not in blocked_set]


def _normalize_operator(value: Any) -> Optional[str]:
    text = _clean_text(value)
    return text if text in OPERATOR_IDS else None


def get_evolution_action(item: Dict[str, Any]) -> str:
    return _clean_text(item.get("evolution_action"))


def should_route_for_evolution(item: Dict[str, Any]) -> bool:
    return get_evolution_action(item) in EVOLUTION_REQUIRED_ACTIONS


def get_sample_profile(item: Dict[str, Any]) -> Dict[str, Any]:
    profile = item.get("sample_profile")
    if not isinstance(profile, dict):
        raise ValueError("record missing sample_profile; run profile_samples.py first")
    return profile


def get_overscore_diagnosis(item: Dict[str, Any]) -> Dict[str, Any]:
    diagnosis = item.get("overscore_diagnosis")
    if not isinstance(diagnosis, dict):
        raise ValueError("record missing overscore_diagnosis; run profile_samples.py first")
    return diagnosis


def get_evolution_state(item: Dict[str, Any]) -> Dict[str, Any]:
    state = item.get("evolution_state")
    return state if isinstance(state, dict) else {}


def build_sample_signature(item: Dict[str, Any]) -> Dict[str, str]:
    profile = get_sample_profile(item)
    diagnosis = get_overscore_diagnosis(item)
    return {
        "core_capability": _clean_text(profile.get("core_capability")),
        "claim_level": _clean_text(profile.get("claim_level")),
        "problem_shape": _clean_text(profile.get("problem_shape")),
        "candidate_overscore_cause": _clean_text(diagnosis.get("candidate_overscore_cause")),
    }


def signature_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    compared = 0
    matched = 0
    for field in SIGNATURE_FIELDS:
        left = _clean_text(a.get(field))
        right = _clean_text(b.get(field))
        if not left or not right:
            continue
        compared += 1
        if left == right:
            matched += 1
    if compared == 0:
        return 0.0
    return matched / compared


def find_memory_matches(
    signature: Dict[str, str],
    memory_records: Sequence[Dict[str, Any]],
    *,
    min_similarity: float = 0.75,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for record in memory_records:
        memory_signature = record.get("sample_signature")
        if not isinstance(memory_signature, dict):
            continue
        similarity = signature_similarity(signature, memory_signature)
        if similarity >= min_similarity:
            match = dict(record)
            match["signature_similarity"] = similarity
            matches.append(match)
    matches.sort(key=lambda item: item.get("signature_similarity", 0), reverse=True)
    return matches


def _base_rule_route(item: Dict[str, Any]) -> Tuple[Optional[str], List[str], str]:
    diagnosis = get_overscore_diagnosis(item)
    cause = _clean_text(diagnosis.get("candidate_overscore_cause"))
    target = _clean_text(diagnosis.get("target_failure_mode"))
    combined = f"{cause} {target}"

    if _has_any(target, ("反常线索主线切换失败", "主线切换")):
        return (
            O9_ABNORMAL_CLUE_MAINLINE_SWITCH,
            [O6_SINGLE_VARIABLE_COUNTERFACTUAL],
            "target_failure_mode indicates abnormal-clue mainline switching.",
        )

    if _has_any(cause, ("漏最小关键事实", "最小关键事实", "最关键缺口")):
        return (
            O1_GAP_CHOICE,
            [O2_SUBCLAIM_LOCALIZATION],
            "candidate_overscore_cause maps to gap-choice routing.",
        )

    if _has_any(cause, ("层级越推", "线索升级", "层级混淆")):
        return (
            O3_STEP_JUMP,
            [O4_NEAR_LEVEL_RANKING],
            "candidate_overscore_cause maps to step-jump routing.",
        )

    if _has_any(cause, ("题外补设", "题干外", "隐藏前提")):
        return (
            O5_EXTRA_PREMISE_DETECTION,
            [],
            "candidate_overscore_cause maps to extra-premise detection.",
        )

    if _has_any(cause, ("泛化罗列", "套话", "事实绑定")):
        return (
            O7_FACT_BINDING_CONSTRAINT,
            [],
            "candidate_overscore_cause maps to fact-binding constraint.",
        )

    if _has_any(cause, ("抓显眼点漏关键层", "双门槛", "漏关键层")):
        return (
            O8_DOUBLE_THRESHOLD_CLAIM,
            [O2_SUBCLAIM_LOCALIZATION],
            "candidate_overscore_cause maps to double-threshold routing.",
        )

    if _has_any(combined, ("近似项分层", "判据内", "判据外", "相关但不可用")):
        return (
            O4_NEAR_LEVEL_RANKING,
            [O5_EXTRA_PREMISE_DETECTION],
            "diagnosis indicates near-level or criterion-boundary ranking.",
        )

    if _has_any(cause, ("受干扰信息带偏", "干扰信息")):
        return (
            O6_SINGLE_VARIABLE_COUNTERFACTUAL,
            [O9_ABNORMAL_CLUE_MAINLINE_SWITCH, O4_NEAR_LEVEL_RANKING],
            "candidate_overscore_cause maps to counterfactual or mainline-switch routing.",
        )

    return (
        O2_SUBCLAIM_LOCALIZATION,
        [O4_NEAR_LEVEL_RANKING],
        "fallback to subclaim localization for evolvable sample.",
    )


def _previous_operator(item: Dict[str, Any]) -> Optional[str]:
    state = get_evolution_state(item)
    operator = _normalize_operator(state.get("previous_operator"))
    if operator:
        return operator

    meta_info = item.get("meta_info")
    if isinstance(meta_info, dict):
        metadata = meta_info.get("question_evolution_metadata")
        if isinstance(metadata, dict):
            return _normalize_operator(metadata.get("operator_used"))
    return None


def _recommended_next_methods(item: Dict[str, Any]) -> List[str]:
    state = get_evolution_state(item)
    values = state.get("recommended_next_methods")
    if not isinstance(values, list):
        return []
    operators: List[str] = []
    for value in values:
        operator = _normalize_operator(value)
        if operator and operator not in operators:
            operators.append(operator)
    return operators


def _is_current_full_score(item: Dict[str, Any], full_score_threshold: float) -> bool:
    score_rate = get_score_rate(item)
    if score_rate is None:
        return False
    return score_rate >= full_score_threshold


def _is_high_value_sample(item: Dict[str, Any]) -> bool:
    diagnosis = get_overscore_diagnosis(item)
    profile = get_sample_profile(item)
    action = get_evolution_action(item)
    cause = _clean_text(diagnosis.get("candidate_overscore_cause"))
    target = _clean_text(diagnosis.get("target_failure_mode"))
    return (
        action in EVOLUTION_REQUIRED_ACTIONS
        and bool(diagnosis.get("is_worth_evolving"))
        and _clean_text(profile.get("external_knowledge_risk")).lower() != "high"
        and _has_any(
            f"{cause} {target}",
            (
                "漏最小关键事实",
                "选错最关键缺口",
                "抓显眼点漏关键层",
                "近似项分层",
                "反常线索主线切换失败",
                "主线抓偏",
            ),
        )
    )


def build_operator_route(
    item: Dict[str, Any],
    *,
    operator_memory: Sequence[Dict[str, Any]] = (),
    failure_memory: Sequence[Dict[str, Any]] = (),
    full_score_threshold: float = 0.99,
) -> Dict[str, Any]:
    action = get_evolution_action(item)
    if action in NON_EVOLUTION_ACTIONS:
        return {
            "primary_operator": None,
            "backup_operators": [],
            "avoid_operators": [],
            "routing_reason": f"evolution_action={action} does not require question evolution.",
            "is_high_value_sample": False,
            "should_use_local_tree_search": False,
            "memory_matches": {"operator": [], "failure": []},
        }
    if action and action not in EVOLUTION_REQUIRED_ACTIONS:
        raise ValueError(f"unsupported evolution_action: {action}")

    get_sample_profile(item)
    get_overscore_diagnosis(item)

    primary, backups, reason = _base_rule_route(item)
    avoid: List[str] = []
    reason_parts = [reason]
    recommended_next = _recommended_next_methods(item)

    signature = build_sample_signature(item)
    operator_matches = find_memory_matches(signature, operator_memory)
    failure_matches = find_memory_matches(signature, failure_memory)

    for match in failure_matches:
        failed_operator = _normalize_operator(match.get("operator_used"))
        if failed_operator:
            _append_unique(avoid, [failed_operator])

    if operator_matches:
        memory_operator = _normalize_operator(operator_matches[0].get("operator_used"))
        if memory_operator and memory_operator not in avoid:
            if primary and primary != memory_operator:
                _append_unique(backups, [primary])
                reason_parts.append(
                    f"operator memory promotes {memory_operator} over rule primary {primary}."
                )
            primary = memory_operator

    previous_operator = _previous_operator(item)
    if previous_operator == O1_GAP_CHOICE and _is_current_full_score(item, full_score_threshold):
        _append_unique(avoid, [O1_GAP_CHOICE])
        if primary == O1_GAP_CHOICE:
            primary = O2_SUBCLAIM_LOCALIZATION
            backups = [O4_NEAR_LEVEL_RANKING, O8_DOUBLE_THRESHOLD_CLAIM] + backups
        else:
            _append_unique(backups, [O2_SUBCLAIM_LOCALIZATION, O4_NEAR_LEVEL_RANKING, O8_DOUBLE_THRESHOLD_CLAIM])
        reason_parts.append("previous O1 full-score result blocks repeating O1.")

    if recommended_next:
        ordered_candidates: List[str] = []
        _append_unique(ordered_candidates, recommended_next)
        _append_unique(ordered_candidates, [primary])
        _append_unique(ordered_candidates, backups)
        ordered_candidates = _remove_values(ordered_candidates, avoid)
        if ordered_candidates:
            primary = ordered_candidates[0]
            backups = ordered_candidates[1:]
            reason_parts.append(
                "recommended_next_methods from evolution_state are prioritized before fallback rule routing."
            )

    backups = _remove_values(backups, [primary] if primary else [])
    backups = _remove_values(backups, avoid)
    if primary in avoid:
        replacement = next((operator for operator in backups if operator not in avoid), None)
        if replacement:
            primary = replacement
            backups = _remove_values(backups, [primary])
        else:
            primary = O2_SUBCLAIM_LOCALIZATION if O2_SUBCLAIM_LOCALIZATION not in avoid else None

    consecutive_full = int(get_evolution_state(item).get("consecutive_full_score_count", 0) or 0)
    should_tree = (
        _is_high_value_sample(item)
        or action == RECONSTRUCT_LOW_SCORE_BOUNDARY
        or consecutive_full >= 2
    )

    return {
        "primary_operator": primary,
        "backup_operators": backups,
        "avoid_operators": avoid,
        "routing_reason": " ".join(reason_parts),
        "is_high_value_sample": _is_high_value_sample(item),
        "should_use_local_tree_search": should_tree,
        "memory_matches": {
            "operator": operator_matches[:3],
            "failure": failure_matches[:3],
        },
    }


def attach_operator_route(
    item: Dict[str, Any],
    *,
    operator_memory: Sequence[Dict[str, Any]] = (),
    failure_memory: Sequence[Dict[str, Any]] = (),
    full_score_threshold: float = 0.99,
) -> Dict[str, Any]:
    result = dict(item)
    result["operator_route"] = build_operator_route(
        item,
        operator_memory=operator_memory,
        failure_memory=failure_memory,
        full_score_threshold=full_score_threshold,
    )
    return result


def route_records(
    records: Sequence[Dict[str, Any]],
    *,
    operator_memory: Sequence[Dict[str, Any]] = (),
    failure_memory: Sequence[Dict[str, Any]] = (),
    full_score_threshold: float = 0.99,
) -> List[Dict[str, Any]]:
    return [
        attach_operator_route(
            record,
            operator_memory=operator_memory,
            failure_memory=failure_memory,
            full_score_threshold=full_score_threshold,
        )
        for record in records
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route profiled evolution candidates to question operators.")
    parser.add_argument("--input", required=True, help="Input profiled_candidates JSON/JSONL path.")
    parser.add_argument("--output", required=True, help="Output routed JSONL path.")
    parser.add_argument("--memory-dir", default="memory", help="Directory containing memory bank JSONL files.")
    parser.add_argument("--operator-memory", default=None, help="Override operator memory JSONL path.")
    parser.add_argument("--failure-memory", default=None, help="Override failure memory JSONL path.")
    parser.add_argument(
        "--full-score-threshold",
        type=float,
        default=0.99,
        help="Score-rate threshold used by no-repeat rules.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    operator_memory_path = args.operator_memory or os.path.join(args.memory_dir, "operator_memory_bank.jsonl")
    failure_memory_path = args.failure_memory or os.path.join(args.memory_dir, "failure_memory_bank.jsonl")
    records = load_json_or_jsonl(args.input)
    routed = route_records(
        records,
        operator_memory=load_jsonl_if_exists(operator_memory_path),
        failure_memory=load_jsonl_if_exists(failure_memory_path),
        full_score_threshold=args.full_score_threshold,
    )
    write_jsonl(routed, args.output)


if __name__ == "__main__":
    main()
