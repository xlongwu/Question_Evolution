import argparse
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analyze_evolution_effect import (
    get_metadata,
    get_operator_used,
    is_question_evolved,
    load_json_or_jsonl,
)
from search_state_contract import (
    DEFAULT_BOUNDARY_AXES,
    TREE_SEARCH_CONFIG_DEFAULTS,
    get_question_evolution_metadata,
    get_root_prompt,
    normalize_search_state,
    sample_key,
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

TREE_SAMPLE_TERMINAL_STOP_STATUSES = {
    "stop_sample",
    "sample_budget_exhausted",
    "sample_boundary_limit_reached",
    "sample_branch_limit_reached",
    "no_new_boundary_stop",
    "recommended_axes_exhausted_stop",
}

BRANCH_TERMINAL_STATUSES = {
    "boundary_hit",
    "exhausted",
    "duplicate",
    "invalid",
}

NEW_BRANCH_ACTIONS = {
    "fork_from_root",
    "fork_from_parent",
}

ALLOW_DEFAULT_AXIS_FALLBACK_CONFIG = "ALLOW_DEFAULT_AXIS_FALLBACK_AFTER_RECOMMENDATION_EXHAUSTED"

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


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = _clean_text(value).lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_non_negative_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _tree_config(overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config = dict(TREE_SEARCH_CONFIG_DEFAULTS)
    for key, default in TREE_SEARCH_CONFIG_DEFAULTS.items():
        env_value = os.getenv(key)
        if env_value is None:
            continue
        if isinstance(default, bool):
            config[key] = _as_bool(env_value, default)
        elif isinstance(default, int):
            config[key] = _as_non_negative_int(env_value, int(default))
        else:
            config[key] = env_value
    if overrides:
        for key, value in overrides.items():
            if key not in config:
                config[key] = value
                continue
            default = TREE_SEARCH_CONFIG_DEFAULTS.get(key)
            if isinstance(default, bool):
                config[key] = _as_bool(value, bool(default))
            elif isinstance(default, int):
                config[key] = _as_non_negative_int(value, int(default))
            else:
                config[key] = value
    return config


def _short_hash(value: Any) -> str:
    return hashlib.sha1(_clean_text(value).encode("utf-8")).hexdigest()[:10]


def _frontier_id(source_node_id: str, action_type: str, target_axis: str) -> str:
    suffix = _short_hash(f"{source_node_id}|{action_type}|{target_axis}")
    return f"frontier_{source_node_id}_{action_type}_{suffix}"


def _int_state(state: Mapping[str, Any], field: str, default: int = 0) -> int:
    return _as_non_negative_int(state.get(field), default)


def _list_state(state: Mapping[str, Any], field: str) -> List[Any]:
    values = state.get(field)
    return list(values) if isinstance(values, list) else []


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


def _effect_axis(item: Dict[str, Any], state: Mapping[str, Any]) -> Optional[str]:
    effect = _effect(item)
    metadata = get_question_evolution_metadata(item)
    for value in (
        effect.get("boundary_axis_detected"),
        effect.get("target_boundary_axis"),
        metadata.get("boundary_axis"),
        metadata.get("target_boundary_axis"),
        state.get("boundary_axis"),
    ):
        text = _clean_text(value)
        if text:
            return text
    return None


def _candidate_generation(item: Dict[str, Any]) -> Dict[str, Any]:
    generation = item.get("candidate_generation")
    return generation if isinstance(generation, dict) else {}


def _tree_decision(item: Dict[str, Any]) -> Dict[str, Any]:
    decision = item.get("tree_search_decision")
    return decision if isinstance(decision, dict) else {}


def _operator_route(item: Dict[str, Any]) -> Dict[str, Any]:
    route = item.get("operator_route")
    return route if isinstance(route, dict) else {}


def _generation_action(item: Dict[str, Any]) -> str:
    metadata = get_question_evolution_metadata(item)
    generation = _candidate_generation(item)
    decision = _tree_decision(item)
    route = _operator_route(item)
    for value in (
        generation.get("generation_action"),
        metadata.get("generation_action"),
        decision.get("action_type"),
        route.get("branch_action"),
    ):
        text = _clean_text(value)
        if text:
            return text
    return "expand_current_branch"


def _new_boundary_entry(item: Dict[str, Any], state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    effect = _effect(item)
    label = _clean_text(effect.get("effect_label"))
    duplicate = bool(effect.get("duplicate_boundary_for_sample") or item.get("discard_as_duplicate"))
    is_new_boundary = bool(effect.get("is_new_boundary_for_sample"))
    if not is_new_boundary and not (label == "effective_boundary_probe" and not duplicate):
        return None

    axis = _effect_axis(item, state)
    if not axis:
        return None
    dedup_signature = _clean_text(effect.get("dedup_signature") or item.get("dedup_signature"))
    current_node_id = _clean_text(state.get("current_node_id"))
    boundary_id_source = f"{state.get('search_root_id')}|{current_node_id}|{axis}|{dedup_signature}"
    return {
        "boundary_id": f"boundary_{_short_hash(boundary_id_source)}",
        "boundary_axis": axis,
        "target_boundary_axis": effect.get("target_boundary_axis") or axis,
        "trigger_node_id": current_node_id,
        "branch_id": state.get("branch_id"),
        "effect_label": label or None,
        "dedup_signature": dedup_signature or _short_hash(boundary_id_source),
        "score_rate_before": effect.get("score_rate_before"),
        "score_rate_after": effect.get("score_rate_after"),
        "hit_confidence": effect.get("hit_confidence"),
    }


def _append_boundary(
    discovered_boundaries: Sequence[Any],
    boundary: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized = [dict(item) for item in discovered_boundaries if isinstance(item, dict)]
    if not boundary:
        return normalized
    signature = _clean_text(boundary.get("dedup_signature"))
    trigger_node_id = _clean_text(boundary.get("trigger_node_id"))
    for existing in normalized:
        existing_signature = _clean_text(existing.get("dedup_signature"))
        existing_trigger = _clean_text(existing.get("trigger_node_id"))
        if signature and existing_signature == signature:
            return normalized
        if trigger_node_id and existing_trigger == trigger_node_id:
            return normalized
    normalized.append(boundary)
    return normalized


def _append_axis(axes: Sequence[Any], axis: Optional[str]) -> List[str]:
    result = [_clean_text(axis_value) for axis_value in axes if _clean_text(axis_value)]
    axis_text = _clean_text(axis)
    if axis_text and axis_text not in result:
        result.append(axis_text)
    return result


def _has_explicit_recommended_axes(state: Mapping[str, Any]) -> bool:
    return any(_clean_text(axis) for axis in _list_state(state, "recommended_next_axes"))


def _filtered_next_axes(
    state: Mapping[str, Any],
    current_axis: Optional[str] = None,
    *,
    allow_default_fallback: bool = True,
) -> List[str]:
    explored = set(_clean_text(axis) for axis in _list_state(state, "already_explored_axes") if _clean_text(axis))
    current_axis_text = _clean_text(current_axis)
    axes = []
    for axis in _list_state(state, "recommended_next_axes"):
        axis_text = _clean_text(axis)
        if axis_text and axis_text not in explored and axis_text not in axes:
            axes.append(axis_text)
    if not axes and (allow_default_fallback or not _has_explicit_recommended_axes(state)):
        for axis in DEFAULT_BOUNDARY_AXES:
            if axis not in explored and axis != current_axis_text:
                axes.append(axis)
    return axes


def _opened_branch_ids(state: Mapping[str, Any]) -> List[str]:
    branch_ids: List[str] = []
    for value in _list_state(state, "opened_branch_ids"):
        branch_id = _clean_text(value)
        if branch_id and branch_id not in branch_ids:
            branch_ids.append(branch_id)

    branch_id = _clean_text(state.get("branch_id"))
    if branch_id and branch_id not in branch_ids:
        branch_ids.append(branch_id)

    for boundary in _list_state(state, "discovered_boundaries"):
        if not isinstance(boundary, Mapping):
            continue
        boundary_branch_id = _clean_text(boundary.get("branch_id"))
        if boundary_branch_id and boundary_branch_id not in branch_ids:
            branch_ids.append(boundary_branch_id)
    return branch_ids


def _max_sample_branches(config: Mapping[str, Any]) -> int:
    return int(config.get("MAX_SAMPLE_BRANCHES", TREE_SEARCH_CONFIG_DEFAULTS["MAX_SAMPLE_BRANCHES"]) or 0)


def _sample_branch_limit_reached(state: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    max_branches = _max_sample_branches(config)
    return max_branches > 0 and len(_opened_branch_ids(state)) >= max_branches


def _allow_default_axis_fallback(config: Mapping[str, Any]) -> bool:
    return _as_bool(config.get(ALLOW_DEFAULT_AXIS_FALLBACK_CONFIG), False)


def _candidate_homogeneity_detected(item: Dict[str, Any]) -> bool:
    effect = _effect(item)
    validation = _validation(item)
    return (
        _clean_text(effect.get("effect_label")) == "repeated_pattern"
        or _clean_text(validation.get("repeat_pattern_risk")) == "high"
        or bool(validation.get("repeated_pattern_with_previous_round"))
    )


def _is_sample_terminal(state: Mapping[str, Any]) -> bool:
    return _clean_text(state.get("stop_status")) in TREE_SAMPLE_TERMINAL_STOP_STATUSES


def _tree_branch_status(
    item: Dict[str, Any],
    state: Mapping[str, Any],
    *,
    branch_budget_remaining: int,
    no_new_boundary_rounds: int,
    config: Mapping[str, Any],
) -> str:
    effect = _effect(item)
    label = _clean_text(effect.get("effect_label"))
    validation = _validation(item)
    duplicate = bool(effect.get("duplicate_boundary_for_sample") or item.get("discard_as_duplicate"))
    max_depth = _int_state(state, "max_search_depth", int(config["MAX_SAMPLE_DEPTH"]))
    search_depth = _int_state(state, "search_depth", 0)
    max_no_new = int(config["MAX_NO_NEW_BOUNDARY_ROUNDS"])

    if duplicate:
        return "duplicate"
    if label == "invalid_complexity" or validation.get("passed") is False:
        return "invalid"
    if effect.get("is_new_boundary_for_sample") or label == "effective_boundary_probe":
        return "boundary_hit"
    if _candidate_homogeneity_detected(item):
        return "exhausted"
    if max_depth > 0 and search_depth >= max_depth:
        return "exhausted"
    if branch_budget_remaining <= 0:
        return "exhausted"
    if max_no_new > 0 and no_new_boundary_rounds >= max_no_new:
        return "exhausted"
    return "exploring"


def _apply_tree_search_state_update(
    item: Dict[str, Any],
    state: Dict[str, Any],
    *,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    if not _as_bool(config.get("ENABLE_TREE_SEARCH")):
        return state

    effect = _effect(item)
    evolved = is_question_evolved(item)
    spent = 1 if evolved else 0
    branch_budget = max(0, _int_state(state, "branch_budget_remaining", int(config["MAX_SAMPLE_DEPTH"])) - spent)
    sample_budget = max(
        0,
        _int_state(state, "sample_budget_remaining", int(config["MAX_SAMPLE_CANDIDATES_TOTAL"])) - spent,
    )

    boundary = _new_boundary_entry(item, state)
    discovered = _append_boundary(_list_state(state, "discovered_boundaries"), boundary)
    axis = _effect_axis(item, state)
    explored_axes = _append_axis(_list_state(state, "already_explored_axes"), axis if boundary else None)
    if effect.get("duplicate_boundary_for_sample"):
        explored_axes = _append_axis(explored_axes, axis)
    previous_recommended_axes = [
        _clean_text(axis_value)
        for axis_value in _list_state(state, "recommended_next_axes")
        if _clean_text(axis_value)
    ]
    next_axes = _filtered_next_axes(
        {**state, "already_explored_axes": explored_axes},
        current_axis=axis,
        allow_default_fallback=_allow_default_axis_fallback(config),
    )
    recommended_axes_exhausted = bool(previous_recommended_axes) and not next_axes

    previous_no_new = _int_state(state, "no_new_boundary_rounds", 0)
    no_new_rounds = 0 if boundary else (previous_no_new + 1 if evolved else previous_no_new)
    branch_ids = _opened_branch_ids({**state, "discovered_boundaries": discovered})
    state["branch_budget_remaining"] = branch_budget
    state["sample_budget_remaining"] = sample_budget
    state["discovered_boundaries"] = discovered
    state["already_explored_axes"] = explored_axes
    state["recommended_next_axes"] = next_axes
    state["recommended_axes_exhausted"] = recommended_axes_exhausted
    state["opened_branch_ids"] = branch_ids
    state["sample_branch_count"] = len(branch_ids)
    state["no_new_boundary_rounds"] = no_new_rounds
    state["branch_status"] = _tree_branch_status(
        item,
        state,
        branch_budget_remaining=branch_budget,
        no_new_boundary_rounds=no_new_rounds,
        config=config,
    )
    if _candidate_homogeneity_detected(item):
        state["candidate_homogeneity_detected"] = True
        state["branch_stop_reason"] = "homogeneous_candidate_stop"

    max_boundaries = int(config["MAX_SAMPLE_BOUNDARIES"])
    max_no_new = int(config["MAX_NO_NEW_BOUNDARY_ROUNDS"])
    if sample_budget <= 0:
        state["stop_status"] = "sample_budget_exhausted"
    elif max_boundaries > 0 and len(discovered) >= max_boundaries:
        state["stop_status"] = "sample_boundary_limit_reached"
    elif state["branch_status"] in BRANCH_TERMINAL_STATUSES and _sample_branch_limit_reached(state, config):
        state["stop_status"] = "sample_branch_limit_reached"
    elif state["branch_status"] in BRANCH_TERMINAL_STATUSES and recommended_axes_exhausted:
        state["stop_status"] = "recommended_axes_exhausted_stop"
    elif max_no_new > 0 and no_new_rounds >= max_no_new:
        state["stop_status"] = "no_new_boundary_stop"
    elif state["branch_status"] in BRANCH_TERMINAL_STATUSES:
        state["stop_status"] = "continue_branch_search"
    return state


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


def build_next_state(
    item: Dict[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config_values = _tree_config(config)
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

    legacy_next_state = {
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

    state_seed = dict(previous_state)
    state_seed.update(legacy_next_state)
    normalized_item = dict(item)
    normalized_item["evolution_state"] = state_seed
    next_state = normalize_search_state(normalized_item, config=config_values)
    next_state.update(legacy_next_state)
    return _apply_tree_search_state_update(item, next_state, config=config_values)


def attach_next_state(
    item: Dict[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result = dict(item)
    result["evolution_state"] = build_next_state(item, config=config)
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


def _graph_node_from_updated_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    state = normalize_search_state(record)
    effect = record.get("effect_analysis")
    effect = effect if isinstance(effect, Mapping) else {}
    metadata = get_question_evolution_metadata(record)
    generation = record.get("candidate_generation")
    generation = generation if isinstance(generation, Mapping) else {}
    dedup_signature = (
        _clean_text(effect.get("dedup_signature"))
        or _clean_text(record.get("dedup_signature"))
        or _short_hash(
            f"{sample_key(record)}|{state.get('boundary_axis')}|{state.get('current_node_id')}|{record.get('prompt')}"
        )
    )
    selected_as_boundary_leaf = bool(
        record.get("selected_as_boundary_leaf")
        or effect.get("is_new_boundary_for_sample")
        or state.get("branch_status") == "boundary_hit"
    )
    discard_as_duplicate = bool(record.get("discard_as_duplicate") or effect.get("duplicate_boundary_for_sample"))
    selected_into_mainline = bool(record.get("selected_into_mainline", not selected_as_boundary_leaf and not discard_as_duplicate))

    return {
        "sample_id": sample_key(record),
        "search_root_id": state["search_root_id"],
        "node_id": state["current_node_id"],
        "parent_node_id": state.get("parent_node_id"),
        "branch_id": state["branch_id"],
        "depth": state["search_depth"],
        "prompt": _clean_text(record.get("prompt")),
        "operator_used": (
            _clean_text(effect.get("operator_used"))
            or _clean_text(record.get("candidate_operator"))
            or _clean_text(metadata.get("operator_used"))
            or _clean_text(generation.get("operator_id"))
            or None
        ),
        "boundary_axis": state.get("boundary_axis") or effect.get("boundary_axis_detected"),
        "score_rate_before": effect.get("score_rate_before"),
        "score_rate_after": effect.get("score_rate_after", record.get("score_rate")),
        "effect_label": effect.get("effect_label") or state.get("previous_effect_status"),
        "is_boundary_hit": state.get("branch_status") == "boundary_hit",
        "selected_into_mainline": selected_into_mainline,
        "selected_as_boundary_leaf": selected_as_boundary_leaf,
        "discard_as_duplicate": discard_as_duplicate,
        "dedup_signature": dedup_signature,
        "branch_status": state.get("branch_status"),
        "stop_status": state.get("stop_status"),
    }


def _graph_key(record: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        _clean_text(record.get("sample_id")),
        _clean_text(record.get("search_root_id")),
        _clean_text(record.get("node_id")),
        _clean_text(record.get("dedup_signature")),
    )


def merge_search_graph_records(
    previous_graph: Sequence[Dict[str, Any]],
    current_graph: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for record in list(previous_graph) + list(current_graph):
        key = _graph_key(record)
        if not any(key):
            continue
        merged[key] = dict(record)
    return list(merged.values())


def _source_prompt_for_frontier(record: Mapping[str, Any], source_node_type: str) -> Tuple[str, str]:
    metadata = get_question_evolution_metadata(record)
    if source_node_type == "root":
        root_prompt = get_root_prompt(record)
        current_prompt = _clean_text(record.get("prompt"))
        return root_prompt, "meta_info.prompt_old" if root_prompt != current_prompt else "prompt"
    if source_node_type == "parent":
        source_prompt = _clean_text(metadata.get("source_prompt"))
        if source_prompt:
            return source_prompt, _clean_text(metadata.get("prompt_source")) or "parent_node"
        return get_root_prompt(record), "parent_node"
    return _clean_text(record.get("prompt")), "prompt"


def _next_frontier_action(
    state: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> Optional[Tuple[str, str]]:
    if _is_sample_terminal(state):
        return None
    branch_status = _clean_text(state.get("branch_status"))
    search_depth = _int_state(state, "search_depth", 0)
    max_depth = _int_state(state, "max_search_depth", int(config["MAX_SAMPLE_DEPTH"]))
    branch_budget = _int_state(state, "branch_budget_remaining", 0)

    if branch_status == "exploring" and branch_budget > 0 and (max_depth <= 0 or search_depth < max_depth):
        return "expand_current_branch", "current"
    if branch_status not in BRANCH_TERMINAL_STATUSES:
        return None

    parent_available = bool(_clean_text(state.get("parent_node_id")))
    if _as_bool(config.get("ENABLE_BRANCH_BACKTRACK")) and parent_available and search_depth > 1:
        return "fork_from_parent", "parent"
    if _as_bool(config.get("ENABLE_ROOT_FORK")):
        return "fork_from_root", "root"
    return None


def _frontier_from_updated_record(
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not _as_bool(config.get("ENABLE_TREE_SEARCH")):
        return None
    state = normalize_search_state(record, config=config)
    decision = _next_frontier_action(state, config=config)
    if decision is None:
        return None
    action_type, source_node_type = decision
    if action_type in NEW_BRANCH_ACTIONS and _sample_branch_limit_reached(state, config):
        return None
    next_axes = _filtered_next_axes(
        state,
        current_axis=state.get("boundary_axis"),
        allow_default_fallback=_allow_default_axis_fallback(config),
    )
    if action_type in NEW_BRANCH_ACTIONS and not next_axes:
        return None
    target_axis = next_axes[0] if action_type in NEW_BRANCH_ACTIONS and next_axes else _clean_text(state.get("boundary_axis"))
    if not target_axis:
        target_axis = DEFAULT_BOUNDARY_AXES[0]

    if source_node_type == "root":
        source_node_id = state["search_root_id"]
        next_depth = 1
        branch_id = ""
    elif source_node_type == "parent":
        source_node_id = _clean_text(state.get("parent_node_id")) or state["search_root_id"]
        next_depth = max(1, _int_state(state, "search_depth", 0))
        branch_id = ""
    else:
        source_node_id = state["current_node_id"]
        next_depth = _int_state(state, "search_depth", 0) + 1
        branch_id = _clean_text(state.get("branch_id")) or "main"

    source_prompt, prompt_source = _source_prompt_for_frontier(record, source_node_type)
    if not source_prompt:
        return None

    branch_budget = _int_state(state, "branch_budget_remaining", 0)
    if action_type in NEW_BRANCH_ACTIONS:
        branch_budget = _int_state(state, "max_search_depth", int(config["MAX_SAMPLE_DEPTH"]))

    return {
        "sample_id": sample_key(record),
        "search_root_id": state["search_root_id"],
        "frontier_node_id": _frontier_id(source_node_id, action_type, target_axis),
        "source_node_id": source_node_id,
        "source_node_type": source_node_type,
        "source_prompt": source_prompt,
        "prompt_source": prompt_source,
        "action_type": action_type,
        "branch_id": branch_id,
        "target_boundary_axis": target_axis,
        "search_depth": _int_state(state, "search_depth", 0),
        "next_depth": next_depth,
        "max_search_depth": _int_state(state, "max_search_depth", int(config["MAX_SAMPLE_DEPTH"])),
        "branch_budget_remaining": branch_budget,
        "sample_budget_remaining": _int_state(state, "sample_budget_remaining", 0),
        "discovered_boundary_count": len(_list_state(state, "discovered_boundaries")),
        "origin_branch_status": state.get("branch_status"),
        "origin_stop_status": state.get("stop_status"),
    }


def build_search_graph_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_graph_node_from_updated_record(record) for record in records]


def build_active_frontier_records(
    records: Sequence[Dict[str, Any]],
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    config_values = _tree_config(config)
    frontier: List[Dict[str, Any]] = []
    seen: set = set()
    for record in records:
        row = _frontier_from_updated_record(record, config=config_values)
        if not row:
            continue
        key = row["frontier_node_id"]
        if key in seen:
            continue
        seen.add(key)
        frontier.append(row)
    return frontier


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


def update_records(
    records: Sequence[Dict[str, Any]],
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    updated = [attach_next_state(record, config=config) for record in records]
    operator_entries, failure_entries, invalid_entries = classify_memory_entries(records)
    return updated, operator_entries, failure_entries, invalid_entries


def update_records_with_artifacts(
    records: Sequence[Dict[str, Any]],
    *,
    config: Optional[Mapping[str, Any]] = None,
    previous_graph: Sequence[Dict[str, Any]] = (),
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    config_values = _tree_config(config)
    updated, operator_entries, failure_entries, invalid_entries = update_records(records, config=config_values)
    current_graph = build_search_graph_records(updated)
    search_graph = merge_search_graph_records(previous_graph, current_graph)
    active_frontier = build_active_frontier_records(updated, config=config_values)
    return updated, operator_entries, failure_entries, invalid_entries, search_graph, active_frontier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update evolution_state and append Stage 5 memory-bank entries.")
    parser.add_argument("--input", required=True, help="Input analyzed JSON/JSONL path.")
    parser.add_argument("--output", required=True, help="Output state-updated JSONL path.")
    parser.add_argument("--memory-dir", default="memory", help="Directory containing memory bank JSONL files.")
    parser.add_argument("--operator-memory", default=None, help="Override operator memory output path.")
    parser.add_argument("--failure-memory", default=None, help="Override failure memory output path.")
    parser.add_argument("--invalid-output", default=None, help="Override invalid generation case output path.")
    parser.add_argument("--active-frontier-output", default=None, help="Optional next-round active_frontier.jsonl output path.")
    parser.add_argument("--search-graph-output", default=None, help="Optional merged search_graph.jsonl output path.")
    parser.add_argument("--previous-search-graph", default=None, help="Optional previous search_graph.jsonl to merge before writing.")
    parser.add_argument("--tree-search-enabled", default=None, help="true/false; overrides ENABLE_TREE_SEARCH for state/frontier updates.")
    parser.add_argument("--max-sample-branches", type=int, default=None, help="Override MAX_SAMPLE_BRANCHES.")
    parser.add_argument("--max-sample-depth", type=int, default=None, help="Override MAX_SAMPLE_DEPTH.")
    parser.add_argument("--max-sample-boundaries", type=int, default=None, help="Override MAX_SAMPLE_BOUNDARIES.")
    parser.add_argument("--max-sample-candidates-total", type=int, default=None, help="Override MAX_SAMPLE_CANDIDATES_TOTAL.")
    parser.add_argument("--max-no-new-boundary-rounds", type=int, default=None, help="Override MAX_NO_NEW_BOUNDARY_ROUNDS.")
    parser.add_argument("--enable-branch-backtrack", default=None, help="true/false; allow parent-node sibling forks.")
    parser.add_argument("--enable-root-fork", default=None, help="true/false; allow root-node forks.")
    parser.add_argument(
        "--allow-default-axis-fallback-after-recommendation-exhausted",
        default=None,
        help="true/false; when true, exhausted explicit recommended_next_axes may fall back to default axes.",
    )
    parser.add_argument("--no-memory-output", action="store_true", help="Do not append memory-bank entries.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.tree_search_enabled is not None:
        overrides["ENABLE_TREE_SEARCH"] = args.tree_search_enabled
    if args.max_sample_branches is not None:
        overrides["MAX_SAMPLE_BRANCHES"] = args.max_sample_branches
    if args.max_sample_depth is not None:
        overrides["MAX_SAMPLE_DEPTH"] = args.max_sample_depth
    if args.max_sample_boundaries is not None:
        overrides["MAX_SAMPLE_BOUNDARIES"] = args.max_sample_boundaries
    if args.max_sample_candidates_total is not None:
        overrides["MAX_SAMPLE_CANDIDATES_TOTAL"] = args.max_sample_candidates_total
    if args.max_no_new_boundary_rounds is not None:
        overrides["MAX_NO_NEW_BOUNDARY_ROUNDS"] = args.max_no_new_boundary_rounds
    if args.enable_branch_backtrack is not None:
        overrides["ENABLE_BRANCH_BACKTRACK"] = args.enable_branch_backtrack
    if args.enable_root_fork is not None:
        overrides["ENABLE_ROOT_FORK"] = args.enable_root_fork
    if args.allow_default_axis_fallback_after_recommendation_exhausted is not None:
        overrides[ALLOW_DEFAULT_AXIS_FALLBACK_CONFIG] = (
            args.allow_default_axis_fallback_after_recommendation_exhausted
        )
    return _tree_config(overrides)


def _load_optional_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    return load_json_or_jsonl(path)


def main() -> None:
    args = parse_args()
    config = _config_from_args(args)
    records = load_json_or_jsonl(args.input)
    previous_graph = _load_optional_jsonl(args.previous_search_graph)
    (
        updated,
        operator_entries,
        failure_entries,
        invalid_entries,
        search_graph,
        active_frontier,
    ) = update_records_with_artifacts(records, config=config, previous_graph=previous_graph)
    write_jsonl(updated, args.output)
    if args.search_graph_output:
        write_jsonl(search_graph, args.search_graph_output)
    if args.active_frontier_output:
        write_jsonl(active_frontier, args.active_frontier_output)

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
