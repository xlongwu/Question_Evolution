import hashlib
import re
from typing import Any, Dict, Mapping, Optional


DEFAULT_BOUNDARY_AXES = [
    "结论分层",
    "最关键缺口识别",
    "伪闭环识别",
    "补强项升级判断",
    "题干外补设识别",
    "反常线索主线切换",
]

BRANCH_STATUSES = {
    "exploring",
    "boundary_hit",
    "exhausted",
    "duplicate",
    "invalid",
}

FRONTIER_ACTION_TYPES = {
    "expand_current_branch",
    "fork_from_root",
    "fork_from_parent",
    "stop_branch",
    "stop_sample",
}

TREE_SEARCH_CONFIG_DEFAULTS = {
    "ENABLE_TREE_SEARCH": False,
    "MAX_SAMPLE_BRANCHES": 4,
    "MAX_SAMPLE_DEPTH": 3,
    "MAX_SAMPLE_BOUNDARIES": 3,
    "MAX_SAMPLE_CANDIDATES_TOTAL": 10,
    "MAX_NO_NEW_BOUNDARY_ROUNDS": 2,
    "MAX_GLOBAL_NEW_BOUNDARY_GAP": 2,
    "ENABLE_BRANCH_BACKTRACK": True,
    "ENABLE_ROOT_FORK": True,
    "ALLOW_DEFAULT_AXIS_FALLBACK_AFTER_RECOMMENDATION_EXHAUSTED": False,
}


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _slug(value: Any) -> str:
    text = _clean_text(value)
    slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_")
    if slug:
        return slug
    return f"id_{_short_hash(text)}"


def _short_hash(value: Any) -> str:
    payload = _clean_text(value)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def sample_key(record: Mapping[str, Any]) -> str:
    for field in ("sample_id", "index"):
        value = record.get(field)
        if _clean_text(value):
            return _clean_text(value)
    prompt = _clean_text(record.get("prompt"))
    return f"anon_{_short_hash(prompt)}"


def get_question_evolution_metadata(record: Mapping[str, Any]) -> Dict[str, Any]:
    meta_info = record.get("meta_info")
    if not isinstance(meta_info, Mapping):
        return {}
    metadata = meta_info.get("question_evolution_metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def get_root_prompt(record: Mapping[str, Any]) -> str:
    meta_info = record.get("meta_info")
    if isinstance(meta_info, Mapping):
        prompt_old = _clean_text(meta_info.get("prompt_old"))
        if prompt_old:
            return prompt_old
    return _clean_text(record.get("prompt"))


def is_evolved_record(record: Mapping[str, Any]) -> bool:
    if record.get("question_evolved") is True:
        return True
    metadata = get_question_evolution_metadata(record)
    if metadata.get("question_evolved") is True:
        return True
    meta_info = record.get("meta_info")
    return isinstance(meta_info, Mapping) and bool(_clean_text(meta_info.get("prompt_old")))


def search_root_id_for(record: Mapping[str, Any]) -> str:
    state = record.get("evolution_state")
    if isinstance(state, Mapping) and _clean_text(state.get("search_root_id")):
        return _clean_text(state.get("search_root_id"))
    return f"sample_{_slug(sample_key(record))}_root"


def current_node_id_for(record: Mapping[str, Any], root_id: Optional[str] = None) -> str:
    state = record.get("evolution_state")
    if isinstance(state, Mapping) and _clean_text(state.get("current_node_id")):
        return _clean_text(state.get("current_node_id"))

    root_id = root_id or search_root_id_for(record)
    if not is_evolved_record(record):
        return root_id

    round_value = _coerce_non_negative_int(record.get("round"), 1)
    prompt_hash = _short_hash(record.get("prompt"))[:8]
    return f"{root_id}_r{round_value}_{prompt_hash}"


def infer_branch_status(record: Mapping[str, Any], stop_status: str) -> str:
    state = record.get("evolution_state")
    if isinstance(state, Mapping):
        branch_status = _clean_text(state.get("branch_status"))
        if branch_status in BRANCH_STATUSES:
            return branch_status

    effect = record.get("effect_analysis")
    effect_label = _clean_text(effect.get("effect_label")) if isinstance(effect, Mapping) else ""
    if stop_status == "effective_boundary_sample" or effect_label == "effective_boundary_probe":
        return "boundary_hit"
    if stop_status in {"invalid_complexity_sample", "unanswerable_or_trap_sample"}:
        return "invalid"
    if stop_status in {"stable_high_score_stop", "stop_branch", "branch_exhausted"}:
        return "exhausted"
    return "exploring"


def normalize_search_state(
    record: Mapping[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config_values = dict(TREE_SEARCH_CONFIG_DEFAULTS)
    if config:
        config_values.update(config)

    previous_state = record.get("evolution_state")
    state = dict(previous_state) if isinstance(previous_state, Mapping) else {}

    root_id = search_root_id_for(record)
    current_node_id = current_node_id_for(record, root_id)
    stop_status = _clean_text(state.get("stop_status")) or "continue"
    search_depth = _coerce_non_negative_int(
        state.get("search_depth"),
        1 if is_evolved_record(record) else 0,
    )
    max_search_depth = _coerce_non_negative_int(
        state.get("max_search_depth"),
        int(config_values["MAX_SAMPLE_DEPTH"]),
    )

    metadata = get_question_evolution_metadata(record)
    boundary_axis = (
        _clean_text(state.get("boundary_axis"))
        or _clean_text(metadata.get("boundary_axis"))
        or _clean_text(metadata.get("ability_axis"))
        or None
    )

    state.setdefault("round", _coerce_non_negative_int(record.get("round"), 0))
    state["search_root_id"] = root_id
    state["current_node_id"] = current_node_id
    state["parent_node_id"] = (
        _clean_text(state.get("parent_node_id"))
        or (root_id if is_evolved_record(record) and current_node_id != root_id else None)
    )
    state["branch_id"] = _clean_text(state.get("branch_id")) or "main"
    state["boundary_axis"] = boundary_axis
    state["branch_status"] = infer_branch_status(record, stop_status)
    state["search_depth"] = search_depth
    state["max_search_depth"] = max_search_depth
    state["branch_budget_remaining"] = _coerce_non_negative_int(
        state.get("branch_budget_remaining"),
        max(0, max_search_depth - search_depth),
    )
    state["sample_budget_remaining"] = _coerce_non_negative_int(
        state.get("sample_budget_remaining"),
        int(config_values["MAX_SAMPLE_CANDIDATES_TOTAL"]),
    )
    if not isinstance(state.get("discovered_boundaries"), list):
        state["discovered_boundaries"] = []
    if not isinstance(state.get("recommended_next_axes"), list):
        state["recommended_next_axes"] = []
    if not isinstance(state.get("already_explored_axes"), list):
        state["already_explored_axes"] = []
    state["stop_status"] = stop_status
    return state


def attach_normalized_search_state(record: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(record)
    result["evolution_state"] = normalize_search_state(record)
    return result


def _score_rate(record: Mapping[str, Any]) -> Optional[float]:
    try:
        score_rate = float(record.get("score_rate"))
    except (TypeError, ValueError):
        return None
    if 0 <= score_rate <= 1:
        return score_rate
    return None


def _operator_used(record: Mapping[str, Any]) -> Optional[str]:
    metadata = get_question_evolution_metadata(record)
    for value in (
        record.get("candidate_operator"),
        record.get("operator_used"),
        metadata.get("operator_used"),
        normalize_search_state(record).get("previous_operator"),
    ):
        text = _clean_text(value)
        if text:
            return text
    return None


def dedup_signature(record: Mapping[str, Any], state: Optional[Mapping[str, Any]] = None) -> str:
    state = state or normalize_search_state(record)
    parts = [
        sample_key(record),
        _clean_text(state.get("boundary_axis")) or "unknown_axis",
        _clean_text(record.get("prompt")),
    ]
    return _short_hash("|||".join(parts))


def build_search_graph_node(record: Mapping[str, Any]) -> Dict[str, Any]:
    state = normalize_search_state(record)
    effect = record.get("effect_analysis")
    effect = effect if isinstance(effect, Mapping) else {}
    effect_label = _clean_text(effect.get("effect_label")) or _clean_text(state.get("previous_effect_status"))

    return {
        "sample_id": sample_key(record),
        "search_root_id": state["search_root_id"],
        "node_id": state["current_node_id"],
        "parent_node_id": state.get("parent_node_id"),
        "branch_id": state["branch_id"],
        "depth": state["search_depth"],
        "prompt": _clean_text(record.get("prompt")),
        "operator_used": _operator_used(record),
        "boundary_axis": state.get("boundary_axis"),
        "score_rate_before": effect.get("score_rate_before"),
        "score_rate_after": effect.get("score_rate_after", _score_rate(record)),
        "effect_label": effect_label or None,
        "is_boundary_hit": state.get("branch_status") == "boundary_hit",
        "selected_into_mainline": record.get("selected_into_mainline", True),
        "dedup_signature": dedup_signature(record, state),
    }


def build_active_frontier_node(
    record: Mapping[str, Any],
    *,
    action_type: str = "expand_current_branch",
    target_boundary_axis: Optional[str] = None,
    source_node_type: str = "current",
) -> Dict[str, Any]:
    if action_type not in FRONTIER_ACTION_TYPES:
        raise ValueError(f"unsupported frontier action_type: {action_type}")
    if source_node_type not in {"current", "root", "parent"}:
        raise ValueError(f"unsupported source_node_type: {source_node_type}")

    state = normalize_search_state(record)
    if source_node_type == "root":
        source_node_id = state["search_root_id"]
        source_prompt = get_root_prompt(record)
        prompt_source = "meta_info.prompt_old" if source_prompt != _clean_text(record.get("prompt")) else "prompt"
        next_depth = 1
    elif source_node_type == "parent":
        source_node_id = state.get("parent_node_id") or state["search_root_id"]
        source_prompt = get_root_prompt(record)
        prompt_source = "parent_node"
        next_depth = max(1, int(state["search_depth"]))
    else:
        source_node_id = state["current_node_id"]
        source_prompt = _clean_text(record.get("prompt"))
        prompt_source = "prompt"
        next_depth = int(state["search_depth"]) + 1

    return {
        "sample_id": sample_key(record),
        "search_root_id": state["search_root_id"],
        "frontier_node_id": f"frontier_{source_node_id}_{action_type}",
        "source_node_id": source_node_id,
        "source_node_type": source_node_type,
        "source_prompt": source_prompt,
        "prompt_source": prompt_source,
        "action_type": action_type,
        "branch_id": state["branch_id"],
        "target_boundary_axis": target_boundary_axis or state.get("boundary_axis") or DEFAULT_BOUNDARY_AXES[0],
        "search_depth": state["search_depth"],
        "next_depth": next_depth,
        "max_search_depth": state["max_search_depth"],
        "branch_budget_remaining": state["branch_budget_remaining"],
        "sample_budget_remaining": state["sample_budget_remaining"],
        "discovered_boundary_count": len(state.get("discovered_boundaries") or []),
    }
