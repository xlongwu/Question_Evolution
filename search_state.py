import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_MAX_SAMPLE_BRANCHES = 3
DEFAULT_MAX_SAMPLE_DEPTH = 2
DEFAULT_MAX_SAMPLE_BOUNDARIES = 2
DEFAULT_MAX_SAMPLE_CANDIDATES_TOTAL = 6


DEFAULT_BOUNDARY_AXES = [
    "最小关键事实识别",
    "子判断定位",
    "近似层级排序",
    "隐藏前提识别",
    "单变量反事实",
    "事实绑定约束",
    "双门槛判断",
    "反常线索主线切换",
]


OPERATOR_AXIS = {
    "O1_gap_choice": "最小关键事实识别",
    "O2_subclaim_localization": "子判断定位",
    "O3_step_jump": "层级跳步识别",
    "O4_near_level_ranking": "近似层级排序",
    "O5_extra_premise_detection": "隐藏前提识别",
    "O6_single_variable_counterfactual": "单变量反事实",
    "O7_fact_binding_constraint": "事实绑定约束",
    "O8_double_threshold_claim": "双门槛判断",
    "O9_abnormal_clue_mainline_switch": "反常线索主线切换",
}


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _slug(value: Any, *, fallback: str = "x", max_len: int = 24) -> str:
    text = _clean_text(value)
    if not text:
        return fallback
    ascii_slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if ascii_slug:
        return ascii_slug[:max_len]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{fallback}_{digest}"


def _coerce_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _append_unique(items: List[str], values: Iterable[Any]) -> List[str]:
    for value in values:
        text = _clean_text(value)
        if text and text not in items:
            items.append(text)
    return items


def sample_key(record: Dict[str, Any]) -> str:
    """Return the stable per-sample key used by tree-search artifacts."""
    for field in ("sample_id", "index"):
        value = record.get(field)
        if value is not None and _clean_text(value):
            return _clean_text(value)
    prompt = _clean_text(record.get("prompt"))
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    return f"prompt_{digest}"


def make_search_root_id(record_or_sample: Any) -> str:
    sample = sample_key(record_or_sample) if isinstance(record_or_sample, dict) else _clean_text(record_or_sample)
    return f"sample_{_slug(sample, fallback='sample')}_root"


def make_node_id(search_root_id: str, branch_index: int, depth: int) -> str:
    branch_index = max(0, int(branch_index or 0))
    depth = max(0, int(depth or 0))
    if depth == 0:
        return search_root_id
    return f"{search_root_id}_b{branch_index}_d{depth}"


def make_branch_id(operator_id: str, axis: str, branch_index: int) -> str:
    operator_part = _slug(operator_id, fallback="op", max_len=32)
    axis_part = _slug(axis, fallback="axis", max_len=20)
    return f"branch_{operator_part}_{axis_part}_{max(1, int(branch_index or 1)):03d}"


def operator_axis(operator_id: Any) -> str:
    return OPERATOR_AXIS.get(_clean_text(operator_id), "")


def normalize_axis_candidates(values: Any = None) -> List[str]:
    axes: List[str] = []
    if isinstance(values, list):
        _append_unique(axes, values)
    elif isinstance(values, str) and values.strip():
        _append_unique(axes, [values])
    _append_unique(axes, DEFAULT_BOUNDARY_AXES)
    return axes


def resolve_boundary_axis(record: Dict[str, Any], operator_id: Optional[str] = None) -> str:
    for source in (
        record.get("boundary_axis"),
        record.get("boundary_axis_detected"),
    ):
        text = _clean_text(source)
        if text:
            return text

    generation = record.get("candidate_generation")
    if isinstance(generation, dict):
        text = _clean_text(generation.get("boundary_axis"))
        if text:
            return text

    route = record.get("operator_route")
    if isinstance(route, dict):
        text = _clean_text(route.get("boundary_axis"))
        if text:
            return text

    state = record.get("evolution_state")
    if isinstance(state, dict):
        text = _clean_text(state.get("boundary_axis"))
        if text:
            return text

    meta_info = record.get("meta_info")
    if isinstance(meta_info, dict):
        metadata = meta_info.get("question_evolution_metadata")
        if isinstance(metadata, dict):
            for field in ("ability_axis", "boundary_axis"):
                text = _clean_text(metadata.get(field))
                if text:
                    return text

    return operator_axis(operator_id or get_operator_used(record))


def get_operator_used(record: Dict[str, Any]) -> str:
    effect = record.get("effect_analysis")
    if isinstance(effect, dict):
        text = _clean_text(effect.get("operator_used"))
        if text:
            return text
    for field in ("candidate_operator", "operator_used"):
        text = _clean_text(record.get(field))
        if text:
            return text
    selection = record.get("candidate_selection")
    if isinstance(selection, dict):
        text = _clean_text(selection.get("selected_operator"))
        if text:
            return text
    meta_info = record.get("meta_info")
    if isinstance(meta_info, dict):
        metadata = meta_info.get("question_evolution_metadata")
        if isinstance(metadata, dict):
            return _clean_text(metadata.get("operator_used"))
    return ""


def boundary_signature(record: Dict[str, Any]) -> str:
    effect = record.get("effect_analysis")
    effect = effect if isinstance(effect, dict) else {}
    axis = (
        _clean_text(effect.get("boundary_axis_detected"))
        or resolve_boundary_axis(record)
        or "unknown_axis"
    )
    operator = _clean_text(effect.get("operator_used")) or get_operator_used(record) or "unknown_operator"
    return f"{sample_key(record)}|{axis}|{operator}"


def init_search_state(
    record: Dict[str, Any],
    *,
    max_depth: int = DEFAULT_MAX_SAMPLE_DEPTH,
    max_branches: int = DEFAULT_MAX_SAMPLE_BRANCHES,
    max_boundaries: int = DEFAULT_MAX_SAMPLE_BOUNDARIES,
    max_candidates_total: int = DEFAULT_MAX_SAMPLE_CANDIDATES_TOTAL,
) -> Dict[str, Any]:
    root_id = make_search_root_id(record)
    return {
        "search_root_id": root_id,
        "current_node_id": root_id,
        "parent_node_id": None,
        "branch_id": None,
        "branch_action": "expand_current_branch",
        "boundary_axis": None,
        "branch_status": "root",
        "search_depth": 0,
        "max_search_depth": max_depth,
        "max_sample_branches": max_branches,
        "max_sample_boundaries": max_boundaries,
        "max_sample_candidates_total": max_candidates_total,
        "branch_budget_remaining": max_depth,
        "sample_budget_remaining": max_candidates_total,
        "sample_candidates_used": 0,
        "branch_count": 0,
        "discovered_boundaries": [],
        "explored_axes": [],
        "recommended_next_axes": list(DEFAULT_BOUNDARY_AXES),
        "sample_stop_status": "continue_branch_search",
    }


def merge_search_state(
    record: Dict[str, Any],
    previous_state: Optional[Dict[str, Any]] = None,
    *,
    max_depth: int = DEFAULT_MAX_SAMPLE_DEPTH,
    max_branches: int = DEFAULT_MAX_SAMPLE_BRANCHES,
    max_boundaries: int = DEFAULT_MAX_SAMPLE_BOUNDARIES,
    max_candidates_total: int = DEFAULT_MAX_SAMPLE_CANDIDATES_TOTAL,
) -> Dict[str, Any]:
    state = init_search_state(
        record,
        max_depth=max_depth,
        max_branches=max_branches,
        max_boundaries=max_boundaries,
        max_candidates_total=max_candidates_total,
    )
    if isinstance(previous_state, dict):
        state.update(previous_state)
        state.setdefault("search_root_id", make_search_root_id(record))
        state.setdefault("current_node_id", state["search_root_id"])
    return state


def discovered_signatures(state: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    boundaries = state.get("discovered_boundaries")
    if isinstance(boundaries, list):
        for boundary in boundaries:
            if isinstance(boundary, dict):
                signature = _clean_text(boundary.get("dedup_signature") or boundary.get("boundary_signature"))
                if signature and signature not in values:
                    values.append(signature)
    return values


def budget_exhausted(state: Dict[str, Any]) -> bool:
    max_depth = _coerce_int(state.get("max_search_depth"), DEFAULT_MAX_SAMPLE_DEPTH)
    max_branches = _coerce_int(state.get("max_sample_branches"), DEFAULT_MAX_SAMPLE_BRANCHES)
    max_boundaries = _coerce_int(state.get("max_sample_boundaries"), DEFAULT_MAX_SAMPLE_BOUNDARIES)
    max_candidates = _coerce_int(
        state.get("max_sample_candidates_total"),
        DEFAULT_MAX_SAMPLE_CANDIDATES_TOTAL,
    )
    depth = _coerce_int(state.get("search_depth"), 0)
    branch_count = _coerce_int(state.get("branch_count"), 0)
    candidates_used = _coerce_int(state.get("sample_candidates_used"), 0)
    boundaries = state.get("discovered_boundaries")
    boundary_count = len(boundaries) if isinstance(boundaries, list) else 0
    return (
        depth >= max_depth
        or branch_count >= max_branches
        or boundary_count >= max_boundaries
        or candidates_used >= max_candidates
        or _coerce_int(state.get("sample_budget_remaining"), max_candidates) <= 0
    )


def first_unexplored_axis(
    *,
    axis_candidates: Sequence[str],
    explored_axes: Sequence[str],
    fallback_axis: str = "",
) -> str:
    explored = {_clean_text(axis) for axis in explored_axes if _clean_text(axis)}
    for axis in normalize_axis_candidates(list(axis_candidates)):
        if axis and axis not in explored:
            return axis
    return _clean_text(fallback_axis)
