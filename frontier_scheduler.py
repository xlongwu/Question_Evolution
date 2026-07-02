import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from search_state import (
    DEFAULT_BOUNDARY_AXES,
    DEFAULT_MAX_SAMPLE_BOUNDARIES,
    DEFAULT_MAX_SAMPLE_BRANCHES,
    DEFAULT_MAX_SAMPLE_CANDIDATES_TOTAL,
    DEFAULT_MAX_SAMPLE_DEPTH,
    boundary_signature,
    first_unexplored_axis,
    init_search_state,
    make_branch_id,
    make_node_id,
    normalize_axis_candidates,
    resolve_boundary_axis,
    sample_key,
)


TERMINAL_SAMPLE_STATUSES = {
    "max_boundaries_reached",
    "budget_exhausted",
    "stop_sample",
    "stable_high_score_stop",
    "invalid_complexity_sample",
    "unanswerable_or_trap_sample",
}

NODE_RECORD_FIELDS = (
    "sample_id",
    "index",
    "prompt",
    "meta_info",
    "rubric",
    "rubric_thought_process",
    "score_prompt",
    "scoring_result",
    "score_rate",
    "sample_profile",
    "overscore_diagnosis",
    "evolution_action",
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


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _optional_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _bool_from_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _state(record: Dict[str, Any]) -> Dict[str, Any]:
    state = record.get("evolution_state")
    return state if isinstance(state, dict) else {}


def _effect(record: Dict[str, Any]) -> Dict[str, Any]:
    effect = record.get("effect_analysis")
    return effect if isinstance(effect, dict) else {}


def _generation(record: Dict[str, Any]) -> Dict[str, Any]:
    generation = record.get("candidate_generation")
    return generation if isinstance(generation, dict) else {}


def _append_unique(items: List[str], values: Iterable[Any]) -> List[str]:
    for value in values:
        text = _clean_text(value)
        if text and text not in items:
            items.append(text)
    return items


def compact_node_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {field: record[field] for field in NODE_RECORD_FIELDS if field in record}


def _node_records(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    records = state.get("node_records")
    if not isinstance(records, dict):
        return {}
    return {str(key): value for key, value in records.items() if isinstance(value, dict)}


def _node_prompts(state: Dict[str, Any]) -> Dict[str, str]:
    prompts = state.get("node_prompts")
    if not isinstance(prompts, dict):
        return {}
    return {str(key): _clean_text(value) for key, value in prompts.items() if _clean_text(value)}


def _with_source_record(record: Dict[str, Any], state: Dict[str, Any], source_node_id: str) -> Dict[str, Any]:
    restored = dict(record)
    source_record = _node_records(state).get(source_node_id)
    if source_record:
        restored.update(source_record)
    else:
        source_prompt = _node_prompts(state).get(source_node_id)
        if source_prompt:
            restored["prompt"] = source_prompt
    return restored


def _max_limits(
    state: Dict[str, Any],
    *,
    max_branches: int,
    max_depth: int,
    max_boundaries: int,
    max_candidates_total: int,
) -> Dict[str, int]:
    return {
        "max_sample_branches": _coerce_int(state.get("max_sample_branches"), max_branches),
        "max_search_depth": _coerce_int(state.get("max_search_depth"), max_depth),
        "max_sample_boundaries": _coerce_int(state.get("max_sample_boundaries"), max_boundaries),
        "max_sample_candidates_total": _coerce_int(
            state.get("max_sample_candidates_total"),
            max_candidates_total,
        ),
    }


def _boundary_count(state: Dict[str, Any]) -> int:
    boundaries = state.get("discovered_boundaries")
    return len(boundaries) if isinstance(boundaries, list) else 0


def _axis_candidates(record: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for source in (
        state.get("recommended_next_axes"),
        record.get("next_best_axes"),
        record.get("boundary_axis_candidates"),
    ):
        if isinstance(source, list):
            _append_unique(candidates, source)
    profile = record.get("sample_profile")
    if isinstance(profile, dict):
        _append_unique(candidates, profile.get("boundary_axis_candidates") or [])
        _append_unique(candidates, profile.get("next_best_axes") or [])
    return normalize_axis_candidates(candidates)


def _search_identity(record: Dict[str, Any], *, branch_index_hint: int = 1) -> Dict[str, Any]:
    state = _state(record)
    generation = _generation(record)
    root_id = _clean_text(
        generation.get("search_root_id")
        or state.get("search_root_id")
        or record.get("search_root_id")
        or init_search_state(record)["search_root_id"]
    )
    branch_index = _coerce_int(
        generation.get("branch_index")
        or record.get("branch_index")
        or state.get("branch_index")
        or state.get("branch_count"),
        branch_index_hint,
    )
    depth = _coerce_int(
        generation.get("search_depth") or state.get("search_depth") or record.get("search_depth"),
        0,
    )
    current_node_id = _clean_text(
        generation.get("node_id")
        or generation.get("candidate_node_id")
        or state.get("current_node_id")
        or record.get("current_node_id")
        or make_node_id(root_id, branch_index, depth)
    )
    parent_node_id = _clean_text(
        generation.get("parent_node_id")
        or state.get("parent_node_id")
        or record.get("parent_node_id")
        or (root_id if depth > 0 else "")
    )
    branch_id = _clean_text(
        generation.get("branch_id")
        or state.get("branch_id")
        or record.get("branch_id")
    )
    return {
        "search_root_id": root_id,
        "current_node_id": current_node_id,
        "parent_node_id": parent_node_id or None,
        "branch_id": branch_id or None,
        "branch_index": branch_index,
        "search_depth": depth,
    }


def _budget_allows_new_entry(
    state: Dict[str, Any],
    *,
    limits: Dict[str, int],
    action: str,
    target_depth: int,
) -> bool:
    if _boundary_count(state) >= limits["max_sample_boundaries"]:
        return False
    if _coerce_int(state.get("sample_candidates_used"), 0) >= limits["max_sample_candidates_total"]:
        return False
    if target_depth > limits["max_search_depth"]:
        return False
    if action in {"fork_from_parent", "fork_from_root"}:
        branch_count = _coerce_int(state.get("branch_count"), 0)
        if branch_count >= limits["max_sample_branches"]:
            return False
    return True


def _frontier_entry(
    record: Dict[str, Any],
    *,
    branch_action: str,
    source_node_id: str,
    parent_node_id: Optional[str],
    boundary_axis: str,
    source_search_depth: int,
    target_search_depth: int,
    branch_index: int,
    max_branches: int,
    max_depth: int,
    max_boundaries: int,
    max_candidates_total: int,
) -> Dict[str, Any]:
    state = dict(_state(record))
    root_id = _clean_text(state.get("search_root_id") or init_search_state(record)["search_root_id"])
    if branch_action == "expand_current_branch":
        parent_node_id = source_node_id
    existing_branch_id = _clean_text(state.get("branch_id"))
    if branch_action == "expand_current_branch" and existing_branch_id:
        branch_id = existing_branch_id
    else:
        branch_id = make_branch_id("pending", boundary_axis, branch_index)
    target_state = dict(state)
    target_state.update(
        {
            "search_root_id": root_id,
            "current_node_id": source_node_id,
            "parent_node_id": parent_node_id,
            "branch_id": branch_id,
            "branch_index": branch_index,
            "branch_action": branch_action,
            "boundary_axis": boundary_axis,
            "source_search_depth": source_search_depth,
            "target_search_depth": target_search_depth,
            "search_depth": source_search_depth,
            "max_search_depth": max_depth,
            "max_sample_branches": max_branches,
            "max_sample_boundaries": max_boundaries,
            "max_sample_candidates_total": max_candidates_total,
            "sample_stop_status": "continue_branch_search",
        }
    )

    entry = _with_source_record(record, target_state, source_node_id)
    entry.update(
        {
            "sample_id": sample_key(record),
            "search_root_id": root_id,
            "source_node_id": source_node_id,
            "parent_node_id": parent_node_id,
            "branch_id": branch_id,
            "branch_index": branch_index,
            "branch_action": branch_action,
            "boundary_axis": boundary_axis,
            "source_search_depth": source_search_depth,
            "target_search_depth": target_search_depth,
            "search_depth": source_search_depth,
            "evolution_state": target_state,
        }
    )
    return entry


def _is_scheduled_frontier_record(record: Dict[str, Any]) -> bool:
    state = _state(record)
    branch_action = _clean_text(record.get("branch_action") or state.get("branch_action"))
    return (
        branch_action in {"expand_current_branch", "fork_from_parent", "fork_from_root"}
        and bool(_clean_text(record.get("source_node_id") or state.get("current_node_id")))
        and _optional_int(record.get("target_search_depth")) is not None
    )


def _preserve_scheduled_frontier_entry(
    record: Dict[str, Any],
    *,
    branch_index_hint: int,
    max_branches: int,
    max_depth: int,
    max_boundaries: int,
    max_candidates_total: int,
) -> Optional[Dict[str, Any]]:
    state = dict(_state(record))
    if not state:
        state = init_search_state(
            record,
            max_depth=max_depth,
            max_branches=max_branches,
            max_boundaries=max_boundaries,
            max_candidates_total=max_candidates_total,
        )
    limits = _max_limits(
        state,
        max_branches=max_branches,
        max_depth=max_depth,
        max_boundaries=max_boundaries,
        max_candidates_total=max_candidates_total,
    )
    if _clean_text(state.get("sample_stop_status") or state.get("stop_status")) in TERMINAL_SAMPLE_STATUSES:
        return None

    root_id = _clean_text(
        record.get("search_root_id")
        or state.get("search_root_id")
        or init_search_state(record)["search_root_id"]
    )
    branch_action = _clean_text(record.get("branch_action") or state.get("branch_action"))
    if branch_action not in {"expand_current_branch", "fork_from_parent", "fork_from_root"}:
        return None

    source_node_id = _clean_text(record.get("source_node_id") or state.get("current_node_id") or root_id)
    parent_node_id = _clean_text(record.get("parent_node_id") or state.get("parent_node_id"))
    source_depth = _optional_int(record.get("source_search_depth"))
    if source_depth is None:
        source_depth = _optional_int(record.get("search_depth"))
    if source_depth is None:
        source_depth = _optional_int(state.get("source_search_depth"))
    if source_depth is None:
        source_depth = _optional_int(state.get("search_depth"))
    if source_depth is None:
        source_depth = 0
    target_depth = _optional_int(record.get("target_search_depth"))
    if target_depth is None:
        target_depth = _optional_int(state.get("target_search_depth"))
    if target_depth is None:
        return None
    if target_depth > limits["max_search_depth"] or target_depth < source_depth:
        return None
    if branch_action == "expand_current_branch" and target_depth <= source_depth:
        return None

    if _boundary_count(state) >= limits["max_sample_boundaries"]:
        return None
    if _coerce_int(state.get("sample_candidates_used"), 0) >= limits["max_sample_candidates_total"]:
        return None

    branch_index = _coerce_int(
        record.get("branch_index") or state.get("branch_index") or state.get("branch_count"),
        branch_index_hint,
    )
    boundary_axis = (
        _clean_text(record.get("boundary_axis") or state.get("boundary_axis"))
        or resolve_boundary_axis(record)
        or first_unexplored_axis(
            axis_candidates=_axis_candidates(record, state),
            explored_axes=state.get("explored_axes") or [],
        )
        or DEFAULT_BOUNDARY_AXES[0]
    )
    branch_id = _clean_text(record.get("branch_id") or state.get("branch_id"))
    if not branch_id:
        branch_id = make_branch_id("pending", boundary_axis, branch_index)

    state.setdefault("node_records", {})
    state.setdefault("node_prompts", {})
    if isinstance(state["node_records"], dict):
        state["node_records"].setdefault(root_id, compact_node_record(record))
    if isinstance(state["node_prompts"], dict):
        state["node_prompts"].setdefault(root_id, _clean_text(record.get("prompt")))

    target_state = dict(state)
    target_state.update(
        {
            "search_root_id": root_id,
            "current_node_id": source_node_id,
            "parent_node_id": parent_node_id or None,
            "branch_id": branch_id,
            "branch_index": branch_index,
            "branch_action": branch_action,
            "boundary_axis": boundary_axis,
            "source_search_depth": source_depth,
            "target_search_depth": target_depth,
            "search_depth": source_depth,
            "max_search_depth": limits["max_search_depth"],
            "max_sample_branches": limits["max_sample_branches"],
            "max_sample_boundaries": limits["max_sample_boundaries"],
            "max_sample_candidates_total": limits["max_sample_candidates_total"],
            "sample_stop_status": "continue_branch_search",
        }
    )

    entry = _with_source_record(record, target_state, source_node_id)
    entry.update(
        {
            "sample_id": sample_key(record),
            "search_root_id": root_id,
            "source_node_id": source_node_id,
            "parent_node_id": parent_node_id or None,
            "branch_id": branch_id,
            "branch_index": branch_index,
            "branch_action": branch_action,
            "boundary_axis": boundary_axis,
            "source_search_depth": source_depth,
            "target_search_depth": target_depth,
            "search_depth": source_depth,
            "evolution_state": target_state,
        }
    )
    return entry


def build_active_frontier(
    records: Sequence[Dict[str, Any]],
    *,
    max_branches: int = DEFAULT_MAX_SAMPLE_BRANCHES,
    max_depth: int = DEFAULT_MAX_SAMPLE_DEPTH,
    max_boundaries: int = DEFAULT_MAX_SAMPLE_BOUNDARIES,
    max_candidates_total: int = DEFAULT_MAX_SAMPLE_CANDIDATES_TOTAL,
) -> List[Dict[str, Any]]:
    frontier: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if _is_scheduled_frontier_record(record):
            preserved = _preserve_scheduled_frontier_entry(
                record,
                branch_index_hint=index,
                max_branches=max_branches,
                max_depth=max_depth,
                max_boundaries=max_boundaries,
                max_candidates_total=max_candidates_total,
            )
            if preserved is not None:
                frontier.append(preserved)
            continue

        state = dict(_state(record)) or init_search_state(
            record,
            max_depth=max_depth,
            max_branches=max_branches,
            max_boundaries=max_boundaries,
            max_candidates_total=max_candidates_total,
        )
        root_id = _clean_text(state.get("search_root_id") or init_search_state(record)["search_root_id"])
        state.setdefault("node_records", {})
        state.setdefault("node_prompts", {})
        if isinstance(state["node_records"], dict):
            state["node_records"].setdefault(root_id, compact_node_record(record))
        if isinstance(state["node_prompts"], dict):
            state["node_prompts"].setdefault(root_id, _clean_text(record.get("prompt")))
        if _clean_text(state.get("sample_stop_status") or state.get("stop_status")) in TERMINAL_SAMPLE_STATUSES:
            continue
        identity = _search_identity({"evolution_state": state, **record}, branch_index_hint=index)
        axis = resolve_boundary_axis(record) or first_unexplored_axis(
            axis_candidates=_axis_candidates(record, state),
            explored_axes=state.get("explored_axes") or [],
        )
        frontier.append(
            _frontier_entry(
                {"evolution_state": state, **record},
                branch_action=_clean_text(record.get("branch_action") or state.get("branch_action") or "expand_current_branch"),
                source_node_id=identity["current_node_id"],
                parent_node_id=identity["parent_node_id"],
                boundary_axis=axis,
                source_search_depth=identity["search_depth"],
                target_search_depth=identity["search_depth"] + 1,
                branch_index=identity["branch_index"],
                max_branches=max_branches,
                max_depth=max_depth,
                max_boundaries=max_boundaries,
                max_candidates_total=max_candidates_total,
            )
        )
    return frontier


def schedule_next_frontier(
    records: Sequence[Dict[str, Any]],
    *,
    max_branches: int = DEFAULT_MAX_SAMPLE_BRANCHES,
    max_depth: int = DEFAULT_MAX_SAMPLE_DEPTH,
    max_boundaries: int = DEFAULT_MAX_SAMPLE_BOUNDARIES,
    max_candidates_total: int = DEFAULT_MAX_SAMPLE_CANDIDATES_TOTAL,
    enable_branch_backtrack: bool = True,
    enable_root_fork: bool = True,
) -> List[Dict[str, Any]]:
    next_frontier: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        state = dict(_state(record))
        if not state:
            state = init_search_state(
                record,
                max_depth=max_depth,
                max_branches=max_branches,
                max_boundaries=max_boundaries,
                max_candidates_total=max_candidates_total,
            )
        limits = _max_limits(
            state,
            max_branches=max_branches,
            max_depth=max_depth,
            max_boundaries=max_boundaries,
            max_candidates_total=max_candidates_total,
        )
        identity = _search_identity({"evolution_state": state, **record}, branch_index_hint=index)
        effect = _effect(record)
        label = _clean_text(effect.get("effect_label") or state.get("previous_effect_status"))
        current_axis = (
            _clean_text(effect.get("boundary_axis_detected"))
            or resolve_boundary_axis(record)
            or _clean_text(state.get("boundary_axis"))
        )
        explored_axes = list(state.get("explored_axes") or [])
        if label in {
            "effective_boundary_probe",
            "repeated_pattern",
            "score_increased",
            "full_score_no_drop",
            "no_clear_effect",
            "invalid_complexity",
        }:
            _append_unique(explored_axes, [current_axis])
        next_axis = first_unexplored_axis(
            axis_candidates=_axis_candidates(record, state),
            explored_axes=explored_axes,
            fallback_axis=current_axis,
        )
        next_branch_index = _coerce_int(state.get("branch_count"), 0) + 1
        current_depth = _coerce_int(identity["search_depth"], _coerce_int(state.get("search_depth"), 0))

        action = "expand_current_branch"
        source_node_id = identity["current_node_id"]
        parent_node_id = identity["parent_node_id"]
        source_depth = current_depth
        target_depth = min(current_depth + 1, limits["max_search_depth"])
        axis = current_axis or next_axis

        if _boundary_count(state) >= limits["max_sample_boundaries"]:
            continue
        if _coerce_int(state.get("sample_candidates_used"), 0) >= limits["max_sample_candidates_total"]:
            continue

        if label == "effective_boundary_probe":
            if enable_branch_backtrack and parent_node_id and next_axis:
                action = "fork_from_parent"
                source_node_id = parent_node_id
                source_depth = max(0, current_depth - 1)
                target_depth = max(1, current_depth)
                axis = next_axis
            elif enable_root_fork and next_axis:
                action = "fork_from_root"
                source_node_id = identity["search_root_id"]
                parent_node_id = identity["search_root_id"]
                source_depth = 0
                target_depth = 1
                axis = next_axis
            else:
                continue
        elif label == "invalid_complexity":
            if enable_branch_backtrack and parent_node_id:
                action = "fork_from_parent"
                source_node_id = parent_node_id
                source_depth = max(0, current_depth - 1)
                target_depth = max(1, current_depth)
            elif enable_root_fork:
                action = "fork_from_root"
                source_node_id = identity["search_root_id"]
                parent_node_id = identity["search_root_id"]
                source_depth = 0
                target_depth = 1
                axis = next_axis
            else:
                continue
        elif label == "repeated_pattern":
            if not enable_root_fork or not next_axis:
                continue
            action = "fork_from_root"
            source_node_id = identity["search_root_id"]
            parent_node_id = identity["search_root_id"]
            source_depth = 0
            target_depth = 1
            axis = next_axis
        elif label == "full_score_no_drop":
            if current_depth < limits["max_search_depth"]:
                action = "expand_current_branch"
                source_depth = current_depth
                target_depth = current_depth + 1
            elif enable_root_fork and next_axis:
                action = "fork_from_root"
                source_node_id = identity["search_root_id"]
                parent_node_id = identity["search_root_id"]
                source_depth = 0
                target_depth = 1
                axis = next_axis
            else:
                continue
        elif label == "no_clear_effect":
            if current_depth < limits["max_search_depth"]:
                action = "expand_current_branch"
                source_depth = current_depth
                target_depth = current_depth + 1
            elif enable_root_fork and next_axis:
                action = "fork_from_root"
                source_node_id = identity["search_root_id"]
                parent_node_id = identity["search_root_id"]
                source_depth = 0
                target_depth = 1
                axis = next_axis
            else:
                continue
        elif label == "score_increased":
            if enable_branch_backtrack and parent_node_id and next_axis:
                action = "fork_from_parent"
                source_node_id = parent_node_id
                source_depth = max(0, current_depth - 1)
                target_depth = max(1, current_depth)
                axis = next_axis
            elif enable_root_fork and next_axis:
                action = "fork_from_root"
                source_node_id = identity["search_root_id"]
                parent_node_id = identity["search_root_id"]
                source_depth = 0
                target_depth = 1
                axis = next_axis
            else:
                continue
        elif label == "pass_through":
            continue
        else:
            continue

        if target_depth < source_depth:
            continue
        if action == "expand_current_branch" and target_depth <= source_depth:
            continue

        if not _budget_allows_new_entry(state, limits=limits, action=action, target_depth=target_depth):
            continue

        next_frontier.append(
            _frontier_entry(
                record,
                branch_action=action,
                source_node_id=source_node_id,
                parent_node_id=parent_node_id,
                boundary_axis=axis or DEFAULT_BOUNDARY_AXES[0],
                source_search_depth=source_depth,
                target_search_depth=target_depth,
                branch_index=next_branch_index if action in {"fork_from_parent", "fork_from_root"} else identity["branch_index"],
                max_branches=limits["max_sample_branches"],
                max_depth=limits["max_search_depth"],
                max_boundaries=limits["max_sample_boundaries"],
                max_candidates_total=limits["max_sample_candidates_total"],
            )
        )
    return next_frontier


def summarize_discovered_boundaries(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    seen: Dict[str, set] = {}
    for record in records:
        state = _state(record)
        sample_id = sample_key(record)
        summary = summaries.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "search_root_id": _clean_text(state.get("search_root_id") or init_search_state(record)["search_root_id"]),
                "boundary_count": 0,
                "boundaries": [],
                "sample_stop_status": _clean_text(state.get("sample_stop_status") or "continue_branch_search"),
            },
        )
        seen.setdefault(sample_id, set())
        boundaries = state.get("discovered_boundaries")
        if isinstance(boundaries, list):
            for boundary in boundaries:
                if not isinstance(boundary, dict):
                    continue
                signature = _clean_text(boundary.get("dedup_signature") or boundary.get("boundary_signature"))
                if signature and signature not in seen[sample_id]:
                    seen[sample_id].add(signature)
                    summary["boundaries"].append(boundary)
        effect = _effect(record)
        if effect.get("effect_label") == "effective_boundary_probe" and effect.get("is_new_boundary_for_sample") is not False:
            signature = _clean_text(effect.get("boundary_signature")) or boundary_signature(record)
            if signature not in seen[sample_id]:
                seen[sample_id].add(signature)
                summary["boundaries"].append(
                    {
                        "boundary_id": f"boundary_{len(seen[sample_id]):03d}",
                        "boundary_axis": _clean_text(effect.get("boundary_axis_detected")) or resolve_boundary_axis(record),
                        "trigger_node_id": _clean_text(state.get("current_node_id")),
                        "node_id": _clean_text(state.get("current_node_id")),
                        "parent_node_id": _clean_text(state.get("parent_node_id")) or None,
                        "depth": _coerce_int(state.get("search_depth"), 0),
                        "branch_id": _clean_text(state.get("branch_id")),
                        "operator_used": _clean_text(effect.get("operator_used")),
                        "score_rate_before": effect.get("score_rate_before"),
                        "score_rate_after": effect.get("score_rate_after"),
                        "effect_label": effect.get("effect_label"),
                        "dedup_signature": signature,
                    }
                )
        summary["sample_stop_status"] = _clean_text(
            state.get("sample_stop_status") or state.get("stop_status") or summary["sample_stop_status"]
        )

    for summary in summaries.values():
        summary["boundary_count"] = len(summary["boundaries"])
    return sorted(summaries.values(), key=lambda item: item["sample_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build active/next frontier files for tree-search mode.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output frontier JSONL path.")
    parser.add_argument("--mode", choices=["active", "schedule"], default="schedule")
    parser.add_argument("--discovered-output", default=None, help="Optional discovered boundaries summary JSONL path.")
    parser.add_argument("--max-sample-branches", type=int, default=DEFAULT_MAX_SAMPLE_BRANCHES)
    parser.add_argument("--max-sample-depth", type=int, default=DEFAULT_MAX_SAMPLE_DEPTH)
    parser.add_argument("--max-sample-boundaries", type=int, default=DEFAULT_MAX_SAMPLE_BOUNDARIES)
    parser.add_argument("--max-sample-candidates-total", type=int, default=DEFAULT_MAX_SAMPLE_CANDIDATES_TOTAL)
    parser.add_argument("--enable-branch-backtrack", default="true")
    parser.add_argument("--enable-root-fork", default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_or_jsonl(args.input)
    if args.mode == "active":
        frontier = build_active_frontier(
            records,
            max_branches=args.max_sample_branches,
            max_depth=args.max_sample_depth,
            max_boundaries=args.max_sample_boundaries,
            max_candidates_total=args.max_sample_candidates_total,
        )
    else:
        frontier = schedule_next_frontier(
            records,
            max_branches=args.max_sample_branches,
            max_depth=args.max_sample_depth,
            max_boundaries=args.max_sample_boundaries,
            max_candidates_total=args.max_sample_candidates_total,
            enable_branch_backtrack=_bool_from_text(args.enable_branch_backtrack),
            enable_root_fork=_bool_from_text(args.enable_root_fork),
        )
    write_jsonl(frontier, args.output)
    if args.discovered_output:
        write_jsonl(summarize_discovered_boundaries(records), args.discovered_output)


if __name__ == "__main__":
    main()
