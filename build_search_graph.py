import argparse
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence

from search_state import (
    boundary_signature,
    get_operator_used,
    init_search_state,
    make_branch_id,
    make_node_id,
    resolve_boundary_axis,
    sample_key,
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


def _coerce_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _effect(record: Dict[str, Any]) -> Dict[str, Any]:
    effect = record.get("effect_analysis")
    return effect if isinstance(effect, dict) else {}


def _generation(record: Dict[str, Any]) -> Dict[str, Any]:
    generation = record.get("candidate_generation")
    return generation if isinstance(generation, dict) else {}


def _state(record: Dict[str, Any]) -> Dict[str, Any]:
    state = record.get("evolution_state")
    return state if isinstance(state, dict) else {}


def build_search_node(record: Dict[str, Any], *, fallback_branch_index: int = 1) -> Dict[str, Any]:
    state = _state(record)
    generation = _generation(record)
    effect = _effect(record)
    base_state = init_search_state(record)
    search_root_id = _clean_text(
        generation.get("search_root_id")
        or state.get("search_root_id")
        or record.get("search_root_id")
        or base_state["search_root_id"]
    )
    operator_used = _clean_text(effect.get("operator_used")) or get_operator_used(record)
    axis = _clean_text(effect.get("boundary_axis_detected")) or resolve_boundary_axis(record, operator_used)
    branch_index = _coerce_int(
        generation.get("branch_index") or state.get("branch_count"),
        fallback_branch_index,
    )
    depth = _coerce_int(
        generation.get("search_depth") or state.get("search_depth") or record.get("search_depth"),
        0 if record.get("question_evolved") is False else 1,
    )
    branch_id = _clean_text(
        generation.get("branch_id")
        or state.get("branch_id")
        or record.get("branch_id")
        or make_branch_id(operator_used or "unknown_operator", axis or "unknown_axis", branch_index)
    )
    node_id = _clean_text(
        generation.get("node_id")
        or generation.get("candidate_node_id")
        or state.get("current_node_id")
        or record.get("current_node_id")
        or make_node_id(search_root_id, branch_index, depth)
    )
    parent_node_id = _clean_text(
        generation.get("parent_node_id")
        or state.get("parent_node_id")
        or record.get("parent_node_id")
        or (search_root_id if depth > 0 else "")
    )
    if depth == 0 or node_id == search_root_id:
        parent_node_id = ""
    signature = _clean_text(effect.get("boundary_signature")) or boundary_signature(record)
    effect_label = _clean_text(effect.get("effect_label"))
    is_boundary_hit = effect_label == "effective_boundary_probe" or bool(effect.get("lightweight_boundary_hit"))
    selection = record.get("candidate_selection")
    selection = selection if isinstance(selection, dict) else {}
    selected_into_mainline = selection.get("selected_into_mainline")
    if selected_into_mainline is None:
        selected_into_mainline = record.get("question_evolved") is not False

    return {
        "sample_id": sample_key(record),
        "search_root_id": search_root_id,
        "node_id": node_id,
        "parent_node_id": parent_node_id or None,
        "branch_id": branch_id,
        "branch_action": _clean_text(generation.get("branch_action") or state.get("branch_action") or "expand_current_branch"),
        "depth": depth,
        "prompt": _clean_text(record.get("prompt")),
        "operator_used": operator_used,
        "boundary_axis": axis,
        "score_rate_before": _coerce_float(effect.get("score_rate_before")),
        "score_rate_after": _coerce_float(effect.get("score_rate_after")),
        "effect_label": effect_label,
        "is_boundary_hit": bool(is_boundary_hit),
        "selected_into_mainline": bool(selected_into_mainline),
        "selected_as_boundary_leaf": bool(is_boundary_hit and not effect.get("discard_as_duplicate")),
        "discard_as_duplicate": bool(effect.get("is_new_boundary_for_sample") is False and is_boundary_hit),
        "dedup_signature": signature,
    }


def build_search_graph(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        nodes.append(build_search_node(record, fallback_branch_index=index))
    return nodes


def merge_search_graphs(
    previous_nodes: Sequence[Dict[str, Any]],
    current_nodes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for node in list(previous_nodes) + list(current_nodes):
        key = (
            _clean_text(node.get("sample_id")),
            _clean_text(node.get("node_id")),
            _clean_text(node.get("branch_id")),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(node)
    return merged


def _boundary_with_node_fields(boundary: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(boundary)
    node_id = _clean_text(enriched.get("node_id") or enriched.get("trigger_node_id"))
    if not node_id:
        node_id = _clean_text(node.get("node_id"))
    enriched["node_id"] = node_id
    enriched["trigger_node_id"] = _clean_text(enriched.get("trigger_node_id") or node_id)
    if node_id == _clean_text(node.get("node_id")):
        enriched.setdefault("parent_node_id", node.get("parent_node_id"))
        enriched.setdefault("depth", node.get("depth"))
    return enriched


def build_discovered_boundaries(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "sample_id": "",
            "search_root_id": "",
            "boundary_count": 0,
            "boundaries": [],
            "sample_stop_status": "continue_branch_search",
        }
    )
    seen: DefaultDict[str, set] = defaultdict(set)
    for record in records:
        state = _state(record)
        node = build_search_node(record)
        sample_id = node["sample_id"]
        bucket = grouped[sample_id]
        bucket["sample_id"] = sample_id
        bucket["search_root_id"] = node["search_root_id"]
        bucket["sample_stop_status"] = _clean_text(
            state.get("sample_stop_status") or state.get("stop_status") or bucket["sample_stop_status"]
        )

        boundaries = state.get("discovered_boundaries")
        if isinstance(boundaries, list) and boundaries:
            for boundary in boundaries:
                if not isinstance(boundary, dict):
                    continue
                signature = _clean_text(boundary.get("dedup_signature") or boundary.get("boundary_signature"))
                if not signature or signature in seen[sample_id]:
                    continue
                seen[sample_id].add(signature)
                bucket["boundaries"].append(_boundary_with_node_fields(boundary, node))
            continue

        if node["is_boundary_hit"] and not node["discard_as_duplicate"]:
            signature = node["dedup_signature"]
            if signature not in seen[sample_id]:
                seen[sample_id].add(signature)
                bucket["boundaries"].append(
                    {
                        "boundary_id": f"boundary_{len(seen[sample_id]):03d}",
                        "boundary_axis": node["boundary_axis"],
                        "trigger_node_id": node["node_id"],
                        "node_id": node["node_id"],
                        "parent_node_id": node["parent_node_id"],
                        "depth": node["depth"],
                        "branch_id": node["branch_id"],
                        "operator_used": node["operator_used"],
                        "score_rate_before": node["score_rate_before"],
                        "score_rate_after": node["score_rate_after"],
                        "effect_label": node["effect_label"],
                        "dedup_signature": signature,
                    }
                )

    summaries: List[Dict[str, Any]] = []
    for bucket in grouped.values():
        bucket["boundary_count"] = len(bucket["boundaries"])
        summaries.append(bucket)
    summaries.sort(key=lambda item: item["sample_id"])
    return summaries


def merge_discovered_boundaries(
    previous_summaries: Sequence[Dict[str, Any]],
    current_summaries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    seen: Dict[str, set] = {}
    for summary in list(previous_summaries) + list(current_summaries):
        sample_id = _clean_text(summary.get("sample_id"))
        if not sample_id:
            continue
        bucket = merged.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "search_root_id": _clean_text(summary.get("search_root_id")),
                "boundary_count": 0,
                "boundaries": [],
                "sample_stop_status": _clean_text(summary.get("sample_stop_status") or "continue_branch_search"),
            },
        )
        if _clean_text(summary.get("search_root_id")):
            bucket["search_root_id"] = _clean_text(summary.get("search_root_id"))
        if _clean_text(summary.get("sample_stop_status")):
            bucket["sample_stop_status"] = _clean_text(summary.get("sample_stop_status"))
        seen.setdefault(sample_id, set())
        boundaries = summary.get("boundaries")
        if not isinstance(boundaries, list):
            continue
        for boundary in boundaries:
            if not isinstance(boundary, dict):
                continue
            signature = _clean_text(boundary.get("dedup_signature") or boundary.get("boundary_signature"))
            if not signature:
                signature = "|".join(
                    [
                        sample_id,
                        _clean_text(boundary.get("boundary_axis")),
                        _clean_text(boundary.get("operator_used")),
                    ]
                )
            if signature in seen[sample_id]:
                continue
            seen[sample_id].add(signature)
            bucket["boundaries"].append(boundary)
    for bucket in merged.values():
        bucket["boundary_count"] = len(bucket["boundaries"])
    return sorted(merged.values(), key=lambda item: item["sample_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tree-search artifacts from state/effect records.")
    parser.add_argument("--input", required=True, help="Input state/effect JSONL path.")
    parser.add_argument("--output", required=True, help="Output search_graph.jsonl path.")
    parser.add_argument("--discovered-output", default=None, help="Optional discovered_boundaries.jsonl output path.")
    parser.add_argument("--previous-graph", default=None, help="Optional previous search_graph JSONL to merge.")
    parser.add_argument("--combined-output", default=None, help="Optional cumulative search_graph JSONL output path.")
    parser.add_argument("--previous-discovered", default=None, help="Optional previous discovered_boundaries JSONL to merge.")
    parser.add_argument("--combined-discovered-output", default=None, help="Optional cumulative discovered_boundaries JSONL output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_or_jsonl(args.input)
    current_graph = build_search_graph(records)
    write_jsonl(current_graph, args.output)
    if args.combined_output:
        previous_graph = load_json_or_jsonl(args.previous_graph) if args.previous_graph and os.path.exists(args.previous_graph) else []
        write_jsonl(merge_search_graphs(previous_graph, current_graph), args.combined_output)
    if args.discovered_output:
        current_discovered = build_discovered_boundaries(records)
        write_jsonl(current_discovered, args.discovered_output)
        if args.combined_discovered_output:
            previous_discovered = (
                load_json_or_jsonl(args.previous_discovered)
                if args.previous_discovered and os.path.exists(args.previous_discovered)
                else []
            )
            write_jsonl(
                merge_discovered_boundaries(previous_discovered, current_discovered),
                args.combined_discovered_output,
            )


if __name__ == "__main__":
    main()
