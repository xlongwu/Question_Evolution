import argparse
import json
import os
from typing import Any, Dict, Iterable, List

from question_evolution import expand_items_from_frontier, load_json_or_jsonl


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_frontier_records(
    base_records: List[Dict[str, Any]],
    frontier_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not frontier_records:
        return list(base_records)
    expanded = expand_items_from_frontier(base_records, frontier_records)
    prepared: List[Dict[str, Any]] = []
    for record in expanded:
        item = dict(record)
        frontier = item.get("_frontier_context")
        frontier = frontier if isinstance(frontier, dict) else {}
        source_prompt = str(frontier.get("source_prompt") or "").strip()
        if source_prompt:
            meta_info = item.get("meta_info")
            meta_info = dict(meta_info) if isinstance(meta_info, dict) else {}
            metadata = meta_info.get("question_evolution_metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            metadata["frontier_prompt_overlaid"] = True
            metadata["frontier_base_prompt"] = item.get("prompt")
            metadata["frontier_prompt_source"] = frontier.get("prompt_source")
            meta_info["question_evolution_metadata"] = metadata
            item["meta_info"] = meta_info
            item["prompt"] = source_prompt
        prepared.append(item)
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay active_frontier rows onto base pipeline records before profile/select/router stages."
    )
    parser.add_argument("--input", required=True, help="Base scored/state JSONL records.")
    parser.add_argument("--frontier-input", required=True, help="active_frontier.jsonl from the previous round.")
    parser.add_argument("--output", required=True, help="frontier-expanded records for this round.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_records = load_json_or_jsonl(args.input)
    frontier_records = load_json_or_jsonl(args.frontier_input)
    write_jsonl(prepare_frontier_records(base_records, frontier_records), args.output)


if __name__ == "__main__":
    main()
