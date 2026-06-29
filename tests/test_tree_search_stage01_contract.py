import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema_validation import validate_records_against_schema
from search_state_contract import (
    TREE_SEARCH_CONFIG_DEFAULTS,
    attach_normalized_search_state,
    build_active_frontier_node,
    build_search_graph_node,
    normalize_search_state,
)


def load_jsonl(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def test_legacy_single_chain_records_get_tree_search_defaults():
    record = load_jsonl(ROOT / "tests" / "fixtures" / "stage01_contract.jsonl")[0]

    state = normalize_search_state(record)

    assert state["search_root_id"] == "sample_fixture-801_root"
    assert state["current_node_id"] == "sample_fixture-801_root"
    assert state["parent_node_id"] is None
    assert state["branch_id"] == "main"
    assert state["branch_status"] == "exploring"
    assert state["search_depth"] == 0
    assert state["max_search_depth"] == TREE_SEARCH_CONFIG_DEFAULTS["MAX_SAMPLE_DEPTH"]
    assert state["branch_budget_remaining"] == TREE_SEARCH_CONFIG_DEFAULTS["MAX_SAMPLE_DEPTH"]
    assert state["sample_budget_remaining"] == TREE_SEARCH_CONFIG_DEFAULTS["MAX_SAMPLE_CANDIDATES_TOTAL"]
    assert state["discovered_boundaries"] == []
    assert state["recommended_next_axes"] == []
    assert state["stop_status"] == "continue"


def test_evolved_record_defaults_to_root_parent_and_depth_one():
    record = load_jsonl(ROOT / "tests" / "fixtures" / "stage01_contract.jsonl")[2]

    state = normalize_search_state(record)

    assert state["search_root_id"] == "sample_fixture-evolved_root"
    assert state["current_node_id"] != state["search_root_id"]
    assert state["parent_node_id"] == state["search_root_id"]
    assert state["branch_id"] == "main"
    assert state["search_depth"] == 1
    assert state["branch_budget_remaining"] == TREE_SEARCH_CONFIG_DEFAULTS["MAX_SAMPLE_DEPTH"] - 1


def test_normalized_state_keeps_pipeline_schema_compatible():
    records = [
        attach_normalized_search_state(record)
        for record in load_jsonl(ROOT / "tests" / "fixtures" / "stage01_contract.jsonl")
    ]

    errors = validate_records_against_schema(records, ROOT / "schemas" / "pipeline_record.schema.json")

    assert errors == []


def test_search_graph_and_frontier_fixtures_match_schemas():
    graph_records = load_jsonl(ROOT / "tests" / "fixtures" / "stage01_search_graph.jsonl")
    frontier_records = load_jsonl(ROOT / "tests" / "fixtures" / "stage01_active_frontier.jsonl")

    graph_errors = validate_records_against_schema(
        graph_records,
        ROOT / "schemas" / "search_graph_node.schema.json",
    )
    frontier_errors = validate_records_against_schema(
        frontier_records,
        ROOT / "schemas" / "active_frontier.schema.json",
    )

    assert graph_errors == []
    assert frontier_errors == []


def test_contract_builders_emit_schema_valid_rows_from_legacy_record():
    record = load_jsonl(ROOT / "tests" / "fixtures" / "stage01_contract.jsonl")[0]

    graph_node = build_search_graph_node(record)
    frontier_node = build_active_frontier_node(
        record,
        action_type="expand_current_branch",
        target_boundary_axis="最关键缺口识别",
    )

    assert validate_records_against_schema([graph_node], ROOT / "schemas" / "search_graph_node.schema.json") == []
    assert validate_records_against_schema([frontier_node], ROOT / "schemas" / "active_frontier.schema.json") == []
    assert graph_node["node_id"] == "sample_fixture-801_root"
    assert graph_node["selected_into_mainline"] is True
    assert frontier_node["source_node_id"] == graph_node["node_id"]
    assert frontier_node["next_depth"] == 1


def test_root_fork_frontier_uses_prompt_old_when_available():
    record = load_jsonl(ROOT / "tests" / "fixtures" / "stage01_contract.jsonl")[2]

    frontier_node = build_active_frontier_node(
        record,
        action_type="fork_from_root",
        source_node_type="root",
        target_boundary_axis="伪闭环识别",
    )

    assert frontier_node["source_node_id"] == "sample_fixture-evolved_root"
    assert frontier_node["source_prompt"] == "Original fixture question."
    assert frontier_node["prompt_source"] == "meta_info.prompt_old"
    assert frontier_node["next_depth"] == 1
