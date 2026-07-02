import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from schema_validation import load_schema, validate_instance
from search_state import (
    budget_exhausted,
    init_search_state,
    make_branch_id,
    make_node_id,
    sample_key,
)


def test_search_state_ids_are_stable_and_budgeted():
    record = {"sample_id": "stage-tree-01", "prompt": "原题。"}
    state = init_search_state(
        record,
        max_depth=2,
        max_branches=3,
        max_boundaries=2,
        max_candidates_total=6,
    )

    assert sample_key(record) == "stage-tree-01"
    assert state["search_root_id"] == "sample_stage_tree_01_root"
    assert make_node_id(state["search_root_id"], 1, 1) != make_node_id(state["search_root_id"], 2, 1)
    assert make_branch_id("O1_gap_choice", "最小关键事实识别", 1).startswith("branch_o1_gap_choice_")
    assert budget_exhausted({**state, "sample_candidates_used": 6}) is True


def test_search_node_and_optional_evolution_state_schema_contracts():
    schema_dir = ROOT / "schemas"
    node_schema = load_schema(schema_dir / "search_node.schema.json")
    state_schema = load_schema(schema_dir / "evolution_state.schema.json")

    validate_instance(
        {
            "sample_id": "801",
            "search_root_id": "sample_801_root",
            "node_id": "sample_801_root_b1_d1",
            "parent_node_id": "sample_801_root",
            "branch_id": "branch_o1_axis_001",
            "branch_action": "fork_from_parent",
            "depth": 1,
            "prompt": "新题",
            "is_boundary_hit": True,
        },
        node_schema,
        schema_dir=schema_dir,
    )
    validate_instance({"round": 0, "stop_status": "continue"}, state_schema, schema_dir=schema_dir)


if __name__ == "__main__":
    test_search_state_ids_are_stable_and_budgeted()
    test_search_node_and_optional_evolution_state_schema_contracts()
    print("tree stage01 contract checks passed")
