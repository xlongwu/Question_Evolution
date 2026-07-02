import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from build_search_graph import (
    build_discovered_boundaries,
    build_search_graph,
    merge_discovered_boundaries,
    merge_search_graphs,
)
from schema_validation import load_schema, validate_instance


def test_search_graph_marks_effective_boundary_node():
    record = {
        "sample_id": "tree-graph",
        "prompt": "进化题。",
        "question_evolved": True,
        "candidate_generation": {
            "search_root_id": "sample_tree_graph_root",
            "candidate_node_id": "sample_tree_graph_root_b1_d1",
            "parent_node_id": "sample_tree_graph_root",
            "branch_id": "branch_o1_axis_001",
            "branch_action": "expand_current_branch",
            "boundary_axis": "最小关键事实识别",
            "search_depth": 1,
        },
        "candidate_selection": {
            "selected_candidate_id": "tree-graph::cand_1",
            "selected_operator": "O1_gap_choice",
            "selected_into_mainline": True,
        },
        "effect_analysis": {
            "effect_label": "effective_boundary_probe",
            "operator_used": "O1_gap_choice",
            "score_rate_before": 1.0,
            "score_rate_after": 0.5,
            "boundary_axis_detected": "最小关键事实识别",
            "boundary_signature": "tree-graph|最小关键事实识别|O1_gap_choice",
            "is_new_boundary_for_sample": True,
        },
    }

    graph = build_search_graph([record])
    assert len(graph) == 1
    node = graph[0]
    assert node["node_id"] == "sample_tree_graph_root_b1_d1"
    assert node["parent_node_id"] == "sample_tree_graph_root"
    assert node["is_boundary_hit"] is True
    assert node["selected_as_boundary_leaf"] is True
    assert node["dedup_signature"] == "tree-graph|最小关键事实识别|O1_gap_choice"

    discovered = build_discovered_boundaries([record])
    assert discovered[0]["boundary_count"] == 1
    assert discovered[0]["boundaries"][0]["trigger_node_id"] == node["node_id"]


def test_search_graph_schema_matches_jsonl_node_and_merge_keeps_history():
    schema = load_schema(ROOT / "schemas" / "search_graph.schema.json")
    previous = [
        {
            "sample_id": "merge-sample",
            "search_root_id": "sample_merge_root",
            "node_id": "sample_merge_root_b1_d1",
            "parent_node_id": "sample_merge_root",
            "branch_id": "branch_o1_axis_001",
            "depth": 1,
            "prompt": "第一轮节点",
            "is_boundary_hit": True,
        }
    ]
    current = [
        {
            "sample_id": "merge-sample",
            "search_root_id": "sample_merge_root",
            "node_id": "sample_merge_root_b2_d1",
            "parent_node_id": "sample_merge_root",
            "branch_id": "branch_o2_axis_002",
            "depth": 1,
            "prompt": "第二轮节点",
            "is_boundary_hit": True,
        }
    ]
    validate_instance(previous[0], schema, schema_dir=ROOT / "schemas")
    merged = merge_search_graphs(previous, current)
    assert [node["node_id"] for node in merged] == ["sample_merge_root_b1_d1", "sample_merge_root_b2_d1"]

    summaries = merge_discovered_boundaries(
        [
            {
                "sample_id": "merge-sample",
                "search_root_id": "sample_merge_root",
                "boundary_count": 1,
                "boundaries": [{"boundary_axis": "最小关键事实识别", "dedup_signature": "sig-1"}],
            }
        ],
        [
            {
                "sample_id": "merge-sample",
                "search_root_id": "sample_merge_root",
                "boundary_count": 1,
                "boundaries": [{"boundary_axis": "子判断定位", "dedup_signature": "sig-2"}],
            }
        ],
    )
    assert summaries[0]["boundary_count"] == 2


def test_root_pass_through_node_has_no_parent_self_reference():
    record = {
        "sample_id": "tree-root-pass",
        "prompt": "未进化透传题。",
        "question_evolved": False,
        "evolution_state": {
            "search_root_id": "sample_tree_root_pass_root",
            "current_node_id": "sample_tree_root_pass_root",
            "parent_node_id": "sample_tree_root_pass_root",
            "search_depth": 0,
            "branch_action": "expand_current_branch",
        },
        "effect_analysis": {
            "effect_label": "pass_through",
        },
    }

    graph = build_search_graph([record])

    assert graph[0]["node_id"] == "sample_tree_root_pass_root"
    assert graph[0]["depth"] == 0
    assert graph[0]["parent_node_id"] is None


if __name__ == "__main__":
    test_search_graph_marks_effective_boundary_node()
    test_search_graph_schema_matches_jsonl_node_and_merge_keeps_history()
    test_root_pass_through_node_has_no_parent_self_reference()
    print("tree stage02 search graph checks passed")
