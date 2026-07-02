import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from build_search_graph import build_discovered_boundaries, build_search_graph
from frontier_scheduler import schedule_next_frontier
from update_sample_state import update_records


def boundary_hit(axis, operator, *, state=None, node_suffix="b1_d1"):
    return {
        "sample_id": "fake-tree",
        "prompt": f"进化题：{axis}",
        "question_evolved": True,
        "candidate_generation": {
            "search_root_id": "sample_fake_tree_root",
            "candidate_node_id": f"sample_fake_tree_root_{node_suffix}",
            "parent_node_id": "sample_fake_tree_root",
            "branch_id": f"branch_{operator}",
            "branch_action": "expand_current_branch" if state is None else "fork_from_parent",
            "boundary_axis": axis,
            "search_depth": 1,
            "branch_index": 1 if state is None else 2,
        },
        "meta_info": {
            "prompt_old": "原始 root 题面：判断结论是否成立。",
            "question_evolution_metadata": {
                "question_evolved": True,
                "operator_used": operator,
                "ability_axis": axis,
            }
        },
        "validation_result": {"passed": True},
        "evolution_state": state
        or {
            "round": 1,
            "stop_status": "continue_with_new_operator",
            "search_root_id": "sample_fake_tree_root",
            "max_sample_branches": 2,
            "max_search_depth": 2,
            "max_sample_boundaries": 2,
            "max_sample_candidates_total": 4,
            "recommended_next_axes": ["最小关键事实识别", "子判断定位"],
        },
        "effect_analysis": {
            "effect_label": "effective_boundary_probe",
            "operator_used": operator,
            "score_rate_before": 1.0,
            "score_rate_after": 0.5,
            "boundary_axis_detected": axis,
            "boundary_signature": f"fake-tree|{axis}|{operator}",
            "is_new_boundary_for_sample": True,
            "complexity_passed": True,
            "lightweight_boundary_hit": True,
        },
    }


def test_fake_backtrack_pipeline_finds_two_boundaries_then_stops_sample():
    first_updated, _, _, _ = update_records(
        [boundary_hit("最小关键事实识别", "O1_gap_choice", node_suffix="b1_d1")]
    )
    first_state = first_updated[0]["evolution_state"]
    assert len(first_state["discovered_boundaries"]) == 1
    assert first_state["sample_stop_status"] == "continue_branch_search"

    next_frontier = schedule_next_frontier(first_updated, max_branches=2, max_boundaries=2, max_candidates_total=4)
    assert next_frontier[0]["branch_action"] == "fork_from_parent"
    assert next_frontier[0]["boundary_axis"] == "子判断定位"
    assert next_frontier[0]["prompt"] == "原始 root 题面：判断结论是否成立。"

    second = boundary_hit(
        "子判断定位",
        "O2_subclaim_localization",
        state=next_frontier[0]["evolution_state"],
        node_suffix="b2_d1",
    )
    second_updated, _, _, _ = update_records([second])
    second_state = second_updated[0]["evolution_state"]
    assert len(second_state["discovered_boundaries"]) == 2
    assert second_state["sample_stop_status"] == "max_boundaries_reached"
    assert schedule_next_frontier(second_updated, max_branches=2, max_boundaries=2) == []

    all_updated = [first_updated[0], second_updated[0]]
    graph = build_search_graph(all_updated)
    assert [node["is_boundary_hit"] for node in graph] == [True, True]
    summary = build_discovered_boundaries(second_updated)
    assert summary[0]["sample_id"] == "fake-tree"
    assert summary[0]["boundary_count"] == 2
    assert {b["boundary_axis"] for b in summary[0]["boundaries"]} == {"最小关键事实识别", "子判断定位"}


def test_fake_backtrack_pipeline_can_fork_from_root_when_parent_backtrack_disabled():
    first_updated, _, _, _ = update_records(
        [boundary_hit("最小关键事实识别", "O1_gap_choice", node_suffix="b1_d1")]
    )

    next_frontier = schedule_next_frontier(
        first_updated,
        max_branches=2,
        max_boundaries=2,
        max_candidates_total=4,
        enable_branch_backtrack=False,
        enable_root_fork=True,
    )

    assert len(next_frontier) == 1
    assert next_frontier[0]["branch_action"] == "fork_from_root"
    assert next_frontier[0]["source_node_id"] == "sample_fake_tree_root"
    assert next_frontier[0]["parent_node_id"] == "sample_fake_tree_root"
    assert next_frontier[0]["boundary_axis"] == "子判断定位"
    assert next_frontier[0]["prompt"] == "原始 root 题面：判断结论是否成立。"


if __name__ == "__main__":
    test_fake_backtrack_pipeline_finds_two_boundaries_then_stops_sample()
    test_fake_backtrack_pipeline_can_fork_from_root_when_parent_backtrack_disabled()
    print("tree stage08 fake backtrack pipeline checks passed")
