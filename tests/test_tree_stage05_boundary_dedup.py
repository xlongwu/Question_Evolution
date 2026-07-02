import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from analyze_evolution_effect import analyze_records
from build_search_graph import build_discovered_boundaries, build_search_graph
from update_sample_state import update_records


def scored_boundary_record(axis, operator, *, previous_boundaries=None):
    signature = f"dedup-sample|{axis}|{operator}"
    return {
        "sample_id": "dedup-sample",
        "prompt": f"进化题：{axis}",
        "question_evolved": True,
        "meta_info": {
            "references": ["参考答案。"],
            "question_evolution_metadata": {
                "question_evolved": True,
                "trigger_score_rate": 1.0,
                "operator_used": operator,
                "ability_axis": axis,
                "expected_evaluation_focus": [axis],
            },
        },
        "validation_result": {"passed": True, "repeat_pattern_risk": "low"},
        "candidate_generation": {
            "search_root_id": "sample_dedup_sample_root",
            "candidate_node_id": f"sample_dedup_sample_root_{operator}",
            "parent_node_id": "sample_dedup_sample_root",
            "branch_id": f"branch_{operator}",
            "branch_action": "fork_from_parent",
            "boundary_axis": axis,
            "search_depth": 1,
        },
        "evolution_state": {
            "round": 1,
            "stop_status": "continue_with_new_operator",
            "search_root_id": "sample_dedup_sample_root",
            "max_sample_boundaries": 2,
            "discovered_boundaries": previous_boundaries or [],
            "explored_axes": [boundary["boundary_axis"] for boundary in previous_boundaries or []],
        },
        "scoring_result": {
            "candidate_answer": f"候选答案错误地讨论了{axis}。",
            "total_awarded": 5,
            "total_possible": 10,
        },
    }


def test_boundary_dedup_keeps_duplicate_axis_out_and_counts_new_axis():
    existing = {
        "boundary_id": "boundary_001",
        "boundary_axis": "最小关键事实识别",
        "trigger_node_id": "sample_dedup_sample_root_b1_d1",
        "branch_id": "branch_o1",
        "operator_used": "O1_gap_choice",
        "effect_label": "effective_boundary_probe",
        "dedup_signature": "dedup-sample|最小关键事实识别|O1_gap_choice",
    }

    duplicate = analyze_records(
        [scored_boundary_record("最小关键事实识别", "O1_gap_choice", previous_boundaries=[existing])]
    )[0]
    assert duplicate["effect_analysis"]["effect_label"] == "effective_boundary_probe"
    assert duplicate["effect_analysis"]["is_new_boundary_for_sample"] is False
    updated_duplicate, operator_memory, _, _ = update_records([duplicate])
    assert len(updated_duplicate[0]["evolution_state"]["discovered_boundaries"]) == 1
    assert operator_memory == []

    new_axis = analyze_records(
        [scored_boundary_record("子判断定位", "O2_subclaim_localization", previous_boundaries=[existing])]
    )[0]
    assert new_axis["effect_analysis"]["is_new_boundary_for_sample"] is True
    updated_new, operator_memory, _, _ = update_records([new_axis])
    state = updated_new[0]["evolution_state"]
    assert len(state["discovered_boundaries"]) == 2
    assert state["sample_stop_status"] == "max_boundaries_reached"
    assert operator_memory
    new_boundary = next(
        boundary for boundary in state["discovered_boundaries"] if boundary["boundary_axis"] == "子判断定位"
    )
    assert new_boundary["node_id"] == "sample_dedup_sample_root_O2_subclaim_localization"
    assert new_boundary["trigger_node_id"] == new_boundary["node_id"]
    assert new_boundary["parent_node_id"] == "sample_dedup_sample_root"
    assert new_boundary["depth"] == 1

    graph = build_search_graph(updated_new)
    assert graph[0]["is_boundary_hit"] is True
    summary = build_discovered_boundaries(updated_new)
    assert summary[0]["boundary_count"] == 2
    summary_boundary = next(
        boundary for boundary in summary[0]["boundaries"] if boundary["boundary_axis"] == "子判断定位"
    )
    assert summary_boundary["node_id"] == new_boundary["node_id"]
    assert summary_boundary["parent_node_id"] == "sample_dedup_sample_root"
    assert summary_boundary["depth"] == 1


if __name__ == "__main__":
    test_boundary_dedup_keeps_duplicate_axis_out_and_counts_new_axis()
    print("tree stage05 boundary dedup checks passed")
