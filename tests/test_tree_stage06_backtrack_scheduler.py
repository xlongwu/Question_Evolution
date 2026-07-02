import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from frontier_scheduler import schedule_next_frontier


def record(label, *, depth=1, boundaries=1, candidates_used=1):
    return {
        "sample_id": f"scheduler-{label}",
        "prompt": "当前题。",
        "evolution_state": {
            "round": 1,
            "stop_status": "effective_boundary_sample" if label == "effective_boundary_probe" else "continue_with_new_operator",
            "search_root_id": f"sample_scheduler_{label}_root",
            "current_node_id": f"sample_scheduler_{label}_root_b1_d{depth}",
            "parent_node_id": f"sample_scheduler_{label}_root",
            "branch_id": "branch_o1",
            "boundary_axis": "最小关键事实识别",
            "search_depth": depth,
            "branch_count": 1,
            "sample_candidates_used": candidates_used,
            "max_sample_branches": 2,
            "max_search_depth": 2,
            "max_sample_boundaries": 2,
            "max_sample_candidates_total": 4,
            "discovered_boundaries": [
                {
                    "boundary_id": "boundary_001",
                    "boundary_axis": "最小关键事实识别",
                    "dedup_signature": f"scheduler-{label}|最小关键事实识别|O1_gap_choice",
                }
            ][:boundaries],
            "explored_axes": ["最小关键事实识别"],
            "recommended_next_axes": ["子判断定位", "近似层级排序"],
        },
        "effect_analysis": {
            "effect_label": label,
            "operator_used": "O1_gap_choice",
            "boundary_axis_detected": "最小关键事实识别",
            "is_new_boundary_for_sample": label == "effective_boundary_probe",
        },
    }


def test_scheduler_prioritizes_parent_fork_after_boundary_hit():
    next_frontier = schedule_next_frontier([record("effective_boundary_probe")], max_branches=2)
    assert len(next_frontier) == 1
    assert next_frontier[0]["branch_action"] == "fork_from_parent"
    assert next_frontier[0]["source_node_id"] == "sample_scheduler_effective_boundary_probe_root"
    assert next_frontier[0]["boundary_axis"] == "子判断定位"


def test_scheduler_handles_invalid_repeated_full_and_budget_stop():
    invalid = schedule_next_frontier([record("invalid_complexity")])
    assert invalid[0]["branch_action"] == "fork_from_parent"

    repeated = schedule_next_frontier([record("repeated_pattern")])
    assert repeated[0]["branch_action"] == "fork_from_root"
    assert repeated[0]["boundary_axis"] == "子判断定位"

    full_score = schedule_next_frontier([record("full_score_no_drop", depth=1, boundaries=0)])
    assert full_score[0]["branch_action"] == "expand_current_branch"
    assert full_score[0]["search_depth"] == 1
    assert full_score[0]["target_search_depth"] == 2
    assert full_score[0]["parent_node_id"] == "sample_scheduler_full_score_no_drop_root_b1_d1"

    exhausted = schedule_next_frontier([record("no_clear_effect", depth=2, boundaries=0, candidates_used=4)])
    assert exhausted == []


def test_score_increased_backtracks_to_new_axis_when_budget_allows():
    next_frontier = schedule_next_frontier([record("score_increased", boundaries=0)])
    assert len(next_frontier) == 1
    assert next_frontier[0]["branch_action"] == "fork_from_parent"
    assert next_frontier[0]["boundary_axis"] == "子判断定位"
    assert next_frontier[0]["target_search_depth"] == 1


def test_depth_limit_failures_fork_to_unexplored_axis():
    full_score = schedule_next_frontier([record("full_score_no_drop", depth=2, boundaries=0)])
    assert full_score[0]["branch_action"] == "fork_from_root"
    assert full_score[0]["boundary_axis"] == "子判断定位"
    assert full_score[0]["source_node_id"] == full_score[0]["search_root_id"]
    assert full_score[0]["target_search_depth"] == 1

    no_clear = schedule_next_frontier([record("no_clear_effect", depth=2, boundaries=0)])
    assert no_clear[0]["branch_action"] == "fork_from_root"
    assert no_clear[0]["boundary_axis"] == "子判断定位"


def test_scheduler_never_outputs_invalid_depth_frontier():
    cases = [
        record("effective_boundary_probe", depth=1),
        record("score_increased", depth=1, boundaries=0),
        record("full_score_no_drop", depth=1, boundaries=0),
        record("full_score_no_drop", depth=2, boundaries=0),
        record("no_clear_effect", depth=2, boundaries=0),
        record("repeated_pattern", depth=1),
    ]

    for item in cases:
        for entry in schedule_next_frontier([item], max_depth=2, max_boundaries=2):
            assert entry["target_search_depth"] <= 2
            assert entry["target_search_depth"] >= entry["source_search_depth"]
            if entry["branch_action"] == "expand_current_branch":
                assert entry["target_search_depth"] > entry["source_search_depth"]

    unknown = record("unknown_effect", depth=2, boundaries=0)
    unknown["effect_analysis"]["effect_label"] = "unexpected_label"
    assert schedule_next_frontier([unknown], max_depth=2, max_boundaries=2) == []


if __name__ == "__main__":
    test_scheduler_prioritizes_parent_fork_after_boundary_hit()
    test_scheduler_handles_invalid_repeated_full_and_budget_stop()
    test_score_increased_backtracks_to_new_axis_when_budget_allows()
    test_depth_limit_failures_fork_to_unexplored_axis()
    test_scheduler_never_outputs_invalid_depth_frontier()
    print("tree stage06 backtrack scheduler checks passed")
