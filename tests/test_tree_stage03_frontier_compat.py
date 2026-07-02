import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from frontier_scheduler import build_active_frontier


RUN_LOOP = (ROOT / "run_loop.sh").read_text(encoding="utf-8")


def assert_in_order(text, snippets):
    cursor = -1
    for snippet in snippets:
        position = text.find(snippet)
        assert position != -1, f"missing snippet: {snippet}"
        assert position > cursor, f"snippet out of order: {snippet}"
        cursor = position


def test_run_loop_keeps_legacy_steps_and_adds_tree_gate():
    assert "ENABLE_TREE_SEARCH=${ENABLE_TREE_SEARCH:-false}" in RUN_LOOP
    assert 'DEFAULT_INPUT_FILE="data/data.jsonl"' in RUN_LOOP
    assert "Tree Step 0a/12: frontier_scheduler.py active_frontier" in RUN_LOOP
    assert "Step 12/13: build_search_graph.py" in RUN_LOOP
    assert "Step 13/13: frontier_scheduler.py next_frontier" in RUN_LOOP
    assert 'run_if_missing "$ROUND_DIR/search_graph.jsonl"' not in RUN_LOOP
    assert '--combined-output "$COMBINED_SEARCH_GRAPH"' in RUN_LOOP
    assert '--combined-discovered-output "$COMBINED_DISCOVERED_BOUNDARIES"' in RUN_LOOP
    assert 'LAST_FINAL_SCORED="$ROUND_RESULT_FOR_FINAL"' in RUN_LOOP
    assert 'LAST_FRONTIER="$ROUND_DIR/next_frontier.jsonl"' in RUN_LOOP
    assert 'cp "$LAST_FINAL_SCORED" "$FINAL_DIR/final_scored.jsonl"' in RUN_LOOP
    assert 'cp "$LAST_FRONTIER" "$FINAL_DIR/final_frontier.jsonl"' in RUN_LOOP
    assert 'cp "$PREV_SCORED" "$FINAL_DIR/final_scored.jsonl"' not in RUN_LOOP
    assert_in_order(
        RUN_LOOP,
        [
            "Step 1/11: profile_samples.py",
            "Step 2/11: select_evolution_candidates.py",
            "Step 3/11: operator_router.py",
            "Step 4/11: question_evolution.py",
            "Step 5/11: validate_evolved_question.py",
            "Step 6/11: candidate_selection.py",
            "Step 7/11: collect_answers.py",
            "Step 8/11: gen_rubric.py",
            "Step 9/11: scoring.py",
            "Step 10/11: analyze_evolution_effect.py",
            "Step 11/11: update_sample_state.py",
        ],
    )


def test_active_frontier_wraps_mainline_and_skips_stopped_samples():
    active = build_active_frontier(
        [
            {
                "sample_id": "frontier-live",
                "prompt": "当前主链题。",
                "evolution_state": {
                    "round": 1,
                    "stop_status": "continue_with_new_operator",
                    "search_root_id": "sample_frontier_live_root",
                    "current_node_id": "sample_frontier_live_root_b1_d1",
                    "search_depth": 1,
                    "recommended_next_axes": ["子判断定位"],
                },
            },
            {
                "sample_id": "frontier-stop",
                "prompt": "停止题。",
                "evolution_state": {"round": 1, "stop_status": "stable_high_score_stop"},
            },
        ],
        max_branches=2,
        max_depth=2,
        max_boundaries=2,
        max_candidates_total=4,
    )

    assert len(active) == 1
    assert active[0]["sample_id"] == "frontier-live"
    assert active[0]["source_node_id"] == "sample_frontier_live_root_b1_d1"
    assert active[0]["boundary_axis"] == "子判断定位"


def test_active_frontier_preserves_scheduled_next_frontier_fields():
    scheduled = {
        "sample_id": "frontier-preserve",
        "prompt": "调度记录中的占位题面。",
        "search_root_id": "sample_frontier_preserve_root",
        "source_node_id": "sample_frontier_preserve_root_b2_d1",
        "parent_node_id": "sample_frontier_preserve_root",
        "branch_id": "branch_pending_axis_002",
        "branch_index": 2,
        "branch_action": "fork_from_parent",
        "boundary_axis": "子判断定位",
        "source_search_depth": 1,
        "target_search_depth": 2,
        "search_depth": 1,
        "evolution_state": {
            "search_root_id": "sample_frontier_preserve_root",
            "current_node_id": "sample_frontier_preserve_root_b2_d1",
            "parent_node_id": "sample_frontier_preserve_root",
            "branch_id": "branch_pending_axis_002",
            "branch_index": 2,
            "branch_action": "fork_from_parent",
            "boundary_axis": "子判断定位",
            "source_search_depth": 1,
            "target_search_depth": 2,
            "search_depth": 1,
            "branch_count": 1,
            "sample_candidates_used": 1,
            "sample_stop_status": "continue_branch_search",
            "max_search_depth": 2,
            "max_sample_branches": 2,
            "max_sample_boundaries": 2,
            "max_sample_candidates_total": 4,
            "node_prompts": {
                "sample_frontier_preserve_root_b2_d1": "恢复后的父节点题面。",
            },
        },
    }

    active = build_active_frontier(
        [scheduled],
        max_branches=2,
        max_depth=2,
        max_boundaries=2,
        max_candidates_total=4,
    )

    assert len(active) == 1
    preserved = active[0]
    for field in (
        "source_node_id",
        "parent_node_id",
        "branch_id",
        "branch_index",
        "branch_action",
        "boundary_axis",
        "source_search_depth",
        "target_search_depth",
        "search_depth",
    ):
        assert preserved[field] == scheduled[field]
    assert preserved["prompt"] == "恢复后的父节点题面。"
    assert preserved["evolution_state"]["target_search_depth"] == 2


def test_active_frontier_rejects_invalid_scheduled_depths():
    base = {
        "sample_id": "frontier-invalid-depth",
        "prompt": "调度记录。",
        "search_root_id": "sample_frontier_invalid_depth_root",
        "source_node_id": "sample_frontier_invalid_depth_root_b1_d2",
        "parent_node_id": "sample_frontier_invalid_depth_root_b1_d1",
        "branch_id": "branch_pending_axis_001",
        "branch_index": 1,
        "branch_action": "expand_current_branch",
        "boundary_axis": "子判断定位",
        "source_search_depth": 2,
        "target_search_depth": 3,
        "search_depth": 2,
        "evolution_state": {
            "search_root_id": "sample_frontier_invalid_depth_root",
            "current_node_id": "sample_frontier_invalid_depth_root_b1_d2",
            "sample_stop_status": "continue_branch_search",
            "max_search_depth": 2,
            "max_sample_boundaries": 2,
            "max_sample_candidates_total": 4,
        },
    }

    assert build_active_frontier([base], max_depth=2, max_boundaries=2, max_candidates_total=4) == []

    inverted = dict(base)
    inverted["branch_action"] = "fork_from_root"
    inverted["source_search_depth"] = 3
    inverted["search_depth"] = 3
    inverted["target_search_depth"] = 2

    assert build_active_frontier([inverted], max_depth=4, max_boundaries=2, max_candidates_total=4) == []


if __name__ == "__main__":
    test_run_loop_keeps_legacy_steps_and_adds_tree_gate()
    test_active_frontier_wraps_mainline_and_skips_stopped_samples()
    test_active_frontier_preserves_scheduled_next_frontier_fields()
    test_active_frontier_rejects_invalid_scheduled_depths()
    print("tree stage03 frontier compatibility checks passed")
