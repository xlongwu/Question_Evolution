import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_evolution_effect import analyze_records, build_effect_matrix
from update_sample_state import update_records, update_records_with_artifacts


def load_jsonl(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def test_effect_analysis_labels_drop_full_invalid_and_review_cases():
    previous = load_jsonl(ROOT / "tests" / "fixtures" / "stage05_previous_scored.jsonl")
    current = load_jsonl(ROOT / "tests" / "fixtures" / "stage05_scored.jsonl")
    analyzed = analyze_records(current, previous_records=previous)
    effects = {record["sample_id"]: record["effect_analysis"] for record in analyzed}

    assert effects["stage05-drop"]["effect_label"] == "effective_boundary_probe"
    assert effects["stage05-drop"]["lightweight_boundary_hit"] is True
    assert round(effects["stage05-drop"]["delta_score_rate"], 2) == -0.38

    assert effects["stage05-full"]["effect_label"] == "full_score_no_drop"
    assert effects["stage05-full"]["is_full_score"] is True

    assert effects["stage05-invalid"]["effect_label"] == "invalid_complexity"
    assert effects["stage05-invalid"]["complexity_passed"] is False

    assert effects["stage05-review"]["effect_label"] == "needs_manual_review"
    assert effects["stage05-review"]["hit_confidence"] == "low"
    assert effects["stage05-review"]["needs_manual_review"] is True
    assert effects["stage05-review"]["focus_answer_alignment"]["matches"] is True


def test_focus_mismatch_does_not_become_effective_boundary_probe():
    previous = [
        {
            "sample_id": "focus-mismatch",
            "prompt": "上一轮题。",
            "scoring_result": {"candidate_answer": "上一轮答案。", "total_awarded": 10, "total_possible": 10},
            "score_rate": 1.0,
        }
    ]
    current = [
        {
            "sample_id": "focus-mismatch",
            "prompt": "请区分判据内和判据外信息。",
            "question_evolved": True,
            "meta_info": {
                "question_evolution_metadata": {
                    "question_evolved": True,
                    "operator_used": "O4_near_level_ranking",
                    "expected_evaluation_focus": ["是否排除判据外信息"],
                }
            },
            "validation_result": {"passed": True, "repeat_pattern_risk": "low"},
            "scoring_result": {
                "candidate_answer": "候选答案只讨论时间连续性缺口和人员轨迹跳步。",
                "total_awarded": 6,
                "total_possible": 10,
            },
            "score_rate": 0.6,
        }
    ]

    analyzed = analyze_records(current, previous_records=previous)
    effect = analyzed[0]["effect_analysis"]
    _, operator_memory, _, _ = update_records(analyzed)

    assert effect["lightweight_boundary_hit"] is False
    assert effect["effect_label"] == "needs_manual_review"
    assert effect["needs_manual_review"] is True
    assert effect["focus_answer_alignment"]["matches"] is False
    assert operator_memory == []


def test_effect_matrix_summarizes_sample_type_by_operator():
    previous = load_jsonl(ROOT / "tests" / "fixtures" / "stage05_previous_scored.jsonl")
    current = load_jsonl(ROOT / "tests" / "fixtures" / "stage05_scored.jsonl")
    analyzed = analyze_records(current, previous_records=previous)
    matrix = build_effect_matrix(analyzed)

    o1_rows = [row for row in matrix if row["operator_used"] == "O1_gap_choice"]
    assert sum(row["sample_count"] for row in o1_rows) == 2
    assert sum(row["lightweight_boundary_hit_count"] for row in o1_rows) == 1
    assert sum(row["full_score_count"] for row in o1_rows) == 1

    invalid_row = next(row for row in matrix if row["operator_used"] == "O4_near_level_ranking")
    assert invalid_row["invalid_complexity_count"] == 1


def test_state_update_and_memory_entries_cover_success_failure_invalid_review():
    previous = load_jsonl(ROOT / "tests" / "fixtures" / "stage05_previous_scored.jsonl")
    current = load_jsonl(ROOT / "tests" / "fixtures" / "stage05_scored.jsonl")
    analyzed = analyze_records(current, previous_records=previous)
    updated, operator_memory, failure_memory, invalid_memory = update_records(analyzed)
    states = {record["sample_id"]: record["evolution_state"] for record in updated}

    assert states["stage05-drop"]["stop_status"] == "effective_boundary_sample"
    assert states["stage05-full"]["stop_status"] == "local_tree_search_needed"
    assert states["stage05-full"]["consecutive_full_score_count"] == 2
    assert states["stage05-invalid"]["stop_status"] == "invalid_complexity_sample"
    assert states["stage05-review"]["stop_status"] == "continue_with_new_operator"

    assert {entry["sample_id"] for entry in operator_memory} == {"stage05-drop", "stage05-review"}
    low_confidence_entry = next(entry for entry in operator_memory if entry["sample_id"] == "stage05-review")
    assert low_confidence_entry["hit_confidence"] == "low"
    assert low_confidence_entry["needs_manual_review"] is True

    assert {entry["sample_id"] for entry in failure_memory} == {"stage05-full"}
    assert failure_memory[0]["failure_type"] == "full_score_no_drop"

    assert {entry["sample_id"] for entry in invalid_memory} == {"stage05-invalid"}
    assert invalid_memory[0]["invalid_type"] == "format_difficulty_dominant"


def test_full_score_after_operator_switch_can_stop_as_stable_high_score():
    record = {
        "sample_id": "stable-stop",
        "question_evolved": True,
        "evolution_state": {
            "previous_operator": "O1_gap_choice",
            "consecutive_full_score_count": 1,
            "consecutive_same_operator_count": 1,
            "stop_status": "continue_with_new_operator",
        },
        "effect_analysis": {
            "effect_label": "full_score_no_drop",
            "operator_used": "O2_subclaim_localization",
            "is_full_score": True,
            "score_rate_after": 1.0,
            "complexity_passed": True,
        },
    }

    updated, _, failure_memory, _ = update_records([record])
    state = updated[0]["evolution_state"]

    assert state["consecutive_full_score_count"] == 2
    assert state["stop_status"] == "stable_high_score_stop"
    assert failure_memory[0]["failure_type"] == "full_score_no_drop"


def stage05_tree_config(**overrides):
    config = {
        "ENABLE_TREE_SEARCH": True,
        "MAX_SAMPLE_BRANCHES": 4,
        "MAX_SAMPLE_DEPTH": 3,
        "MAX_SAMPLE_BOUNDARIES": 3,
        "MAX_SAMPLE_CANDIDATES_TOTAL": 6,
        "MAX_NO_NEW_BOUNDARY_ROUNDS": 2,
        "ENABLE_BRANCH_BACKTRACK": True,
        "ENABLE_ROOT_FORK": True,
    }
    config.update(overrides)
    return config


def make_tree_effect_record(
    sample_id,
    *,
    current_node_id,
    parent_node_id="sample_tree_root",
    branch_id="branch_gap_01",
    boundary_axis="最关键缺口识别",
    search_depth=1,
    branch_budget_remaining=2,
    sample_budget_remaining=5,
    effect_label="effective_boundary_probe",
    is_new_boundary=True,
    duplicate=False,
):
    return {
        "sample_id": sample_id,
        "round": 2,
        "prompt": f"{sample_id} evolved prompt",
        "question_evolved": True,
        "meta_info": {
            "prompt_old": f"{sample_id} root prompt",
            "question_evolution_metadata": {
                "question_evolved": True,
                "operator_used": "O1_gap_choice",
                "search_root_id": "sample_tree_root",
                "source_node_id": parent_node_id,
                "source_prompt": f"{sample_id} parent prompt",
                "generated_node_id": current_node_id,
                "parent_node_id": parent_node_id,
                "branch_id": branch_id,
                "boundary_axis": boundary_axis,
                "generation_action": "expand_current_branch",
                "search_depth": search_depth,
            },
        },
        "evolution_state": {
            "round": 1,
            "search_root_id": "sample_tree_root",
            "current_node_id": current_node_id,
            "parent_node_id": parent_node_id,
            "branch_id": branch_id,
            "boundary_axis": boundary_axis,
            "branch_status": "exploring",
            "search_depth": search_depth,
            "max_search_depth": 3,
            "branch_budget_remaining": branch_budget_remaining,
            "sample_budget_remaining": sample_budget_remaining,
            "recommended_next_axes": ["结论分层", "伪闭环识别"],
            "already_explored_axes": [],
            "discovered_boundaries": [],
            "stop_status": "continue",
        },
        "validation_result": {"passed": True, "repeat_pattern_risk": "low"},
        "effect_analysis": {
            "effect_label": effect_label,
            "operator_used": "O1_gap_choice",
            "score_rate_before": 1.0,
            "score_rate_after": 0.52,
            "is_full_score": False,
            "complexity_passed": True,
            "hit_confidence": "high",
            "target_boundary_axis": boundary_axis,
            "boundary_axis_detected": boundary_axis,
            "dedup_signature": f"boundary:{sample_id}:{boundary_axis}",
            "duplicate_boundary_for_sample": duplicate,
            "is_new_boundary_for_sample": is_new_boundary,
            "boundary_candidate_status": "new_boundary" if is_new_boundary else "duplicate_boundary",
        },
    }


def test_tree_state_update_records_boundary_and_root_frontier():
    record = make_tree_effect_record("tree-root-fork", current_node_id="sample_tree_b1_d1")

    updated, _, _, _, graph, frontier = update_records_with_artifacts(
        [record],
        config=stage05_tree_config(),
    )
    state = updated[0]["evolution_state"]

    assert state["branch_status"] == "boundary_hit"
    assert state["stop_status"] == "continue_branch_search"
    assert state["sample_budget_remaining"] == 4
    assert state["branch_budget_remaining"] == 1
    assert state["discovered_boundaries"][0]["boundary_axis"] == "最关键缺口识别"
    assert state["already_explored_axes"] == ["最关键缺口识别"]

    assert graph[0]["node_id"] == "sample_tree_b1_d1"
    assert graph[0]["is_boundary_hit"] is True
    assert graph[0]["selected_as_boundary_leaf"] is True

    assert len(frontier) == 1
    assert frontier[0]["action_type"] == "fork_from_root"
    assert frontier[0]["source_node_type"] == "root"
    assert frontier[0]["branch_id"] == ""
    assert frontier[0]["target_boundary_axis"] == "结论分层"


def test_tree_state_update_backtracks_to_parent_for_deep_terminal_branch():
    record = make_tree_effect_record(
        "tree-parent-fork",
        current_node_id="sample_tree_b1_d2",
        parent_node_id="sample_tree_b1_d1",
        search_depth=2,
        effect_label="effective_boundary_probe",
        is_new_boundary=True,
    )

    updated, _, _, _, _, frontier = update_records_with_artifacts(
        [record],
        config=stage05_tree_config(),
    )

    assert updated[0]["evolution_state"]["branch_status"] == "boundary_hit"
    assert frontier[0]["action_type"] == "fork_from_parent"
    assert frontier[0]["source_node_type"] == "parent"
    assert frontier[0]["source_node_id"] == "sample_tree_b1_d1"
    assert frontier[0]["branch_id"] == ""


def test_tree_state_update_stops_sample_when_budget_exhausted():
    record = make_tree_effect_record(
        "tree-budget-stop",
        current_node_id="sample_tree_b1_d1",
        branch_budget_remaining=1,
        sample_budget_remaining=1,
        effect_label="full_score_no_drop",
        is_new_boundary=False,
    )

    updated, _, _, _, _, frontier = update_records_with_artifacts(
        [record],
        config=stage05_tree_config(),
    )

    assert updated[0]["evolution_state"]["sample_budget_remaining"] == 0
    assert updated[0]["evolution_state"]["stop_status"] == "sample_budget_exhausted"
    assert frontier == []


def test_tree_state_update_does_not_expand_invalid_branch():
    record = make_tree_effect_record(
        "tree-invalid-fork",
        current_node_id="sample_tree_b1_d1",
        effect_label="invalid_complexity",
        is_new_boundary=False,
    )
    record["validation_result"] = {"passed": False, "invalid_type": "format_difficulty_dominant"}
    record["effect_analysis"]["complexity_passed"] = False

    updated, _, _, _, _, frontier = update_records_with_artifacts(
        [record],
        config=stage05_tree_config(),
    )

    assert updated[0]["evolution_state"]["branch_status"] == "invalid"
    assert updated[0]["evolution_state"]["stop_status"] == "continue_branch_search"
    assert frontier[0]["action_type"] == "fork_from_root"
    assert frontier[0]["source_node_type"] == "root"


def test_tree_state_update_respects_disabled_fork_switches():
    record = make_tree_effect_record("tree-no-fork", current_node_id="sample_tree_b1_d1")

    _, _, _, _, _, frontier = update_records_with_artifacts(
        [record],
        config=stage05_tree_config(ENABLE_BRANCH_BACKTRACK=False, ENABLE_ROOT_FORK=False),
    )

    assert frontier == []


def test_tree_state_update_respects_max_sample_branches_one():
    record = make_tree_effect_record("tree-branch-limit-one", current_node_id="sample_tree_b1_d1")

    updated, _, _, _, _, frontier = update_records_with_artifacts(
        [record],
        config=stage05_tree_config(MAX_SAMPLE_BRANCHES=1),
    )
    state = updated[0]["evolution_state"]

    assert state["sample_branch_count"] == 1
    assert state["opened_branch_ids"] == ["branch_gap_01"]
    assert state["stop_status"] == "sample_branch_limit_reached"
    assert frontier == []


def test_tree_state_update_respects_max_sample_branches_two_before_third_branch():
    record = make_tree_effect_record(
        "tree-branch-limit-two",
        current_node_id="sample_tree_b2_d1",
        branch_id="branch_layer_02",
        boundary_axis="结论分层",
    )
    record["evolution_state"]["opened_branch_ids"] = ["branch_gap_01", "branch_layer_02"]

    updated, _, _, _, _, frontier = update_records_with_artifacts(
        [record],
        config=stage05_tree_config(MAX_SAMPLE_BRANCHES=2),
    )
    state = updated[0]["evolution_state"]

    assert state["sample_branch_count"] == 2
    assert state["stop_status"] == "sample_branch_limit_reached"
    assert frontier == []


def test_tree_state_update_stops_when_explicit_recommended_axes_are_exhausted():
    record = make_tree_effect_record(
        "tree-axis-exhausted",
        current_node_id="sample_tree_b1_d1",
        boundary_axis="最关键缺口识别",
    )
    record["evolution_state"]["recommended_next_axes"] = ["最关键缺口识别"]

    updated, _, _, _, _, frontier = update_records_with_artifacts(
        [record],
        config=stage05_tree_config(),
    )
    state = updated[0]["evolution_state"]

    assert state["recommended_next_axes"] == []
    assert state["recommended_axes_exhausted"] is True
    assert state["stop_status"] == "recommended_axes_exhausted_stop"
    assert frontier == []


def test_tree_state_update_marks_homogeneous_candidate_branch_exhausted():
    record = make_tree_effect_record(
        "tree-homogeneous",
        current_node_id="sample_tree_b1_d1",
        effect_label="no_clear_effect",
        is_new_boundary=False,
    )
    record["validation_result"]["repeat_pattern_risk"] = "high"
    record["effect_analysis"]["is_new_boundary_for_sample"] = False
    record["effect_analysis"]["boundary_candidate_status"] = "duplicate_pattern"

    updated, _, _, _, _, _ = update_records_with_artifacts(
        [record],
        config=stage05_tree_config(),
    )
    state = updated[0]["evolution_state"]

    assert state["branch_status"] == "exhausted"
    assert state["candidate_homogeneity_detected"] is True
    assert state["branch_stop_reason"] == "homogeneous_candidate_stop"


if __name__ == "__main__":
    test_effect_analysis_labels_drop_full_invalid_and_review_cases()
    test_focus_mismatch_does_not_become_effective_boundary_probe()
    test_effect_matrix_summarizes_sample_type_by_operator()
    test_state_update_and_memory_entries_cover_success_failure_invalid_review()
    test_full_score_after_operator_switch_can_stop_as_stable_high_score()
    test_tree_state_update_records_boundary_and_root_frontier()
    test_tree_state_update_backtracks_to_parent_for_deep_terminal_branch()
    test_tree_state_update_stops_sample_when_budget_exhausted()
    test_tree_state_update_does_not_expand_invalid_branch()
    test_tree_state_update_respects_disabled_fork_switches()
    test_tree_state_update_respects_max_sample_branches_one()
    test_tree_state_update_respects_max_sample_branches_two_before_third_branch()
    test_tree_state_update_stops_when_explicit_recommended_axes_are_exhausted()
    test_tree_state_update_marks_homogeneous_candidate_branch_exhausted()
    print("stage05 effect analysis and state update checks passed")
