import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_evolution_effect import analyze_records, build_boundary_dedup_signature
from candidate_selection import select_candidates


def previous_record(sample_id="stage04-boundary"):
    return {
        "sample_id": sample_id,
        "prompt": "上一轮题：判断两个事实中哪一个才是真正关键。",
        "scoring_result": {
            "candidate_answer": "上一轮答案完整。",
            "total_awarded": 10,
            "total_possible": 10,
        },
        "score_rate": 1.0,
    }


def scored_boundary_candidate(
    *,
    sample_id="stage04-boundary",
    candidate_id="cand_1",
    target_axis="最关键缺口识别",
    score_rate=0.55,
    discovered_boundaries=None,
):
    return {
        "sample_id": sample_id,
        "candidate_group_id": sample_id,
        "candidate_id": f"{sample_id}::{candidate_id}",
        "candidate_operator": "O1_gap_choice",
        "prompt": "请比较 A 与 B 两个候选事实，判断哪一个才是支撑结论的最小关键事实。",
        "question_evolved": True,
        "meta_info": {
            "question_evolution_metadata": {
                "question_evolved": True,
                "operator_used": "O1_gap_choice",
                "boundary_axis": target_axis,
                "expected_evaluation_focus": ["是否抓住真正最小关键事实"],
                "expected_qwen_failure": "选错最关键缺口",
            }
        },
        "validation_result": {
            "passed": True,
            "main_axis_count": 1,
            "new_facts_count": 1,
            "output_tasks_count": 1,
            "candidate_options_count": 2,
            "counterfactual_count": 0,
            "estimated_prompt_chars": 45,
            "external_knowledge_risk": "low",
            "format_difficulty_risk": "low",
            "repeat_pattern_risk": "low",
            "why_passed": "ok",
            "reject_reason": None,
        },
        "scoring_result": {
            "candidate_answer": "候选答案把旁证 B 当成最小关键事实，没有识别 A 才是独立必要条件。",
            "total_awarded": score_rate * 10,
            "total_possible": 10,
        },
        "score_rate": score_rate,
        "evolution_state": {
            "search_root_id": f"sample_{sample_id}_root",
            "current_node_id": f"sample_{sample_id}_b1_d1",
            "branch_id": "branch_gap_01",
            "boundary_axis": target_axis,
            "discovered_boundaries": discovered_boundaries or [],
        },
    }


def test_effect_analysis_marks_new_boundary_with_axis_and_signature():
    current = scored_boundary_candidate()
    analyzed = analyze_records([current], previous_records=[previous_record()])
    effect = analyzed[0]["effect_analysis"]

    assert effect["target_boundary_axis"] == "最关键缺口识别"
    assert effect["boundary_axis_detected"] == "最关键缺口识别"
    assert effect["is_new_boundary_for_sample"] is True
    assert effect["duplicate_boundary_for_sample"] is False
    assert effect["boundary_candidate_status"] == "new_boundary"
    assert effect["dedup_signature"].startswith("boundary:")


def test_effect_analysis_marks_duplicate_boundary_from_discovered_state():
    current = scored_boundary_candidate()
    signature = build_boundary_dedup_signature(current, "最关键缺口识别")
    duplicate = scored_boundary_candidate(
        discovered_boundaries=[
            {
                "boundary_id": "existing-gap",
                "boundary_axis": "最关键缺口识别",
                "dedup_signature": signature,
            }
        ]
    )

    analyzed = analyze_records([duplicate], previous_records=[previous_record()])
    effect = analyzed[0]["effect_analysis"]

    assert effect["dedup_signature"] == signature
    assert effect["duplicate_boundary_for_sample"] is True
    assert effect["is_new_boundary_for_sample"] is False
    assert effect["boundary_candidate_status"] == "duplicate_boundary"


def test_effect_analysis_keeps_axis_mismatch_audit_for_structural_signal():
    current = scored_boundary_candidate(target_axis="结论分层", score_rate=1.0)
    analyzed = analyze_records([current], previous_records=[previous_record()])
    effect = analyzed[0]["effect_analysis"]

    assert effect["effect_label"] == "full_score_no_drop"
    assert effect["target_boundary_axis"] == "结论分层"
    assert effect["boundary_axis_detected"] == "最关键缺口识别"
    assert effect["structural_boundary_signal"] is True
    assert effect["boundary_axis_mismatch"] is True
    assert effect["is_new_boundary_for_sample"] is True
    assert effect["boundary_candidate_status"] == "axis_mismatch_needs_review"


def test_candidate_selection_rejects_duplicate_and_preserves_boundary_leaf():
    new_candidate = analyze_records(
        [scored_boundary_candidate(sample_id="stage04-select-boundary", candidate_id="cand_1")],
        previous_records=[previous_record("stage04-select-boundary")],
    )[0]
    duplicate_input = scored_boundary_candidate(
        sample_id="stage04-select-boundary",
        candidate_id="cand_2",
        discovered_boundaries=[
            {
                "boundary_id": "existing-gap",
                "boundary_axis": "最关键缺口识别",
                "dedup_signature": new_candidate["effect_analysis"]["dedup_signature"],
            }
        ],
    )
    duplicate_candidate = analyze_records(
        [duplicate_input],
        previous_records=[previous_record("stage04-select-boundary")],
    )[0]

    selected, invalid_cases = select_candidates([duplicate_candidate, new_candidate])
    selection = selected[0]["candidate_selection"]

    assert invalid_cases == []
    assert selection["selected_candidate_id"] == "stage04-select-boundary::cand_1"
    assert selection["selected_as_boundary_leaf"] is True
    assert selection["selected_into_mainline"] is False
    assert selected[0]["selected_as_boundary_leaf"] is True
    assert selected[0]["discard_as_duplicate"] is False
    assert selection["rejected_candidates"][0]["discard_as_duplicate"] is True
