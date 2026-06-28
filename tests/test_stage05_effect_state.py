import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_evolution_effect import analyze_records, build_effect_matrix
from update_sample_state import update_records


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


if __name__ == "__main__":
    test_effect_analysis_labels_drop_full_invalid_and_review_cases()
    test_focus_mismatch_does_not_become_effective_boundary_probe()
    test_effect_matrix_summarizes_sample_type_by_operator()
    test_state_update_and_memory_entries_cover_success_failure_invalid_review()
    test_full_score_after_operator_switch_can_stop_as_stable_high_score()
    print("stage05 effect analysis and state update checks passed")
