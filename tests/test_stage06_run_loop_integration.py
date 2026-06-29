import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_LOOP = (ROOT / "run_loop.sh").read_text(encoding="utf-8")


def assert_in_order(text, snippets):
    cursor = -1
    for snippet in snippets:
        position = text.find(snippet)
        assert position != -1, f"missing snippet: {snippet}"
        assert position > cursor, f"snippet out of order: {snippet}"
        cursor = position


def test_run_loop_uses_stage06_full_pipeline_order():
    assert_in_order(
        RUN_LOOP,
        [
            "Step 1/12: prepare_frontier_records.py",
            "Step 2/12: profile_samples.py",
            "Step 3/12: select_evolution_candidates.py",
            "Step 4/12: operator_router.py",
            "Step 5/12: question_evolution.py",
            "Step 6/12: validate_evolved_question.py",
            "Step 7/12: collect_answers.py",
            "Step 8/12: gen_rubric.py",
            "Step 9/12: scoring.py",
            "Step 10/12: analyze_evolution_effect.py",
            "Step 11/12: candidate_selection.py boundary-aware final selection",
            "Step 12/12: update_sample_state.py",
        ],
    )


def test_run_loop_carries_state_forward_and_guards_memory_writes():
    assert 'MEMORY_DIR="$EXP_DIR/memory"' in RUN_LOOP
    assert 'run_if_missing "$ROUND_DIR/state_updated.jsonl"' in RUN_LOOP
    assert 'ROUND_OUTPUT_FOR_NEXT="$ROUND_DIR/state_updated.jsonl"' in RUN_LOOP
    assert 'PREV_SCORED="$ROUND_OUTPUT_FOR_NEXT"' in RUN_LOOP
    assert '--memory-dir "$MEMORY_DIR"' in RUN_LOOP
    assert '--invalid-output "$ROUND_DIR/invalid_generation_cases.jsonl"' in RUN_LOOP
    assert 'PREV_FRONTIER=""' in RUN_LOOP
    assert 'PREV_SEARCH_GRAPH=""' in RUN_LOOP
    assert 'ROUND_STAGE_INPUT="$ROUND_DIR/frontier_records.jsonl"' in RUN_LOOP
    assert 'cp "$ROUND_DIR/scored.jsonl" "$ROUND_DIR/evolved.jsonl"' in RUN_LOOP


def test_run_loop_defaults_to_existing_data_input():
    assert 'DEFAULT_INPUT_FILE="data/data.jsonl"' in RUN_LOOP
    assert 'LEGACY_INPUT_FILE="data/data.jsonl"' in RUN_LOOP
    assert 'INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT_FILE}' in RUN_LOOP
    assert "请设置 INPUT_FILE 指向 admitted_seed_samples.jsonl" in RUN_LOOP


def test_run_loop_uses_existing_stage_cli_flags():
    assert "--high-score-threshold \"$MIN_SCORE_RATE\"" in RUN_LOOP
    assert "--min-score-rate \"$MIN_SCORE_RATE\"" in RUN_LOOP
    assert "--num-candidates \"$NUM_CANDIDATES\"" in RUN_LOOP
    assert "--max-candidate-budget \"$MAX_CANDIDATE_BUDGET\"" in RUN_LOOP
    assert "--validation-retries \"$VALIDATION_RETRIES\"" in RUN_LOOP
    assert "python prepare_frontier_records.py" in RUN_LOOP
    assert '--frontier-input "$PREV_FRONTIER"' in RUN_LOOP
    question_call_start = RUN_LOOP.find("python question_evolution.py")
    validate_call_start = RUN_LOOP.find("python validate_evolved_question.py")
    question_call = RUN_LOOP[question_call_start:validate_call_start]
    assert "--frontier-input" not in question_call
    assert "--judge-base-url \"$QWEN_BASE_URL\"" in RUN_LOOP
    assert "--judge-api-key \"$QWEN_API_KEY\"" in RUN_LOOP
    assert "--base-url \"$ANSWER_BASE_URL\"" in RUN_LOOP
    assert "--base-url \"$RUBRIC_BASE_URL\"" in RUN_LOOP
    select_call_start = RUN_LOOP.find("python select_evolution_candidates.py")
    route_call_start = RUN_LOOP.find("python operator_router.py")
    assert select_call_start != -1
    assert route_call_start != -1
    select_call = RUN_LOOP[select_call_start:route_call_start]
    assert "--min-score-rate" not in select_call


def test_run_loop_integrates_stage05_tree_search_artifacts():
    assert "ENABLE_TREE_SEARCH=${ENABLE_TREE_SEARCH:-false}" in RUN_LOOP
    assert "MAX_SAMPLE_BRANCHES=${MAX_SAMPLE_BRANCHES:-4}" in RUN_LOOP
    assert "MAX_SAMPLE_DEPTH=${MAX_SAMPLE_DEPTH:-3}" in RUN_LOOP
    assert "MAX_SAMPLE_BOUNDARIES=${MAX_SAMPLE_BOUNDARIES:-3}" in RUN_LOOP
    assert "MAX_SAMPLE_CANDIDATES_TOTAL=${MAX_SAMPLE_CANDIDATES_TOTAL:-10}" in RUN_LOOP
    assert "MAX_NO_NEW_BOUNDARY_ROUNDS=${MAX_NO_NEW_BOUNDARY_ROUNDS:-2}" in RUN_LOOP
    assert "MAX_GLOBAL_NEW_BOUNDARY_GAP=${MAX_GLOBAL_NEW_BOUNDARY_GAP:-2}" in RUN_LOOP
    assert "ENABLE_BRANCH_BACKTRACK=${ENABLE_BRANCH_BACKTRACK:-true}" in RUN_LOOP
    assert "ENABLE_ROOT_FORK=${ENABLE_ROOT_FORK:-true}" in RUN_LOOP
    assert "ALLOW_DEFAULT_AXIS_FALLBACK_AFTER_RECOMMENDATION_EXHAUSTED=${ALLOW_DEFAULT_AXIS_FALLBACK_AFTER_RECOMMENDATION_EXHAUSTED:-false}" in RUN_LOOP
    assert '--active-frontier-output "$ROUND_DIR/active_frontier.jsonl"' in RUN_LOOP
    assert '--search-graph-output "$ROUND_DIR/search_graph.jsonl"' in RUN_LOOP
    assert '--previous-search-graph "$PREV_SEARCH_GRAPH"' in RUN_LOOP
    assert "--allow-default-axis-fallback-after-recommendation-exhausted" in RUN_LOOP
    assert 'PREV_FRONTIER="$ROUND_DIR/active_frontier.jsonl"' in RUN_LOOP
    assert 'PREV_SEARCH_GRAPH="$ROUND_DIR/search_graph.jsonl"' in RUN_LOOP
    assert "frontier_empty_stop" in RUN_LOOP
    assert "global_boundary_gap_stop" in RUN_LOOP
    assert "extract_new_boundary_count" in RUN_LOOP
    assert 'effect.get("is_new_boundary_for_sample") is True' in RUN_LOOP


def test_run_loop_keeps_rubric_and_scoring_as_closed_loop_steps_only():
    rubric_call_start = RUN_LOOP.find("python gen_rubric.py")
    scoring_call_start = RUN_LOOP.find("Step 9/12: scoring.py")
    effect_call_start = RUN_LOOP.find("Step 10/12: analyze_evolution_effect.py")
    selection_call_start = RUN_LOOP.find("Step 11/12: candidate_selection.py")
    assert rubric_call_start != -1
    assert scoring_call_start != -1
    assert rubric_call_start < scoring_call_start
    assert scoring_call_start < effect_call_start < selection_call_start

    rubric_call = RUN_LOOP[rubric_call_start:scoring_call_start]
    assert "--prompt-version" not in rubric_call
    assert "expected_evaluation_focus" not in RUN_LOOP
    assert "judge agreement" not in RUN_LOOP.lower()


if __name__ == "__main__":
    test_run_loop_uses_stage06_full_pipeline_order()
    test_run_loop_carries_state_forward_and_guards_memory_writes()
    test_run_loop_defaults_to_existing_data_input()
    test_run_loop_uses_existing_stage_cli_flags()
    test_run_loop_integrates_stage05_tree_search_artifacts()
    test_run_loop_keeps_rubric_and_scoring_as_closed_loop_steps_only()
    print("stage06 run loop integration checks passed")
