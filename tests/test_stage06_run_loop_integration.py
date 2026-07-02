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


def test_run_loop_carries_state_forward_and_guards_memory_writes():
    assert 'MEMORY_DIR="$EXP_DIR/memory"' in RUN_LOOP
    assert 'run_if_missing "$ROUND_DIR/state_updated.jsonl"' in RUN_LOOP
    assert 'ROUND_RESULT_FOR_FINAL="$ROUND_DIR/state_updated.jsonl"' in RUN_LOOP
    assert 'ROUND_OUTPUT_FOR_NEXT="$ROUND_RESULT_FOR_FINAL"' in RUN_LOOP
    assert 'PREV_SCORED="$ROUND_OUTPUT_FOR_NEXT"' in RUN_LOOP
    assert '--memory-dir "$MEMORY_DIR"' in RUN_LOOP
    assert '--invalid-output "$ROUND_DIR/invalid_generation_cases.jsonl"' in RUN_LOOP


def test_run_loop_defaults_to_data_jsonl():
    assert 'DEFAULT_INPUT_FILE="data/data.jsonl"' in RUN_LOOP
    assert 'INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT_FILE}' in RUN_LOOP
    assert "请设置 INPUT_FILE 指向 data/data.jsonl" in RUN_LOOP


def test_run_loop_uses_local_config_defaults_before_legacy_shell_defaults():
    assert "config_value()" in RUN_LOOP
    assert 'CONFIG_OPENAI_BASE_URL=$(config_value "" "OPENAI_BASE_URL" "BASE_URL")' in RUN_LOOP
    assert 'CONFIG_QWEN_BASE_URL=$(config_value "" "QWEN_BASE_URL" "JUDGE_BASE_URL" "BASE_URL" "OPENAI_BASE_URL")' in RUN_LOOP
    assert 'CONFIG_QWEN_API_KEY=$(config_value "" "QWEN_API_KEY" "JUDGE_API_KEY" "JUDGE_API_KEYS" "HIAPI_KEYS_BIG" "OPENAI_API_KEY")' in RUN_LOOP
    assert 'QWEN_BASE_URL=${QWEN_BASE_URL:-$CONFIG_QWEN_BASE_URL}' in RUN_LOOP
    assert 'QWEN_BASE_URL=${QWEN_BASE_URL:-"http://127.0.0.1:18011/v1"}' in RUN_LOOP
    assert 'GPT_MODEL=${GPT_MODEL:-$CONFIG_GPT_MODEL}' in RUN_LOOP
    assert 'OPENAI_BASE_URL=${OPENAI_BASE_URL:-$CONFIG_OPENAI_BASE_URL}' in RUN_LOOP


def test_run_loop_uses_existing_stage_cli_flags():
    assert "--high-score-threshold \"$MIN_SCORE_RATE\"" in RUN_LOOP
    assert "--min-score-rate \"$MIN_SCORE_RATE\"" in RUN_LOOP
    assert "--num-candidates \"$NUM_CANDIDATES\"" in RUN_LOOP
    assert "--max-candidate-budget \"$MAX_CANDIDATE_BUDGET\"" in RUN_LOOP
    assert "--validation-retries \"$VALIDATION_RETRIES\"" in RUN_LOOP
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


def test_run_loop_stops_on_empty_required_jsonl_outputs():
    assert "count_jsonl_records()" in RUN_LOOP
    assert_in_order(
        RUN_LOOP,
        [
            'BASELINE_RECORD_COUNT=$(count_jsonl_records "$ROUND_DIR/scored.jsonl")',
            'Round 0 baseline 没有产生有效评分记录',
            'AVG_RATE=$(compute_avg_score_rate "$ROUND_DIR/scored.jsonl")',
        ],
    )
    assert 'ROUND_INPUT_COUNT=$(count_jsonl_records "$ROUND_DIR/input.jsonl")' in RUN_LOOP
    assert '"input_empty"' in RUN_LOOP
    assert 'CANDIDATE_COUNT=$(count_jsonl_records "$ROUND_DIR/candidates.jsonl")' in RUN_LOOP
    assert '"candidate_empty"' in RUN_LOOP
    assert 'ROUND_SCORED_COUNT=$(count_jsonl_records "$ROUND_DIR/scored.jsonl")' in RUN_LOOP
    assert 'Round $ROUND 没有产生有效评分记录' in RUN_LOOP


def test_run_loop_avg_score_prefers_score_rate_and_clamps_scoring_result():
    assert "score_rate = clamp_rate(item.get(\"score_rate\"))" in RUN_LOOP
    assert "if score_rate is not None:" in RUN_LOOP
    assert "rates.append(score_rate)" in RUN_LOOP
    assert "rates.append(max(0.0, min(1.0, awarded / possible)))" in RUN_LOOP


def test_run_loop_keeps_rubric_and_scoring_as_closed_loop_steps_only():
    rubric_call_start = RUN_LOOP.find("python gen_rubric.py")
    scoring_call_start = RUN_LOOP.find("Step 9/11: scoring.py")
    assert rubric_call_start != -1
    assert scoring_call_start != -1
    assert rubric_call_start < scoring_call_start

    rubric_call = RUN_LOOP[rubric_call_start:scoring_call_start]
    assert "--prompt-version" not in rubric_call
    assert "expected_evaluation_focus" not in RUN_LOOP
    assert "judge agreement" not in RUN_LOOP.lower()


if __name__ == "__main__":
    test_run_loop_uses_stage06_full_pipeline_order()
    test_run_loop_carries_state_forward_and_guards_memory_writes()
    test_run_loop_defaults_to_data_jsonl()
    test_run_loop_uses_local_config_defaults_before_legacy_shell_defaults()
    test_run_loop_uses_existing_stage_cli_flags()
    test_run_loop_stops_on_empty_required_jsonl_outputs()
    test_run_loop_keeps_rubric_and_scoring_as_closed_loop_steps_only()
    print("stage06 run loop integration checks passed")
