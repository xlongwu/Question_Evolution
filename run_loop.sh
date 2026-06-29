#!/bin/bash
# Question Evolution 循环流水线
# 每轮把上一轮 scored/state 结果接入画像、分流、路由、多候选进化、复杂度校验、
# 标准采答案/rubric/评分闭环、效果统计和状态更新。
# 支持断点续跑：每一步的目标文件已存在且非空时跳过该步，避免重复写 memory。

set -euo pipefail

# ===================== 可配置参数 =====================
MAX_ROUNDS=${MAX_ROUNDS:-5}                      # 最大迭代轮数
EARLY_STOP_RATE=${EARLY_STOP_RATE:-0.5}          # 平均得分率低于该值时停止
NO_INFO_STOP_ROUNDS=${NO_INFO_STOP_ROUNDS:-2}    # 连续多少轮无新信息时停止
NO_INFO_MIN_DELTA=${NO_INFO_MIN_DELTA:-0.0001}   # 平均分变化小于该值视为无新信息
MAX_GLOBAL_NEW_BOUNDARY_GAP=${MAX_GLOBAL_NEW_BOUNDARY_GAP:-2}  # 连续多少轮没有新增边界时全局停止；0 表示禁用
MIN_SCORE_RATE=${MIN_SCORE_RATE:-0.8}            # legacy question_evolution 触发阈值
NUM_CANDIDATES=${NUM_CANDIDATES:-2}              # 每条待进化样本最多生成候选数，范围 1-4
MAX_CANDIDATE_BUDGET=${MAX_CANDIDATE_BUDGET:-0}  # 单轮候选总预算；0 表示待进化样本数 * 2
VALIDATION_RETRIES=${VALIDATION_RETRIES:-1}      # validate-retry 次数；当前最多 1 次
ENABLE_TREE_SEARCH=${ENABLE_TREE_SEARCH:-false}  # true 时使用 active_frontier 驱动下一轮 evolution
MAX_SAMPLE_BRANCHES=${MAX_SAMPLE_BRANCHES:-4}
MAX_SAMPLE_DEPTH=${MAX_SAMPLE_DEPTH:-3}
MAX_SAMPLE_BOUNDARIES=${MAX_SAMPLE_BOUNDARIES:-3}
MAX_SAMPLE_CANDIDATES_TOTAL=${MAX_SAMPLE_CANDIDATES_TOTAL:-10}
MAX_NO_NEW_BOUNDARY_ROUNDS=${MAX_NO_NEW_BOUNDARY_ROUNDS:-2}
ENABLE_BRANCH_BACKTRACK=${ENABLE_BRANCH_BACKTRACK:-true}
ENABLE_ROOT_FORK=${ENABLE_ROOT_FORK:-true}
ALLOW_DEFAULT_AXIS_FALLBACK_AFTER_RECOMMENDATION_EXHAUSTED=${ALLOW_DEFAULT_AXIS_FALLBACK_AFTER_RECOMMENDATION_EXHAUSTED:-false}

DEFAULT_INPUT_FILE="data/data.jsonl"
LEGACY_INPUT_FILE="data/data.jsonl"
INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT_FILE}    # 推荐输入：已完成准入的种子样本
EXP_ROOT=${EXP_ROOT:-"experiments"}              # 实验结果根目录

# Qwen（候选模型 / 评分模型）配置
QWEN_BASE_URL=${QWEN_BASE_URL:-"http://127.0.0.1:18011/v1"}
QWEN_API_KEY=${QWEN_API_KEY:-""}
QWEN_MODEL=${QWEN_MODEL:-"hjl_Qwen3.6-27B"}

# GPT / OpenAI-compatible 配置。API key 优先使用各脚本支持的环境变量：
# PROFILE_API_KEYS、EVOLVE_API_KEYS、OPENAI_API_KEYS 或 OPENAI_API_KEY。
GPT_MODEL=${GPT_MODEL:-"gpt-5.4"}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-""}
PROFILE_MODEL=${PROFILE_MODEL:-$GPT_MODEL}
PROFILE_BASE_URL=${PROFILE_BASE_URL:-$OPENAI_BASE_URL}
EVOLVE_MODEL=${EVOLVE_MODEL:-$GPT_MODEL}
EVOLVE_BASE_URL=${EVOLVE_BASE_URL:-$OPENAI_BASE_URL}
ANSWER_BASE_URL=${ANSWER_BASE_URL:-$OPENAI_BASE_URL}
RUBRIC_BASE_URL=${RUBRIC_BASE_URL:-$OPENAI_BASE_URL}

# 并发数
SCORING_CONCURRENCY=${SCORING_CONCURRENCY:-10}
PROFILE_CONCURRENCY=${PROFILE_CONCURRENCY:-5}
EVO_CONCURRENCY=${EVO_CONCURRENCY:-10}
ANSWER_CONCURRENCY=${ANSWER_CONCURRENCY:-10}
RUBRIC_CONCURRENCY=${RUBRIC_CONCURRENCY:-10}
# ======================================================

if [ ! -f "$INPUT_FILE" ] && [ "$INPUT_FILE" = "$DEFAULT_INPUT_FILE" ] && [ -f "$LEGACY_INPUT_FILE" ]; then
    echo "未找到 $DEFAULT_INPUT_FILE，回退到旧输入文件: $LEGACY_INPUT_FILE"
    INPUT_FILE="$LEGACY_INPUT_FILE"
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "输入文件不存在: $INPUT_FILE"
    echo "请设置 INPUT_FILE 指向 admitted_seed_samples.jsonl 或其他已准入 JSONL。"
    exit 1
fi

# 为当天运行自动选择实验目录：
#   experiments/YYYY-MM-DD/exp
#   experiments/YYYY-MM-DD/exp1
#   experiments/YYYY-MM-DD/exp2
#   ...
RUN_DATE=$(date +%F)
DAY_DIR="$EXP_ROOT/$RUN_DATE"
mkdir -p "$DAY_DIR"

EXP_DIR="$DAY_DIR/exp"
if [ -e "$EXP_DIR" ]; then
    EXP_INDEX=1
    while [ -e "$DAY_DIR/exp$EXP_INDEX" ]; do
        EXP_INDEX=$((EXP_INDEX + 1))
    done
    EXP_DIR="$DAY_DIR/exp$EXP_INDEX"
fi
mkdir -p "$EXP_DIR"

MEMORY_DIR="$EXP_DIR/memory"
mkdir -p "$MEMORY_DIR"
for bank_file in operator_memory_bank.jsonl failure_memory_bank.jsonl invalid_generation_cases.jsonl; do
    if [ ! -f "$MEMORY_DIR/$bank_file" ]; then
        : > "$MEMORY_DIR/$bank_file"
    fi
done

echo "本次实验目录: $EXP_DIR"
echo "Memory 目录: $MEMORY_DIR"

run_if_missing() {
    local output_file="$1"
    local step_label="$2"
    shift 2

    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        echo "检测到已存在 $output_file，跳过 $step_label"
    else
        echo "$step_label"
        "$@"
    fi
}

is_true() {
    case "$1" in
        1|true|TRUE|True|yes|YES|Yes|y|Y|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

# 辅助函数：计算 jsonl 的平均得分率
compute_avg_score_rate() {
    local scored_file="$1"
    python - "$scored_file" <<'PY'
import json, sys
path = sys.argv[1]
rates = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        sr = item.get("scoring_result", {})
        awarded = sr.get("total_awarded", 0) or 0
        possible = sr.get("total_possible", 0) or 0
        if possible > 0:
            rates.append(awarded / possible)
avg = sum(rates) / len(rates) if rates else 0.0
print(f"{avg:.4f}")
PY
}

# 辅助函数：比较两个浮点数，输出 true/false
lt_float() {
    python -c "print('true' if float('$1') < float('$2') else 'false')"
}

abs_diff_float() {
    python -c "print(abs(float('$1') - float('$2')))"
}

extract_new_boundary_count() {
    local analyzed_file="$1"
    python - "$analyzed_file" <<'PY'
import json, sys
path = sys.argv[1]
count = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        effect = item.get("effect_analysis", {})
        if (
            isinstance(effect, dict)
            and effect.get("is_new_boundary_for_sample") is True
            and effect.get("duplicate_boundary_for_sample") is not True
        ):
            count += 1
print(count)
PY
}

SUMMARY_FILE="$EXP_DIR/summary.txt"
echo "Question Evolution Loop Summary" > "$SUMMARY_FILE"
echo "================================" >> "$SUMMARY_FILE"
echo "Input file: $INPUT_FILE" >> "$SUMMARY_FILE"
echo "Memory dir: $MEMORY_DIR" >> "$SUMMARY_FILE"
echo "Max rounds: $MAX_ROUNDS" >> "$SUMMARY_FILE"
echo "Early stop rate: $EARLY_STOP_RATE" >> "$SUMMARY_FILE"
echo "No-info stop rounds: $NO_INFO_STOP_ROUNDS" >> "$SUMMARY_FILE"
echo "No-info min delta: $NO_INFO_MIN_DELTA" >> "$SUMMARY_FILE"
echo "Max global new boundary gap: $MAX_GLOBAL_NEW_BOUNDARY_GAP" >> "$SUMMARY_FILE"
echo "Evolution trigger rate: $MIN_SCORE_RATE" >> "$SUMMARY_FILE"
echo "Num candidates: $NUM_CANDIDATES" >> "$SUMMARY_FILE"
echo "Max candidate budget: $MAX_CANDIDATE_BUDGET" >> "$SUMMARY_FILE"
echo "Validation retries: $VALIDATION_RETRIES" >> "$SUMMARY_FILE"
echo "Tree search enabled: $ENABLE_TREE_SEARCH" >> "$SUMMARY_FILE"
echo "Max sample branches: $MAX_SAMPLE_BRANCHES" >> "$SUMMARY_FILE"
echo "Max sample depth: $MAX_SAMPLE_DEPTH" >> "$SUMMARY_FILE"
echo "Max sample boundaries: $MAX_SAMPLE_BOUNDARIES" >> "$SUMMARY_FILE"
echo "Max sample candidates total: $MAX_SAMPLE_CANDIDATES_TOTAL" >> "$SUMMARY_FILE"
echo "Max no-new-boundary rounds: $MAX_NO_NEW_BOUNDARY_ROUNDS" >> "$SUMMARY_FILE"
echo "Enable branch backtrack: $ENABLE_BRANCH_BACKTRACK" >> "$SUMMARY_FILE"
echo "Enable root fork: $ENABLE_ROOT_FORK" >> "$SUMMARY_FILE"
echo "Allow default axis fallback after recommendation exhausted: $ALLOW_DEFAULT_AXIS_FALLBACK_AFTER_RECOMMENDATION_EXHAUSTED" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Round | Avg Score Rate | New Boundaries | Status" >> "$SUMMARY_FILE"
echo "------|----------------|----------------|--------" >> "$SUMMARY_FILE"

# ===================== Round 0: 初始评分 =====================
ROUND=0
ROUND_DIR="$EXP_DIR/round_$ROUND"
mkdir -p "$ROUND_DIR"

echo ""
echo "========================================"
echo "Round $ROUND: 初始评分（baseline）"
echo "========================================"

run_if_missing "$ROUND_DIR/input.jsonl" "[Round $ROUND] Step 0/2: 准备 baseline input" \
    cp "$INPUT_FILE" "$ROUND_DIR/input.jsonl"

run_if_missing "$ROUND_DIR/scored.jsonl" "[Round $ROUND] Step 1/2: scoring.py baseline" \
    python scoring.py \
        --input "$ROUND_DIR/input.jsonl" \
        --output "$ROUND_DIR/scored.jsonl" \
        --answer-mode llm \
        --answer-base-url "$QWEN_BASE_URL" \
        --answer-api-key "$QWEN_API_KEY" \
        --answer-model "$QWEN_MODEL" \
        --judge-base-url "$QWEN_BASE_URL" \
        --judge-api-key "$QWEN_API_KEY" \
        --judge-model "$QWEN_MODEL" \
        --concurrency "$SCORING_CONCURRENCY"

AVG_RATE=$(compute_avg_score_rate "$ROUND_DIR/scored.jsonl")
echo "Round $ROUND 平均得分率: $AVG_RATE"
printf "%5s | %14s | %14s | %s\n" "$ROUND" "$AVG_RATE" "-" "baseline" >> "$SUMMARY_FILE"

PREV_SCORED="$ROUND_DIR/scored.jsonl"
PREV_AVG_RATE="$AVG_RATE"
PREV_FRONTIER=""
PREV_SEARCH_GRAPH=""
NO_INFO_STREAK=0
GLOBAL_NEW_BOUNDARY_GAP_STREAK=0

# ===================== Round 1..N: 循环进化 =====================
for ROUND in $(seq 1 "$MAX_ROUNDS"); do
    ROUND_DIR="$EXP_DIR/round_$ROUND"
    mkdir -p "$ROUND_DIR"

    echo ""
    echo "========================================"
    echo "Round $ROUND: Question Evolution"
    echo "========================================"

    run_if_missing "$ROUND_DIR/input.jsonl" "[Round $ROUND] Step 0/12: 复制上一轮 scored/state 输入" \
        cp "$PREV_SCORED" "$ROUND_DIR/input.jsonl"

    ROUND_STAGE_INPUT="$ROUND_DIR/input.jsonl"
    if is_true "$ENABLE_TREE_SEARCH" && [ -n "$PREV_FRONTIER" ] && [ -f "$PREV_FRONTIER" ] && [ -s "$PREV_FRONTIER" ]; then
        run_if_missing "$ROUND_DIR/frontier_records.jsonl" "[Round $ROUND] Step 1/12: prepare_frontier_records.py" \
            python prepare_frontier_records.py \
                --input "$ROUND_DIR/input.jsonl" \
                --frontier-input "$PREV_FRONTIER" \
                --output "$ROUND_DIR/frontier_records.jsonl"
        ROUND_STAGE_INPUT="$ROUND_DIR/frontier_records.jsonl"
        echo "Round $ROUND 使用 frontier-expanded 输入: $ROUND_STAGE_INPUT"
    fi

    if [ -f "$ROUND_DIR/scored.jsonl" ] && [ -s "$ROUND_DIR/scored.jsonl" ]; then
        echo "检测到已存在 $ROUND_DIR/scored.jsonl，跳过本轮生成闭环"
    else
        run_if_missing "$ROUND_DIR/profiled.jsonl" "[Round $ROUND] Step 2/12: profile_samples.py" \
            python profile_samples.py \
                --input "$ROUND_STAGE_INPUT" \
                --output "$ROUND_DIR/profiled.jsonl" \
                --model "$PROFILE_MODEL" \
                --base-url "$PROFILE_BASE_URL" \
                --concurrency "$PROFILE_CONCURRENCY"

        run_if_missing "$ROUND_DIR/profiled_candidates.jsonl" "[Round $ROUND] Step 3/12: select_evolution_candidates.py" \
            python select_evolution_candidates.py \
                --input "$ROUND_DIR/profiled.jsonl" \
                --output "$ROUND_DIR/profiled_candidates.jsonl" \
                --high-score-threshold "$MIN_SCORE_RATE"

        run_if_missing "$ROUND_DIR/routed.jsonl" "[Round $ROUND] Step 4/12: operator_router.py" \
            python operator_router.py \
                --input "$ROUND_DIR/profiled_candidates.jsonl" \
                --output "$ROUND_DIR/routed.jsonl" \
                --memory-dir "$MEMORY_DIR"

        run_if_missing "$ROUND_DIR/candidates.jsonl" "[Round $ROUND] Step 5/12: question_evolution.py" \
            python question_evolution.py \
                --input "$ROUND_DIR/routed.jsonl" \
                --output "$ROUND_DIR/candidates.jsonl" \
                --min-score-rate "$MIN_SCORE_RATE" \
                --model "$EVOLVE_MODEL" \
                --base-url "$EVOLVE_BASE_URL" \
                --concurrency "$EVO_CONCURRENCY" \
                --num-candidates "$NUM_CANDIDATES" \
                --max-candidate-budget "$MAX_CANDIDATE_BUDGET" \
                --validation-retries "$VALIDATION_RETRIES"

        run_if_missing "$ROUND_DIR/validated_candidates.jsonl" "[Round $ROUND] Step 6/12: validate_evolved_question.py" \
            python validate_evolved_question.py \
                --input "$ROUND_DIR/candidates.jsonl" \
                --output "$ROUND_DIR/validated_candidates.jsonl"

        run_if_missing "$ROUND_DIR/with_answers.jsonl" "[Round $ROUND] Step 7/12: collect_answers.py" \
            python collect_answers.py \
                --input "$ROUND_DIR/validated_candidates.jsonl" \
                --output "$ROUND_DIR/with_answers.jsonl" \
                --concurrency "$ANSWER_CONCURRENCY" \
                --samples 1 \
                --model "$GPT_MODEL" \
                --base-url "$ANSWER_BASE_URL"

        run_if_missing "$ROUND_DIR/rubric.jsonl" "[Round $ROUND] Step 8/12: gen_rubric.py" \
            python gen_rubric.py \
                --input "$ROUND_DIR/with_answers.jsonl" \
                --output "$ROUND_DIR/rubric.jsonl" \
                --concurrency "$RUBRIC_CONCURRENCY" \
                --model "$GPT_MODEL" \
                --base-url "$RUBRIC_BASE_URL"

        run_if_missing "$ROUND_DIR/scored_candidates.jsonl" "[Round $ROUND] Step 9/12: scoring.py" \
            python scoring.py \
                --input "$ROUND_DIR/rubric.jsonl" \
                --output "$ROUND_DIR/scored_candidates.jsonl" \
                --answer-mode llm \
                --answer-base-url "$QWEN_BASE_URL" \
                --answer-api-key "$QWEN_API_KEY" \
                --answer-model "$QWEN_MODEL" \
                --judge-base-url "$QWEN_BASE_URL" \
                --judge-api-key "$QWEN_API_KEY" \
                --judge-model "$QWEN_MODEL" \
                --concurrency "$SCORING_CONCURRENCY"

        run_if_missing "$ROUND_DIR/effect_analysis.jsonl" "[Round $ROUND] Step 10/12: analyze_evolution_effect.py" \
            python analyze_evolution_effect.py \
                --before "$PREV_SCORED" \
                --input "$ROUND_DIR/scored_candidates.jsonl" \
                --output "$ROUND_DIR/effect_analysis.jsonl" \
                --matrix-output "$ROUND_DIR/effect_matrix.jsonl"

        run_if_missing "$ROUND_DIR/scored.jsonl" "[Round $ROUND] Step 11/12: candidate_selection.py boundary-aware final selection" \
            python candidate_selection.py \
                --input "$ROUND_DIR/effect_analysis.jsonl" \
                --output "$ROUND_DIR/scored.jsonl" \
                --invalid-output "$ROUND_DIR/invalid_generation_cases.jsonl"

        if [ ! -f "$ROUND_DIR/evolved.jsonl" ] || [ ! -s "$ROUND_DIR/evolved.jsonl" ]; then
            cp "$ROUND_DIR/scored.jsonl" "$ROUND_DIR/evolved.jsonl"
        fi
    fi

    UPDATE_TREE_ARGS=(
        --tree-search-enabled "$ENABLE_TREE_SEARCH"
        --max-sample-branches "$MAX_SAMPLE_BRANCHES"
        --max-sample-depth "$MAX_SAMPLE_DEPTH"
        --max-sample-boundaries "$MAX_SAMPLE_BOUNDARIES"
        --max-sample-candidates-total "$MAX_SAMPLE_CANDIDATES_TOTAL"
        --max-no-new-boundary-rounds "$MAX_NO_NEW_BOUNDARY_ROUNDS"
        --enable-branch-backtrack "$ENABLE_BRANCH_BACKTRACK"
        --enable-root-fork "$ENABLE_ROOT_FORK"
        --allow-default-axis-fallback-after-recommendation-exhausted "$ALLOW_DEFAULT_AXIS_FALLBACK_AFTER_RECOMMENDATION_EXHAUSTED"
        --active-frontier-output "$ROUND_DIR/active_frontier.jsonl"
        --search-graph-output "$ROUND_DIR/search_graph.jsonl"
        --previous-search-graph "$PREV_SEARCH_GRAPH"
    )

    run_if_missing "$ROUND_DIR/state_updated.jsonl" "[Round $ROUND] Step 12/12: update_sample_state.py" \
        python update_sample_state.py \
            --input "$ROUND_DIR/scored.jsonl" \
            --output "$ROUND_DIR/state_updated.jsonl" \
            --memory-dir "$MEMORY_DIR" \
            "${UPDATE_TREE_ARGS[@]}"

    # 计算本轮平均得分率
    AVG_RATE=$(compute_avg_score_rate "$ROUND_DIR/scored.jsonl")
    echo "Round $ROUND 平均得分率: $AVG_RATE"
    NEW_BOUNDARY_COUNT=$(extract_new_boundary_count "$ROUND_DIR/scored.jsonl")
    AVG_DELTA=$(abs_diff_float "$AVG_RATE" "$PREV_AVG_RATE")

    ROUND_OUTPUT_FOR_NEXT="$ROUND_DIR/scored.jsonl"
    if [ -f "$ROUND_DIR/state_updated.jsonl" ] && [ -s "$ROUND_DIR/state_updated.jsonl" ]; then
        ROUND_OUTPUT_FOR_NEXT="$ROUND_DIR/state_updated.jsonl"
    fi

    if [ -f "$ROUND_DIR/search_graph.jsonl" ] && [ -s "$ROUND_DIR/search_graph.jsonl" ]; then
        PREV_SEARCH_GRAPH="$ROUND_DIR/search_graph.jsonl"
    fi

    if is_true "$ENABLE_TREE_SEARCH" && [ -f "$ROUND_DIR/active_frontier.jsonl" ] && [ -s "$ROUND_DIR/active_frontier.jsonl" ]; then
        PREV_FRONTIER="$ROUND_DIR/active_frontier.jsonl"
    else
        PREV_FRONTIER=""
    fi

    # 检查提前停止条件
    SHOULD_STOP=$(lt_float "$AVG_RATE" "$EARLY_STOP_RATE")
    if [ "$SHOULD_STOP" = "true" ]; then
        echo "提前停止：Round $ROUND 平均得分率 $AVG_RATE < $EARLY_STOP_RATE"
        printf "%5s | %14s | %14s | %s\n" "$ROUND" "$AVG_RATE" "$NEW_BOUNDARY_COUNT" "early_stop" >> "$SUMMARY_FILE"
        PREV_SCORED="$ROUND_OUTPUT_FOR_NEXT"
        break
    fi

    if [ "$NEW_BOUNDARY_COUNT" -eq 0 ]; then
        GLOBAL_NEW_BOUNDARY_GAP_STREAK=$((GLOBAL_NEW_BOUNDARY_GAP_STREAK + 1))
    else
        GLOBAL_NEW_BOUNDARY_GAP_STREAK=0
    fi

    if [ "$MAX_GLOBAL_NEW_BOUNDARY_GAP" -gt 0 ] && [ "$GLOBAL_NEW_BOUNDARY_GAP_STREAK" -ge "$MAX_GLOBAL_NEW_BOUNDARY_GAP" ]; then
        echo "提前停止：连续 $GLOBAL_NEW_BOUNDARY_GAP_STREAK 轮没有新增边界（new_boundary_count=0）"
        printf "%5s | %14s | %14s | %s\n" "$ROUND" "$AVG_RATE" "$NEW_BOUNDARY_COUNT" "global_boundary_gap_stop" >> "$SUMMARY_FILE"
        PREV_SCORED="$ROUND_OUTPUT_FOR_NEXT"
        break
    fi

    if [ "$NEW_BOUNDARY_COUNT" -eq 0 ] && [ "$(lt_float "$AVG_DELTA" "$NO_INFO_MIN_DELTA")" = "true" ]; then
        NO_INFO_STREAK=$((NO_INFO_STREAK + 1))
    else
        NO_INFO_STREAK=0
    fi

    if [ "$NO_INFO_STREAK" -ge "$NO_INFO_STOP_ROUNDS" ]; then
        echo "提前停止：连续 $NO_INFO_STREAK 轮无新信息（new_boundary_count=0 且 avg_delta=$AVG_DELTA < $NO_INFO_MIN_DELTA）"
        printf "%5s | %14s | %14s | %s\n" "$ROUND" "$AVG_RATE" "$NEW_BOUNDARY_COUNT" "no_info_stop" >> "$SUMMARY_FILE"
        PREV_SCORED="$ROUND_OUTPUT_FOR_NEXT"
        break
    fi

    if is_true "$ENABLE_TREE_SEARCH" && [ -z "$PREV_FRONTIER" ]; then
        echo "提前停止：树状搜索没有生成下一轮 active frontier"
        printf "%5s | %14s | %14s | %s\n" "$ROUND" "$AVG_RATE" "$NEW_BOUNDARY_COUNT" "frontier_empty_stop" >> "$SUMMARY_FILE"
        PREV_SCORED="$ROUND_OUTPUT_FOR_NEXT"
        break
    fi

    printf "%5s | %14s | %14s | %s\n" "$ROUND" "$AVG_RATE" "$NEW_BOUNDARY_COUNT" "continue" >> "$SUMMARY_FILE"

    PREV_SCORED="$ROUND_OUTPUT_FOR_NEXT"
    PREV_AVG_RATE="$AVG_RATE"
done

# ===================== 保存最终结果 =====================
FINAL_DIR="$EXP_DIR/final"
mkdir -p "$FINAL_DIR"
cp "$PREV_SCORED" "$FINAL_DIR/final_scored.jsonl"

echo ""
echo "========================================"
echo "循环结束"
echo "最终结果: $FINAL_DIR/final_scored.jsonl"
echo "各轮汇总: $SUMMARY_FILE"
echo "========================================"
cat "$SUMMARY_FILE"
