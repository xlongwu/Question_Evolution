#!/bin/bash
# Question Evolution 循环流水线
# 每轮把上一轮 scored/state 结果接入画像、分流、路由、多候选进化、复杂度校验、
# 标准采答案/rubric/评分闭环、效果统计和状态更新。
# 支持断点续跑：每一步的目标文件已存在且非空时跳过该步，避免重复写 memory。

set -euo pipefail

config_value() {
    local default_value="$1"
    shift
    python - "$default_value" "$@" <<'PY'
import sys
from local_api_config import get_config_value

default = sys.argv[1]
names = sys.argv[2:]
print(get_config_value(*names, default=default))
PY
}

# ===================== 可配置参数 =====================
MAX_ROUNDS=${MAX_ROUNDS:-5}                      # 最大迭代轮数
EARLY_STOP_RATE=${EARLY_STOP_RATE:-0.5}          # 平均得分率低于该值时停止
NO_INFO_STOP_ROUNDS=${NO_INFO_STOP_ROUNDS:-2}    # 连续多少轮无新信息时停止
NO_INFO_MIN_DELTA=${NO_INFO_MIN_DELTA:-0.0001}   # 平均分变化小于该值视为无新信息
MIN_SCORE_RATE=${MIN_SCORE_RATE:-0.8}            # legacy question_evolution 触发阈值
NUM_CANDIDATES=${NUM_CANDIDATES:-2}              # 每条待进化样本最多生成候选数，范围 1-4
MAX_CANDIDATE_BUDGET=${MAX_CANDIDATE_BUDGET:-0}  # 单轮候选总预算；0 表示待进化样本数 * 2
VALIDATION_RETRIES=${VALIDATION_RETRIES:-1}      # validate-retry 次数；当前最多 1 次
ENABLE_TREE_SEARCH=${ENABLE_TREE_SEARCH:-false}  # 是否启用树搜索/frontier 调度
MAX_SAMPLE_BRANCHES=${MAX_SAMPLE_BRANCHES:-3}
MAX_SAMPLE_DEPTH=${MAX_SAMPLE_DEPTH:-2}
MAX_SAMPLE_BOUNDARIES=${MAX_SAMPLE_BOUNDARIES:-2}
MAX_SAMPLE_CANDIDATES_TOTAL=${MAX_SAMPLE_CANDIDATES_TOTAL:-6}
ENABLE_BRANCH_BACKTRACK=${ENABLE_BRANCH_BACKTRACK:-true}
ENABLE_ROOT_FORK=${ENABLE_ROOT_FORK:-true}

DEFAULT_INPUT_FILE="data/data.jsonl"
INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT_FILE}    # 默认输入：当前仓库 data/data.jsonl
EXP_ROOT=${EXP_ROOT:-"experiments"}              # 实验结果根目录

CONFIG_OPENAI_BASE_URL=$(config_value "" "OPENAI_BASE_URL" "BASE_URL")
CONFIG_GPT_MODEL=$(config_value "gpt-5.4" "GPT_MODEL" "QA_MODEL")
CONFIG_PROFILE_MODEL=$(config_value "$CONFIG_GPT_MODEL" "PROFILE_MODEL" "EVOLVE_MODEL" "QA_MODEL" "GPT_MODEL")
CONFIG_EVOLVE_MODEL=$(config_value "$CONFIG_GPT_MODEL" "EVOLVE_MODEL" "QA_MODEL" "GPT_MODEL")
CONFIG_PROFILE_BASE_URL=$(config_value "$CONFIG_OPENAI_BASE_URL" "PROFILE_BASE_URL" "EVOLVE_BASE_URL" "BASE_URL" "OPENAI_BASE_URL")
CONFIG_EVOLVE_BASE_URL=$(config_value "$CONFIG_OPENAI_BASE_URL" "EVOLVE_BASE_URL" "BASE_URL" "OPENAI_BASE_URL")
CONFIG_ANSWER_BASE_URL=$(config_value "$CONFIG_OPENAI_BASE_URL" "ANSWER_BASE_URL" "BASE_URL" "OPENAI_BASE_URL")
CONFIG_RUBRIC_BASE_URL=$(config_value "$CONFIG_OPENAI_BASE_URL" "RUBRIC_BASE_URL" "BASE_URL" "OPENAI_BASE_URL")
CONFIG_QWEN_BASE_URL=$(config_value "" "QWEN_BASE_URL" "JUDGE_BASE_URL" "BASE_URL" "OPENAI_BASE_URL")
CONFIG_QWEN_API_KEY=$(config_value "" "QWEN_API_KEY" "JUDGE_API_KEY" "JUDGE_API_KEYS" "HIAPI_KEYS_BIG" "OPENAI_API_KEY")
CONFIG_QWEN_MODEL=$(config_value "hjl_Qwen3.6-27B" "QWEN_MODEL" "JUDGE_MODEL" "ANSWER_MODEL")

# Qwen（候选模型 / 评分模型）配置
QWEN_BASE_URL=${QWEN_BASE_URL:-$CONFIG_QWEN_BASE_URL}
QWEN_BASE_URL=${QWEN_BASE_URL:-"http://127.0.0.1:18011/v1"}
QWEN_API_KEY=${QWEN_API_KEY:-$CONFIG_QWEN_API_KEY}
QWEN_MODEL=${QWEN_MODEL:-$CONFIG_QWEN_MODEL}

# GPT / OpenAI-compatible 配置。API key 优先使用各脚本支持的环境变量：
# PROFILE_API_KEYS、EVOLVE_API_KEYS、OPENAI_API_KEYS 或 OPENAI_API_KEY。
GPT_MODEL=${GPT_MODEL:-$CONFIG_GPT_MODEL}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-$CONFIG_OPENAI_BASE_URL}
PROFILE_MODEL=${PROFILE_MODEL:-$CONFIG_PROFILE_MODEL}
PROFILE_BASE_URL=${PROFILE_BASE_URL:-$CONFIG_PROFILE_BASE_URL}
EVOLVE_MODEL=${EVOLVE_MODEL:-$CONFIG_EVOLVE_MODEL}
EVOLVE_BASE_URL=${EVOLVE_BASE_URL:-$CONFIG_EVOLVE_BASE_URL}
ANSWER_BASE_URL=${ANSWER_BASE_URL:-$CONFIG_ANSWER_BASE_URL}
RUBRIC_BASE_URL=${RUBRIC_BASE_URL:-$CONFIG_RUBRIC_BASE_URL}

# 并发数
SCORING_CONCURRENCY=${SCORING_CONCURRENCY:-10}
PROFILE_CONCURRENCY=${PROFILE_CONCURRENCY:-5}
EVO_CONCURRENCY=${EVO_CONCURRENCY:-10}
ANSWER_CONCURRENCY=${ANSWER_CONCURRENCY:-10}
RUBRIC_CONCURRENCY=${RUBRIC_CONCURRENCY:-10}
# ======================================================

if [ ! -f "$INPUT_FILE" ]; then
    echo "输入文件不存在: $INPUT_FILE"
    echo "请设置 INPUT_FILE 指向 data/data.jsonl 或其他 JSONL。"
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

# 辅助函数：计算 jsonl 的平均得分率
compute_avg_score_rate() {
    local scored_file="$1"
    python - "$scored_file" <<'PY'
import json, sys
path = sys.argv[1]
rates = []
def clamp_rate(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        score_rate = clamp_rate(item.get("score_rate"))
        if score_rate is not None:
            rates.append(score_rate)
            continue
        sr = item.get("scoring_result", {})
        if not isinstance(sr, dict):
            continue
        try:
            awarded = float(sr.get("total_awarded", 0) or 0)
            possible = float(sr.get("total_possible", 0) or 0)
        except (TypeError, ValueError):
            continue
        if possible > 0:
            rates.append(max(0.0, min(1.0, awarded / possible)))
avg = sum(rates) / len(rates) if rates else 0.0
print(f"{avg:.4f}")
PY
}

count_jsonl_records() {
    local jsonl_file="$1"
    python - "$jsonl_file" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    with path.open("r", encoding="utf-8") as f:
        print(sum(1 for line in f if line.strip()))
PY
}

# 辅助函数：比较两个浮点数，输出 true/false
lt_float() {
    python -c "print('true' if float('$1') < float('$2') else 'false')"
}

abs_diff_float() {
    python -c "print(abs(float('$1') - float('$2')))"
}

extract_effect_count() {
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
        if isinstance(effect, dict) and effect.get("effect_label") == "effective_boundary_probe":
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
echo "Evolution trigger rate: $MIN_SCORE_RATE" >> "$SUMMARY_FILE"
echo "Num candidates: $NUM_CANDIDATES" >> "$SUMMARY_FILE"
echo "Max candidate budget: $MAX_CANDIDATE_BUDGET" >> "$SUMMARY_FILE"
echo "Validation retries: $VALIDATION_RETRIES" >> "$SUMMARY_FILE"
echo "Tree search enabled: $ENABLE_TREE_SEARCH" >> "$SUMMARY_FILE"
echo "Tree search budget: branches=$MAX_SAMPLE_BRANCHES depth=$MAX_SAMPLE_DEPTH boundaries=$MAX_SAMPLE_BOUNDARIES candidates=$MAX_SAMPLE_CANDIDATES_TOTAL" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Round | Avg Score Rate | Status" >> "$SUMMARY_FILE"
echo "------|----------------|--------" >> "$SUMMARY_FILE"

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

BASELINE_RECORD_COUNT=$(count_jsonl_records "$ROUND_DIR/scored.jsonl")
if [ "$BASELINE_RECORD_COUNT" -eq 0 ]; then
    echo "Round 0 baseline 没有产生有效评分记录，停止运行。请检查 $ROUND_DIR/scored.jsonl.failed 以及 API base_url/model/key 配置。" >&2
    exit 1
fi

AVG_RATE=$(compute_avg_score_rate "$ROUND_DIR/scored.jsonl")
echo "Round $ROUND 平均得分率: $AVG_RATE"
printf "%5s | %14s | %s\n" "$ROUND" "$AVG_RATE" "baseline" >> "$SUMMARY_FILE"

PREV_SCORED="$ROUND_DIR/scored.jsonl"
PREV_AVG_RATE="$AVG_RATE"
PREV_EFFECT_COUNT=0
NO_INFO_STREAK=0
LAST_SEARCH_GRAPH=""
LAST_DISCOVERED_BOUNDARIES=""
LAST_FINAL_SCORED="$PREV_SCORED"
LAST_FRONTIER=""
COMBINED_SEARCH_GRAPH="$EXP_DIR/search_graph.jsonl"
COMBINED_DISCOVERED_BOUNDARIES="$EXP_DIR/discovered_boundaries.jsonl"

# ===================== Round 1..N: 循环进化 =====================
for ROUND in $(seq 1 "$MAX_ROUNDS"); do
    ROUND_DIR="$EXP_DIR/round_$ROUND"
    mkdir -p "$ROUND_DIR"

    echo ""
    echo "========================================"
    echo "Round $ROUND: Question Evolution"
    echo "========================================"

    if [ "$ENABLE_TREE_SEARCH" = "true" ]; then
        run_if_missing "$ROUND_DIR/active_frontier.jsonl" "[Round $ROUND] Tree Step 0a/12: frontier_scheduler.py active_frontier" \
            python frontier_scheduler.py \
                --mode active \
                --input "$PREV_SCORED" \
                --output "$ROUND_DIR/active_frontier.jsonl" \
                --max-sample-branches "$MAX_SAMPLE_BRANCHES" \
                --max-sample-depth "$MAX_SAMPLE_DEPTH" \
                --max-sample-boundaries "$MAX_SAMPLE_BOUNDARIES" \
                --max-sample-candidates-total "$MAX_SAMPLE_CANDIDATES_TOTAL"

        run_if_missing "$ROUND_DIR/input.jsonl" "[Round $ROUND] Step 0/11: 复制 active_frontier 输入" \
            cp "$ROUND_DIR/active_frontier.jsonl" "$ROUND_DIR/input.jsonl"
    else
        run_if_missing "$ROUND_DIR/input.jsonl" "[Round $ROUND] Step 0/11: 复制上一轮 scored/state 输入" \
            cp "$PREV_SCORED" "$ROUND_DIR/input.jsonl"
    fi

    ROUND_INPUT_COUNT=$(count_jsonl_records "$ROUND_DIR/input.jsonl")
    if [ "$ROUND_INPUT_COUNT" -eq 0 ]; then
        echo "Round $ROUND 输入为空，停止循环。"
        printf "%5s | %14s | %s\n" "$ROUND" "$PREV_AVG_RATE" "input_empty" >> "$SUMMARY_FILE"
        break
    fi

    if [ -f "$ROUND_DIR/scored.jsonl" ] && [ -s "$ROUND_DIR/scored.jsonl" ]; then
        echo "检测到已存在 $ROUND_DIR/scored.jsonl，跳过本轮生成闭环"
    else
        run_if_missing "$ROUND_DIR/profiled.jsonl" "[Round $ROUND] Step 1/11: profile_samples.py" \
            python profile_samples.py \
                --input "$ROUND_DIR/input.jsonl" \
                --output "$ROUND_DIR/profiled.jsonl" \
                --model "$PROFILE_MODEL" \
                --base-url "$PROFILE_BASE_URL" \
                --concurrency "$PROFILE_CONCURRENCY"

        run_if_missing "$ROUND_DIR/profiled_candidates.jsonl" "[Round $ROUND] Step 2/11: select_evolution_candidates.py" \
            python select_evolution_candidates.py \
                --input "$ROUND_DIR/profiled.jsonl" \
                --output "$ROUND_DIR/profiled_candidates.jsonl" \
                --high-score-threshold "$MIN_SCORE_RATE"

        run_if_missing "$ROUND_DIR/routed.jsonl" "[Round $ROUND] Step 3/11: operator_router.py" \
            python operator_router.py \
                --input "$ROUND_DIR/profiled_candidates.jsonl" \
                --output "$ROUND_DIR/routed.jsonl" \
                --memory-dir "$MEMORY_DIR"

        run_if_missing "$ROUND_DIR/candidates.jsonl" "[Round $ROUND] Step 4/11: question_evolution.py" \
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

        CANDIDATE_COUNT=$(count_jsonl_records "$ROUND_DIR/candidates.jsonl")
        if [ "$CANDIDATE_COUNT" -eq 0 ]; then
            echo "Round $ROUND 未生成候选题，停止循环。"
            printf "%5s | %14s | %s\n" "$ROUND" "$PREV_AVG_RATE" "candidate_empty" >> "$SUMMARY_FILE"
            break
        fi

        run_if_missing "$ROUND_DIR/validated_candidates.jsonl" "[Round $ROUND] Step 5/11: validate_evolved_question.py" \
            python validate_evolved_question.py \
                --input "$ROUND_DIR/candidates.jsonl" \
                --output "$ROUND_DIR/validated_candidates.jsonl"

        run_if_missing "$ROUND_DIR/evolved.jsonl" "[Round $ROUND] Step 6/11: candidate_selection.py" \
            python candidate_selection.py \
                --input "$ROUND_DIR/validated_candidates.jsonl" \
                --output "$ROUND_DIR/evolved.jsonl" \
                --invalid-output "$ROUND_DIR/invalid_generation_cases.jsonl"

        run_if_missing "$ROUND_DIR/with_answers.jsonl" "[Round $ROUND] Step 7/11: collect_answers.py" \
            python collect_answers.py \
                --input "$ROUND_DIR/evolved.jsonl" \
                --output "$ROUND_DIR/with_answers.jsonl" \
                --concurrency "$ANSWER_CONCURRENCY" \
                --samples 1 \
                --model "$GPT_MODEL" \
                --base-url "$ANSWER_BASE_URL"

        run_if_missing "$ROUND_DIR/rubric.jsonl" "[Round $ROUND] Step 8/11: gen_rubric.py" \
            python gen_rubric.py \
                --input "$ROUND_DIR/with_answers.jsonl" \
                --output "$ROUND_DIR/rubric.jsonl" \
                --concurrency "$RUBRIC_CONCURRENCY" \
                --model "$GPT_MODEL" \
                --base-url "$RUBRIC_BASE_URL"

        run_if_missing "$ROUND_DIR/scored.jsonl" "[Round $ROUND] Step 9/11: scoring.py" \
            python scoring.py \
                --input "$ROUND_DIR/rubric.jsonl" \
                --output "$ROUND_DIR/scored.jsonl" \
                --answer-mode llm \
                --answer-base-url "$QWEN_BASE_URL" \
                --answer-api-key "$QWEN_API_KEY" \
                --answer-model "$QWEN_MODEL" \
                --judge-base-url "$QWEN_BASE_URL" \
                --judge-api-key "$QWEN_API_KEY" \
                --judge-model "$QWEN_MODEL" \
                --concurrency "$SCORING_CONCURRENCY"

        ROUND_SCORED_COUNT=$(count_jsonl_records "$ROUND_DIR/scored.jsonl")
        if [ "$ROUND_SCORED_COUNT" -eq 0 ]; then
            echo "Round $ROUND 没有产生有效评分记录，停止运行。请检查 $ROUND_DIR/scored.jsonl.failed 以及 API base_url/model/key 配置。" >&2
            exit 1
        fi
    fi

    run_if_missing "$ROUND_DIR/effect_analysis.jsonl" "[Round $ROUND] Step 10/11: analyze_evolution_effect.py" \
        python analyze_evolution_effect.py \
            --before "$PREV_SCORED" \
            --input "$ROUND_DIR/scored.jsonl" \
            --output "$ROUND_DIR/effect_analysis.jsonl" \
            --matrix-output "$ROUND_DIR/effect_matrix.jsonl"

    run_if_missing "$ROUND_DIR/state_updated.jsonl" "[Round $ROUND] Step 11/11: update_sample_state.py" \
        python update_sample_state.py \
            --input "$ROUND_DIR/effect_analysis.jsonl" \
            --output "$ROUND_DIR/state_updated.jsonl" \
            --memory-dir "$MEMORY_DIR"

    if [ "$ENABLE_TREE_SEARCH" = "true" ]; then
        echo "[Round $ROUND] Step 12/13: build_search_graph.py（刷新本轮与累计树搜索产物）"
        python build_search_graph.py \
            --input "$ROUND_DIR/state_updated.jsonl" \
            --output "$ROUND_DIR/search_graph.jsonl" \
            --discovered-output "$ROUND_DIR/discovered_boundaries.jsonl" \
            --previous-graph "$COMBINED_SEARCH_GRAPH" \
            --combined-output "$COMBINED_SEARCH_GRAPH" \
            --previous-discovered "$COMBINED_DISCOVERED_BOUNDARIES" \
            --combined-discovered-output "$COMBINED_DISCOVERED_BOUNDARIES"

        run_if_missing "$ROUND_DIR/next_frontier.jsonl" "[Round $ROUND] Step 13/13: frontier_scheduler.py next_frontier" \
            python frontier_scheduler.py \
                --mode schedule \
                --input "$ROUND_DIR/state_updated.jsonl" \
                --output "$ROUND_DIR/next_frontier.jsonl" \
                --discovered-output "$ROUND_DIR/discovered_boundaries.jsonl" \
                --max-sample-branches "$MAX_SAMPLE_BRANCHES" \
                --max-sample-depth "$MAX_SAMPLE_DEPTH" \
                --max-sample-boundaries "$MAX_SAMPLE_BOUNDARIES" \
                --max-sample-candidates-total "$MAX_SAMPLE_CANDIDATES_TOTAL" \
                --enable-branch-backtrack "$ENABLE_BRANCH_BACKTRACK" \
                --enable-root-fork "$ENABLE_ROOT_FORK"

        LAST_SEARCH_GRAPH="$COMBINED_SEARCH_GRAPH"
        LAST_DISCOVERED_BOUNDARIES="$COMBINED_DISCOVERED_BOUNDARIES"
    fi

    # 计算本轮平均得分率
    AVG_RATE=$(compute_avg_score_rate "$ROUND_DIR/scored.jsonl")
    echo "Round $ROUND 平均得分率: $AVG_RATE"
    EFFECT_COUNT=$(extract_effect_count "$ROUND_DIR/effect_analysis.jsonl")
    AVG_DELTA=$(abs_diff_float "$AVG_RATE" "$PREV_AVG_RATE")

    ROUND_RESULT_FOR_FINAL="$ROUND_DIR/scored.jsonl"
    if [ -f "$ROUND_DIR/state_updated.jsonl" ] && [ -s "$ROUND_DIR/state_updated.jsonl" ]; then
        ROUND_RESULT_FOR_FINAL="$ROUND_DIR/state_updated.jsonl"
    fi
    LAST_FINAL_SCORED="$ROUND_RESULT_FOR_FINAL"
    ROUND_OUTPUT_FOR_NEXT="$ROUND_RESULT_FOR_FINAL"
    if [ "$ENABLE_TREE_SEARCH" = "true" ] && [ -f "$ROUND_DIR/next_frontier.jsonl" ] && [ -s "$ROUND_DIR/next_frontier.jsonl" ]; then
        LAST_FRONTIER="$ROUND_DIR/next_frontier.jsonl"
        ROUND_OUTPUT_FOR_NEXT="$ROUND_DIR/next_frontier.jsonl"
    fi

    # 检查提前停止条件
    SHOULD_STOP=$(lt_float "$AVG_RATE" "$EARLY_STOP_RATE")
    if [ "$SHOULD_STOP" = "true" ]; then
        echo "提前停止：Round $ROUND 平均得分率 $AVG_RATE < $EARLY_STOP_RATE"
        printf "%5s | %14s | %s\n" "$ROUND" "$AVG_RATE" "early_stop" >> "$SUMMARY_FILE"
        PREV_SCORED="$ROUND_OUTPUT_FOR_NEXT"
        break
    fi

    if [ "$EFFECT_COUNT" -eq 0 ] && [ "$(lt_float "$AVG_DELTA" "$NO_INFO_MIN_DELTA")" = "true" ]; then
        NO_INFO_STREAK=$((NO_INFO_STREAK + 1))
    else
        NO_INFO_STREAK=0
    fi

    if [ "$NO_INFO_STREAK" -ge "$NO_INFO_STOP_ROUNDS" ]; then
        echo "提前停止：连续 $NO_INFO_STREAK 轮无新信息（effect_count=0 且 avg_delta=$AVG_DELTA < $NO_INFO_MIN_DELTA）"
        printf "%5s | %14s | %s\n" "$ROUND" "$AVG_RATE" "no_info_stop" >> "$SUMMARY_FILE"
        PREV_SCORED="$ROUND_OUTPUT_FOR_NEXT"
        break
    fi

    if [ "$ENABLE_TREE_SEARCH" = "true" ] && { [ ! -f "$ROUND_DIR/next_frontier.jsonl" ] || [ ! -s "$ROUND_DIR/next_frontier.jsonl" ]; }; then
        echo "树搜索停止：Round $ROUND 未生成新的 next_frontier。"
        printf "%5s | %14s | %s\n" "$ROUND" "$AVG_RATE" "tree_frontier_empty" >> "$SUMMARY_FILE"
        LAST_FRONTIER=""
        PREV_SCORED="$ROUND_RESULT_FOR_FINAL"
        break
    fi

    printf "%5s | %14s | %s\n" "$ROUND" "$AVG_RATE" "continue" >> "$SUMMARY_FILE"

    PREV_SCORED="$ROUND_OUTPUT_FOR_NEXT"
    PREV_AVG_RATE="$AVG_RATE"
    PREV_EFFECT_COUNT="$EFFECT_COUNT"
done

# ===================== 保存最终结果 =====================
FINAL_DIR="$EXP_DIR/final"
mkdir -p "$FINAL_DIR"
cp "$LAST_FINAL_SCORED" "$FINAL_DIR/final_scored.jsonl"
if [ "$ENABLE_TREE_SEARCH" = "true" ]; then
    if [ -n "$LAST_FRONTIER" ] && [ -f "$LAST_FRONTIER" ]; then
        cp "$LAST_FRONTIER" "$FINAL_DIR/final_frontier.jsonl"
    fi
    if [ -n "$LAST_SEARCH_GRAPH" ] && [ -f "$LAST_SEARCH_GRAPH" ]; then
        cp "$LAST_SEARCH_GRAPH" "$FINAL_DIR/search_graph.jsonl"
    fi
    if [ -n "$LAST_DISCOVERED_BOUNDARIES" ] && [ -f "$LAST_DISCOVERED_BOUNDARIES" ]; then
        cp "$LAST_DISCOVERED_BOUNDARIES" "$FINAL_DIR/discovered_boundaries.jsonl"
    fi
fi

echo ""
echo "========================================"
echo "循环结束"
echo "最终结果: $FINAL_DIR/final_scored.jsonl"
echo "各轮汇总: $SUMMARY_FILE"
echo "========================================"
cat "$SUMMARY_FILE"
