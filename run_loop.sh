# run_loop.sh
#!/bin/bash
# Question Evolution 循环流水线
# 每轮把上一轮 scored 结果中的高分题进行 question evolution，并重新采集 reference、生成 rubric、复测。
# 支持提前停止：当某轮 Qwen 平均得分率 < EARLY_STOP_RATE 时停止。
# 每轮数据单独保存在 当天日期/exp*/round_N/ 子文件夹中。

set -uo pipefail

# ===================== 可配置参数 =====================
MAX_ROUNDS=5                      # 最大迭代轮数
EARLY_STOP_RATE=0.5                # 提前停止阈值：平均得分率低于该值则停止
MIN_SCORE_RATE=0.8                 # question_evolution 触发阈值

INPUT_FILE="data/police_qa_testset_v2.8_iteration_subset_selected.jsonl"   # 初始输入（含 prompt + references + rubric）
EXP_ROOT="experiments"                            # 实验结果根目录；每天在其下创建 YYYY-MM-DD/exp*

# Qwen（候选模型 / 评分模型）配置
QWEN_BASE_URL="http://127.0.0.1:18011/v1"
QWEN_API_KEY=""
QWEN_MODEL="hjl_Qwen3.6-27B"

# GPT（参考答案 / rubric 生成模型）配置
GPT_MODEL="gpt-5.4"

# 并发数
SCORING_CONCURRENCY=10
EVO_CONCURRENCY=10
ANSWER_CONCURRENCY=10
RUBRIC_CONCURRENCY=10
# ======================================================

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

echo "本次实验目录: $EXP_DIR"

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

SUMMARY_FILE="$EXP_DIR/summary.txt"
echo "Question Evolution Loop Summary" > "$SUMMARY_FILE"
echo "================================" >> "$SUMMARY_FILE"
echo "Max rounds: $MAX_ROUNDS" >> "$SUMMARY_FILE"
echo "Early stop rate: $EARLY_STOP_RATE" >> "$SUMMARY_FILE"
echo "Evolution trigger rate: $MIN_SCORE_RATE" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Round | Avg Score Rate | Status" >> "$SUMMARY_FILE"
echo "------|----------------|--------" >> "$SUMMARY_FILE"

# ===================== Round 0: 初始评分 =====================
ROUND=0
ROUND_DIR="$EXP_DIR/round_$ROUND"
mkdir -p "$ROUND_DIR"

echo ""
echo "========================================"
echo "Round $ROUND: 初始评分（ baseline ）"
echo "========================================"

if [ -f "$ROUND_DIR/scored.jsonl" ] && [ -s "$ROUND_DIR/scored.jsonl" ]; then
    echo "检测到已存在 $ROUND_DIR/scored.jsonl，跳过本轮执行"
else
    cp "$INPUT_FILE" "$ROUND_DIR/input.jsonl"
    python scoring.py \
        --input "$ROUND_DIR/input.jsonl" \
        --output "$ROUND_DIR/scored.jsonl" \
        --answer-mode llm \
        --answer-base-url "$QWEN_BASE_URL" \
        --answer-api-key "$QWEN_API_KEY" \
        --answer-model "$QWEN_MODEL" \
        --concurrency "$SCORING_CONCURRENCY"
fi

AVG_RATE=$(compute_avg_score_rate "$ROUND_DIR/scored.jsonl")
echo "Round $ROUND 平均得分率: $AVG_RATE"
printf "%5s | %14s | %s\n" "$ROUND" "$AVG_RATE" "baseline" >> "$SUMMARY_FILE"

PREV_SCORED="$ROUND_DIR/scored.jsonl"

# ===================== Round 1..N: 循环进化 =====================
for ROUND in $(seq 1 "$MAX_ROUNDS"); do
    ROUND_DIR="$EXP_DIR/round_$ROUND"
    mkdir -p "$ROUND_DIR"

    echo ""
    echo "========================================"
    echo "Round $ROUND: Question Evolution"
    echo "========================================"

    # 断点续跑：如果本轮 scored.jsonl 已存在且非空，则跳过执行，只做评估
    if [ -f "$ROUND_DIR/scored.jsonl" ] && [ -s "$ROUND_DIR/scored.jsonl" ]; then
        echo "检测到已存在 $ROUND_DIR/scored.jsonl，跳过本轮执行"
    else
        # 1) 复制上一轮 scored 结果作为本轮输入
        cp "$PREV_SCORED" "$ROUND_DIR/input.jsonl"

        # 2) Question Evolution：对高分题升级
        echo "[Round $ROUND] Step 1/4: question_evolution.py"
        python question_evolution.py \
            --input "$ROUND_DIR/input.jsonl" \
            --output "$ROUND_DIR/evolved.jsonl" \
            --min-score-rate "$MIN_SCORE_RATE" \
            --concurrency "$EVO_CONCURRENCY"

        # 3) 为进化后的题目重新采集参考答案
        echo "[Round $ROUND] Step 2/4: collect_answers.py"
        python collect_answers.py \
            --input "$ROUND_DIR/evolved.jsonl" \
            --output "$ROUND_DIR/with_answers.jsonl" \
            --concurrency "$ANSWER_CONCURRENCY" \
            --samples 1 \
            --model "$GPT_MODEL"

        # 4) 针对新题和新 reference 生成 rubric
        echo "[Round $ROUND] Step 3/4: gen_rubric.py"
        python gen_rubric.py \
            --input "$ROUND_DIR/with_answers.jsonl" \
            --output "$ROUND_DIR/rubric.jsonl" \
            --concurrency "$RUBRIC_CONCURRENCY" \
            --model "$GPT_MODEL"

        # 5) 用 Qwen 重新答题并评分
        echo "[Round $ROUND] Step 4/4: scoring.py"
        python scoring.py \
            --input "$ROUND_DIR/rubric.jsonl" \
            --output "$ROUND_DIR/scored.jsonl" \
            --answer-mode llm \
            --answer-base-url "$QWEN_BASE_URL" \
            --answer-api-key "$QWEN_API_KEY" \
            --answer-model "$QWEN_MODEL" \
            --concurrency "$SCORING_CONCURRENCY"
    fi

    # 计算本轮平均得分率
    AVG_RATE=$(compute_avg_score_rate "$ROUND_DIR/scored.jsonl")
    echo "Round $ROUND 平均得分率: $AVG_RATE"

    # 检查提前停止条件
    SHOULD_STOP=$(lt_float "$AVG_RATE" "$EARLY_STOP_RATE")
    if [ "$SHOULD_STOP" = "true" ]; then
        echo "提前停止：Round $ROUND 平均得分率 $AVG_RATE < $EARLY_STOP_RATE"
        printf "%5s | %14s | %s\n" "$ROUND" "$AVG_RATE" "early_stop" >> "$SUMMARY_FILE"
        PREV_SCORED="$ROUND_DIR/scored.jsonl"
        break
    else
        printf "%5s | %14s | %s\n" "$ROUND" "$AVG_RATE" "continue" >> "$SUMMARY_FILE"
    fi

    PREV_SCORED="$ROUND_DIR/scored.jsonl"
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