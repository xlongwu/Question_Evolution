# 阶段06流水线集成验收清单

本文档记录阶段06的可审查验收项。阶段06只负责把阶段01-05产物接入 `run_loop.sh` 并验证闭环，不修改 `collect_answers.py`、`gen_rubric.py`、`scoring.py` 的内部逻辑，不修改 rubric prompt、score prompt、rubric item、权重、扣分项或 judge 策略。

## 当前集成链路

`run_loop.sh` 当前每轮编排顺序为：

```text
round_0/input.jsonl
-> scoring.py
-> round_0/scored.jsonl

round_N/input.jsonl
-> profile_samples.py
-> select_evolution_candidates.py
-> operator_router.py
-> question_evolution.py --num-candidates
-> validate_evolved_question.py
-> candidate_selection.py
-> collect_answers.py
-> gen_rubric.py
-> scoring.py
-> analyze_evolution_effect.py
-> update_sample_state.py
-> round_N/state_updated.jsonl
```

`state_updated.jsonl` 优先作为下一轮输入，保证 `evolution_state` 能进入后续画像、路由和进化决策。

## 必需产物

一次 `MAX_ROUNDS=1` 的验收运行至少应生成：

- `round_0/input.jsonl`
- `round_0/scored.jsonl`
- `round_1/input.jsonl`
- `round_1/profiled.jsonl`
- `round_1/profiled_candidates.jsonl`
- `round_1/routed.jsonl`
- `round_1/candidates.jsonl`
- `round_1/validated_candidates.jsonl`
- `round_1/evolved.jsonl`
- `round_1/with_answers.jsonl`
- `round_1/rubric.jsonl`
- `round_1/scored.jsonl`
- `round_1/effect_analysis.jsonl`
- `round_1/effect_matrix.jsonl`
- `round_1/state_updated.jsonl`
- `memory/operator_memory_bank.jsonl`
- `memory/failure_memory_bank.jsonl`
- `memory/invalid_generation_cases.jsonl`
- `final/final_scored.jsonl`

## 本地离线验收

不依赖 Bash 或外部 API 的验收命令：

```bash
python tests/test_stage01_contract.py
python tests/test_stage02_profile_candidates.py
python tests/test_stage03_operator_routing.py
python tests/test_stage04_complexity_selection.py
python tests/test_stage05_effect_state.py
python tests/test_stage06_run_loop_integration.py
python tests/test_stage06_tiny_pipeline_artifacts.py
python tests/test_score_prompt_contract.py
python tests/test_runtime_environment_check.py
python -m py_compile tests/test_stage06_run_loop_integration.py tests/test_stage06_tiny_pipeline_artifacts.py tests/test_score_prompt_contract.py tests/test_runtime_environment_check.py schema_validation.py check_runtime_environment.py question_evolution.py operator_router.py profile_samples.py select_evolution_candidates.py validate_evolved_question.py candidate_selection.py analyze_evolution_effect.py update_sample_state.py collect_answers.py gen_rubric.py scoring.py
```

这些检查覆盖：

- `run_loop.sh` 编排顺序。
- 阶段脚本 CLI 参数接线。
- `state_updated.jsonl` 进入下一轮输入。
- memory 写入由 `state_updated.jsonl` 断点保护。
- mock tiny sample 产出 profile、route、validation、selection、effect、state、memory 和 final 文件。
- `expected_evaluation_focus` 没有接入 rubric/judge 相关步骤。

## 真实环境验收

根目录已提供 8 条指定样本的 `admitted_seed_samples.jsonl`，用于阶段06 tiny admitted seed 验收入口。在具备真实 API 配置后补跑：

```bash
python check_runtime_environment.py
bash -n run_loop.sh
INPUT_FILE=admitted_seed_samples.jsonl MAX_ROUNDS=1 bash run_loop.sh
```

运行前至少确认：

- 根目录存在 `admitted_seed_samples.jsonl`，或显式设置 `INPUT_FILE` 指向其他已准入样本 JSONL。
- 配置了 profile/evolution/answer/rubric/scoring 所需的模型、base URL 和 API key；可使用环境变量，也可使用被 `.gitignore` 忽略的本地 `config.py`。
- Qwen 评分服务可访问，或相应 `QWEN_BASE_URL`、`QWEN_MODEL` 已按本机环境设置；本地服务不需要 key 时，`QWEN_API_KEY` 保持空字符串即可。

真实验收通过后，应检查本次实验目录中的必需产物，并确认 `summary.txt` 中至少包含 baseline 和 round 1 记录。

## 当前环境限制

当前本机环境尚不能完成真实验收，原因包括：

- 当前进程未发现 `OPENAI`、`EVOLVE`、`PROFILE`、`QWEN`、`GPT` 或 `API` 相关配置变量名。
- `.venv` 下 `python check_runtime_environment.py` 显示 `bash_available=true`、`python_dependencies_ready=true`、`admitted_seed_ready=true`，但 `api_config_ready=false`。
- `admitted_seed_samples.jsonl` 已存在，包含 8 条指定样本；该项不再是当前阻塞。
