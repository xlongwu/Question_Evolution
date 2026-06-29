# 阶段05：状态更新、frontier 编排与主流程集成

阶段编号：05

阶段名称：状态更新、frontier 编排与主流程集成

阶段目标：把前面阶段形成的能力轴、候选、边界判断结果收束为可持续运行的树状搜索循环，统一维护预算、分支状态、已发现边界和下一轮待扩展节点。

阶段范围：覆盖 `update_sample_state.py`、`active_frontier.jsonl` 生成、`search_graph.jsonl` 追加、停止条件、`run_loop.sh` 接入和 round 语义弱化。该阶段是全流程集成阶段。

主要任务：

1. 在 `update_sample_state.py` 中实现单分支停止条件：有效边界命中后不再深入、连续无新信息、重复题型、候选同质化、达到最大深度。
2. 实现单样本停止条件：边界点数量达到上限、推荐能力轴均已探索、剩余分支只会重复、样本预算耗尽、连续若干轮无新增边界。
3. 根据阶段04输出更新 `discovered_boundaries`、`branch_status`、`sample_budget_remaining`、`branch_budget_remaining`、`no_new_boundary_rounds`。
4. 生成下一轮 `active_frontier.jsonl`，支持继续当前分支、回根节点开新分支、回父节点开兄弟分支。
5. 将候选节点和边界叶子追加到 `search_graph.jsonl`，记录父子关系、能力轴、选中状态、去重签名和停止原因。
6. 在 `run_loop.sh` 中让 round 消费 frontier，而不是默认所有样本同步前进一步。
7. 保留 `MAX_ROUNDS` 作为批次上限，但优先使用单样本预算和无新增边界停止条件。
8. 增加配置开关，使禁用回溯或禁用 root fork 时可退回接近单链的行为。

涉及模块/文件范围：`update_sample_state.py`、`run_loop.sh`、`active_frontier.jsonl`、`search_graph.jsonl`、各阶段中间产物、实验输出目录结构。

前置依赖：依赖阶段02动作输出、阶段03候选元数据和阶段04边界选择结果。需要人工确认输出目录命名、旧实验是否需要迁移，以及是否允许在每轮输出目录新增 `search_graph.jsonl` 和 `active_frontier.jsonl`。

完成标准：

1. 一个样本命中某个边界后，系统可以回到根节点或父节点继续探索其他能力轴。
2. 样本预算耗尽、分支无新信息或能力轴探索完成时可以正确停止。
3. 每轮输出中可追踪 frontier、graph 和主链状态。
4. `search_graph.jsonl` 能复盘节点父子关系、分支状态和边界命中情况。
5. 禁用树状搜索相关配置时，流程可退回旧单链行为或近似单链行为。

测试与验证方式：

1. 用 2 到 3 个样本跑 dry run 或 mock run，覆盖继续当前分支、根节点 fork、父节点 fork、停止分支、停止样本。
2. 检查预算扣减只发生一次，且分支预算和样本预算一致。
3. 检查 frontier 数量、graph 节点数量和停止状态是否符合预期。
4. 验证失败重跑或重复输入不会产生不可解释的重复节点。
5. 验证禁用 `ENABLE_BRANCH_BACKTRACK` 和 `ENABLE_ROOT_FORK` 时行为符合配置。

风险与注意事项：

1. 这是状态一致性风险最高的阶段，应先用小样本和 mock 数据验证。
2. 预算扣减必须集中，避免画像、生成、选择和状态更新各自扣减。
3. 如果 frontier 和 graph 写入不具备幂等性，失败重跑会产生重复节点。
4. round 语义弱化后，日志和目录命名仍需保持可读，避免实验复盘困难。
5. 不应在集成阶段顺手重构无关脚本，否则回滚成本会升高。

预计交付物：

1. 状态更新闭环。
2. frontier 编排逻辑。
3. search graph 追加逻辑。
4. run_loop 集成。
5. 停止条件实现。
6. 可回退配置。

Codex Goal 建议：适合作为一次独立 Codex Goal，但如果现有 `run_loop.sh` 与 `update_sample_state.py` 很复杂，可在执行时拆成两个连续子任务：先实现状态更新和 frontier，再接入 run_loop。
