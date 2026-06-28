import json
from typing import Any, Dict, Optional


ROUTER_RULE_SUMMARY = """
规则路由优先级：
- 漏最小关键事实 -> O1_gap_choice，备选 O2_subclaim_localization
- 层级越推 -> O3_step_jump，备选 O4_near_level_ranking
- 题外补设 -> O5_extra_premise_detection
- 泛化罗列 -> O7_fact_binding_constraint
- 抓显眼点漏关键层 -> O8_double_threshold_claim
- 受干扰信息带偏 -> O6_single_variable_counterfactual 或 O9_abnormal_clue_mainline_switch
- target_failure_mode 为反常线索主线切换失败 -> O9_abnormal_clue_mainline_switch
- 上一轮 O1 且本轮满分 -> 禁止继续 O1，优先 O2/O4/O8
""".strip()


def build_router_prompt(record: Dict[str, Any], memory_summary: Optional[Dict[str, Any]] = None) -> str:
    """Optional LLM-router prompt; production routing is deterministic in operator_router.py."""
    payload = {
        "sample_id": record.get("sample_id", record.get("index")),
        "sample_profile": record.get("sample_profile", {}),
        "overscore_diagnosis": record.get("overscore_diagnosis", {}),
        "evolution_state": record.get("evolution_state", {}),
        "evolution_action": record.get("evolution_action"),
        "score_rate": record.get("score_rate"),
        "memory_summary": memory_summary or {},
    }
    return f"""
# 角色
你是 question evolution 的算子路由器。你只负责选择 operator，不生成新题，不修改 rubric，不推荐评分规则。

# 路由规则
{ROUTER_RULE_SUMMARY}

# 输入
{json.dumps(payload, ensure_ascii=False, indent=2)}

# 输出
返回合法 JSON 对象，不要输出 Markdown：
{{
  "operator_route": {{
    "primary_operator": "O1_gap_choice",
    "backup_operators": ["O2_subclaim_localization"],
    "avoid_operators": [],
    "routing_reason": "简要说明为什么选择该 operator",
    "is_high_value_sample": true,
    "should_use_local_tree_search": false
  }}
}}
""".strip()
