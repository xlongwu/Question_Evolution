import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from prompts.operators import build_operator_prompt, get_operator_spec
from select_evolution_candidates import (
    EVOLVE_HIGH_SCORE_OVERSCORE,
    EXPAND_CURRENT_BRANCH,
    FORK_FROM_PARENT,
    FORK_FROM_ROOT,
    PASS_THROUGH_OR_SCORING_NOISE,
    RECONSTRUCT_LOW_SCORE_BOUNDARY,
    STOP_EVOLUTION,
)
from search_state_contract import (
    FRONTIER_ACTION_TYPES,
    TREE_SEARCH_CONFIG_DEFAULTS,
    get_root_prompt,
    normalize_search_state,
    sample_key,
)
from local_api_config import get_config_list, get_config_value
import validate_evolved_question as validation_stage


# 默认使用与 rubric_evolution 相同的 strong model，可通过 CLI 或环境变量覆盖。
EVOLVE_MODEL = (
    os.getenv("EVOLVE_MODEL")
    or get_config_value("EVOLVE_MODEL", "QA_MODEL", "GPT_MODEL", default="gpt-5.4")
)
EVOLVE_BASE_URL = (
    os.getenv("EVOLVE_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or get_config_value("EVOLVE_BASE_URL", "BASE_URL", "OPENAI_BASE_URL", default="https://hanbbq.labpilot.top/v1")
)

REQUEST_TIMEOUT_SECONDS = 180.0
MAX_OUTPUT_TOKENS = 32768
DEFAULT_MAX_VALIDATION_RETRIES = 1
EVOLUTION_REQUIRED_ACTIONS = {
    EVOLVE_HIGH_SCORE_OVERSCORE,
    RECONSTRUCT_LOW_SCORE_BOUNDARY,
}
NON_EVOLUTION_ACTIONS = {
    PASS_THROUGH_OR_SCORING_NOISE,
    STOP_EVOLUTION,
}
FRONTIER_FIELD_NAMES = {
    "sample_id",
    "search_root_id",
    "frontier_node_id",
    "source_node_id",
    "source_node_type",
    "source_prompt",
    "prompt_source",
    "action_type",
    "branch_id",
    "target_boundary_axis",
    "search_depth",
    "next_depth",
    "max_search_depth",
    "branch_budget_remaining",
    "sample_budget_remaining",
    "discovered_boundary_count",
}
FRONTIER_CONTEXT_KEYS = ("_frontier_context", "frontier_context", "active_frontier")
SOURCE_NODE_TYPES = {"current", "root", "parent"}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_api_keys(cli_keys: Optional[List[str]] = None) -> List[str]:
    if cli_keys:
        keys = [key.strip() for key in cli_keys if key and key.strip()]
        if keys:
            return keys
    raw = os.getenv("EVOLVE_API_KEYS") or os.getenv("OPENAI_API_KEYS") or os.getenv("OPENAI_API_KEY") or ""
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    if keys:
        return keys
    return get_config_list(
        "EVOLVE_API_KEYS",
        "GPT_API_KEYS",
        "HIAPI_KEYS_BIG",
        "OPENAI_API_KEYS",
        "OPENAI_API_KEY",
        "API_KEYS",
    )


def append_validation_retry_instruction(user_prompt: str, reject_reason: Optional[str]) -> str:
    reason = str(reject_reason or "").strip()
    if not reason:
        return user_prompt
    return (
        user_prompt.rstrip()
        + "\n\n# 上一轮候选题未通过独立复杂度/可回答性校验\n"
        + f"reject_reason: {reason}\n"
        + "请继续使用同一个 operator，不要更换题型主轴；只修正上述问题后重新生成。"
        + "新题必须可回答、单主轴、不过度依赖格式复杂度，也不得引入题干外知识。\n"
    )

#########################################################
'''
V1版本的原始prompt：
"""
# 角色
你是一位专门设计大模型评测题目的专家。当前题目对模型的区分度不足：较强模型（参考答案）和较弱模型（候选答案）的得分都过高。你的任务是把原题升级为"更难版本"，使其能更有效地区分模型的真实能力。

# 目标
分析原题、参考答案和候选答案。候选答案虽然得分高，但可能存在：泛泛而谈、缺少深度推理、遗漏边界条件、未能紧扣核心逻辑等问题。你需要生成一个新题目，使得：
1. 参考答案中的核心知识和推理链条仍然适用，或只需合理扩展即可回答新题；
2. 候选答案中的 superficial / 模板化 / 泛化 / 堆叠术语式的回答不再能轻易得高分；
3. 新题能迫使回答展现以下至少一种能力：因果链推理、边界意识、反事实分析、多条件综合判断、精确操作化、或对干扰信息的甄别。

# 可选进化策略（根据题目特点选择最合适的 1-3 种，不要全部堆砌）
1. **增加约束条件**：要求回答必须基于特定视角、法律条款、技术约束、场景限制或证据条件。
2. **要求显式推理**：不要只问"是什么"，而是问"为什么""如何排除其他可能""在什么条件下结论不成立"。
3. **引入边界或反事实**：改变原题中的某个条件，问结论会如何变化，或要求指出适用边界与例外。
4. **提升综合复杂度**：把两个相关子问题合并，要求比较、权衡、排序或推导优先级。
5. **抑制泛化回答**：要求回答必须紧扣本题具体情境，避免通用套话；或限制回答长度，要求"用最精炼的语言给出最关键的两点"。

# 原则
1. **可回答性**：新题必须仍能被强模型基于参考答案合理回答，不能引入需要外部未提供知识的隐藏条件。
2. **区分度**：新题应让"覆盖关键词但缺乏深度"的回答明显失分，让紧扣 reference 主线的回答得高分。
3. **不过拟合**：不要为了刁难某个候选答案而设置极窄陷阱；进化应提升题目本身质量，而不是针对特定措辞找茬。
4. **语言一致**：新题语言必须与原题一致。
5. **保持核心主题**：不要改变题目的领域和核心事实，只在深度、约束、推理要求上升级。

# 输入
## 原题
{|prompt|}

## 参考答案
{|response1|}

## 候选答案（当前得分过高，需要题目升级以区分强弱模型）
{|response2|}

## 现有评分标准
{|rubrics|}

# 输出
返回合法 JSON 对象，不要输出 Markdown 标记或额外解释：
```json
{
  "evolved_prompt": "升级后的新题目。必须是一个完整、可独立作答的问题。",
  "evolution_strategy": "说明采用了哪些策略（如'增加约束/要求反事实推理/要求比较权衡'），以及为什么这些策略能提升区分度",
  "notes_for_reference": "如果参考答案需要补充或调整才能完美回答新题，请简要说明；如果基本适用则写'基本适用'"
}
```
"""
'''
########################################################

QUESTION_EVOLUTION_PROMPT_TEMPLATE_V1 = """
# 角色
你是一位负责“能力边界定向压测”的评测题目设计专家。你的任务不是把题目机械改难，也不是把题目改成长、多步骤、强格式的结构题，而是要把原题升级成一道：
1. 单主轴；
2. 可回答；
3. 可稳定评分；
4. 能让模型在一个可归因的关键判断点上暴露错误
的新题。

# 核心目标
请分析原题、参考答案、候选答案和现有评分标准，生成一道升级后的新题，使其满足：
1. 仍然考查原题的核心领域、核心事实和核心能力，不改变题目主题；
2. 新题的主要失分点集中在 1 个关键判断上，而不是因为任务太多、格式太复杂、篇幅太长而失分；
3. 新题优先考以下几类真实短板，而不是泛泛增加复杂度：
   - 分不清“已经异常”与“已经能写成结论”；
   - 分不清两个都像有用的补强事实里，哪一个才真正推动结论上升一层；
   - 抓住了显眼事实，但漏掉真正决定定性的那一步；
   - 看起来知道“不够”，但说不准最少到底缺哪一条；
   - 把“看起来像闭环”误当成“已经形成闭环”；
4. 后续 judge 能根据 rubric 稳定判断对错，而不是靠术语密度、结构完整性或格式服从度打分。

# 第零步：先判断这道题下一步该往哪种失误点上压
在内部先做判断，但不要把全部中间推理输出：

1. 先判断原题当前更像哪一类：
   - 开放题太散，模型靠泛泛扩写保分；
   - 边界已经比较清楚，但模型还没被逼到真正关键缺口；
   - 题目如果继续“讲清楚”，反而会更容易答；
   - 题目已经能稳定暴露边界，继续改信息增量很低。

2. 再判断这道题当前最值得压的主错误是哪一种，只能选 1 个：
   - A. 泛泛罗列，缺少题干事实绑定；
   - B. 只会说大方向，不会指出真正关键缺口；
   - C. 混淆层级，把线索/异常/嫌疑直接上推成结论成立；
   - D. 把两个都重要的补强事实，错认成同等关键；
   - E. 把“看起来像闭环”误当成“已经排他成立”；
   - F. 被显眼信息带偏，漏掉真正决定结论的主轴。

3. 如果你发现“把题改得更清楚”会让候选答案更容易顺着答对，那么禁止继续沿着“先承认第一层，再问还缺什么”的方式改写，必须换题型。

# 第一步：先拆结论，再决定是否继续考这一层
1. 如果原题目标结论可以拆成 2-3 个相邻层级，先拆开。
2. 只能选择其中 1 个最值得压测的相邻层级差异作为主轴。
3. 如果候选答案已经能稳定回答“哪一层成立、哪一层不成立”，本轮不得继续复用同类问法，必须改成更细的比较题。
4. 如果新题会把真正难点提前说破，只剩一个显而易见的缺口可答，则这次改写无效，必须重做。

# 第二步：题型选择
从下面题型里只选 1 种主方式；如必须组合，只允许“1 个主方式 + 1 个轻量辅助方式”。

1. 候选缺口二选一
   适用：模型已经知道“大方向还不够”，但分不清哪个补强事实才是真正关键。
   目标：逼它在两个都像有帮助的补强事实里，选出真正推动结论上升一层的那一个。

2. 子判断定位
   适用：模型容易把“已支持层”和“未支持层”混在一起。
   目标：逼它指出到底是哪一个子判断还不能成立，以及最少还缺什么。

3. 近似项分层
   适用：模型会做粗分层，但不会分更细的相邻层级。
   目标：给 2-3 个都“看起来不能直接下结论”的近似项，要求继续分高低，而不是只做可写/不可写二分。

4. 伪闭环识别
   适用：模型容易把“局部成立”误写成“整体连续成立”。
   目标：让模型区分“局部画面已成立”“连续性仍未闭环”“排他性仍未闭环”。

5. 补强项升级判断
   适用：模型知道某事实会让异常更强，但不会分“更异常”和“结论上升一层”。
   目标：让模型判断哪个补充事实只是让原判断更稳，哪个才真正改变结论层级。

6. 单变量反事实
   适用：只差一条条件变化就能看出模型会不会重新排列判断顺序。
   目标：只改 1 个变量，逼模型说明哪一层变化、哪一层不变。

# 第三步：反复题型与“讲清楚反而变简单”检查
生成新题前必须逐项自检：

1. 这次新题是否仍然只是“哪一层成立 + 还缺什么”的换壳版本？
   如果是，且候选答案已经能稳定处理这一类题，则必须换题型。

2. 这次新题是否把真正难点提前说破了？
   如果候选答案只要顺着题面复述“第一层成立、第二层不成立、缺连续性/缺直接状态/缺排他闭环”就容易拿高分，则必须重做。

3. 这次新题是否只是把题面写得更标准、更聚焦，但没有制造两个足够接近的竞争判断？
   如果是，这种改写大概率只会让题更容易答，不应采用。

4. 如果存在两个以上都说得通的最小缺口，不得做开放式“还缺什么”，必须改成候选项区分题。

# 第四步：复杂度预算
1. 新题必须只有 1 个清晰主轴。读者应能用一句话说清“本题到底考什么”。
2. 新增事实或场景条件最多 3 条。
3. 输出任务最多 2 个，优先 1 个。
4. 候选项最多 3 个；如果是缺口比较题，优先 2 个近似候选。
5. 不得要求大表格、多层标签体系、固定句数、复杂编号系统。
6. 不得把难度主要建立在长篇格式、繁琐约束、复杂任务编排上。
7. 如果上一轮已经很长，本轮优先收主轴，不要继续加材料。

# 第五步：可回答性与唯一失分点检查
1. 题干必须提供完成任务所需事实，不得要求题外知识才能回答。
2. 新题应让弱模型主要在 1 个核心错误上失分，而不是在多个任务点上同时失分。
3. 不得设置只能靠猜、靠经验、靠题外法律评价才能作答的陷阱。
4. 如果参考答案需要大量新增知识才能回答，说明改写失败。
5. 如果删掉题目中的一小段要求后，核心考点明显变化，说明本题混入了多个能力点，必须继续简化。

# 禁止的进化方式
1. 禁止把难度主要建立在题目更长、任务更多、格式更复杂上。
2. 禁止连续多轮只复用“最小关键事实/最小前提/最小跳步/哪一层成立”这一类同家族题型。
3. 禁止把开放题统一改写成边界题后就停住，不再继续追问真正决定胜负的差异。
4. 禁止让题目主要考“遵循复杂指令”，而不是考原题对应的专业判断能力。

# 输入
## 原题
{|prompt|}

## 参考答案
{|response1|}

## 候选答案
{|response2|}

## 现有评分标准
{|rubrics|}

# 输出要求
返回合法 JSON 对象，不要输出 Markdown 标记或额外解释。
必须包含以下字段：

{
  "evolved_prompt": "升级后的新题目。必须完整、可独立作答、聚焦一个主轴，并遵守复杂度预算。",
  "evolution_strategy": "说明本轮选择了哪一个主错误、哪一种题型、为什么没有继续沿用更容易让题变清楚的改法。",
  "evaluation_focus": [
    "后续评分时最该检查的错误1",
    "后续评分时最该检查的错误2",
    "后续评分时最该检查的错误3"
  ],
  "complexity_budget": {
    "main_axis": "一句话说明本题核心考什么",
    "chosen_primary_cause": "A/B/C/D/E/F 之一",
    "chosen_primary_method": "候选缺口二选一/子判断定位/近似项分层/伪闭环识别/补强项升级判断/单变量反事实",
    "target_subclaim": "若结论可拆分，这一轮具体考哪一个子判断或哪一段相邻层级差异",
    "new_facts_count": 0,
    "output_tasks_count": 0,
    "candidate_options_count": 0,
    "estimated_prompt_chars": 0,
    "clarity_trap_checked": true,
    "why_within_budget": "说明为什么没有复杂度堆叠，也没有把题改得更清楚却更容易答"
  },
  "notes_for_reference": "如果参考答案需要补充或调整才能完美回答新题，请简要说明；如果基本适用则写'基本适用'"
}

# 最终质量自检
输出前逐项确认：
1. 新题不是简单加长版；
2. 新题只有 1 个主轴；
3. 如果原题已经被压成边界题，这次是否真的换到了更细的竞争判断；
4. 如果题目要求比较缺口，是否给出了足够接近但不等价的候选；
5. 新题不会因为“更清楚”而让候选答案更容易顺着答；
6. 新题的主要失分点能被 rubric 稳定捕捉。
""".strip()


QUESTION_EVOLUTION_PROMPT_TEMPLATE_V2 = """
# 角色
你是一位负责"题目难度升级与反模板化"的评测专家。当前题目的瓶颈是：候选答案（通常来自较小模型）靠泛化扩写、专业术语堆砌或通用流程覆盖拿到了高分，但实际质量明显弱于参考答案。你需要把原题改写成一道能压制这类"虚高回答"的升级题。

# 目标
生成的新题必须同时满足：
1. 参考答案中的核心判断和关键依据仍然可以直接使用，或经过简单延伸即可回答；
2. 候选答案那种"看起来很长、很专业，但缺乏对本题具体因果链的聚焦"的回答应失分；
3. 新题能迫使模型给出：紧扣题意的因果链、具体情境下的边界判断、对干扰项的排除、或受约束的精确结论。

# 推荐升级方向（择其适用者，不要全部使用）
1. **要求说明"为什么不是其他选项/其他可能"**：迫使模型给出排除性推理，而不是罗列知识点。
2. **加入具体但合理的限制**：如"假设只能使用视频中出现的证据""不考虑额外的技术鉴定""在资源受限的情况下"，测试模型能否在约束下聚焦。
3. **把开放性问题改为带条件的判断**：如"如果嫌疑人声称X，该如何根据现有证据反驳/支持？"要求模型把证据和结论绑定。
4. **要求给出最小充分条件**：如"要证明该结论，最关键的两条证据是什么？"抑制泛泛而谈。
5. **要求识别题目中的误导或冗余信息**：如"上述情境中哪些信息对判断没有实质帮助？"测试模型是否能聚焦主线。

# 约束
1. 不得改变原题的核心事实、领域和基本案情。
2. 不得引入需要外部未提供知识才能回答的条件。
3. 新题语言与原题一致。
4. 不要设置只能靠猜的陷阱；升级应体现在推理深度和聚焦度上。

# 输入
## 原题
{|prompt|}

## 参考答案
{|response1|}

## 候选答案（当前得分虚高，需要题目升级以压制泛化回答）
{|response2|}

## 现有评分标准
{|rubrics|}

# 输出
返回合法 JSON 对象，不要输出 Markdown 标记或额外解释：
```json
{
  "evolved_prompt": "升级后的新题目。必须是一个完整、可独立作答的问题。",
  "evolution_strategy": "说明采用了哪些策略，以及为什么能压制候选答案的虚高并拉开模型差距",
  "notes_for_reference": "如果参考答案需要补充或调整才能完美回答新题，请简要说明；如果基本适用则写'基本适用'"
}
```
""".strip()


def get_evolution_prompt_template(version: str) -> str:
    version = (version or "").strip().lower()
    if version in {"v1", "baseline", "default", ""}:
        return QUESTION_EVOLUTION_PROMPT_TEMPLATE_V1
    if version in {"v2", "anti-verbosity", "focused"}:
        return QUESTION_EVOLUTION_PROMPT_TEMPLATE_V2
    raise ValueError(f"不支持的 question evolution prompt 版本: {version}")


def extract_answer(resp) -> str:
    choices = getattr(resp, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        return (getattr(message, "content", "") or "").strip()

    if hasattr(resp, "model_dump"):
        payload = resp.model_dump()
        choices = payload.get("choices")
        if choices:
            message = choices[0].get("message", {})
            return (message.get("content", "") or "").strip()

    if isinstance(resp, str):
        payload = resp
        if payload.startswith("data:"):
            payload = payload[len("data:"):].strip()
        parsed = json.loads(payload)
        return (parsed["choices"][0]["message"]["content"] or "").strip()

    raise TypeError(f"Unsupported or empty response type: {type(resp)}")


class RotatingAPIClient:
    """支持自动切换 API Key 的 OpenAI 兼容客户端包装器。"""

    def __init__(self, base_url: str, api_keys: List[str], request_timeout: float = REQUEST_TIMEOUT_SECONDS):
        if not api_keys:
            raise ValueError("api_keys 不能为空")
        self.base_url = base_url
        self.api_keys = api_keys
        self.request_timeout = request_timeout
        self.current_key_index = 0
        self.client: Optional[AsyncOpenAI] = None
        self._lock = asyncio.Lock()
        self._init_client()

    def _init_client(self):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install openai to run question_evolution.py.") from exc

        current_key = self.api_keys[self.current_key_index]
        kwargs = {
            "api_key": current_key,
            "timeout": self.request_timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.client = AsyncOpenAI(**kwargs)
        logger.info(
            f"使用 question evolution API Key [{self.current_key_index + 1}/{len(self.api_keys)}]: "
            f"{current_key[:8]}..."
        )

    async def close(self):
        if self.client:
            await self.client.close()

    def _is_token_exhausted_error(self, error: Exception) -> bool:
        error_str = str(error)
        return (
            "401" in error_str and
            ("TokenStatusExhausted" in error_str or "令牌额度已用尽" in error_str)
        )

    async def switch_to_next_key(self) -> bool:
        async with self._lock:
            self.current_key_index += 1
            if self.current_key_index >= len(self.api_keys):
                logger.error("所有 question evolution API Key 额度已用尽")
                return False
            self._init_client()
            return True

    async def chat_completions_create(self, **kwargs):
        for _ in range(len(self.api_keys)):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if self._is_token_exhausted_error(e):
                    logger.warning(f"question evolution API Key [{self.current_key_index + 1}] 额度用尽: {str(e)[:100]}")
                    if await self.switch_to_next_key():
                        continue
                    raise Exception("所有 question evolution API Key 额度已用尽") from e
                raise
        raise Exception("所有 question evolution API Key 额度已用尽")


def collect_json_candidate_texts(response_text: str) -> List[str]:
    text = response_text if isinstance(response_text, str) else str(response_text)
    stripped = text.strip()
    candidates: List[str] = []
    seen = set()

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    if not stripped:
        return candidates

    code_fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)
    for match in reversed(list(code_fence_pattern.finditer(stripped))):
        add(match.group(1))

    outside_text = re.sub(r"```[\s\S]+?```", "\n", stripped).strip()
    if outside_text and outside_text != stripped:
        add(outside_text)

    object_start, object_end = stripped.find("{"), stripped.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        add(stripped[object_start:object_end + 1])

    if not candidates:
        add(stripped)

    return candidates


def loads_json_with_repair(json_str: str) -> Any:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([\]}])", r"\1", json_str.strip())
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(repaired.lstrip())
            return obj
        except Exception:
            object_start, object_end = repaired.find("{"), repaired.rfind("}")
            if object_start != -1 and object_end != -1 and object_end > object_start:
                return json.loads(repaired[object_start:object_end + 1])
            raise


def parse_evolution_response(response_text: str) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for candidate in collect_json_candidate_texts(response_text):
        try:
            parsed = loads_json_with_repair(candidate)
            if not isinstance(parsed, dict):
                raise ValueError("evolution 响应必须是 JSON 对象")
            if "evolved_prompt" not in parsed:
                raise ValueError("evolution 响应缺少 evolved_prompt 字段")
            evolved_prompt = str(parsed["evolved_prompt"]).strip()
            if not evolved_prompt:
                raise ValueError("evolved_prompt 不能为空")
            parsed["evolved_prompt"] = evolved_prompt
            parsed["evolution_strategy"] = str(parsed.get("evolution_strategy", "")).strip()
            parsed["notes_for_reference"] = str(parsed.get("notes_for_reference", "")).strip()
            return parsed
        except Exception as e:
            last_error = e
    raise ValueError(f"无法解析有效 question evolution JSON: {last_error}")


def validate_evolved_question(original_prompt: str, evolved_prompt: str) -> None:
    if not evolved_prompt or not evolved_prompt.strip():
        raise ValueError("进化后的问题不能为空")
    if evolved_prompt.strip() == original_prompt.strip():
        raise ValueError("进化后的问题与原题完全相同")
    if len(evolved_prompt) < 0.5 * len(original_prompt):
        raise ValueError("进化后的问题明显短于原题，疑似丢失信息")


def load_json_or_jsonl(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON 输入必须是数组")
        return data
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return parsed if parsed >= 0 else default


def short_hash(value: Any) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()[:8]


def safe_id_segment(value: Any, fallback: str = "node") -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^0-9a-zA-Z_-]+", "_", text).strip("_")
    return text[:48] or fallback


def get_frontier_key_suffix(item: Dict[str, Any]) -> str:
    for container_name in FRONTIER_CONTEXT_KEYS:
        container = item.get(container_name)
        if isinstance(container, dict):
            frontier_node_id = clean_text(container.get("frontier_node_id"))
            if frontier_node_id:
                return frontier_node_id

    candidate_generation = item.get("candidate_generation")
    if isinstance(candidate_generation, dict):
        frontier_node_id = clean_text(candidate_generation.get("frontier_node_id"))
        if frontier_node_id:
            return frontier_node_id

    meta_info = item.get("meta_info")
    if isinstance(meta_info, dict):
        metadata = meta_info.get("question_evolution_metadata")
        if isinstance(metadata, dict):
            frontier_node_id = clean_text(metadata.get("frontier_node_id"))
            if frontier_node_id:
                return frontier_node_id

    frontier_node_id = clean_text(item.get("frontier_node_id"))
    if frontier_node_id:
        return frontier_node_id

    source_node_id = clean_text(item.get("source_node_id"))
    action_type = clean_text(item.get("action_type"))
    target_axis = clean_text(item.get("target_boundary_axis"))
    if source_node_id or action_type or target_axis:
        return f"{source_node_id}|{action_type}|{target_axis}"
    return ""


def get_item_key(item: Dict[str, Any]) -> str:
    index = item.get("index", "")
    prompt = item.get("prompt", "")
    frontier_suffix = get_frontier_key_suffix(item)
    if frontier_suffix:
        return f"{index}|||{prompt}|||frontier:{frontier_suffix}"
    return f"{index}|||{prompt}"


def record_lookup_keys(record: Dict[str, Any]) -> List[str]:
    keys = []
    for field in ("sample_id", "index"):
        value = clean_text(record.get(field))
        if value:
            keys.append(value)
    try:
        state = normalize_search_state(record)
        root_id = clean_text(state.get("search_root_id"))
        if root_id:
            keys.append(root_id)
    except Exception:
        root_id = clean_text(record.get("search_root_id"))
        if root_id:
            keys.append(root_id)
    return list(dict.fromkeys(keys))


def expand_items_from_frontier(
    base_items: List[Dict[str, Any]],
    frontier_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    base_by_key: Dict[str, Dict[str, Any]] = {}
    for item in base_items:
        for key in record_lookup_keys(item):
            base_by_key.setdefault(key, item)

    expanded: List[Dict[str, Any]] = []
    for frontier in frontier_items:
        lookup_candidates = [
            clean_text(frontier.get("sample_id")),
            clean_text(frontier.get("index")),
            clean_text(frontier.get("search_root_id")),
        ]
        base_item = None
        for key in lookup_candidates:
            if key and key in base_by_key:
                base_item = base_by_key[key]
                break
        if base_item is None:
            raise ValueError(
                "frontier row 无法在 --input 中找到匹配的基础记录: "
                f"frontier_node_id={frontier.get('frontier_node_id')}"
            )

        item = dict(base_item)
        item["_frontier_context"] = dict(frontier)

        decision = item.get("tree_search_decision")
        decision = dict(decision) if isinstance(decision, dict) else {}
        if clean_text(frontier.get("action_type")):
            decision["action_type"] = clean_text(frontier.get("action_type"))
            decision["branch_intent"] = clean_text(frontier.get("action_type"))
        if clean_text(frontier.get("source_node_type")):
            decision["source_node_type"] = clean_text(frontier.get("source_node_type"))
        if clean_text(frontier.get("target_boundary_axis")):
            decision["target_boundary_axis"] = clean_text(frontier.get("target_boundary_axis"))
        item["tree_search_decision"] = decision

        route = item.get("operator_route")
        if isinstance(route, dict):
            route = dict(route)
            if clean_text(frontier.get("action_type")):
                route["branch_action"] = clean_text(frontier.get("action_type"))
                route["branch_intent"] = clean_text(frontier.get("action_type"))
            if clean_text(frontier.get("source_node_type")):
                route["source_node_type"] = clean_text(frontier.get("source_node_type"))
            if clean_text(frontier.get("target_boundary_axis")):
                route["target_boundary_axis"] = clean_text(frontier.get("target_boundary_axis"))
                route["boundary_axis"] = clean_text(frontier.get("target_boundary_axis"))
            item["operator_route"] = route

        expanded.append(item)
    return expanded


def load_processed_keys(output_path: str) -> set:
    processed_keys = set()
    if not os.path.exists(output_path):
        return processed_keys

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                processed_keys.add(get_item_key(item))
        logger.info(f"从输出文件加载了 {len(processed_keys)} 条已输出记录")
    except Exception as e:
        logger.warning(f"读取已有输出文件失败: {e}，将从头开始处理")

    return processed_keys


def _coerce_score_rate(value: Any) -> Optional[float]:
    try:
        score_rate = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= score_rate <= 1:
        return score_rate
    return None


def get_score_rate(item: Dict[str, Any]) -> Optional[float]:
    top_level_score_rate = _coerce_score_rate(item.get("score_rate"))
    if top_level_score_rate is not None:
        return top_level_score_rate

    scoring_result = item.get("scoring_result")
    if not isinstance(scoring_result, dict):
        return None

    try:
        awarded = float(scoring_result.get("total_awarded", 0) or 0)
        possible = float(scoring_result.get("total_possible", 0) or 0)
    except Exception:
        return None

    if possible <= 0:
        return None
    return awarded / possible


def _normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def get_evolution_action(item: Dict[str, Any]) -> str:
    return str(item.get("evolution_action", "") or "").strip()


def uses_stage_action_contract(item: Dict[str, Any]) -> bool:
    return bool(get_evolution_action(item))


def action_requires_evolution(item: Dict[str, Any]) -> bool:
    return get_evolution_action(item) in EVOLUTION_REQUIRED_ACTIONS


def get_operator_route(item: Dict[str, Any]) -> Dict[str, Any]:
    route = item.get("operator_route")
    return route if isinstance(route, dict) else {}


def get_evolution_state(item: Dict[str, Any]) -> Dict[str, Any]:
    state = item.get("evolution_state")
    return state if isinstance(state, dict) else {}


def get_sample_profile(item: Dict[str, Any]) -> Dict[str, Any]:
    profile = item.get("sample_profile")
    return profile if isinstance(profile, dict) else {}


def get_overscore_diagnosis(item: Dict[str, Any]) -> Dict[str, Any]:
    diagnosis = item.get("overscore_diagnosis")
    return diagnosis if isinstance(diagnosis, dict) else {}


def get_tree_search_decision(item: Dict[str, Any]) -> Dict[str, Any]:
    decision = item.get("tree_search_decision")
    return decision if isinstance(decision, dict) else {}


def collect_frontier_context(item: Dict[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for container_name in FRONTIER_CONTEXT_KEYS:
        container = item.get(container_name)
        if isinstance(container, dict):
            context.update(container)
    for field in FRONTIER_FIELD_NAMES:
        if field in item:
            context[field] = item[field]
    return context


def infer_source_node_type(action_type: str) -> str:
    if action_type == FORK_FROM_ROOT:
        return "root"
    if action_type == FORK_FROM_PARENT:
        return "parent"
    return "current"


def get_parent_prompt(item: Dict[str, Any]) -> str:
    meta_info = item.get("meta_info")
    if isinstance(meta_info, dict):
        for field in ("parent_prompt", "source_parent_prompt", "prompt_parent"):
            value = clean_text(meta_info.get(field))
            if value:
                return value
    return get_root_prompt(item)


def get_generation_context(item: Dict[str, Any]) -> Dict[str, Any]:
    state = normalize_search_state(item)
    route = get_operator_route(item)
    decision = get_tree_search_decision(item)
    frontier = collect_frontier_context(item)

    action_type = (
        clean_text(frontier.get("action_type"))
        or clean_text(route.get("branch_action"))
        or clean_text(decision.get("action_type"))
        or clean_text(route.get("branch_intent"))
        or clean_text(decision.get("branch_intent"))
        or EXPAND_CURRENT_BRANCH
    )
    if action_type not in FRONTIER_ACTION_TYPES:
        action_type = EXPAND_CURRENT_BRANCH

    source_node_type = (
        clean_text(frontier.get("source_node_type"))
        or clean_text(route.get("source_node_type"))
        or clean_text(decision.get("source_node_type"))
        or infer_source_node_type(action_type)
    )
    if source_node_type not in SOURCE_NODE_TYPES:
        source_node_type = infer_source_node_type(action_type)

    source_prompt = clean_text(frontier.get("source_prompt"))
    prompt_source = clean_text(frontier.get("prompt_source"))
    if not source_prompt:
        if source_node_type == "root":
            source_prompt = get_root_prompt(item)
            prompt_source = prompt_source or (
                "meta_info.prompt_old" if source_prompt != clean_text(item.get("prompt")) else "prompt"
            )
        elif source_node_type == "parent":
            source_prompt = get_parent_prompt(item)
            prompt_source = prompt_source or "parent_node"
        else:
            source_prompt = clean_text(item.get("prompt"))
            prompt_source = prompt_source or "prompt"

    if not source_prompt:
        raise ValueError("缺少有效 source prompt，无法执行回溯式候选生成")

    search_root_id = clean_text(frontier.get("search_root_id")) or clean_text(state.get("search_root_id"))
    if source_node_type == "root":
        source_node_id = clean_text(frontier.get("source_node_id")) or search_root_id
        default_next_depth = 1
    elif source_node_type == "parent":
        source_node_id = (
            clean_text(frontier.get("source_node_id"))
            or clean_text(state.get("parent_node_id"))
            or search_root_id
        )
        default_next_depth = max(1, coerce_non_negative_int(state.get("search_depth"), 0))
    else:
        source_node_id = (
            clean_text(frontier.get("source_node_id"))
            or clean_text(state.get("current_node_id"))
            or search_root_id
        )
        default_next_depth = coerce_non_negative_int(state.get("search_depth"), 0) + 1

    explicit_branch_id = bool(clean_text(frontier.get("branch_id")))
    branch_id = clean_text(frontier.get("branch_id")) or clean_text(state.get("branch_id")) or "main"
    target_boundary_axis = (
        clean_text(frontier.get("target_boundary_axis"))
        or clean_text(route.get("target_boundary_axis"))
        or clean_text(route.get("boundary_axis"))
        or clean_text(decision.get("target_boundary_axis"))
        or clean_text(state.get("boundary_axis"))
    )

    search_depth = coerce_non_negative_int(frontier.get("search_depth"), coerce_non_negative_int(state.get("search_depth"), 0))
    next_depth = coerce_non_negative_int(frontier.get("next_depth"), default_next_depth)
    max_search_depth = coerce_non_negative_int(
        frontier.get("max_search_depth"),
        coerce_non_negative_int(state.get("max_search_depth"), int(TREE_SEARCH_CONFIG_DEFAULTS["MAX_SAMPLE_DEPTH"])),
    )
    branch_budget_remaining = coerce_non_negative_int(
        frontier.get("branch_budget_remaining"),
        coerce_non_negative_int(
            state.get("branch_budget_remaining"),
            max(0, max_search_depth - search_depth),
        ),
    )
    sample_budget_remaining = coerce_non_negative_int(
        frontier.get("sample_budget_remaining"),
        coerce_non_negative_int(
            state.get("sample_budget_remaining"),
            int(TREE_SEARCH_CONFIG_DEFAULTS["MAX_SAMPLE_CANDIDATES_TOTAL"]),
        ),
    )
    discovered_boundary_count = coerce_non_negative_int(
        frontier.get("discovered_boundary_count"),
        len(state.get("discovered_boundaries") or []),
    )

    return {
        "sample_id": clean_text(frontier.get("sample_id")) or sample_key(item),
        "search_root_id": search_root_id,
        "frontier_node_id": clean_text(frontier.get("frontier_node_id"))
        or f"frontier_{safe_id_segment(source_node_id)}_{safe_id_segment(action_type)}",
        "source_node_id": source_node_id,
        "source_node_type": source_node_type,
        "source_prompt": source_prompt,
        "prompt_source": prompt_source or "prompt",
        "action_type": action_type,
        "branch_id": branch_id,
        "target_boundary_axis": target_boundary_axis,
        "search_depth": search_depth,
        "next_depth": next_depth,
        "max_search_depth": max_search_depth,
        "branch_budget_remaining": branch_budget_remaining,
        "sample_budget_remaining": sample_budget_remaining,
        "discovered_boundary_count": discovered_boundary_count,
        "explicit_branch_id": explicit_branch_id,
    }


def action_starts_new_branch(action_type: str) -> bool:
    return action_type in {FORK_FROM_ROOT, FORK_FROM_PARENT}


def resolve_candidate_branch_id(
    generation_context: Dict[str, Any],
    *,
    candidate_index: int,
    operator_id: Optional[str],
) -> str:
    branch_id = clean_text(generation_context.get("branch_id")) or "main"
    action_type = clean_text(generation_context.get("action_type"))
    if action_starts_new_branch(action_type) and not generation_context.get("explicit_branch_id"):
        axis = clean_text(generation_context.get("target_boundary_axis")) or operator_id or action_type
        suffix = short_hash(
            f"{generation_context.get('search_root_id')}|{generation_context.get('source_node_id')}|"
            f"{action_type}|{axis}|{candidate_index}|{operator_id}"
        )
        return f"branch_{safe_id_segment(axis, 'axis')}_{suffix}_{candidate_index:02d}"
    return branch_id


def resolve_generated_node_id(
    generation_context: Dict[str, Any],
    *,
    branch_id: str,
    candidate_index: int,
    operator_id: Optional[str],
) -> str:
    root_id = safe_id_segment(generation_context.get("search_root_id"), "sample")
    branch_segment = safe_id_segment(branch_id, "branch")
    depth = coerce_non_negative_int(generation_context.get("next_depth"), 0)
    suffix = short_hash(
        f"{generation_context.get('source_node_id')}|{branch_id}|{depth}|{candidate_index}|{operator_id}"
    )
    return f"{root_id}_{branch_segment}_d{depth}_c{candidate_index}_{suffix}"


def build_generation_metadata(
    generation_context: Dict[str, Any],
    *,
    candidate_index: int,
    operator_id: Optional[str],
) -> Dict[str, Any]:
    branch_id = resolve_candidate_branch_id(
        generation_context,
        candidate_index=candidate_index,
        operator_id=operator_id,
    )
    node_id = resolve_generated_node_id(
        generation_context,
        branch_id=branch_id,
        candidate_index=candidate_index,
        operator_id=operator_id,
    )
    action_type = clean_text(generation_context.get("action_type"))
    is_new_branch = action_starts_new_branch(action_type)
    parent_node_id = clean_text(generation_context.get("source_node_id"))
    return {
        "search_root_id": clean_text(generation_context.get("search_root_id")),
        "frontier_node_id": clean_text(generation_context.get("frontier_node_id")),
        "source_node_id": parent_node_id,
        "source_node_type": clean_text(generation_context.get("source_node_type")),
        "source_prompt": clean_text(generation_context.get("source_prompt")),
        "prompt_source": clean_text(generation_context.get("prompt_source")),
        "parent_node_id": parent_node_id,
        "generated_node_id": node_id,
        "branch_id": branch_id,
        "boundary_axis": clean_text(generation_context.get("target_boundary_axis")) or None,
        "is_new_branch": is_new_branch,
        "operator_used": operator_id,
        "search_depth": coerce_non_negative_int(generation_context.get("next_depth"), 0),
        "generation_action": action_type,
        "branch_budget_remaining": coerce_non_negative_int(generation_context.get("branch_budget_remaining"), 0),
        "sample_budget_remaining": coerce_non_negative_int(generation_context.get("sample_budget_remaining"), 0),
    }


def candidate_budget_cap(item: Dict[str, Any], requested_max: int) -> int:
    if not should_evolve(item, 0):
        return 1
    context = get_generation_context(item)
    caps = [
        requested_max,
        int(TREE_SEARCH_CONFIG_DEFAULTS["MAX_SAMPLE_BRANCHES"]),
        coerce_non_negative_int(context.get("branch_budget_remaining"), requested_max),
        coerce_non_negative_int(context.get("sample_budget_remaining"), requested_max),
    ]
    return max(0, min(caps))


def resolve_operator_id(item: Dict[str, Any]) -> str:
    action = get_evolution_action(item)
    if action not in EVOLUTION_REQUIRED_ACTIONS:
        raise ValueError(f"evolution_action={action or '<missing>'} does not require operator evolution")

    route = get_operator_route(item)
    if not route:
        raise ValueError("缺少 operator_route；请先运行 operator_router.py")

    operator_id = str(route.get("primary_operator") or "").strip()
    if not operator_id:
        raise ValueError("operator_route.primary_operator 不能为空")

    get_operator_spec(operator_id)
    return operator_id


def get_candidate_group_id(item: Dict[str, Any]) -> str:
    frontier_suffix = get_frontier_key_suffix(item)
    suffix = f"::{safe_id_segment(frontier_suffix, 'frontier')}" if frontier_suffix else ""
    for field in ("sample_id", "index"):
        value = item.get(field)
        if value is not None and str(value).strip():
            return f"{str(value).strip()}{suffix}"
    prompt = str(item.get("prompt", "") or "")
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    return f"prompt_{digest}{suffix}"


def resolve_candidate_operator_ids(item: Dict[str, Any], max_candidates: int) -> List[str]:
    if max_candidates < 1:
        raise ValueError("--num-candidates 必须大于等于 1")
    if not uses_stage_action_contract(item):
        return [resolve_operator_id(item)] if should_evolve(item, 0) else []
    if not action_requires_evolution(item):
        return []

    route = get_operator_route(item)
    if not route:
        raise ValueError("缺少 operator_route；请先运行 operator_router.py")

    avoid = {
        str(operator).strip()
        for operator in route.get("avoid_operators", [])
        if isinstance(operator, str) and operator.strip()
    }
    candidates: List[str] = []
    for operator_id in [route.get("primary_operator")] + list(route.get("backup_operators", [])):
        if not isinstance(operator_id, str):
            continue
        operator_id = operator_id.strip()
        if not operator_id or operator_id in avoid or operator_id in candidates:
            continue
        get_operator_spec(operator_id)
        candidates.append(operator_id)
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        raise ValueError("operator_route 未提供可用候选算子")
    return candidates


def make_passthrough_record(item: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(item)
    result["question_evolved"] = False

    meta_info = result.get("meta_info")
    if not isinstance(meta_info, dict):
        meta_info = {}
    else:
        meta_info = dict(meta_info)

    existing_metadata = meta_info.get("question_evolution_metadata")
    if isinstance(existing_metadata, dict):
        metadata = dict(existing_metadata)
    else:
        metadata = {}
    metadata["question_evolved"] = False

    score_rate = get_score_rate(item)
    if score_rate is not None:
        metadata.setdefault("trigger_score_rate", score_rate)

    meta_info["question_evolution_metadata"] = metadata
    result["meta_info"] = meta_info
    return result


def make_passthrough_candidate_record(item: Dict[str, Any], requested_candidates: int) -> Dict[str, Any]:
    result = make_passthrough_record(item)
    group_id = get_candidate_group_id(item)
    result["candidate_group_id"] = group_id
    result["candidate_id"] = f"{group_id}::pass_through"
    result["candidate_generation"] = {
        "candidate_index": 0,
        "num_candidates_requested": requested_candidates,
        "operator_id": None,
        "operator_source": "pass_through",
    }
    return result


def should_evolve(item: Dict[str, Any], min_score_rate: float) -> bool:
    if uses_stage_action_contract(item):
        return action_requires_evolution(item)
    score_rate = get_score_rate(item)
    if score_rate is None:
        return False
    return score_rate >= min_score_rate


def get_reference_answer(item: Dict[str, Any]) -> str:
    meta_info = item.get("meta_info")
    if isinstance(meta_info, dict):
        references = meta_info.get("references")
        if isinstance(references, list) and references and isinstance(references[0], str) and references[0].strip():
            return references[0].strip()

        answers_list = meta_info.get("answers_list")
        if isinstance(answers_list, list) and answers_list and isinstance(answers_list[0], str) and answers_list[0].strip():
            return answers_list[0].strip()

    for field in ("reference_answer", "answer_from_book"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError("缺少有效 reference_answer/meta_info.references[0]")


def get_candidate_answer(item: Dict[str, Any]) -> str:
    scoring_result = item.get("scoring_result")
    if isinstance(scoring_result, dict):
        value = scoring_result.get("candidate_answer")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = item.get("candidate_answer")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("缺少有效 scoring_result.candidate_answer")


def build_evolution_prompt(
    item: Dict[str, Any],
    prompt_version: str = "v1",
    operator_id: Optional[str] = None,
    validation_reject_reason: Optional[str] = None,
) -> str:
    generation_context = get_generation_context(item)
    prompt = generation_context["source_prompt"]
    rubric = item.get("rubric")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("缺少有效 source prompt")

    if operator_id:
        evolution_state = dict(get_evolution_state(item))
        evolution_state.update(
            {
                "search_root_id": generation_context["search_root_id"],
                "current_node_id": generation_context["source_node_id"],
                "branch_id": generation_context["branch_id"],
                "boundary_axis": generation_context["target_boundary_axis"],
                "search_depth": generation_context["search_depth"],
                "next_search_depth": generation_context["next_depth"],
                "generation_action": generation_context["action_type"],
                "source_node_type": generation_context["source_node_type"],
            }
        )
        user_prompt = build_operator_prompt(
            operator_id,
            prompt=prompt.strip(),
            reference_answer=get_reference_answer(item),
            candidate_answer=get_candidate_answer(item),
            rubric=rubric if isinstance(rubric, list) else [],
            sample_profile=get_sample_profile(item),
            overscore_diagnosis=get_overscore_diagnosis(item),
            evolution_state=evolution_state,
            operator_route=get_operator_route(item),
        )
        return append_validation_retry_instruction(user_prompt, validation_reject_reason)

    replacements = {
        "{|prompt|}": prompt.strip(),
        "{|rubrics|}": json.dumps(rubric if isinstance(rubric, list) else [], ensure_ascii=False, indent=2),
        "{|response1|}": get_reference_answer(item),
        "{|response2|}": get_candidate_answer(item),
    }
    user_prompt = get_evolution_prompt_template(prompt_version)
    for placeholder, value in replacements.items():
        user_prompt = user_prompt.replace(placeholder, value)
    return append_validation_retry_instruction(user_prompt, validation_reject_reason)


def build_validation_probe_record(item: Dict[str, Any], evolved: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(item)
    generation_context = get_generation_context(item)
    original_prompt = generation_context["source_prompt"]
    result["prompt"] = str(evolved.get("evolved_prompt", "") or "").strip()
    result["question_evolved"] = True

    meta_info = result.get("meta_info")
    if not isinstance(meta_info, dict):
        meta_info = {}
    else:
        meta_info = dict(meta_info)
    meta_info.setdefault("prompt_old", original_prompt)

    metadata = meta_info.get("question_evolution_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata["question_evolved"] = True

    complexity_budget = evolved.get("complexity_budget")
    if isinstance(complexity_budget, dict):
        metadata["complexity_budget"] = complexity_budget

    operator_used = evolved.get("operator_used")
    if isinstance(operator_used, str) and operator_used.strip():
        metadata["operator_used"] = operator_used.strip()

    meta_info["question_evolution_metadata"] = metadata
    result["meta_info"] = meta_info
    if isinstance(operator_used, str) and operator_used.strip():
        result["candidate_operator"] = operator_used.strip()
    return result


def validate_evolved_result_against_stage_rules(
    item: Dict[str, Any],
    evolved: Dict[str, Any],
) -> Dict[str, Any]:
    probe = build_validation_probe_record(item, evolved)
    return validation_stage.validate_record(probe)


def enrich_evolution_result_with_operator(
    evolved: Dict[str, Any],
    item: Dict[str, Any],
    operator_id: str,
) -> Dict[str, Any]:
    spec = get_operator_spec(operator_id)
    route = get_operator_route(item)
    diagnosis = get_overscore_diagnosis(item)
    enriched = dict(evolved)
    enriched.setdefault("operator_used", operator_id)
    enriched.setdefault("ability_axis", spec.ability_axis)

    target_failure = str(diagnosis.get("target_failure_mode", "") or "").strip()
    cause = str(diagnosis.get("candidate_overscore_cause", "") or "").strip()
    routing_reason = str(route.get("routing_reason", "") or "").strip()

    if routing_reason:
        enriched.setdefault("boundary_hypothesis", routing_reason)
    if target_failure or cause:
        enriched.setdefault("expected_qwen_failure", target_failure or cause)
    if not _normalize_string_list(
        enriched.get("expected_evaluation_focus", enriched.get("evaluation_focus"))
    ):
        enriched["expected_evaluation_focus"] = list(spec.default_evaluation_focus)
    return enriched


def make_evolved_record(
    item: Dict[str, Any],
    evolved: Dict[str, Any],
    score_rate: Optional[float],
    model: str,
    *,
    generation_context: Optional[Dict[str, Any]] = None,
    candidate_index: int = 1,
    operator_id: Optional[str] = None,
) -> Dict[str, Any]:
    """构造进化后的记录。注意：rubric/score_prompt/scoring_result 对已改变 prompt 的题目已失效，移到 meta_info 中保存。"""
    result = dict(item)
    result.pop("_frontier_context", None)
    generation_context = generation_context or get_generation_context(item)
    generation_metadata = build_generation_metadata(
        generation_context,
        candidate_index=candidate_index,
        operator_id=operator_id or evolved.get("operator_used"),
    )
    original_prompt = generation_context["source_prompt"]
    evolved_prompt = evolved["evolved_prompt"]

    # 保留旧 prompt 与旧评分产物
    meta_info = result.get("meta_info")
    if not isinstance(meta_info, dict):
        meta_info = {}
    meta_info["prompt_old"] = original_prompt
    meta_info["stale_rubric"] = result.pop("rubric", None)
    meta_info["stale_score_prompt"] = result.pop("score_prompt", None)
    meta_info["stale_scoring_result"] = result.pop("scoring_result", None)

    # 写入 question evolution 元数据。expected_evaluation_focus 只保存在这里，
    # 不传入 rubric 生成或评分 prompt。
    metadata = {
        "question_evolved": True,
        "trigger_score_rate": score_rate,
        "question_evolution_model": model,
        "evolution_strategy": evolved.get("evolution_strategy", ""),
        "notes_for_reference": evolved.get("notes_for_reference", ""),
        "question_evolution_raw_response": evolved.get("question_evolution_raw_response", ""),
    }

    expected_evaluation_focus = _normalize_string_list(
        evolved.get("expected_evaluation_focus", evolved.get("evaluation_focus"))
    )
    if expected_evaluation_focus:
        metadata["expected_evaluation_focus"] = expected_evaluation_focus

    for field in (
        "operator_used",
        "ability_axis",
        "target_subclaim",
        "boundary_hypothesis",
        "expected_qwen_failure",
    ):
        value = evolved.get(field)
        if isinstance(value, str) and value.strip():
            metadata[field] = value.strip()

    for field in (
        "search_root_id",
        "frontier_node_id",
        "source_node_id",
        "source_node_type",
        "parent_node_id",
        "generated_node_id",
        "branch_id",
        "boundary_axis",
        "prompt_source",
        "source_prompt",
        "generation_action",
        "search_depth",
        "is_new_branch",
    ):
        value = generation_metadata.get(field)
        if value is not None and value != "":
            metadata[field] = value

    complexity_budget = evolved.get("complexity_budget")
    if isinstance(complexity_budget, dict):
        metadata["complexity_budget"] = complexity_budget

    validation_retry = evolved.get("validation_retry")
    if isinstance(validation_retry, dict):
        metadata["validation_retry"] = validation_retry

    meta_info["question_evolution_metadata"] = metadata

    state = get_evolution_state(result)
    state = dict(state) if isinstance(state, dict) else {}
    state.update(
        {
            "search_root_id": generation_metadata["search_root_id"],
            "current_node_id": generation_metadata["generated_node_id"],
            "parent_node_id": generation_metadata["parent_node_id"],
            "branch_id": generation_metadata["branch_id"],
            "boundary_axis": generation_metadata["boundary_axis"],
            "branch_status": "exploring",
            "search_depth": generation_metadata["search_depth"],
            "max_search_depth": generation_context["max_search_depth"],
            "branch_budget_remaining": generation_context["branch_budget_remaining"],
            "sample_budget_remaining": generation_context["sample_budget_remaining"],
        }
    )

    result["prompt"] = evolved_prompt
    result["meta_info"] = meta_info
    result["evolution_state"] = state
    result["question_evolved"] = True
    return result


def make_evolved_candidate_record(
    item: Dict[str, Any],
    evolved: Dict[str, Any],
    score_rate: Optional[float],
    model: str,
    *,
    candidate_index: int,
    requested_candidates: int,
    operator_id: Optional[str],
) -> Dict[str, Any]:
    generation_context = get_generation_context(item)
    result = make_evolved_record(
        item,
        evolved,
        score_rate,
        model,
        generation_context=generation_context,
        candidate_index=candidate_index,
        operator_id=operator_id,
    )
    group_id = get_candidate_group_id(item)
    metadata = result["meta_info"]["question_evolution_metadata"]
    result["candidate_group_id"] = group_id
    result["candidate_id"] = f"{group_id}::cand_{candidate_index}"
    result["candidate_operator"] = operator_id or evolved.get("operator_used")
    result["candidate_generation"] = {
        "candidate_index": candidate_index,
        "num_candidates_requested": requested_candidates,
        "operator_id": operator_id,
        "operator_source": "primary" if candidate_index == 1 else f"backup_{candidate_index - 1}",
        "search_root_id": metadata.get("search_root_id"),
        "frontier_node_id": metadata.get("frontier_node_id"),
        "source_node_id": metadata.get("source_node_id"),
        "source_node_type": metadata.get("source_node_type"),
        "parent_node_id": metadata.get("parent_node_id"),
        "generated_node_id": metadata.get("generated_node_id"),
        "branch_id": metadata.get("branch_id"),
        "boundary_axis": metadata.get("boundary_axis"),
        "is_new_branch": metadata.get("is_new_branch"),
        "search_depth": metadata.get("search_depth"),
        "generation_action": metadata.get("generation_action"),
        "prompt_source": metadata.get("prompt_source"),
    }
    return result


class QuestionEvolutionProcessor:
    def __init__(
        self,
        client: RotatingAPIClient,
        model: str,
        max_concurrent: int = 20,
        max_retries: int = 3,
        min_score_rate: float = 0.8,
        prompt_version: str = "v1",
        num_candidates: int = 1,
        max_validation_retries: int = DEFAULT_MAX_VALIDATION_RETRIES,
        max_candidate_budget: int = 0,
    ):
        self.client = client
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.write_lock = asyncio.Lock()
        self.max_retries = max_retries
        self.min_score_rate = min_score_rate
        self.prompt_version = prompt_version
        self.num_candidates = num_candidates
        self.max_validation_retries = max(0, max_validation_retries)
        self.max_candidate_budget = max_candidate_budget

    async def evolve_once(
        self,
        item: Dict[str, Any],
        operator_id: Optional[str] = None,
        validation_reject_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if operator_id:
            get_operator_spec(operator_id)
        elif uses_stage_action_contract(item):
            operator_id = resolve_operator_id(item)
        user_prompt = build_evolution_prompt(
            item,
            self.prompt_version,
            operator_id=operator_id,
            validation_reject_reason=validation_reject_reason,
        )
        response = await self.client.chat_completions_create(
            model=self.model,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=MAX_OUTPUT_TOKENS
        )
        content = extract_answer(response)
        original_prompt = str(item.get("prompt", "")).strip()
        evolved = parse_evolution_response(content)
        if operator_id:
            evolved = enrich_evolution_result_with_operator(evolved, item, operator_id)
        evolved["question_evolution_raw_response"] = content
        return evolved

    async def _evolve_once_with_model_retry(
        self,
        item: Dict[str, Any],
        operator_id: Optional[str],
        validation_reject_reason: Optional[str],
    ) -> Dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                return await self.evolve_once(
                    item,
                    operator_id=operator_id,
                    validation_reject_reason=validation_reject_reason,
                )
            except Exception as e:
                logger.warning(
                    f"question 进化失败 (尝试 {attempt + 1}/{self.max_retries + 1}) "
                    f"index={item.get('index')}: {str(e)[:200]}"
                )
                if attempt < self.max_retries:
                    error_text = str(e)
                    if "调用频率" in error_text or "qpm" in error_text.lower() or "0x04030020" in error_text:
                        await asyncio.sleep(30)
                    else:
                        await asyncio.sleep(attempt + 1)
                else:
                    raise
        raise RuntimeError("question 进化重试逻辑异常退出")

    async def evolve_with_retry(self, item: Dict[str, Any], operator_id: Optional[str] = None) -> Dict[str, Any]:
        reject_reason: Optional[str] = None
        first_reject_reason: Optional[str] = None
        for validation_attempt in range(self.max_validation_retries + 1):
            evolved = await self._evolve_once_with_model_retry(
                item,
                operator_id=operator_id,
                validation_reject_reason=reject_reason,
            )
            validation_result = validate_evolved_result_against_stage_rules(item, evolved)
            if validation_result.get("passed") is True:
                if validation_attempt:
                    evolved["validation_retry"] = {
                        "attempts": validation_attempt,
                        "max_validation_retries": self.max_validation_retries,
                        "first_reject_reason": first_reject_reason,
                        "final_reject_reason": None,
                    }
                return evolved

            reject_reason = str(validation_result.get("reject_reason") or "未通过复杂度/可回答性校验").strip()
            first_reject_reason = first_reject_reason or reject_reason
            if validation_attempt < self.max_validation_retries:
                logger.warning(
                    "候选题未通过独立校验，将带 reject_reason 使用同一 operator 重试 "
                    f"({validation_attempt + 1}/{self.max_validation_retries}) "
                    f"index={item.get('index')} reason={reject_reason[:200]}"
                )
                continue

            evolved["validation_retry"] = {
                "attempts": validation_attempt,
                "max_validation_retries": self.max_validation_retries,
                "first_reject_reason": first_reject_reason,
                "final_reject_reason": reject_reason,
            }
            return evolved

    async def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        async with self.semaphore:
            if not should_evolve(item, self.min_score_rate) or candidate_budget_cap(item, 1) < 1:
                # 未触发进化，原样输出，仅加标记
                return make_passthrough_record(item)

            score_rate = get_score_rate(item)
            evolved = await self.evolve_with_retry(item)
            return make_evolved_record(item, evolved, score_rate, self.model)

    async def process_item_candidates(
        self,
        item: Dict[str, Any],
        requested_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        async with self.semaphore:
            candidate_count = requested_candidates or self.num_candidates
            candidate_count = candidate_budget_cap(item, candidate_count)
            if not should_evolve(item, self.min_score_rate) or candidate_count < 1:
                return [make_passthrough_candidate_record(item, candidate_count)]

            score_rate = get_score_rate(item)
            if uses_stage_action_contract(item):
                operator_ids = resolve_candidate_operator_ids(item, candidate_count)
            else:
                operator_ids = [None]

            candidates: List[Dict[str, Any]] = []
            for candidate_index, operator_id in enumerate(operator_ids, start=1):
                evolved = await self.evolve_with_retry(item, operator_id=operator_id)
                candidates.append(
                    make_evolved_candidate_record(
                        item,
                        evolved,
                        score_rate,
                        self.model,
                        candidate_index=candidate_index,
                        requested_candidates=candidate_count,
                        operator_id=operator_id,
                    )
                )
            return candidates

    def recommended_candidate_count(self, item: Dict[str, Any]) -> int:
        if not should_evolve(item, self.min_score_rate):
            return 1

        count = 1
        route = get_operator_route(item)
        state = get_evolution_state(item)
        action = get_evolution_action(item)

        if route.get("is_high_value_sample") is True:
            count = max(count, 2)
        if route.get("should_use_local_tree_search") is True:
            count = max(count, 2)
        if action == RECONSTRUCT_LOW_SCORE_BOUNDARY:
            count = max(count, 2)

        try:
            full_score_count = int(state.get("consecutive_full_score_count", 0) or 0)
        except (TypeError, ValueError):
            full_score_count = 0
        if full_score_count >= 1:
            count = max(count, 2)
        if full_score_count >= 2:
            count = max(count, 3)

        previous_effect = str(state.get("previous_effect_status", "") or "")
        try:
            invalid_count = int(state.get("consecutive_invalid_generation_count", 0) or 0)
        except (TypeError, ValueError):
            invalid_count = 0
        if invalid_count >= 2 or previous_effect in {"invalid_complexity", "no_clear_effect"}:
            count = max(count, 3)

        return max(1, candidate_budget_cap(item, min(self.num_candidates, count)))

    def allocate_candidate_counts(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        target_items = [item for item in items if should_evolve(item, self.min_score_rate)]
        if not target_items:
            return {}

        budget = self.max_candidate_budget
        if budget <= 0:
            budget = len(target_items) * 2
        if budget < len(target_items):
            raise ValueError(
                f"max_candidate_budget={budget} 小于待进化样本数 {len(target_items)}，无法保证每个样本至少 1 个候选"
            )

        counts = {get_item_key(item): 1 for item in target_items}
        remaining = budget - len(target_items)
        ranked_items = sorted(
            target_items,
            key=lambda item: self.recommended_candidate_count(item),
            reverse=True,
        )
        for item in ranked_items:
            key = get_item_key(item)
            desired = self.recommended_candidate_count(item)
            while counts[key] < desired and remaining > 0:
                counts[key] += 1
                remaining -= 1
            if remaining <= 0:
                break
        return counts

    async def process_file(
        self,
        input_path: str,
        output_path: str,
        frontier_input_path: Optional[str] = None,
    ):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        items = load_json_or_jsonl(input_path)
        if frontier_input_path:
            if not os.path.exists(frontier_input_path):
                raise FileNotFoundError(f"frontier 输入文件不存在: {frontier_input_path}")
            frontier_items = load_json_or_jsonl(frontier_input_path)
            items = expand_items_from_frontier(items, frontier_items)
            logger.info(
                f"读取到 {len(frontier_items)} 条 frontier 记录，展开为 {len(items)} 条待生成记录"
            )
        target_items = [item for item in items if should_evolve(item, self.min_score_rate)]
        logger.info(
            f"读取到 {len(items)} 条记录，其中需要 question evolution 的记录 {len(target_items)} 条"
        )

        processed_keys = load_processed_keys(output_path)
        pending_items = [item for item in items if get_item_key(item) not in processed_keys]
        if processed_keys:
            logger.info(f"跳过 {len(items) - len(pending_items)} 条已输出记录")

        if not pending_items:
            logger.info("所有输入记录均已输出，无需继续")
            return

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        failed_path = output_path + ".failed"
        file_mode = "a" if processed_keys else "w"
        candidate_counts = self.allocate_candidate_counts(pending_items) if self.num_candidates > 1 else {}

        async def run_one(item: Dict[str, Any], out_f, fail_f):
            try:
                if self.num_candidates > 1:
                    processed_items = await self.process_item_candidates(
                        item,
                        requested_candidates=candidate_counts.get(get_item_key(item), 1),
                    )
                else:
                    processed_items = [await self.process_item(item)]
                async with self.write_lock:
                    for processed_item in processed_items:
                        out_f.write(json.dumps(processed_item, ensure_ascii=False) + "\n")
                    out_f.flush()
            except Exception as e:
                failed_item = dict(item)
                failed_item["question_evolution_error"] = str(e)
                logger.error(
                    f"question 进化失败 index={item.get('index')} "
                    f"prompt={str(item.get('prompt', ''))[:80]} error={e}"
                )
                async with self.write_lock:
                    fail_f.write(json.dumps(failed_item, ensure_ascii=False) + "\n")
                    fail_f.flush()

        pending_target_count = sum(1 for item in pending_items if should_evolve(item, self.min_score_rate))
        logger.info(
            f"开始输出 {len(pending_items)} 条记录，其中待进化记录 {pending_target_count} 条，"
            f"并发限制 {self.semaphore._value}"
        )
        with open(output_path, file_mode, encoding="utf-8") as out_f, \
             open(failed_path, file_mode, encoding="utf-8") as fail_f:
            tasks = [run_one(item, out_f, fail_f) for item in pending_items]
            try:
                from tqdm.asyncio import tqdm
                await tqdm.gather(*tasks)
            except ImportError:
                await asyncio.gather(*tasks)

        if os.path.exists(failed_path) and os.path.getsize(failed_path) == 0:
            os.remove(failed_path)
        elif os.path.exists(failed_path):
            logger.warning(f"存在失败数据，已保存至: {failed_path}")

        logger.info(f"question 进化/全量输出完成，结果保存至: {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="全量输出 scoring 结果，并为得分率过高的记录生成更难、更具区分度的进化后问题"
    )
    parser.add_argument("--input", type=str, required=True, help="scoring.py 输出的 jsonl/json 文件路径")
    parser.add_argument("--output", type=str, help="输出 jsonl 文件路径，默认在输入文件名后追加 _question_evolved")
    parser.add_argument("--concurrency", type=int, default=20, help="并行处理的题目数量")
    parser.add_argument("--retries", type=int, default=3, help="模型调用失败时的重试次数")
    parser.add_argument(
        "--min-score-rate",
        type=float,
        default=0.8,
        help="触发 question 进化的最低得分率，默认 0.8"
    )
    parser.add_argument("--model", type=str, default=EVOLVE_MODEL, help="question evolution 模型名称")
    parser.add_argument("--base-url", type=str, default=EVOLVE_BASE_URL, help="OpenAI 兼容 base_url")
    parser.add_argument("--api-key", action="append", default=None, help="API key；可多次传入覆盖脚本默认 key")
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT_SECONDS, help="单次请求 timeout 秒数")
    parser.add_argument(
        "--prompt-version",
        default="v1",
        help="question evolution prompt 版本: v1=baseline, v2=反模板化/聚焦"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="每条需进化样本生成的候选题数量；大于 1 时输出候选记录供 validate/select 阶段消费"
    )
    parser.add_argument(
        "--max-candidate-budget",
        type=int,
        default=0,
        help="单轮待进化样本的候选总预算；<=0 时默认为待进化样本数 * 2"
    )
    parser.add_argument(
        "--frontier-input",
        type=str,
        default=None,
        help="可选 active_frontier.jsonl；提供时会按 sample_id/search_root_id 叠加到 --input 基础记录上生成候选"
    )
    parser.add_argument(
        "--validation-retries",
        type=int,
        default=DEFAULT_MAX_VALIDATION_RETRIES,
        help="候选题未通过 validate_evolved_question 规则校验时，使用同一 operator 带 reject_reason 重试的次数"
    )
    args = parser.parse_args()

    if args.min_score_rate < 0 or args.min_score_rate > 1:
        raise ValueError("--min-score-rate 必须在 [0, 1] 之间")
    if args.num_candidates < 1 or args.num_candidates > 4:
        raise ValueError("--num-candidates 必须在 [1, 4] 之间")
    if args.validation_retries < 0 or args.validation_retries > 1:
        raise ValueError("--validation-retries 当前只允许 0 或 1，避免无限修正循环")

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_question_evolved{ext or '.jsonl'}"

    api_keys = parse_api_keys(args.api_key)
    client = RotatingAPIClient(
        base_url=args.base_url or EVOLVE_BASE_URL,
        api_keys=api_keys,
        request_timeout=args.request_timeout
    )

    processor = QuestionEvolutionProcessor(
        client=client,
        model=args.model or EVOLVE_MODEL,
        max_concurrent=args.concurrency,
        max_retries=args.retries,
        min_score_rate=args.min_score_rate,
        prompt_version=args.prompt_version,
        num_candidates=args.num_candidates,
        max_validation_retries=args.validation_retries,
        max_candidate_budget=args.max_candidate_budget,
    )

    try:
        await processor.process_file(args.input, args.output, frontier_input_path=args.frontier_input)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
