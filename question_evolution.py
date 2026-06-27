# question_evolution.py
import argparse
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


def _env_list(name: str) -> List[str]:
    return [value.strip() for value in os.getenv(name, "").split(",") if value.strip()]


# Defaults can be overridden through CLI flags or environment variables.
EVOLVE_MODEL = os.getenv("EVOLVE_MODEL", "gpt-5.4")
EVOLVE_BASE_URL = os.getenv("EVOLVE_BASE_URL", "https://api.openai.com/v1")
EVOLVE_API_KEYS = (
    _env_list("EVOLVE_API_KEYS")
    or _env_list("OPENAI_API_KEYS")
    or _env_list("OPENAI_API_KEY")
)

REQUEST_TIMEOUT_SECONDS = 180.0
MAX_OUTPUT_TOKENS = 32768


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
你是一位负责“能力边界定向挖掘”的评测题目设计专家。你的任务不是把题目机械改难，也不是把题目改成长篇结构题，而是要把原题升级成一道**单主轴、可回答、可稳定评分、并且能让弱模型在一个可归因的错误点上失分**的新题。

# 核心目标
请分析原题、参考答案、候选答案和现有评分标准，生成一道升级后的新题，使其满足：
1. 仍然考查原题的核心领域、核心事实和核心能力，不改变题目主题；
2. 新题的主要失分点尽量集中在 1 个核心错误上；
3. 该错误最好不是“不会写长答案”或“没跟上复杂格式”，而是：
   - 说不准最少到底缺哪一条；
   - 分不清两个都像有用的补充事实谁才更关键；
   - 抓住了显眼动作，但漏掉真正决定定性的那一层；
   - 把“高度像”误当成“已经能写成结论”；
4. 后续 judge 能根据 rubric 稳定判断对错，而不是靠篇幅、术语密度或格式完整性打分。

# 第零步：先拆结论，再决定考哪一层
在内部先做这一步，但不要把中间推理全部输出：
1. 如果原题或候选答案里有一个目标结论，请先把它拆成 1-3 个子判断。
   例如：
   - “实施了强行脱拽式猥亵”可拆成“发生了特定拉扯/脱拽动作”与“该动作已达到猥亵定性所需的性质层”；
   - “两段画面属于同一连续夺枪动作”可拆成“起点动作发生”“中间连续性未断”“终点控制完成”。
2. 本轮只能选择其中 **1 个最值得压测的子判断** 作为主轴，不得同时考多个子判断。
3. 如果不先拆结论就会出现“最小缺口有两个都说得通”的情况，则必须先改题型，不得直接做填空式“还缺哪一条”。

# 第一步：诊断候选答案的虚高主因
你必须先在内部从以下主因中选 1 个最主要的：
- A. 泛泛罗列，缺少本题具体事实绑定；
- B. 只说结论，缺少必要中间推理链；
- C. 混淆层级，把线索/可能/待核查升级成可写结论；
- D. 引入题干外事实、常识、经验或新前提；
- E. 知道“不够”，但说不准哪一条才是最小关键事实；
- F. 结论里有两个门槛时，只抓住更显眼的那个，漏掉真正决定定性的那一个；
- G. 被干扰信息带偏，没有识别真正主轴。

本轮只能围绕 1 个主因设计，不得把多个主因堆在一起。

# 第二步：题型选择与去重复约束
从下列主方式中只选 1 种；如必须组合，只允许“1 个主方式 + 1 个轻量辅助方式”：
1. 候选缺口二选一：给 2 个都像有用的补充事实，要求判断谁才是真正最小关键事实，并说明另一个为什么不够。
2. 子判断定位：先指出当前结论里到底是哪一个子判断还不能成立，再说明最少还缺什么。
3. 单步跳跃识别：要求指出“从哪一步跳到了哪一步”，而不是泛泛说证据不足。
4. 近似项分层：给 2-3 个都“不太能直接写结论”的说法，要求继续分高低，而不是只做可写/不可写二分。
5. 题干外补设识别：要求指出哪一个关键判断偷偷引入了题干外前提。
6. 单变量反事实：只改 1 个条件，要求说明哪一层判断因此变化，哪一层不变。
7. 具体化约束：要求答案必须绑定题干事实，不能退回套话。

去重复要求：
1. 如果同一样本上一轮已经是“最小关键事实/最小前提/最小跳步”题，本轮不得只换壳复用同一问法。
2. 如果上一轮已满分，本轮优先使用：
   - 候选缺口二选一
   - 子判断定位
   - 近似项分层
   而不是继续问“还缺哪一条”。
3. 如果上一轮已经是边界分类题，本轮不得再做同样的三分类换表述，除非原题唯一核心能力本来就是证据层级判断。

# 第三步：复杂度预算
1. 新题必须只有 1 个清晰主轴。读者应能用一句话说清“本题到底考什么”。
2. 新增事实或场景条件最多 3 条。
3. 输出任务最多 2 个；优先 1 个。
4. 候选项最多 3 个；如果是二选一题，优先只给 2 个近似候选。
5. 不得要求大表格、多层标签体系、固定句数、复杂编号系统。
6. 新题长度建议控制在原题的 3-8 倍；若原题很短，原则上不超过 900 个中文字符。
7. 如果上一轮已经很长，本轮优先删约束、收主轴，不要继续加材料。

# 第四步：可回答性与“唯一正确点”自检
生成新题前必须确保：
1. 题干已提供完成任务所需事实，不得要求靠题外专业知识作实体判断；
2. 如果一个问题存在 2 个以上都合理的“最小缺口”，不要做填空式问答，改成候选项区分题；
3. 不得把 rubric 的暗规则藏在题外，题面必须让人知道到底在比较哪一层；
4. 不得设置只能靠猜、靠经验、靠法律评价或靠主观常识才能回答的陷阱；
5. 新题要让弱模型主要在一个主错误上失分，而不是因为任务太多或格式太复杂而广泛失分；
6. 如果删掉题目中的一段要求后，核心考点明显变了，说明本题混入了多个能力点，必须继续简化。

# 禁止的进化方式
1. 禁止把难度主要建立在长篇格式、表格、固定模板或复杂编号上。
2. 禁止把“最小关键事实 / 最小前提 / 最小跳步”三种问法反复换壳当作题型创新。
3. 禁止一次性加入多个阶段、多套反事实、多组排序。
4. 禁止让题目主要考“遵循复杂指令”，而不是考原题对应的专业判断或推理能力。
5. 禁止在结论可拆成多个子判断时，直接问一个含糊的“最少还缺什么”，导致答案与 rubric 只能靠暗约定对齐。

# 输入
## 原题
{|prompt|}

## 参考答案
{|response1|}

## 候选答案（当前得分过高，需要题目升级以区分强弱模型）
{|response2|}

## 现有评分标准
{|rubrics|}

# 输出要求
返回合法 JSON 对象，不要输出 Markdown 标记或额外解释。
必须包含以下字段：

{
  "evolved_prompt": "升级后的新题目。必须完整、可独立作答、聚焦一个主轴，并遵守复杂度预算。",
  "evolution_strategy": "说明本轮选择了哪一个主因、哪一种主方式、为什么这比继续做旧题型更合适。",
  "evaluation_focus": [
    "后续评分时最该检查的错误1",
    "后续评分时最该检查的错误2",
    "后续评分时最该检查的错误3"
  ],
  "complexity_budget": {
    "main_axis": "一句话说明本题核心考什么",
    "chosen_primary_cause": "A/B/C/D/E/F/G 之一",
    "chosen_primary_method": "候选缺口二选一/子判断定位/单步跳跃识别/近似项分层/题干外补设识别/单变量反事实/具体化约束",
    "target_subclaim": "若结论可拆分，这一轮具体考哪一个子判断",
    "new_facts_count": 0,
    "output_tasks_count": 0,
    "candidate_options_count": 0,
    "estimated_prompt_chars": 0,
    "is_repeated_pattern": false,
    "why_within_budget": "说明为什么没有复杂度堆叠，也没有重复上一轮题型"
  },
  "notes_for_reference": "如果参考答案需要补充或调整才能完美回答新题，请简要说明；如果基本适用则写'基本适用'"
}

# 最终质量自检
输出前逐项确认：
1. 新题不是简单加长版；
2. 新题只有 1 个主轴；
3. 如果题目要求回答“最小关键事实”，是否已经先把目标子判断说清；
4. 如果存在两个都说得通的缺口，是否已改成候选项区分，而不是留给 rubric 暗判；
5. 新题不会诱导候选答案靠填满格式得高分；
6. 新题没有要求使用题干未给出的事实；
7. 本题最可能导致弱模型失分的核心原因是否只有 1 个；
8. 本题是否与上一轮同一样本题型明显不同，而不是“最小前提/最小事实/最小跳步”换表述。
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
        current_key = self.api_keys[self.current_key_index]
        self.client = AsyncOpenAI(
            api_key=current_key,
            base_url=self.base_url,
            timeout=self.request_timeout
        )
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


def get_item_key(item: Dict[str, Any]) -> str:
    index = item.get("index", "")
    prompt = item.get("prompt", "")
    return f"{index}|||{prompt}"


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


def get_score_rate(item: Dict[str, Any]) -> Optional[float]:
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


def should_evolve(item: Dict[str, Any], min_score_rate: float) -> bool:
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


def build_evolution_prompt(item: Dict[str, Any], prompt_version: str = "v1") -> str:
    prompt = item.get("prompt")
    rubric = item.get("rubric")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("缺少有效 prompt")

    replacements = {
        "{|prompt|}": prompt.strip(),
        "{|rubrics|}": json.dumps(rubric if isinstance(rubric, list) else [], ensure_ascii=False, indent=2),
        "{|response1|}": get_reference_answer(item),
        "{|response2|}": get_candidate_answer(item),
    }
    user_prompt = get_evolution_prompt_template(prompt_version)
    for placeholder, value in replacements.items():
        user_prompt = user_prompt.replace(placeholder, value)
    return user_prompt


def make_evolved_record(item: Dict[str, Any], evolved: Dict[str, Any], score_rate: float, model: str) -> Dict[str, Any]:
    """构造进化后的记录。注意：rubric/score_prompt/scoring_result 对已改变 prompt 的题目已失效，移到 meta_info 中保存。"""
    result = dict(item)
    original_prompt = str(item.get("prompt", "")).strip()
    evolved_prompt = evolved["evolved_prompt"]

    # 保留旧 prompt 与旧评分产物
    meta_info = result.get("meta_info")
    if not isinstance(meta_info, dict):
        meta_info = {}
    meta_info["prompt_old"] = original_prompt
    meta_info["stale_rubric"] = result.pop("rubric", None)
    meta_info["stale_score_prompt"] = result.pop("score_prompt", None)
    meta_info["stale_scoring_result"] = result.pop("scoring_result", None)

    # 写入 question evolution 元数据
    meta_info["question_evolution_metadata"] = {
        "question_evolved": True,
        "trigger_score_rate": score_rate,
        "question_evolution_model": model,
        "evolution_strategy": evolved.get("evolution_strategy", ""),
        "notes_for_reference": evolved.get("notes_for_reference", ""),
        "question_evolution_raw_response": evolved.get("question_evolution_raw_response", ""),
    }

    result["prompt"] = evolved_prompt
    result["meta_info"] = meta_info
    result["question_evolved"] = True
    return result


class QuestionEvolutionProcessor:
    def __init__(
        self,
        client: RotatingAPIClient,
        model: str,
        max_concurrent: int = 20,
        max_retries: int = 3,
        min_score_rate: float = 0.8,
        prompt_version: str = "v1"
    ):
        self.client = client
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.write_lock = asyncio.Lock()
        self.max_retries = max_retries
        self.min_score_rate = min_score_rate
        self.prompt_version = prompt_version

    async def evolve_once(self, item: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = build_evolution_prompt(item, self.prompt_version)
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
        validate_evolved_question(original_prompt, evolved["evolved_prompt"])
        evolved["question_evolution_raw_response"] = content
        return evolved

    async def evolve_with_retry(self, item: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                return await self.evolve_once(item)
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

    async def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        async with self.semaphore:
            if not should_evolve(item, self.min_score_rate):
                # 未触发进化，原样输出，仅加标记
                result = dict(item)
                result["question_evolved"] = False
                return result

            score_rate = get_score_rate(item)
            evolved = await self.evolve_with_retry(item)
            return make_evolved_record(item, evolved, score_rate, self.model)

    async def process_file(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        items = load_json_or_jsonl(input_path)
        target_items = [item for item in items if should_evolve(item, self.min_score_rate)]
        logger.info(
            f"读取到 {len(items)} 条评分结果，其中得分率 >= {self.min_score_rate:.2%} 的记录 "
            f"{len(target_items)} 条"
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

        async def run_one(item: Dict[str, Any], out_f, fail_f):
            try:
                processed_item = await self.process_item(item)
                async with self.write_lock:
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
    args = parser.parse_args()

    if args.min_score_rate < 0 or args.min_score_rate > 1:
        raise ValueError("--min-score-rate 必须在 [0, 1] 之间")

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_question_evolved{ext or '.jsonl'}"

    api_keys = args.api_key if args.api_key else EVOLVE_API_KEYS
    if not api_keys:
        raise ValueError("Set EVOLVE_API_KEYS or OPENAI_API_KEY, or pass --api-key.")
    client = RotatingAPIClient(
        base_url=args.base_url,
        api_keys=api_keys,
        request_timeout=args.request_timeout
    )

    processor = QuestionEvolutionProcessor(
        client=client,
        model=args.model,
        max_concurrent=args.concurrency,
        max_retries=args.retries,
        min_score_rate=args.min_score_rate,
        prompt_version=args.prompt_version
    )

    try:
        await processor.process_file(args.input, args.output)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
