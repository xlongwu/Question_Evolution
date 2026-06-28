import os
import json
import asyncio
import aiofiles
import logging
import re
import random
from collections import defaultdict
from typing import Any, Dict, List
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
# from bak.rubrichub_prompt import rubric_hub_prompt_cn

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from local_api_config import get_config_list, get_config_value

QA_MODEL = (
    os.getenv("RUBRIC_MODEL")
    or os.getenv("GPT_MODEL")
    or get_config_value("RUBRIC_MODEL", "QA_MODEL", "GPT_MODEL", default="gpt-5.4")
)
BASE_URL = (
    os.getenv("RUBRIC_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or get_config_value("RUBRIC_BASE_URL", "BASE_URL", "OPENAI_BASE_URL", default="")
)


def parse_api_keys(cli_keys: List[str] = None) -> List[str]:
    if cli_keys:
        keys = [key.strip() for key in cli_keys if key and key.strip()]
        if keys:
            return keys
    raw = (
        os.getenv("RUBRIC_API_KEYS")
        or os.getenv("GPT_API_KEYS")
        or os.getenv("OPENAI_API_KEYS")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    if keys:
        return keys
    return get_config_list(
        "RUBRIC_API_KEYS",
        "GPT_API_KEYS",
        "HIAPI_KEYS_BIG",
        "OPENAI_API_KEYS",
        "OPENAI_API_KEY",
        "API_KEYS",
    )


class RotatingAPIClient:
    """
    支持自动切换 API Key 的 OpenAI 客户端包装器
    当遇到 401 令牌额度用尽错误时，自动切换到下一个 key
    """
    def __init__(self, base_url: str, api_keys: List[str]):
        self.base_url = base_url
        self.api_keys = api_keys
        self.current_key_index = 0
        self.client = None
        self._lock = asyncio.Lock()
        self._init_client()
    
    def _init_client(self):
        """使用当前 key 初始化客户端"""
        if self.client:
            asyncio.create_task(self.client.close())
        current_key = self.api_keys[self.current_key_index]
        self.client = AsyncOpenAI(
            api_key=current_key,
            base_url=self.base_url
        )
        logger.info(f"使用 API Key [{self.current_key_index + 1}/{len(self.api_keys)}]: {current_key[:8]}...")
    
    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.close()
    
    def _is_token_exhausted_error(self, error) -> bool:
        """检查错误是否是令牌额度用尽"""
        error_str = str(error)
        return (
            "401" in error_str and 
            ("TokenStatusExhausted" in error_str or "令牌额度已用尽" in error_str)
        )
    
    async def switch_to_next_key(self) -> bool:
        """
        切换到下一个 API Key
        Returns: 是否成功切换（还有剩余 key）
        """
        async with self._lock:
            self.current_key_index += 1
            if self.current_key_index >= len(self.api_keys):
                logger.error("所有 API Key 额度已用尽！")
                return False
            self._init_client()
            return True
    
    async def chat_completions_create(self, **kwargs):
        """
        调用 chat.completions.create，自动处理 key 切换
        """
        request_timeout = getattr(self, "request_timeout", None)
        if request_timeout is not None and "timeout" not in kwargs:
            kwargs["timeout"] = request_timeout

        max_key_switches = len(self.api_keys)
        
        for attempt in range(max_key_switches):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if self._is_token_exhausted_error(e):
                    logger.warning(f"API Key [{self.current_key_index + 1}] 额度用尽: {str(e)[:100]}")
                    if await self.switch_to_next_key():
                        logger.info(f"已切换到下一个 API Key，重试请求...")
                        continue
                    else:
                        raise Exception("所有 API Key 额度已用尽") from e
                else:
                    # 其他错误直接抛出
                    raise
        
        raise Exception("所有 API Key 额度已用尽")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rubric_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

REQUIRED_RUBRIC_FIELDS = ("title", "description", "weight")
RUBRIC_MAX_OUTPUT_TOKENS = 32768


def _collect_json_candidate_texts(response_text: str) -> List[str]:
    """
    提取响应里可能的 JSON 片段，保序去重。

    优先尝试代码块内容，其次尝试去掉代码块后的纯文本。
    """
    text = response_text if isinstance(response_text, str) else str(response_text)
    stripped = text.strip()
    candidates: List[str] = []
    seen = set()

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    if not stripped:
        return candidates

    code_fence_pattern = re.compile(r"```(?:[^\n`]*)\s*([\s\S]+?)\s*```", re.IGNORECASE)
    code_blocks = [match.group(1).strip() for match in code_fence_pattern.finditer(stripped)]
    for block in reversed(code_blocks):
        add(block)

    outside_text = re.sub(r"```[\s\S]+?```", "\n", stripped).strip()
    if outside_text and outside_text != stripped:
        add(outside_text)

    if not candidates:
        add(stripped)

    return candidates

# 评分标准生成提示模板
RUBRIC_PROMPT_TEMPLATE_v3 = """
你是一名面向“大模型训练与评测（GRPO 等组相对策略优化场景）”的高级 Rubric 设计师。
你的任务是围绕[问题]与[参考答案]提炼出可判定、能形成正确训练激励且能拉开组内回答分差的评分维度（包含加分项与扣分项）。
# 输入
- [问题]：
`
{question}
`

- [参考答案]：
`
{reference}
`

# 任务目标
设计的 Rubric 核心目标是从多个维度锚定参考答案，使越接近参考答案（包括内容、格式、逻辑、约束）的回答得分越高，出现幻觉或违背约束的回答得分受惩罚。同时能够区分 “劣质”、“基础”、“优秀”与“卓越”，尤其是怎么区分“优秀”与“卓越”，你应主动寻找参考答案中的精妙之处，将之变成打分点，由此避免“优秀”答案也能获得满分。
# 核心设计原则
1. **多维度强制覆盖**：严禁只罗列内容要点！你必须从以下四个维度拆解参考答案：
   - **核心内容**：关键信息、实体、核心观点、判断结论（区分基础点与深度点，根据重要性赋予不同权重，拉开分差）。
   - **结构与格式**：参考答案的物理形态。如果有列表、多级标题、特定排版、字数特征、关键点数量，必须单独设为打分项。
   - **逻辑与顺序**：参考答案的前后推导关系、步骤的先后顺序。必须单独设为打分项，顺序一致的满分，部分一致得部分分，完全不一致不得分。
   - **惩罚与限制（扣分项）**：严重违背题目要求、语言混乱、编造专业词汇、堆叠重复内容、与题目无关废话、格式严重错误、冗长不易读等。
2. **闭环自包含**：后续的 Grader（打分模型）**看不到**参考答案，因此每条 rubric 必须写清楚具体的匹配词、语义锚点或判断边界。严禁出现“与参考答案一致”这种表述。
3. **边界清晰**：避免“合理即可”、“尽可能多”等模糊词。必须量化，例如“每包含一个以下关键字[A,B,C]得1分，最高3分”。
# 输出要求
只输出符合以下格式的严格 JSON 对象，不要 Markdown 标记符（如 ```json），不要额外解释。
{
  "_thought_process": "在编写 rubric 前的简要思考：1. 参考答案包含哪些核心内容要点？ 2. 参考答案有什么显式的格式、结构、长度或顺序特征？ 3. 假设答题者能力不错，怎么拉开'优秀'与'卓越'的差距？ 4. 哪些常见错误或幻觉必须设立扣分项？",
  "rubric": [
    {
      "title": "评分项标题",
      "description": "具体的打分指南。对于加分项，说明获得不同分数的具体条件（子项累加或分级给分）；对于扣分项，说明触发扣分的具体失败情况。子项分数之和等于 weight；末尾写满分X分。",
      "weight": "对于加分项为 1 到 10 的正整数；对于扣分项为 -1 到 -10 的负整数"
    },
    ...
  ]
}
# 格式约束
- `_thought_process` 必须有，用于强制你多角度分析。
- 每项只有 `title`、`description`、`weight` 三个字段。
- 至少包含 1 个“扣分项”。
- `weight` 必须是整数（扣分项必须是负数）。
- 每条 description 中的子项分数之和等于 `weight`；若写“满分X分”，X 等于 `weight`。
"""

RUBRIC_PROMPT_TEMPLATE_v4 = """
你是一名面向大模型训练与评测的高级 Rubric 设计师。
你的任务是围绕[问题]与[参考答案]提炼出可判定、能形成正确训练激励、并且能稳定拉开“reference / 高质量回答 / 普通回答 / 明显偏离回答”分差的评分维度。

# 输入
- [问题]：
`
{question}
`

- [参考答案]：
`
{reference}
`

# 任务目标
设计一个“参考答案导向、但不过度贴字”的 Rubric。
要求：
1. 必须优先锚定参考答案中真实出现的核心事实、判断、步骤和边界。
2. 必须能区分“答到了核心点”与“补充了少量合理扩展”的差异。
3. 可以鼓励优秀答案更完整，但不要把参考答案里未显式要求的冷门细节、行业黑话、固定话术、排版样式，强行变成硬性得分点。
4. 必须保留负向约束，识别明显幻觉、反向推理、逻辑自相矛盾、答非所问、关键条件遗漏。

# 核心设计原则
1. **先事实，后区分度**
   - 先围绕参考答案的主要事实和判断建立基础分，再补少量真正能拉开差距的深度分。
   - 不要把“参考答案的所有可写细节”都变成评分项。

2. **题目相关性优先**
   - 评分项必须直接服务于题目本身。
   - 不要把模型输出中的修辞、模板化格式、额外说明、交互引导，当作硬评分项，除非题目本身明确要求。

3. **区分度但不过拟合**
   - 可以把“优秀”和“卓越”的差异写进 rubric，但这类差异必须是“同题通用”的，而不是只对这条样本的特定措辞有效。
   - 禁止用只在本题参考答案/候选答案里才成立的特殊词组、极窄场景、专有顺序，去做硬门槛。

4. **边界可解释**
   - 每条 rubric 必须给出清楚的判断边界：什么算命中，什么算未命中。
   - 对于开放式问题，优先评价“是否覆盖核心判断链条”，而不是要求逐字复述。

5. **扣分项克制**
   - 负分项只应针对明显错误、严重偏题、关键事实冲突、逻辑硬伤、明显幻觉。
   - 不要把“少写了某个扩展点”直接升级成重扣分，避免 rubric 过度找茬。

# 输出要求
只输出符合以下格式的严格 JSON 对象，不要 Markdown 标记符，不要额外解释。
{
  "_thought_process": "在编写 rubric 前的简要思考：1. 参考答案的核心结论是什么？ 2. 哪些是必须命中的事实点，哪些只是可选加分点？ 3. 哪些维度能真正拉开高质量答案和普通答案的差距？ 4. 哪些要求若写得太细会导致过拟合？ 5. 哪些明显错误必须设扣分项？",
  "rubric": [
    {
      "title": "评分项标题",
      "description": "具体的打分指南。加分项写清楚基础命中条件与深度加分条件；扣分项写清楚触发条件。尽量使用同题通用、可复用的描述，不要只针对某个候选答案的措辞。子项分数之和等于 weight；末尾写满分X分。",
      "weight": "对于加分项为 1 到 10 的正整数；对于扣分项为 -1 到 -10 的负整数"
    }
  ]
}
"""


async def extract_json_from_response(response_text):
    """
    从LLM响应中提取JSON对象或数组
    支持两种格式：
    1. ```json {...} ``` / ```json [...] ```
    2. ``` {...} ``` / ``` [...] ```
    """
    candidates = _collect_json_candidate_texts(response_text)

    # 依次尝试候选片段，优先返回真正可解析的 JSON 片段。
    for candidate in candidates:
        try:
            loads_json_with_repair(candidate)
            return candidate
        except Exception:
            continue

    preview = (response_text or "").strip().replace("\n", "\\n")[:200]
    raise ValueError(
        f"无法从响应中提取有效的JSON内容（候选 {len(candidates)} 个）；响应前200字: {preview}"
    )


def loads_json_with_repair(json_str: str):
    """解析 LLM 输出的 JSON；仅对常见格式瑕疵做保守修复。"""
    def _escape_control_chars_inside_strings(text: str) -> str:
        """仅在 JSON 字符串内部转义控制字符，避免把有效结构也改坏。"""
        result = []
        in_string = False
        escaped = False
        for ch in text:
            if in_string:
                if escaped:
                    result.append(ch)
                    escaped = False
                    continue
                if ch == "\\":
                    result.append(ch)
                    escaped = True
                    continue
                if ch == '"':
                    result.append(ch)
                    in_string = False
                    continue
                if ch == "\n":
                    result.append("\\n")
                    continue
                if ch == "\r":
                    result.append("\\r")
                    continue
                if ch == "\t":
                    result.append("\\t")
                    continue
            else:
                if ch == '"':
                    in_string = True
            result.append(ch)
        return "".join(result)

    def _raw_decode_first(candidate: str):
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(candidate.lstrip())
        return obj

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        repaired = json_str.strip()
        # 去掉数组/对象闭合前的尾逗号，这是长 JSON 最常见的模型输出瑕疵。
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
        # 处理字符串内部未转义的控制字符（尤其是换行），这是 rubric 长文本里常见的失败原因。
        repaired = _escape_control_chars_inside_strings(repaired)
        # 如果代码块里混入了前后解释，尽量截取最外层数组或对象。
        candidates = [(0, repaired)]
        array_start, array_end = repaired.find("["), repaired.rfind("]")
        if array_start != -1 and array_end != -1 and array_end > array_start:
            candidates.append((array_start, repaired[array_start:array_end + 1]))
        object_start, object_end = repaired.find("{"), repaired.rfind("}")
        if object_start != -1 and object_end != -1 and object_end > object_start:
            candidates.append((object_start, repaired[object_start:object_end + 1]))
        last_error = None
        for _, candidate in sorted(candidates, key=lambda item: item[0]):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as error:
                last_error = error
            try:
                return _raw_decode_first(candidate)
            except Exception as error:
                last_error = error
        raise last_error

def build_user_prompt(question: str, answers: List[str]) -> str:
    """构造 rubric 生成的 user prompt，使用 references 列表的前两项作为 A/B。"""
    cleaned_answers = []
    for answer in answers:
        if not answer:
            continue
        if not isinstance(answer, str):
            answer = str(answer)
        answer = answer.strip()
        if answer and answer not in cleaned_answers:
            cleaned_answers.append(answer)

    if not cleaned_answers:
        raise ValueError("缺少有效参考答案，无法生成 rubric")

    answer_a = cleaned_answers[0]
    answer_b = cleaned_answers[1] if len(cleaned_answers) > 1 else None

    parts = [
        f"问题：{question}",
        "",
        f"参考答案：{answer_a}",
        "",
    ]

    if answer_b:
        parts.extend([
            f"参考答案B：{answer_b}",
            "",
            "补充说明：若A与B存在冲突，严格以参考答案A为准。",
        ])
    else:
        parts.extend([
            "参考答案B：未提供",
            "",
            "补充说明：当前只提供一份参考答案，请严格以参考答案A为准，不要自行补充外部信息。",
        ])

    return "\n".join(parts)


def _append_unique_answer(answers: List[str], answer) -> None:
    """将候选答案清洗后追加到列表，去重且保序。"""
    if answer is None:
        return
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()
    if answer and answer not in answers:
        answers.append(answer)


def collect_reference_answers(item: Dict[str, Any]) -> List[str]:
    """
    提取用于生成 rubric 的参考答案。

    仅支持 `meta_info.references`，不再兼容任何历史字段。
    """
    meta_info = item.get("meta_info", {})
    if not isinstance(meta_info, dict):
        raise ValueError("缺少 meta_info，无法读取 references")

    references = meta_info.get("references")
    if not isinstance(references, list):
        raise ValueError("meta_info.references 必须是数组")

    answers: List[str] = []
    for answer in references:
        _append_unique_answer(answers, answer)

    return answers

def build_score_prompt(record: Dict[str, Any], rubric: List[Dict[str, Any]]) -> str:
    """组装给 judger 的完整打分指令（score_prompt），待评答案处使用占位符。"""
    # v3 支持扣分项（负 weight），满分只累加正分，避免被扣分项拉低
    total_possible = sum(max(item.get("weight", 0), 0) for item in rubric)
    return (
        "你是严格的模型评测打分员。请根据题目、待评答案和完整rubric逐条评分。\n"
        "对每一条rubric item，逐字复制该条目的title，并给出实际得分（awarded）和简要理由（brief_reason）。\n"
        "item_scores的长度必须与rubric条数一致，title必须与rubric逐条完全一致，不要合并、跳过、改写、重命名或新增条目。\n"
        "每个item_scores元素都必须包含且仅需包含 title、awarded、brief_reason；其中title用于程序和rubric做严格匹配。\n"
        "如果某条rubric不得分，也必须输出该条title，并将awarded设为0或对应扣分值，brief_reason说明不得分/扣分依据。\n"
        "不要输出总分 total_awarded；主程序会根据每条得分自动累加。\n"
        "评估是否满足rubric中的实质要求。\n"
        "每条得分必须遵循rubric中的得分标准。\n"
        "输出必须是合法JSON，不要输出markdown或额外解释。\n\n"
        "JSON格式：\n"
        "{\n"
        '  "item_scores": [\n'
        "    {\n"
        '      "title": "逐字复制rubric中的title",\n'
        '      "awarded": 0,\n'
        '      "brief_reason": "简要说明该条得分依据和主要扣分点"\n'
        "    },\n"
        "    ...\n"
        "  ],\n"
        '  "overall_comment": "用2-4句话说明整体主要得分依据"\n'
        "}\n\n"
        f"# 题目\n{record.get('prompt')}\n\n"
        "# 待评答案\n<<<待评答案>>\n\n"
        f"# Rubric（共 {len(rubric)} 条，满分 {total_possible} 分）\n{json.dumps(rubric, ensure_ascii=False, indent=2)}\n\n"
        "# 输出完整性校验要点\n"
        f"- item_scores 必须正好输出 {len(rubric)} 条。\n"
        "- 每条输出的 title 必须从上方 Rubric 中逐字复制。\n"
        "- 不允许只输出 awarded 和 brief_reason；缺少 title 的 JSON 会被判为无效。\n"
        "- 不允许依赖数组顺序省略 title；程序会按 title 与 Rubric 做严格对应。"
    )

def _normalize_text_field(value, field_name: str) -> str:
    """规范化 rubric 文本字段。"""
    if value is None:
        raise ValueError(f"rubric 字段 `{field_name}` 不能为空")

    if not isinstance(value, str):
        value = str(value)

    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        raise ValueError(f"rubric 字段 `{field_name}` 不能为空字符串")
    return normalized

def _normalize_weight(value) -> int:
    """规范化 weight 字段，输出整数（v3 支持负数扣分项）。"""
    if value is None or value == "":
        raise ValueError("rubric 字段 `weight` 不能为空")

    if isinstance(value, bool):
        raise ValueError("rubric 字段 `weight` 不能为布尔值")

    if isinstance(value, (int, float)):
        weight = value
    elif isinstance(value, str):
        weight_str = value.strip()
        if not weight_str:
            raise ValueError("rubric 字段 `weight` 不能为空字符串")
        weight_match = re.search(r"-?\d+(?:\.\d+)?", weight_str)
        if not weight_match:
            raise ValueError(f"rubric 字段 `weight` 无法解析为数字: {value}")
        weight = float(weight_match.group(0))
    else:
        raise ValueError(f"rubric 字段 `weight` 类型非法: {type(value).__name__}")

    # if weight <= 0:
    #     raise ValueError("rubric 字段 `weight` 必须大于 0")

    return int(round(weight))

def validate_and_normalize_rubric(rubric):
    """
    校验并清洗 rubric，确保每一项都包含:
    - title: 非空字符串
    - description: 非空字符串
    - weight: 整数（允许负数扣分项）
    """
    if isinstance(rubric, dict) and "rubric" in rubric:
        rubric = rubric["rubric"]
    if not isinstance(rubric, list):
        raise ValueError("rubric 必须是数组")

    if not rubric:
        raise ValueError("rubric 不能为空数组")

    normalized_rubric = []
    seen_keys = set()

    for index, criterion in enumerate(rubric):
        if not isinstance(criterion, dict):
            raise ValueError(f"rubric 第 {index + 1} 项必须是对象")

        if "weight" not in criterion and isinstance(criterion.get("description"), str):
            max_score_match = re.search(r"满分\s*(-?\d+)\s*分", criterion["description"])
            if max_score_match:
                criterion["weight"] = int(max_score_match.group(1))

        missing_fields = [field for field in REQUIRED_RUBRIC_FIELDS if field not in criterion]
        if missing_fields:
            raise ValueError(f"rubric 第 {index + 1} 项缺少字段: {', '.join(missing_fields)}")

        title = _normalize_text_field(criterion.get("title"), "title")
        description = _normalize_text_field(criterion.get("description"), "description")
        weight = _normalize_weight(criterion.get("weight"))

        dedup_key = (title, description)
        if dedup_key in seen_keys:
            logger.warning(f"rubric 第 {index + 1} 项与前文重复，已跳过: {title}")
            continue
        seen_keys.add(dedup_key)

        normalized_rubric.append({
            "title": title,
            "description": description,
            "weight": weight,
        })

    if not normalized_rubric:
        raise ValueError("rubric 清洗后不能为空数组")

    return normalized_rubric


def validate_rubric_extra(rubric):
    """通用 QA rubric 的最小确定性校验，语义质量主要由 prompt 约束。"""
    external_reference_words = ("参考答案", "答案A", "答案B", "教师答案", "标准答案")
    for index, item in enumerate(rubric, start=1):
        title = item.get("title", "")
        description = item.get("description", "")
        weight = item.get("weight")

        for word in external_reference_words:
            if word in title:
                title = title.replace(word, "题目核心要点")
            if word in description:
                description = description.replace(word, "题目核心要点")
        item["title"] = title
        item["description"] = description

        max_scores = re.findall(r"满分\s*(-?\d+)\s*分", description)
        if max_scores:
            normalized_max_scores = {int(max_score) for max_score in max_scores}
            if len(normalized_max_scores) == 1:
                max_score = normalized_max_scores.pop()
                if max_score != weight:
                    # v3 支持负数扣分项：避免 description 中的正数标记把负分错误修正为正分
                    if weight < 0 and max_score > 0:
                        logger.warning(
                            f"rubric 第 {index} 项为扣分项(weight={weight})但 description 标记为正分({max_score})，保留原负分: {title}"
                        )
                    else:
                        item["weight"] = max_score
            else:
                raise ValueError(
                    f"rubric 第 {index} 项存在多个不一致满分标记({max_scores}): {title}"
                )


def compact_generated_item(item):
    """保持输出记录原样。"""
    return item

def extract_answer(resp) -> str:
    if hasattr(resp, "choices"):
        return (resp.choices[0].message.content or "").strip()

    if isinstance(resp, str):
        payload = resp
        if payload.startswith("data:"):
            payload = payload[len("data:"):].strip()
        parsed = json.loads(payload)
        return (parsed["choices"][0]["message"]["content"] or "").strip()

    raise TypeError(f"Unsupported response type: {type(resp)}")

def get_rubric_prompt_template(version: str):
    version = (version or "").strip().lower()
    if version in {"v3", "baseline", "default", ""}:
        return RUBRIC_PROMPT_TEMPLATE_v3
    if version in {"v4", "experiment", "balanced"}:
        return RUBRIC_PROMPT_TEMPLATE_v4
    raise ValueError(f"不支持的 rubric prompt 版本: {version}")


async def generate_rubric(client: RotatingAPIClient, model, question, answers, max_retries=3, prompt_version="v3"):
    """
    调用LLM生成评分标准
    """
    user_prompt = build_user_prompt(question, answers)
    template = get_rubric_prompt_template(prompt_version)
    user_prompt = template.replace("{question}", question).replace("{reference}", answers[0])
    
    for attempt in range(max_retries):
        try:
            response = await client.chat_completions_create(
                model=model,
                messages=[
                    # {"role": "system", "content": RUBRIC_PROMPT_TEMPLATE_test},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                # 显式提高输出上限，避免长 rubric 在默认配额下被截断成半截 JSON。
                max_tokens=RUBRIC_MAX_OUTPUT_TOKENS,
                # timeout=60.0
            )
            
            
            # content = response.choices[0].message.content
            content = extract_answer(response)
            json_str = await extract_json_from_response(content)
            
            # 验证JSON有效性
            parsed = loads_json_with_repair(json_str)
            
            # 兼容 v3 模板：提取 _thought_process 后，把 rubric 数组单独传给校验
            thought_process = None
            if isinstance(parsed, dict):
                thought_process = parsed.get("_thought_process")
                if "rubric" in parsed:
                    parsed = parsed["rubric"]
            
            rubric = validate_and_normalize_rubric(parsed)
            validate_rubric_extra(rubric)
            
            logger.info(f"成功生成评分标准 (尝试次数: {attempt + 1})")
            return rubric, thought_process
            
        except Exception as e:
            error_str = str(e)
            # 检查是否是令牌用尽错误，如果是，client 已经自动切换了 key，这里只需要重试
            if "所有 API Key 额度已用尽" in error_str:
                logger.error(f"生成评分标准失败: {error_str}")
                raise
            
            logger.warning(f"生成评分标准失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:200]}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
            else:
                logger.error(f"达到最大重试次数，放弃生成: {str(e)}")
                raise

async def process_item(item, client: RotatingAPIClient, model, writer_queue, failed_queue, progress_bar, prompt_version="v3"):
    """
    处理单条数据。只有 rubric 生成、校验、score_prompt 组装全部成功，才写入输出队列；
    任何失败（输入异常、API 失败、格式校验失败等）都进入失败队列，最终写入 .failed 文件。
    """
    try:
        # question_evolution.py 会全量输出；未进化样本的 prompt/reference/rubric
        # 均应沿用上一轮结果，避免 rubric 漂移污染 question evolution 的分数归因。
        if item.get("question_evolved") is False:
            await writer_queue.put(item)
            try:
                logger.debug(f"透传未进化样本 index={item['index']}，不重新生成 rubric")
            except Exception:
                logger.debug("透传未进化样本，不重新生成 rubric")
            return

        # 获取问题
        question = item.get("prompt")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("无效的问题字段")

        # 获取参考答案：仅使用 meta_info.references
        answers = collect_reference_answers(item)
        if not answers:
            raise ValueError("没有有效的参考答案")

        # 生成评分标准
        rubric, thought_process = await generate_rubric(client, model, question, answers, prompt_version=prompt_version)

        # 添加 rubric 字段
        item["rubric"] = rubric

        # 单独保存 v3 模板的思维过程（不影响原 rubric 格式）
        if thought_process is not None:
            item["rubric_thought_process"] = thought_process

        # 组装 score_prompt（待评答案处使用占位符，便于 judger 阶段动态替换）
        item["score_prompt"] = build_score_prompt(item, rubric)

        # 放入成功写入队列
        await writer_queue.put(item)
        try:
            logger.debug(f"完成索引 {item['index']} 的处理")
        except Exception:
            logger.debug(f"完成索引 {item['meta_data']['script_id']} 的处理")

    except Exception as e:
        failed_item = dict(item)
        failed_item["rubric_generation_error"] = str(e)
        await failed_queue.put(failed_item)
        try:
            logger.error(f"处理索引 {item['index']} 时出错，已转入失败文件: {str(e)}")
        except Exception:
            logger.error(f"处理索引 {item['meta_data']['script_id']} 时出错，已转入失败文件: {str(e)}")
    finally:
        progress_bar.update(1)

def deduplicate_by_prompt(items):
    """
    根据prompt字段去重，从相同prompt的条目中随机选择一条

    Args:
        items: 原始数据列表

    Returns:
        去重后的数据列表
    """
    # 按prompt分组
    prompt_groups = defaultdict(list)
    for item in items:
        if 'prompt' in item and item['prompt'] is not None:
            prompt_groups[item['prompt']].append(item)
        else:
            logger.warning(f"发现没有prompt字段或prompt为空的条目: {item}")

    # 从每组中随机选择一条
    deduplicated_items = []
    for prompt, group in prompt_groups.items():
        selected_item = random.choice(group)
        deduplicated_items.append(selected_item)
        if len(group) > 1:
            logger.info(f"Prompt '{prompt[:30]}...' 有 {len(group)} 条重复数据，已随机选择一条")

    logger.info(f"去重前: {len(items)} 条，去重后: {len(deduplicated_items)} 条")
    return deduplicated_items

async def writer_worker(writer_queue, failed_queue, output_file, file_mode='a'):
    """
    异步写入工作者：只有再次通过 rubric 校验的记录才会写入正式输出文件；
    校验失败的记录被转入失败队列，最终写入 .failed 文件。
    """
    async with aiofiles.open(output_file, file_mode, encoding='utf-8') as f:
        while True:
            item = await writer_queue.get()
            if item is None:  # 退出信号
                break

            try:
                rubric = validate_and_normalize_rubric(item.get("rubric"))
                item["rubric"] = rubric
                item = compact_generated_item(item)
                # 序列化并写入
                line = json.dumps(item, ensure_ascii=False) + '\n'
                await f.write(line)
                await f.flush()  # 确保立即写入磁盘
            except Exception as e:
                failed_item = dict(item)
                failed_item["rubric_generation_error"] = f"写入前校验失败: {e}"
                await failed_queue.put(failed_item)
                try:
                    logger.error(f"写入索引 {item['index']} 时出错，已转入失败文件: {str(e)}")
                except Exception:
                    logger.error(f"写入索引 {item['meta_data']['script_id']} 时出错，已转入失败文件: {str(e)}")
            finally:
                writer_queue.task_done()


async def failed_writer_worker(failed_queue, failed_file, file_mode='a'):
    """
    异步失败写入工作者，确保失败记录也能安全落盘。
    """
    async with aiofiles.open(failed_file, file_mode, encoding='utf-8') as f:
        while True:
            item = await failed_queue.get()
            if item is None:  # 退出信号
                break

            try:
                line = json.dumps(item, ensure_ascii=False) + '\n'
                await f.write(line)
                await f.flush()
            except Exception as e:
                logger.error(f"写入失败文件时出错: {str(e)}")
            finally:
                failed_queue.task_done()

async def main(
    input_file,
    output_file,
    concurrency=5,
    model=QA_MODEL,
    request_timeout=None,
    prompt_version="v3",
    base_url=BASE_URL,
    api_keys=None,
):
    """
    主处理函数
    """
    api_keys = list(api_keys) if api_keys is not None else parse_api_keys()
    if not api_keys:
        raise ValueError("缺少 RUBRIC_API_KEYS/GPT_API_KEYS/OPENAI_API_KEY 或 --api-key")
    client = RotatingAPIClient(
        base_url=base_url,
        api_keys=api_keys
    )
    client.request_timeout = request_timeout

    # 读取已存在的输出文件（用于resume功能）
    existing_prompts = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_items = []
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    try:
                        item["rubric"] = validate_and_normalize_rubric(item.get("rubric"))
                        existing_items.append(item)
                    except Exception as e:
                        logger.warning(f"跳过输出文件中的非法 rubric 记录: {str(e)}")
                existing_prompts = {item.get("prompt") for item in existing_items if item.get("prompt") is not None}
            logger.info(f"检测到已存在的输出文件，包含 {len(existing_prompts)} 个已处理的prompt")
        except Exception as e:
            logger.warning(f"读取已存在的输出文件失败: {str(e)}，将重新开始处理")
            existing_prompts = set()
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f if line.strip()]

    logger.info(f"读取到 {len(items)} 条原始数据")

    # 过滤掉已经处理过的条目（基于prompt字段）
    original_count = len(items)
    items = [item for item in items if item.get("prompt") not in existing_prompts]
    logger.info(f"跳过 {original_count - len(items)} 个已处理的条目，剩余 {len(items)} 条需要处理")

    # 去重处理
    items = deduplicate_by_prompt(items)

    # 失败文件路径与打开模式
    failed_path = output_file + ".failed"
    file_mode = 'a' if existing_prompts else 'w'

    logger.info(f"开始处理 {len(items)} 条数据，输出到 {output_file}，失败文件 {failed_path}")

    # 初始化队列和进度条
    writer_queue = asyncio.Queue()
    failed_queue = asyncio.Queue()
    progress_bar = tqdm_asyncio(total=len(items), desc="生成评分标准")

    # 启动写入工作者
    writer_task = asyncio.create_task(writer_worker(writer_queue, failed_queue, output_file, file_mode))
    failed_writer_task = asyncio.create_task(failed_writer_worker(failed_queue, failed_path, file_mode))

    # 信号量控制并发
    semaphore = asyncio.Semaphore(concurrency)

    # 处理任务
    tasks = []
    for item in items:
        async def process_with_semaphore(item):
            async with semaphore:
                await process_item(item, client, model, writer_queue, failed_queue, progress_bar, prompt_version=prompt_version)

        tasks.append(process_with_semaphore(item))

    # 执行所有任务
    await asyncio.gather(*tasks)

    # 通知写入工作者退出
    await writer_queue.join()
    await writer_queue.put(None)
    await writer_task

    await failed_queue.join()
    await failed_queue.put(None)
    await failed_writer_task

    # 关闭客户端
    await client.close()

    progress_bar.close()

    # 清理空失败文件
    if os.path.exists(failed_path) and os.path.getsize(failed_path) == 0:
        os.remove(failed_path)
    elif os.path.exists(failed_path):
        logger.warning(f"存在失败的数据，已保存至: {failed_path}")

    logger.info("所有任务完成")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='生成评分标准')
    parser.add_argument('--input', default="output/police_Q_公安教材-1/policeQA_formatted_公安教材第一批_0126.jsonl", help='输入JSONL文件路径')
    parser.add_argument('--output', default="output/police_Q_公安教材-1/policeQA_Rubric_公安教材第一批_0126.jsonl", help='输出JSONL文件路径')
    parser.add_argument('--concurrency', type=int, default=20, help='并发数量 (默认: 5)')
    parser.add_argument('--model', default=QA_MODEL, help='使用的LLM模型')
    parser.add_argument('--base-url', default=BASE_URL, help='OpenAI-compatible base_url')
    parser.add_argument('--api-key', action='append', default=None, help='API key；可多次传入，默认读取 RUBRIC_API_KEYS/GPT_API_KEYS/OPENAI_API_KEY')
    parser.add_argument('--request-timeout', type=float, default=60.0, help='OpenAI SDK 单次请求 timeout 秒数')
    parser.add_argument('--prompt-version', default='v4', help='rubric prompt 版本: v3=baseline, v4=实验版')
    
    args = parser.parse_args()
    
    # 验证文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        exit(1)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 运行主程序
    try:
        asyncio.run(main(
            input_file=args.input,
            output_file=args.output,
            concurrency=args.concurrency,
            model=args.model or QA_MODEL,
            request_timeout=args.request_timeout,
            prompt_version=args.prompt_version,
            base_url=args.base_url or BASE_URL,
            api_keys=parse_api_keys(args.api_key),
        ))
    except Exception as e:
        logger.critical(f"程序异常终止: {str(e)}", exc_info=True)
        exit(1)
