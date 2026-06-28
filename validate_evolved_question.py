import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from local_api_config import get_config_list, get_config_value
from prompts.validation_prompt import build_validation_prompt
from schema_validation import validate_records_against_schema


FORMAT_DIFFICULTY_TERMS = (
    "表格",
    "大表格",
    "多层标签",
    "固定句数",
    "复杂编号",
    "编号体系",
    "JSON",
    "yaml",
    "逐项打分",
)
EXTERNAL_KNOWLEDGE_TERMS = (
    "查阅",
    "搜索",
    "检索",
    "外部资料",
    "自行查询",
    "未提供的信息",
    "行业惯例",
    "常识判断",
)
COUNTERFACTUAL_TERMS = ("如果", "假设", "若将", "反事实")
TASK_SPLIT_PATTERN = re.compile(r"[？?]|(?:^|\n)\s*(?:\d+[\.、)]|[一二三四五六七八九十]+[、.])")
OPTION_PATTERN = re.compile(r"(?:^|[\s，。；;（(])(?:[A-Da-d][\.、)]|选项[一二三四A-Da-d]|[甲乙丙丁][：:、])")
VALIDATION_MODEL = (
    os.getenv("VALIDATION_MODEL")
    or os.getenv("EVOLVE_MODEL")
    or get_config_value("VALIDATION_MODEL", "EVOLVE_MODEL", "QA_MODEL", "GPT_MODEL", default="gpt-5.4")
)
VALIDATION_BASE_URL = (
    os.getenv("VALIDATION_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or get_config_value("VALIDATION_BASE_URL", "EVOLVE_BASE_URL", "BASE_URL", "OPENAI_BASE_URL", default="")
)


def load_json_or_jsonl(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON input must be an array")
        return data
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def parse_api_keys(cli_keys: Optional[List[str]] = None) -> List[str]:
    if cli_keys:
        keys = [key.strip() for key in cli_keys if key and key.strip()]
        if keys:
            return keys
    raw = (
        os.getenv("VALIDATION_API_KEYS")
        or os.getenv("EVOLVE_API_KEYS")
        or os.getenv("OPENAI_API_KEYS")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    if keys:
        return keys
    return get_config_list(
        "VALIDATION_API_KEYS",
        "EVOLVE_API_KEYS",
        "GPT_API_KEYS",
        "HIAPI_KEYS_BIG",
        "OPENAI_API_KEYS",
        "OPENAI_API_KEY",
        "API_KEYS",
    )


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "是"}:
            return True
        if text in {"false", "no", "0", "否"}:
            return False
    return None


def _record_key(item: Dict[str, Any]) -> str:
    for field in ("candidate_id", "sample_id", "index"):
        value = item.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return _clean_text(item.get("prompt"))


def _extract_json_object(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()
    candidates = [text]
    code_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if code_match:
        candidates.insert(0, code_match.group(1).strip())
    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match:
        candidates.append(object_match.group(0))
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法解析 LLM validation JSON: {last_error}")


def normalize_llm_validation(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for field in (
        "main_axis_clear",
        "answerable",
        "external_knowledge_required",
        "repeated_pattern_with_previous_round",
        "format_difficulty_dominant",
    ):
        value = _coerce_bool(raw.get(field))
        if value is not None:
            normalized[field] = value
    reject_reason = _clean_text(raw.get("reject_reason"))
    if reject_reason:
        normalized["reject_reason"] = reject_reason
    why = _clean_text(raw.get("why_passed") or raw.get("reason"))
    if why:
        normalized["llm_validation_reason"] = why
    return normalized


def merge_llm_validation_result(
    rule_result: Dict[str, Any],
    llm_validation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized = normalize_llm_validation(llm_validation)
    result = dict(rule_result)
    result["llm_validation_used"] = bool(normalized)
    if not normalized:
        result.setdefault("main_axis_clear", result.get("main_axis_count", 0) <= 1)
        result.setdefault("answerable", result.get("external_knowledge_risk") != "high")
        result.setdefault("external_knowledge_required", result.get("external_knowledge_risk") == "high")
        result.setdefault("repeated_pattern_with_previous_round", result.get("repeat_pattern_risk") == "high")
        result.setdefault("format_difficulty_dominant", result.get("format_difficulty_risk") == "high")
        return result

    reject_reasons = list(result.get("reject_reasons") or [])
    invalid_type = result.get("invalid_type")

    if normalized.get("main_axis_clear") is False:
        reject_reasons.append("LLM 校验认为主轴不清晰")
        invalid_type = invalid_type or "multi_axis"
    if normalized.get("answerable") is False:
        reject_reasons.append("LLM 校验认为题目不可回答")
        invalid_type = invalid_type or "unanswerable"
    if normalized.get("external_knowledge_required") is True:
        reject_reasons.append("LLM 校验认为需要题干外知识")
        invalid_type = invalid_type or "external_knowledge_required"
    if normalized.get("repeated_pattern_with_previous_round") is True:
        reject_reasons.append("LLM 校验认为与上一轮题型重复")
        invalid_type = invalid_type or "repeated_pattern"
    if normalized.get("format_difficulty_dominant") is True:
        reject_reasons.append("LLM 校验认为难度主要来自格式复杂度")
        invalid_type = invalid_type or "format_difficulty_dominant"

    llm_reject_reason = _clean_text(normalized.get("reject_reason"))
    if llm_reject_reason and llm_reject_reason not in reject_reasons:
        reject_reasons.append(llm_reject_reason)

    passed = not reject_reasons
    result.update(normalized)
    result["passed"] = passed
    result["reject_reasons"] = reject_reasons
    result["reject_reason"] = None if passed else "；".join(reject_reasons)
    result["invalid_type"] = None if passed else invalid_type or "llm_validation_failed"
    if passed and normalized.get("llm_validation_reason"):
        result["why_passed"] = normalized["llm_validation_reason"]
    return result


class LLMValidationClient:
    def __init__(self, base_url: str, api_keys: List[str], model: str):
        if not api_keys:
            raise ValueError("启用 LLM validation 时必须提供 VALIDATION_API_KEYS/OPENAI_API_KEY 或 --llm-api-key")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install openai to enable LLM validation.") from exc
        self.model = model
        self.clients = []
        for key in api_keys:
            kwargs = {"api_key": key}
            if base_url:
                kwargs["base_url"] = base_url
            self.clients.append(AsyncOpenAI(**kwargs))
        self.current = 0

    async def close(self) -> None:
        for client in self.clients:
            await client.close()

    async def validate(self, item: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_validation_prompt(
            original_prompt=get_original_prompt(item),
            evolved_prompt=get_current_prompt(item),
            metadata=_metadata(item),
            validation_result=validate_record(item),
        )
        last_error: Optional[Exception] = None
        for offset in range(len(self.clients)):
            idx = (self.current + offset) % len(self.clients)
            try:
                response = await self.clients[idx].chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                content = response.choices[0].message.content
                self.current = idx
                return _extract_json_object(content or "")
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"LLM validation failed: {last_error}")


def _metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    meta_info = item.get("meta_info")
    if not isinstance(meta_info, dict):
        return {}
    metadata = meta_info.get("question_evolution_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _complexity_budget(item: Dict[str, Any]) -> Dict[str, Any]:
    budget = _metadata(item).get("complexity_budget")
    return budget if isinstance(budget, dict) else {}


def get_original_prompt(item: Dict[str, Any]) -> str:
    meta_info = item.get("meta_info")
    if isinstance(meta_info, dict):
        old_prompt = meta_info.get("prompt_old")
        if isinstance(old_prompt, str) and old_prompt.strip():
            return old_prompt.strip()
    return _clean_text(item.get("prompt"))


def get_current_prompt(item: Dict[str, Any]) -> str:
    return _clean_text(item.get("prompt"))


def _coerce_nonnegative_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _budget_int(item: Dict[str, Any], field: str) -> Optional[int]:
    return _coerce_nonnegative_int(_complexity_budget(item).get(field))


def _risk_from_terms(prompt: str, terms: Tuple[str, ...]) -> str:
    return "high" if any(term and term in prompt for term in terms) else "low"


def estimate_output_tasks_count(prompt: str) -> int:
    matches = [match.group(0) for match in TASK_SPLIT_PATTERN.finditer(prompt)]
    question_marks = sum(1 for match in matches if "?" in match or "？" in match)
    enumerated = len(matches) - question_marks
    if question_marks == 0 and enumerated == 0:
        return 1 if prompt.strip() else 0
    return max(question_marks, enumerated, 1)


def estimate_candidate_options_count(prompt: str) -> int:
    option_matches = OPTION_PATTERN.findall(prompt)
    slash_options = re.findall(r"[A-Da-d]\s*/\s*[A-Da-d]", prompt)
    explicit_options = len(set(option_matches))
    if explicit_options:
        return explicit_options
    if slash_options:
        return max(2, len(slash_options) + 1)
    return 0


def estimate_counterfactual_count(prompt: str) -> int:
    return sum(prompt.count(term) for term in COUNTERFACTUAL_TERMS)


def estimate_main_axis_count(item: Dict[str, Any], prompt: str) -> int:
    budget_value = _budget_int(item, "main_axis_count")
    if budget_value is not None:
        return budget_value
    main_axis = _clean_text(_complexity_budget(item).get("main_axis"))
    if main_axis:
        return 1
    task_count = estimate_output_tasks_count(prompt)
    return 1 if task_count <= 2 else task_count


def estimate_new_facts_count(item: Dict[str, Any], prompt: str) -> int:
    budget_value = _budget_int(item, "new_facts_count")
    if budget_value is not None:
        return budget_value
    return len(re.findall(r"(?:新增|补充|假设|如果|若)", prompt))


def _previous_operator(item: Dict[str, Any]) -> str:
    state = item.get("evolution_state")
    if isinstance(state, dict):
        value = _clean_text(state.get("previous_operator"))
        if value:
            return value
    return ""


def _current_operator(item: Dict[str, Any]) -> str:
    for field in ("candidate_operator", "operator_used"):
        value = _clean_text(item.get(field))
        if value:
            return value
    return _clean_text(_metadata(item).get("operator_used"))


def detect_repeat_pattern_risk(item: Dict[str, Any], prompt: str) -> Tuple[str, Optional[str]]:
    previous_operator = _previous_operator(item)
    current_operator = _current_operator(item)
    if previous_operator and current_operator and previous_operator == current_operator:
        return "high", f"与上一轮重复使用 {current_operator}"

    state = item.get("evolution_state")
    if isinstance(state, dict):
        avoid_methods = state.get("avoid_methods")
        if isinstance(avoid_methods, list):
            for method in avoid_methods:
                method_text = _clean_text(method)
                if method_text and method_text in prompt:
                    return "high", f"命中上一轮应避免问法：{method_text}"

    return "low", None


def validate_record(
    item: Dict[str, Any],
    *,
    max_prompt_chars: int = 1200,
    max_output_tasks: int = 2,
    max_candidate_options: int = 3,
    max_counterfactuals: int = 1,
    llm_validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if item.get("question_evolved") is False:
        return merge_llm_validation_result({
            "passed": True,
            "main_axis_count": 0,
            "new_facts_count": 0,
            "output_tasks_count": 0,
            "candidate_options_count": 0,
            "counterfactual_count": 0,
            "estimated_prompt_chars": len(get_current_prompt(item)),
            "external_knowledge_risk": "low",
            "format_difficulty_risk": "low",
            "repeat_pattern_risk": "low",
            "why_passed": "透传样本不做复杂度校验。",
            "reject_reason": None,
            "invalid_type": None,
        }, llm_validation)

    original_prompt = get_original_prompt(item)
    prompt = get_current_prompt(item)
    main_axis_count = estimate_main_axis_count(item, prompt)
    new_facts_count = estimate_new_facts_count(item, prompt)
    output_tasks_count = _budget_int(item, "output_tasks_count")
    if output_tasks_count is None:
        output_tasks_count = estimate_output_tasks_count(prompt)
    candidate_options_count = _budget_int(item, "candidate_options_count")
    if candidate_options_count is None:
        candidate_options_count = estimate_candidate_options_count(prompt)
    counterfactual_count = _budget_int(item, "counterfactual_count")
    if counterfactual_count is None:
        counterfactual_count = estimate_counterfactual_count(prompt)
    prompt_chars = len(prompt)
    external_risk = _risk_from_terms(prompt, EXTERNAL_KNOWLEDGE_TERMS)
    format_risk = _risk_from_terms(prompt, FORMAT_DIFFICULTY_TERMS)
    repeat_risk, repeat_reason = detect_repeat_pattern_risk(item, prompt)

    reject_reasons: List[str] = []
    invalid_type = None
    if not prompt:
        reject_reasons.append("进化题为空")
        invalid_type = "empty_prompt"
    if original_prompt and prompt == original_prompt:
        reject_reasons.append("进化题与原题完全相同")
        invalid_type = invalid_type or "repeated_original"
    if prompt_chars > max_prompt_chars:
        reject_reasons.append(f"题长 {prompt_chars} 超过上限 {max_prompt_chars}")
        invalid_type = invalid_type or "invalid_complexity"
    if main_axis_count > 1:
        reject_reasons.append(f"主轴数 {main_axis_count} 超过 1")
        invalid_type = invalid_type or "multi_axis"
    if output_tasks_count > max_output_tasks:
        reject_reasons.append(f"输出任务数 {output_tasks_count} 超过上限 {max_output_tasks}")
        invalid_type = invalid_type or "too_many_tasks"
    if candidate_options_count > max_candidate_options:
        reject_reasons.append(f"候选项数 {candidate_options_count} 超过上限 {max_candidate_options}")
        invalid_type = invalid_type or "too_many_options"
    if counterfactual_count > max_counterfactuals:
        reject_reasons.append(f"反事实数 {counterfactual_count} 超过上限 {max_counterfactuals}")
        invalid_type = invalid_type or "too_many_counterfactuals"
    if external_risk == "high":
        reject_reasons.append("存在题外知识依赖风险")
        invalid_type = invalid_type or "external_knowledge_required"
    if format_risk == "high":
        reject_reasons.append("存在格式复杂度压分风险")
        invalid_type = invalid_type or "format_difficulty_dominant"
    if repeat_risk == "high":
        reject_reasons.append(repeat_reason or "存在重复题型风险")
        invalid_type = invalid_type or "repeated_pattern"

    passed = not reject_reasons
    rule_result = {
        "passed": passed,
        "main_axis_count": main_axis_count,
        "new_facts_count": new_facts_count,
        "output_tasks_count": output_tasks_count,
        "candidate_options_count": candidate_options_count,
        "counterfactual_count": counterfactual_count,
        "estimated_prompt_chars": prompt_chars,
        "external_knowledge_risk": external_risk,
        "format_difficulty_risk": format_risk,
        "repeat_pattern_risk": repeat_risk,
        "why_passed": "主轴、题长、任务数、候选项、反事实、题外知识和重复题型均在预算内。" if passed else "",
        "reject_reason": None if passed else "；".join(reject_reasons),
        "reject_reasons": reject_reasons,
        "invalid_type": None if passed else invalid_type or "invalid_complexity",
    }
    return merge_llm_validation_result(rule_result, llm_validation)


def attach_validation_result(
    item: Dict[str, Any],
    *,
    max_prompt_chars: int = 1200,
    max_output_tasks: int = 2,
    max_candidate_options: int = 3,
    max_counterfactuals: int = 1,
    llm_validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = dict(item)
    result["validation_result"] = validate_record(
        item,
        max_prompt_chars=max_prompt_chars,
        max_output_tasks=max_output_tasks,
        max_candidate_options=max_candidate_options,
        max_counterfactuals=max_counterfactuals,
        llm_validation=llm_validation,
    )
    return result


def validate_records(
    records: Iterable[Dict[str, Any]],
    *,
    max_prompt_chars: int = 1200,
    max_output_tasks: int = 2,
    max_candidate_options: int = 3,
    max_counterfactuals: int = 1,
    llm_validations: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    return [
        attach_validation_result(
            record,
            max_prompt_chars=max_prompt_chars,
            max_output_tasks=max_output_tasks,
            max_candidate_options=max_candidate_options,
            max_counterfactuals=max_counterfactuals,
            llm_validation=(llm_validations or {}).get(_record_key(record)),
        )
        for record in records
    ]


async def collect_llm_validations(
    records: List[Dict[str, Any]],
    *,
    base_url: str,
    api_keys: List[str],
    model: str,
    concurrency: int = 5,
) -> Dict[str, Dict[str, Any]]:
    client = LLMValidationClient(base_url=base_url, api_keys=api_keys, model=model)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    validations: Dict[str, Dict[str, Any]] = {}

    async def run_one(record: Dict[str, Any]) -> None:
        async with semaphore:
            validations[_record_key(record)] = await client.validate(record)

    try:
        await asyncio.gather(*(run_one(record) for record in records))
    finally:
        await client.close()
    return validations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate evolved question candidates for complexity and answerability.")
    parser.add_argument("--input", required=True, help="Input candidate JSON/JSONL path.")
    parser.add_argument("--output", required=True, help="Output validated JSONL path.")
    parser.add_argument("--max-prompt-chars", type=int, default=1200, help="Maximum evolved prompt characters.")
    parser.add_argument("--max-output-tasks", type=int, default=2, help="Maximum output tasks.")
    parser.add_argument("--max-candidate-options", type=int, default=3, help="Maximum candidate options.")
    parser.add_argument("--max-counterfactuals", type=int, default=1, help="Maximum counterfactual groups.")
    parser.add_argument("--enable-llm-validation", action="store_true", help="Enable optional LLM answerability validation.")
    parser.add_argument("--llm-model", default=VALIDATION_MODEL, help="LLM validation model.")
    parser.add_argument("--llm-base-url", default=VALIDATION_BASE_URL, help="OpenAI-compatible base_url for LLM validation.")
    parser.add_argument("--llm-api-key", action="append", default=None, help="LLM validation API key; can be provided multiple times.")
    parser.add_argument("--llm-concurrency", type=int, default=5, help="LLM validation concurrency.")
    parser.add_argument("--validate-schema", action="store_true", help="Validate input/output records against local JSON schemas.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_prompt_chars <= 0:
        raise ValueError("--max-prompt-chars must be positive")
    records = load_json_or_jsonl(args.input)
    if args.validate_schema:
        schema_errors = validate_records_against_schema(records, Path("schemas") / "pipeline_record.schema.json")
        if schema_errors:
            raise ValueError(f"input schema validation failed: {schema_errors[0]}")
    llm_validations = None
    if args.enable_llm_validation:
        llm_validations = asyncio.run(
            collect_llm_validations(
                records,
                base_url=args.llm_base_url or VALIDATION_BASE_URL,
                api_keys=parse_api_keys(args.llm_api_key),
                model=args.llm_model or VALIDATION_MODEL,
                concurrency=args.llm_concurrency,
            )
        )
    validated = validate_records(
        records,
        max_prompt_chars=args.max_prompt_chars,
        max_output_tasks=args.max_output_tasks,
        max_candidate_options=args.max_candidate_options,
        max_counterfactuals=args.max_counterfactuals,
        llm_validations=llm_validations,
    )
    if args.validate_schema:
        schema_errors = validate_records_against_schema(validated, Path("schemas") / "pipeline_record.schema.json")
        if schema_errors:
            raise ValueError(f"output schema validation failed: {schema_errors[0]}")
    write_jsonl(validated, args.output)


if __name__ == "__main__":
    main()
