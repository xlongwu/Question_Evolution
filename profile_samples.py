import argparse
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from local_api_config import get_config_list, get_config_value
from prompts.profile_prompt import build_profile_prompt
from search_state_contract import DEFAULT_BOUNDARY_AXES, normalize_search_state


DEFAULT_PROFILE_MODEL = (
    os.getenv("PROFILE_MODEL")
    or os.getenv("EVOLVE_MODEL")
    or get_config_value("PROFILE_MODEL", "EVOLVE_MODEL", "QA_MODEL", "GPT_MODEL", default="gpt-5.4")
)
DEFAULT_PROFILE_BASE_URL = (
    os.getenv("PROFILE_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or get_config_value("PROFILE_BASE_URL", "EVOLVE_BASE_URL", "BASE_URL", "OPENAI_BASE_URL", default="")
)
REQUEST_TIMEOUT_SECONDS = 180.0
MAX_OUTPUT_TOKENS = 4096

RISK_LEVELS = {"low", "medium", "high"}
PROFILE_REQUIRED_STRING_FIELDS = (
    "core_capability",
    "claim_level",
    "problem_shape",
    "reasoning_granularity",
    "answer_mode_expected",
)
PROFILE_RISK_FIELDS = (
    "easy_judgment_risk",
    "external_knowledge_risk",
    "complexity_expansion_risk",
)
DIAGNOSIS_REQUIRED_STRING_FIELDS = (
    "candidate_overscore_cause",
    "target_failure_mode",
    "why_high_score_is_suspicious",
)
FORBIDDEN_OPERATOR_KEYS = {
    "operator",
    "operator_used",
    "recommended_operator",
    "primary_operator",
    "backup_operators",
    "operator_route",
    "route_operator",
}
BOUNDARY_AXIS_KEYWORDS = (
    ("反常线索主线切换", ("反常线索", "主线切换", "主线抓偏", "受干扰信息带偏")),
    ("最关键缺口识别", ("漏最小关键事实", "最小关键事实", "最关键缺口", "选错最关键缺口", "独立必要条件")),
    ("题干外补设识别", ("题外补设", "题干外", "隐藏前提", "答案写太满超题", "超题")),
    ("补强项升级判断", ("补强", "线索升级", "抓显眼点漏关键层", "双门槛", "动作层与性质层")),
    ("结论分层", ("层级越推", "层级混淆", "近似项分层", "判据内", "判据外", "漏关键层")),
    ("伪闭环识别", ("泛化罗列", "套话", "事实绑定", "闭环", "格式失分")),
)
PROFILE_CAPABILITY_AXIS_DEFAULTS = {
    "证据链补强": ["最关键缺口识别", "补强项升级判断", "伪闭环识别"],
    "行为模式识别": ["反常线索主线切换", "结论分层", "最关键缺口识别"],
    "边界判断": ["结论分层", "题干外补设识别", "伪闭环识别"],
    "反事实推理": ["补强项升级判断", "最关键缺口识别", "结论分层"],
    "排他性认定": ["最关键缺口识别", "题干外补设识别", "伪闭环识别"],
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_api_keys(raw_value: Optional[str] = None) -> List[str]:
    raw = (
        raw_value
        or os.getenv("PROFILE_API_KEYS")
        or os.getenv("EVOLVE_API_KEYS")
        or os.getenv("OPENAI_API_KEYS")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    if keys:
        return keys
    return get_config_list(
        "PROFILE_API_KEYS",
        "EVOLVE_API_KEYS",
        "GPT_API_KEYS",
        "HIAPI_KEYS_BIG",
        "OPENAI_API_KEYS",
        "OPENAI_API_KEY",
        "API_KEYS",
    )


def extract_answer(resp: Any) -> str:
    choices = getattr(resp, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        return (getattr(message, "content", "") or "").strip()

    if hasattr(resp, "model_dump"):
        payload = resp.model_dump()
        choices = payload.get("choices")
        if choices:
            message = choices[0].get("message", {})
            return (message.get("content", "") or "").strip()

    if isinstance(resp, str):
        payload = resp.strip()
        if payload.startswith("data:"):
            payload = payload[len("data:"):].strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return payload
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        if choices:
            return (choices[0].get("message", {}).get("content", "") or "").strip()
        return payload

    raise TypeError(f"Unsupported or empty response type: {type(resp)}")


class RotatingAPIClient:
    """OpenAI-compatible async client with API-key rotation."""

    def __init__(
        self,
        base_url: str,
        api_keys: List[str],
        request_timeout: float = REQUEST_TIMEOUT_SECONDS,
    ):
        if not api_keys:
            raise ValueError("api_keys cannot be empty. Set PROFILE_API_KEYS or OPENAI_API_KEY.")
        self.base_url = base_url
        self.api_keys = api_keys
        self.request_timeout = request_timeout
        self.current_key_index = 0
        self.client = None
        self._lock = asyncio.Lock()
        self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install openai to run profile_samples.py.") from exc

        kwargs = {
            "api_key": self.api_keys[self.current_key_index],
            "timeout": self.request_timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.client = AsyncOpenAI(**kwargs)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()

    async def switch_to_next_key(self) -> bool:
        async with self._lock:
            if len(self.api_keys) <= 1:
                return False
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            if self.client is not None:
                await self.client.close()
            self._init_client()
            logger.warning("Switched to API key index %s", self.current_key_index)
            return True

    async def chat_completions_create(self, **kwargs):
        if self.client is None:
            self._init_client()
        return await self.client.chat.completions.create(**kwargs)


def collect_json_candidate_texts(response_text: str) -> List[str]:
    stripped = response_text.strip()
    if not stripped:
        return []

    candidates: List[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    code_fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)
    for match in reversed(list(code_fence_pattern.finditer(stripped))):
        add(match.group(1))

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


def find_forbidden_operator_keys(value: Any, path: str = "$") -> List[str]:
    matches: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_OPERATOR_KEYS:
                matches.append(child_path)
            matches.extend(find_forbidden_operator_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(find_forbidden_operator_keys(child, f"{path}[{index}]"))
    return matches


def _as_clean_string(value: Any, default: str = "unknown") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _as_risk(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in RISK_LEVELS else "medium"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "是", "值得", "需要"}
    return False


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, str):
        raw_values = re.split(r"[,，、/|]\s*", value)
    else:
        raw_values = []

    values: List[str] = []
    for item in raw_values:
        text = _as_clean_string(item, default="")
        if text and text not in values:
            values.append(text)
    return values


def normalize_boundary_axis(value: Any) -> Optional[str]:
    text = _as_clean_string(value, default="")
    if not text:
        return None
    if text in DEFAULT_BOUNDARY_AXES:
        return text
    for axis in DEFAULT_BOUNDARY_AXES:
        if axis in text or text in axis:
            return axis
    for axis, keywords in BOUNDARY_AXIS_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return axis
    return None


def _append_unique_axes(axes: List[str], values: Iterable[Any]) -> None:
    for value in values:
        axis = normalize_boundary_axis(value)
        if axis and axis not in axes:
            axes.append(axis)


def infer_boundary_axis_candidates(
    sample_profile: Dict[str, Any],
    overscore_diagnosis: Dict[str, Any],
) -> List[str]:
    axes: List[str] = []
    diagnosis_text = " ".join(
        _as_clean_string(overscore_diagnosis.get(field), default="")
        for field in (
            "candidate_overscore_cause",
            "target_failure_mode",
            "why_high_score_is_suspicious",
        )
    )

    for axis, keywords in BOUNDARY_AXIS_KEYWORDS:
        if any(keyword in diagnosis_text for keyword in keywords):
            axes.append(axis)

    core_capability = _as_clean_string(sample_profile.get("core_capability"), default="")
    _append_unique_axes(axes, PROFILE_CAPABILITY_AXIS_DEFAULTS.get(core_capability, []))
    _append_unique_axes(axes, DEFAULT_BOUNDARY_AXES)
    return axes[:4]


def collect_already_explored_axes(item: Dict[str, Any]) -> List[str]:
    state = normalize_search_state(item)
    axes: List[str] = []
    _append_unique_axes(axes, state.get("already_explored_axes", []))

    discovered = state.get("discovered_boundaries")
    if isinstance(discovered, list):
        for boundary in discovered:
            if isinstance(boundary, dict):
                _append_unique_axes(axes, [boundary.get("boundary_axis")])

    branch_status = str(state.get("branch_status") or "")
    if branch_status in {"boundary_hit", "exhausted", "duplicate", "invalid"}:
        _append_unique_axes(axes, [state.get("boundary_axis")])
    return axes


def compute_next_best_axes(
    boundary_axis_candidates: Iterable[Any],
    already_explored_axes: Iterable[Any],
    *,
    is_worth_evolving: bool = True,
) -> List[str]:
    if not is_worth_evolving:
        return []
    explored = {axis for axis in (normalize_boundary_axis(value) for value in already_explored_axes) if axis}
    next_axes: List[str] = []
    _append_unique_axes(next_axes, boundary_axis_candidates)
    return [axis for axis in next_axes if axis not in explored][:4]


def normalize_profile_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    forbidden = find_forbidden_operator_keys(parsed)
    if forbidden:
        raise ValueError(f"profile response must not recommend operators: {', '.join(forbidden)}")

    raw_profile = parsed.get("sample_profile")
    raw_diagnosis = parsed.get("overscore_diagnosis")
    if not isinstance(raw_profile, dict):
        raise ValueError("profile response missing object field: sample_profile")
    if not isinstance(raw_diagnosis, dict):
        raise ValueError("profile response missing object field: overscore_diagnosis")

    sample_profile = dict(raw_profile)
    for field in PROFILE_REQUIRED_STRING_FIELDS:
        sample_profile[field] = _as_clean_string(sample_profile.get(field))
    for field in PROFILE_RISK_FIELDS:
        sample_profile[field] = _as_risk(sample_profile.get(field))

    overscore_diagnosis = dict(raw_diagnosis)
    overscore_diagnosis["is_worth_evolving"] = _as_bool(
        overscore_diagnosis.get("is_worth_evolving")
    )
    for field in DIAGNOSIS_REQUIRED_STRING_FIELDS:
        overscore_diagnosis[field] = _as_clean_string(overscore_diagnosis.get(field))

    raw_axis_candidates = _as_string_list(sample_profile.get("boundary_axis_candidates"))
    boundary_axis_candidates: List[str] = []
    _append_unique_axes(boundary_axis_candidates, raw_axis_candidates)
    if not boundary_axis_candidates:
        boundary_axis_candidates = infer_boundary_axis_candidates(sample_profile, overscore_diagnosis)
    sample_profile["boundary_axis_candidates"] = boundary_axis_candidates[:4]
    sample_profile["already_explored_axes"] = []
    sample_profile["next_best_axes"] = compute_next_best_axes(
        sample_profile["boundary_axis_candidates"],
        [],
        is_worth_evolving=bool(overscore_diagnosis.get("is_worth_evolving")),
    )

    return {
        "sample_profile": sample_profile,
        "overscore_diagnosis": overscore_diagnosis,
    }


def parse_profile_response(response_text: str) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for candidate in collect_json_candidate_texts(response_text):
        try:
            parsed = loads_json_with_repair(candidate)
            if not isinstance(parsed, dict):
                raise ValueError("profile response must be a JSON object")
            return normalize_profile_payload(parsed)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not parse profile JSON: {last_error}")


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


def coerce_score_rate(value: Any) -> Optional[float]:
    try:
        score_rate = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= score_rate <= 1:
        return score_rate
    return None


def get_score_rate(item: Dict[str, Any]) -> Optional[float]:
    top_level = coerce_score_rate(item.get("score_rate"))
    if top_level is not None:
        return top_level

    scoring_result = item.get("scoring_result")
    if not isinstance(scoring_result, dict):
        return None

    try:
        awarded = float(scoring_result.get("total_awarded", 0) or 0)
        possible = float(scoring_result.get("total_possible", 0) or 0)
    except (TypeError, ValueError):
        return None
    if possible <= 0:
        return None
    return awarded / possible


def get_reference_answer(item: Dict[str, Any]) -> str:
    meta_info = item.get("meta_info")
    if isinstance(meta_info, dict):
        references = meta_info.get("references")
        if isinstance(references, list) and references and isinstance(references[0], str):
            return references[0].strip()
    for field in ("reference_answer", "answer_from_book"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_candidate_answer(item: Dict[str, Any]) -> str:
    scoring_result = item.get("scoring_result")
    if isinstance(scoring_result, dict):
        value = scoring_result.get("candidate_answer")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = item.get("candidate_answer")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def build_scoring_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    scoring_result = item.get("scoring_result")
    if not isinstance(scoring_result, dict):
        return {}
    summary: Dict[str, Any] = {}
    for field in ("total_awarded", "total_possible", "answer_mode", "answer_model"):
        if field in scoring_result:
            summary[field] = scoring_result[field]
    item_scores = scoring_result.get("item_scores")
    if isinstance(item_scores, list):
        summary["item_scores"] = [
            {
                key: score.get(key)
                for key in ("title", "weight", "awarded", "brief_reason")
                if isinstance(score, dict) and key in score
            }
            for score in item_scores[:8]
        ]
    return summary


def build_prompt_for_item(item: Dict[str, Any]) -> str:
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("record missing non-empty prompt")
    return build_profile_prompt(
        prompt=prompt,
        reference_answer=get_reference_answer(item),
        candidate_answer=get_candidate_answer(item),
        score_rate=get_score_rate(item),
        scoring_summary=build_scoring_summary(item),
        metadata=item.get("meta_info") if isinstance(item.get("meta_info"), dict) else {},
    )


def attach_profile_result(
    item: Dict[str, Any],
    profile_result: Dict[str, Any],
    *,
    model: str,
    raw_response: str,
) -> Dict[str, Any]:
    result = dict(item)
    sample_profile = dict(profile_result["sample_profile"])
    already_explored_axes = collect_already_explored_axes(result)
    next_best_axes = compute_next_best_axes(
        sample_profile.get("boundary_axis_candidates", []),
        already_explored_axes,
        is_worth_evolving=bool(profile_result["overscore_diagnosis"].get("is_worth_evolving")),
    )
    sample_profile["already_explored_axes"] = already_explored_axes
    sample_profile["next_best_axes"] = next_best_axes

    state = normalize_search_state(result)
    state["already_explored_axes"] = already_explored_axes
    state["recommended_next_axes"] = next_best_axes

    result["sample_profile"] = sample_profile
    result["overscore_diagnosis"] = profile_result["overscore_diagnosis"]
    result["evolution_state"] = state
    result["profile_metadata"] = {
        "profile_model": model,
        "profile_raw_response": raw_response,
    }
    return result


class ProfileProcessor:
    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_PROFILE_MODEL,
        max_concurrent: int = 5,
    ):
        self.client = client
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def profile_once(self, item: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = build_prompt_for_item(item)
        response = await self.client.chat_completions_create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You diagnose evaluation samples and return strict JSON only.",
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        response_text = extract_answer(response)
        profile_result = parse_profile_response(response_text)
        return attach_profile_result(
            item,
            profile_result,
            model=self.model,
            raw_response=response_text,
        )

    async def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        async with self.semaphore:
            return await self.profile_once(item)

    async def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = [self.process_item(item) for item in records]
        return await asyncio.gather(*tasks)

    async def process_file(self, input_path: str, output_path: str) -> None:
        records = load_json_or_jsonl(input_path)
        logger.info("Loaded %s records from %s", len(records), input_path)
        profiled = await self.process_records(records)
        write_jsonl(profiled, output_path)
        logger.info("Wrote profiled records to %s", output_path)


async def async_main(args: argparse.Namespace) -> None:
    api_keys = parse_api_keys(args.api_keys)
    client = RotatingAPIClient(
        base_url=args.base_url or DEFAULT_PROFILE_BASE_URL,
        api_keys=api_keys,
        request_timeout=args.request_timeout,
    )
    try:
        processor = ProfileProcessor(
            client=client,
            model=args.model or DEFAULT_PROFILE_MODEL,
            max_concurrent=args.concurrency,
        )
        await processor.process_file(args.input, args.output)
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample profiles for question evolution.")
    parser.add_argument("--input", required=True, help="Input scored JSON/JSONL path.")
    parser.add_argument("--output", required=True, help="Output profiled JSONL path.")
    parser.add_argument("--model", default=DEFAULT_PROFILE_MODEL, help="Profile model name.")
    parser.add_argument("--base-url", default=DEFAULT_PROFILE_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-keys", default=None, help="Comma-separated API keys. Defaults to env vars.")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent profile requests.")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
        help="Request timeout in seconds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
