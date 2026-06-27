# scoring.py
import argparse
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


def _env_list(name: str) -> List[str]:
    return [value.strip() for value in os.getenv(name, "").split(",") if value.strip()]


JUDGE_MODEL = os.getenv("JUDGE_MODEL", "hjl_Qwen3.6-27B")  # 作为judge完全够用
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://127.0.0.1:18011/v1")
JUDGE_API_KEYS = (
    _env_list("JUDGE_API_KEYS")
    or _env_list("OPENAI_API_KEYS")
    or _env_list("OPENAI_API_KEY")
    or ["key"]
)


ANSWER_PLACEHOLDER = "<<<待评答案>>"
REQUEST_TIMEOUT_SECONDS = 180.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_answer(resp) -> str:
    choices = getattr(resp, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "")
        return (content or "").strip()

    if hasattr(resp, "model_dump"):
        payload = resp.model_dump()
        choices = payload.get("choices")
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            return (content or "").strip()

    if isinstance(resp, str):
        payload = resp
        if payload.startswith("data:"):
            payload = payload[len("data:"):].strip()
        parsed = json.loads(payload)
        return (parsed["choices"][0]["message"]["content"] or "").strip()

    raise TypeError(f"Unsupported or empty response type: {type(resp)}")


class RotatingAPIClient:
    """
    支持自动切换 API Key 的 OpenAI 客户端包装器。
    当遇到 401 令牌额度用尽错误时，自动切换到下一个 key。
    """
    def __init__(self, base_url: str, api_keys: List[str]):
        if not api_keys:
            raise ValueError("api_keys 不能为空")
        self.base_url = base_url
        self.api_keys = api_keys
        self.current_key_index = 0
        self.client: Optional[AsyncOpenAI] = None
        self._lock = asyncio.Lock()
        self._init_client()

    def _init_client(self):
        current_key = self.api_keys[self.current_key_index]
        self.client = AsyncOpenAI(
            api_key=current_key,
            base_url=self.base_url,
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        logger.info(
            f"使用评分 API Key [{self.current_key_index + 1}/{len(self.api_keys)}]: "
            f"{current_key[:8]}..."
        )

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
                logger.error("所有评分 API Key 额度已用尽")
                return False
            self._init_client()
            return True

    async def chat_completions_create(self, **kwargs):
        max_key_switches = len(self.api_keys)
        for _ in range(max_key_switches):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if self._is_token_exhausted_error(e):
                    logger.warning(f"评分 API Key [{self.current_key_index + 1}] 额度用尽: {str(e)[:100]}")
                    if await self.switch_to_next_key():
                        continue
                    raise Exception("所有评分 API Key 额度已用尽") from e
                raise
        raise Exception("所有评分 API Key 额度已用尽")


class AnswerLLMClient:
    """用于自由配置的待评答案模型。"""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key if api_key else "EMPTY_KEY"
        self.model = model
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=REQUEST_TIMEOUT_SECONDS
        )

    async def generate_answer(self, question: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return extract_answer(response)


def extract_json_from_response(response_text: str) -> str:
    """从模型响应中提取 JSON 对象或代码块。"""
    response_text = response_text.strip()
    try:
        json.loads(response_text)
        return response_text
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text)
    if json_match:
        return json_match.group(1).strip()

    code_match = re.search(r"```\s*([\s\S]+?)\s*```", response_text)
    if code_match:
        return code_match.group(1).strip()

    object_start, object_end = response_text.find("{"), response_text.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        return response_text[object_start:object_end + 1].strip()

    raise ValueError("无法从评分响应中提取有效 JSON")


def loads_json_with_repair(json_str: str) -> Any:
    """解析评分 JSON；仅做保守修复。"""
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


def _rubric_title(rubric_item: Dict[str, Any], index: int) -> str:
    title = rubric_item.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"rubric 第 {index + 1} 条缺少非空 title，无法安全对齐评分结果")
    return title.strip()


def _parse_awarded_score(awarded_raw: Any) -> int:
    try:
        if isinstance(awarded_raw, bool):
            return 0
        if isinstance(awarded_raw, (int, float)):
            return int(round(float(awarded_raw)))
        match = re.search(r"-?\d+(?:\.\d+)?", str(awarded_raw))
        return int(round(float(match.group(0)))) if match else 0
    except Exception:
        return 0


def _validate_rubric_titles(rubric: List[Dict[str, Any]]) -> List[str]:
    titles = [_rubric_title(item, index) for index, item in enumerate(rubric)]
    duplicate_titles = sorted({title for title in titles if titles.count(title) > 1})
    if duplicate_titles:
        raise ValueError(f"rubric 存在重复 title，无法按 title 安全对齐: {duplicate_titles}")
    return titles


def normalize_item_scores(item_scores: Any, rubric: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """按 rubric title 严格校验并清洗 item_scores，计算总分。"""
    if not isinstance(item_scores, list):
        raise ValueError("评分结果中的 item_scores 必须为数组")

    rubric_titles = _validate_rubric_titles(rubric)
    if len(item_scores) != len(rubric):
        raise ValueError(f"评分结果 item_scores 数量为 {len(item_scores)}，rubric 数量为 {len(rubric)}")

    score_by_title: Dict[str, Dict[str, Any]] = {}
    for index, raw_item in enumerate(item_scores):
        if not isinstance(raw_item, dict):
            raise ValueError(f"评分结果 item_scores[{index}] 必须为对象")
        title = raw_item.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"评分结果 item_scores[{index}] 缺少非空 title")
        title = title.strip()
        if title in score_by_title:
            raise ValueError(f"评分结果存在重复 title: {title}")
        score_by_title[title] = raw_item

    expected_titles = set(rubric_titles)
    actual_titles = set(score_by_title)
    missing_titles = [title for title in rubric_titles if title not in actual_titles]
    extra_titles = [title for title in score_by_title if title not in expected_titles]
    if missing_titles or extra_titles:
        raise ValueError(
            "评分结果 title 与 rubric 不一致: "
            f"missing={missing_titles}, extra={extra_titles}"
        )

    normalized_scores = []
    total_awarded = 0

    for index, rubric_item in enumerate(rubric):
        title = rubric_titles[index]
        weight = int(rubric_item.get("weight", 0) or 0)
        raw_item = score_by_title[title]
        awarded_raw = raw_item.get("awarded", 0)
        brief_reason = raw_item.get("brief_reason", "")

        awarded = _parse_awarded_score(awarded_raw)
        if weight < 0:
            # 负分项（扣分项）：awarded 应在 [weight, 0] 之间
            awarded = max(weight, min(0, awarded))
        else:
            # 正分项：awarded 应在 [0, weight] 之间
            awarded = max(0, min(weight, awarded))
        total_awarded += awarded

        if not isinstance(brief_reason, str):
            brief_reason = str(brief_reason)

        normalized_scores.append({
            "title": title,
            "weight": weight,
            "awarded": awarded,
            "brief_reason": brief_reason.strip()
        })

    return normalized_scores, total_awarded


def build_scoring_prompt(score_prompt: str, answer_text: str) -> str:
    if ANSWER_PLACEHOLDER not in score_prompt:
        raise ValueError(f"score_prompt 中缺少占位符 {ANSWER_PLACEHOLDER}")
    return score_prompt.replace(ANSWER_PLACEHOLDER, answer_text)


class ScoringProcessor:
    def __init__(
        self,
        judge_client: RotatingAPIClient,
        judge_model: str,
        answer_mode: str,
        max_concurrent: int = 20,
        max_retries: int = 3,
        answer_client: Optional[AnswerLLMClient] = None,
        answer_model_name: str = ""
    ):
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.answer_mode = answer_mode
        self.answer_client = answer_client
        self.answer_model_name = answer_model_name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.write_lock = asyncio.Lock()
        self.max_retries = max_retries

    def load_processed_keys(self, output_path: str) -> set:
        processed_keys = set()
        if not os.path.exists(output_path):
            return processed_keys

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    key = self.get_item_key(data)
                    if key:
                        processed_keys.add(key)
            logger.info(f"从输出文件加载了 {len(processed_keys)} 条已处理记录")
        except Exception as e:
            logger.warning(f"读取已有输出文件时出错: {e}，将从头开始处理")

        return processed_keys

    def get_item_key(self, item: Dict[str, Any]) -> str:
        prompt = item.get("prompt", "")
        index = item.get("index", "")
        return f"{index}|||{prompt}"

    def get_reference_answer(self, item: Dict[str, Any]) -> str:
        outputs = item.get("meta_info").get("references")
        if isinstance(outputs, list) and outputs:
            answer = outputs[0]
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
        raise ValueError("回测模式要求输入数据包含非空 meta_info.references[0]")

    async def generate_candidate_answer(self, item: Dict[str, Any]) -> str:
        if self.answer_mode == "reference":
            return self.get_reference_answer(item)

        if not self.answer_client:
            raise ValueError("自由 LLM 模式下缺少 answer_client")

        question = item.get("prompt")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("缺少有效 prompt，无法生成待评答案")
        return await self.answer_client.generate_answer(question.strip())

    async def generate_candidate_answer_with_retry(self, item: Dict[str, Any]) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                answer = await self.generate_candidate_answer(item)
                if isinstance(answer, str) and answer.strip():
                    return answer.strip()
                raise ValueError("待评答案为空")
            except Exception as e:
                logger.warning(f"生成待评答案失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {str(e)[:200]}")
                if attempt < self.max_retries:
                    await asyncio.sleep(attempt + 1)
                else:
                    raise
        raise RuntimeError("待评答案重试逻辑异常退出")

    async def score_once(self, score_prompt: str) -> Dict[str, Any]:
        response = await self.judge_client.chat_completions_create(
            model=self.judge_model,
            messages=[
                {"role": "user", "content": score_prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content or ""
        json_str = extract_json_from_response(content)
        parsed = loads_json_with_repair(json_str)
        if not isinstance(parsed, dict):
            raise ValueError("评分结果必须是 JSON 对象")
        parsed["_raw_response"] = content.strip()
        return parsed

    async def score_with_retry(self, score_prompt: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                return await self.score_once(score_prompt)
            except Exception as e:
                logger.warning(f"评分失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {str(e)[:200]}")
                if attempt < self.max_retries:
                    error_text = str(e)
                    if "调用频率" in error_text or "qpm" in error_text.lower() or "0x04030020" in error_text:
                        await asyncio.sleep(30)
                    else:
                        await asyncio.sleep(attempt + 1)
                else:
                    raise
        raise RuntimeError("评分重试逻辑异常退出")

    async def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        async with self.semaphore:
            # question evolution 循环中，未进化样本应完全复用上一轮评分结果，
            # 避免重复答题/重评带来的随机波动污染本轮进化效果。
            if item.get("question_evolved") is False:
                scoring_result = item.get("scoring_result")
                if not isinstance(scoring_result, dict) or not scoring_result:
                    raise ValueError("question_evolved=False 但缺少可复用的 scoring_result")
                logger.info(f"透传未进化样本 index={item.get('index')}，不重新答题/评分")
                return item

            score_prompt_template = item.get("score_prompt")
            rubric = item.get("rubric")

            if not isinstance(score_prompt_template, str) or not score_prompt_template.strip():
                raise ValueError("输入数据缺少非空 score_prompt")
            if not isinstance(rubric, list) or not rubric:
                raise ValueError("输入数据缺少非空 rubric")

            if self.answer_mode == "reference":
                candidate_answer = self.get_reference_answer(item)
            else:
                existing_answer = item.get("scoring_result", {}).get("candidate_answer")
                if isinstance(existing_answer, str) and existing_answer.strip():
                    logger.info(f"读取已有 candidate_answer (index={item.get('index')})")
                    candidate_answer = existing_answer.strip()
                else:
                    candidate_answer = await self.generate_candidate_answer_with_retry(item)

            final_prompt = build_scoring_prompt(score_prompt_template, candidate_answer.strip())
            score_result = await self.score_with_retry(final_prompt)
            normalized_item_scores, total_awarded = normalize_item_scores(
                score_result.get("item_scores", []),
                rubric
            )
            total_possible = sum(max(0, int(criterion.get("weight", 0) or 0)) for criterion in rubric)

            item["scoring_result"] = {
                "answer_mode": self.answer_mode,
                "answer_model": self.answer_model_name if self.answer_mode == "llm" else "meta_info.references[0]",
                "candidate_answer": candidate_answer.strip(),
                "item_scores": normalized_item_scores,
                "overall_comment": str(score_result.get("overall_comment", "")).strip(),
                "total_awarded": total_awarded,
                "total_possible": total_possible,
                "judge_model": self.judge_model,
                "judge_raw_response": score_result.get("_raw_response", "")
            }
            return item

    def _print_scoring_stats(self, results: List[Dict[str, Any]]):
        """自动统计并打印得分率。"""
        if not results:
            return

        stats = []
        total_score = 0
        total_possible = 0

        for item in results:
            idx = item.get("index", "N/A")
            sr = item.get("scoring_result", {})
            item_scores = sr.get("item_scores", [])
            awarded = sr.get("total_awarded", 0)

            # 判断是否含负分项：按正项权重之和作为满分
            has_negative = any(it.get("weight", 0) < 0 for it in item_scores)
            if has_negative:
                possible = sum(it.get("weight", 0) for it in item_scores if it.get("weight", 0) > 0)
            else:
                possible = sr.get("total_possible", 0)

            rate = awarded / possible if possible > 0 else 0
            stats.append({
                "index": idx,
                "awarded": awarded,
                "possible": possible,
                "rate": rate,
                "has_negative": has_negative
            })
            total_score += awarded
            total_possible += possible

        overall_rate = total_score / total_possible if total_possible > 0 else 0

        print("\n" + "=" * 60)
        print("评分统计结果")
        print("=" * 60)
        print(f"总样本数: {len(stats)}")
        print(f"总体平均得分率: {overall_rate:.2%} ({total_score}/{total_possible})")
        print("-" * 60)
        print(f"{'Index':>8s} {'得分':>10s} {'满分':>10s} {'得分率':>10s} {'负分项':>8s}")
        print("-" * 60)
        # 按 index 从小到大排序，最多打印前10个
        sorted_stats = sorted(stats, key=lambda x: x["index"])
        for s in sorted_stats[:10]:
            neg_flag = "是" if s["has_negative"] else "否"
            print(f"{s['index']:>8} {s['awarded']:>10} {s['possible']:>10} {s['rate']:>9.2%} {neg_flag:>8}")
        if len(sorted_stats) > 10:
            print(f"{'...':>8} ({len(sorted_stats) - 10} 条已省略)")
        print("=" * 60 + "\n")

    async def process_file(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        items = []
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                # JSON array format
                items = json.loads(content)
            else:
                # JSONL format
                for line in content.splitlines():
                    if line.strip():
                        items.append(json.loads(line))

        processed_keys = self.load_processed_keys(output_path)
        original_count = len(items)
        items = [item for item in items if self.get_item_key(item) not in processed_keys]
        skipped_count = original_count - len(items)
        if skipped_count > 0:
            logger.info(f"跳过 {skipped_count} 条已处理数据")

        if not items:
            logger.info("所有数据已处理完成，无需继续")
            return

        logger.info(f"开始评分 {len(items)} 条数据，并发限制 {self.semaphore._value}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        failed_path = output_path + ".failed"
        file_mode = "a" if processed_keys else "w"
        results: List[Dict[str, Any]] = []

        async def run_one(item: Dict[str, Any], out_f, fail_f):
            try:
                processed_item = await self.process_item(item)
                async with self.write_lock:
                    out_f.write(json.dumps(processed_item, ensure_ascii=False) + "\n")
                    out_f.flush()
                    results.append(processed_item)
            except Exception as e:
                failed_item = dict(item)
                failed_item["scoring_error"] = str(e)
                logger.error(f"评分失败 index={item.get('index')} prompt={str(item.get('prompt', ''))[:80]} error={e}")
                async with self.write_lock:
                    fail_f.write(json.dumps(failed_item, ensure_ascii=False) + "\n")
                    fail_f.flush()

        with open(output_path, file_mode, encoding="utf-8") as out_f, \
             open(failed_path, file_mode, encoding="utf-8") as fail_f:
            tasks = [run_one(item, out_f, fail_f) for item in items]
            try:
                from tqdm.asyncio import tqdm
                await tqdm.gather(*tasks)
            except ImportError:
                await asyncio.gather(*tasks)

        self._print_scoring_stats(results)

        logger.info(f"评分完成，结果保存至: {output_path}")
        if os.path.exists(failed_path) and os.path.getsize(failed_path) == 0:
            os.remove(failed_path)
        elif os.path.exists(failed_path):
            logger.warning(f"存在失败数据，已保存至: {failed_path}")


async def main():
    parser = argparse.ArgumentParser(description="基于 gen_rubric.py 产出的 score_prompt 对答案进行自动评分")
    parser.add_argument("--input", type=str, required=True, help="gen_rubric.py 输出的 jsonl 文件路径")
    parser.add_argument("--output", type=str, help="输出 jsonl 文件路径，默认在输入文件名后追加 _scored")
    parser.add_argument("--concurrency", type=int, default=50, help="并行处理的题目数量")
    parser.add_argument("--retries", type=int, default=3, help="评分调用失败时的重试次数")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL, help="评分模型名称")
    parser.add_argument(
        "--answer-mode",
        type=str,
        choices=["reference", "llm"],
        default="reference",
        help="待评答案来源：reference=直接使用 meta_info.references[0]；llm=读取已有 candidate_answer 或调用自由配置模型生成答案"
    )
    parser.add_argument("--answer-base-url", type=str, default="", help="待评答案模型的 base_url")
    parser.add_argument("--answer-api-key", type=str, default="", help="待评答案模型的 api_key，可为空字符串")
    parser.add_argument("--answer-model", type=str, default="", help="待评答案模型名称")
    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_scored{ext}"

    answer_client = None
    answer_model_name = ""
    if args.answer_mode == "llm":
        if not args.answer_base_url.strip():
            raise ValueError("自由 LLM 模式下必须提供 --answer-base-url")
        if not args.answer_model.strip():
            raise ValueError("自由 LLM 模式下必须提供 --answer-model")
        answer_client = AnswerLLMClient(
            base_url=args.answer_base_url.strip(),
            api_key=args.answer_api_key,
            model=args.answer_model.strip()
        )
        answer_model_name = args.answer_model.strip()

    judge_client = RotatingAPIClient(
        base_url=JUDGE_BASE_URL,
        api_keys=JUDGE_API_KEYS
    )

    processor = ScoringProcessor(
        judge_client=judge_client,
        judge_model=args.judge_model,
        answer_mode=args.answer_mode,
        max_concurrent=args.concurrency,
        max_retries=args.retries,
        answer_client=answer_client,
        answer_model_name=answer_model_name
    )

    await processor.process_file(args.input, args.output)


if __name__ == "__main__":
    asyncio.run(main())
