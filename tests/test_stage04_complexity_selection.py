import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from candidate_selection import select_candidates
from operator_router import route_records
from question_evolution import QuestionEvolutionProcessor
from validate_evolved_question import attach_validation_result


def load_jsonl(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeEvolutionClient:
    def __init__(self):
        self.calls = []

    async def chat_completions_create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if "O2_subclaim_localization" in prompt:
            operator = "O2"
            evolved_prompt = (
                "请只围绕原题结论中的一个子判断作答：现有题干事实已经支持哪一层，"
                "哪一层仍缺少独立必要事实？请先说明不能成立的具体子判断，再说明最少还缺哪一类事实。"
            )
        else:
            operator = "O1"
            evolved_prompt = (
                "请在原题事实基础上比较两个候选补充事实：A 是否直接决定结论成立，B 是否只是旁证。"
                "若只能补充一项，哪一项才是最小关键事实？请说明另一个为什么不能单独支撑结论。"
            )
        content = json.dumps(
            {
                "evolved_prompt": evolved_prompt,
                "evolution_strategy": f"{operator} candidate",
                "complexity_budget": {
                    "main_axis_count": 1,
                    "new_facts_count": 1,
                    "output_tasks_count": 1,
                    "candidate_options_count": 2,
                    "counterfactual_count": 0,
                },
            },
            ensure_ascii=False,
        )
        return FakeResponse(content)


class FakeValidateRetryClient:
    def __init__(self, original_prompt):
        self.original_prompt = original_prompt
        self.calls = []

    async def chat_completions_create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if len(self.calls) == 1:
            evolved_prompt = self.original_prompt
        else:
            assert "reject_reason" in prompt
            assert "完全相同" in prompt
            evolved_prompt = (
                "请比较 A 与 B 两个候选事实：A 是能直接决定结论成立的独立必要条件，"
                "B 只是辅助旁证。若只能补充一项，哪一项才是最小关键事实？"
            )
        return FakeResponse(
            json.dumps(
                {
                    "evolved_prompt": evolved_prompt,
                    "evolution_strategy": "validate retry candidate",
                    "complexity_budget": {
                        "main_axis_count": 1,
                        "new_facts_count": 1,
                        "output_tasks_count": 1,
                        "candidate_options_count": 2,
                        "counterfactual_count": 0,
                    },
                },
                ensure_ascii=False,
            )
        )


def make_candidate(sample_id, candidate_id, prompt, validation=None, *, operator="O1_gap_choice"):
    record = {
        "sample_id": sample_id,
        "candidate_group_id": sample_id,
        "candidate_id": f"{sample_id}::{candidate_id}",
        "candidate_operator": operator,
        "prompt": prompt,
        "question_evolved": True,
        "meta_info": {
            "prompt_old": "原题：判断两个事实中哪一个能支撑结论。",
            "question_evolution_metadata": {
                "question_evolved": True,
                "operator_used": operator,
                "expected_evaluation_focus": ["是否抓住最小关键事实"],
            },
        },
    }
    if validation is not None:
        record["validation_result"] = validation
    return record


def test_question_evolution_can_emit_primary_and_backup_candidates():
    records = load_jsonl(ROOT / "tests" / "fixtures" / "stage03_routing_input.jsonl")
    routed = route_records(records)
    item = {record["sample_id"]: record for record in routed}["stage03-o1"]
    fake_client = FakeEvolutionClient()
    processor = QuestionEvolutionProcessor(
        fake_client,
        model="mock-evolution-model",
        max_concurrent=1,
        max_retries=0,
        num_candidates=2,
    )

    candidates = asyncio.run(processor.process_item_candidates(item))

    assert len(candidates) == 2
    assert [candidate["candidate_operator"] for candidate in candidates] == [
        "O1_gap_choice",
        "O2_subclaim_localization",
    ]
    assert all(candidate["candidate_id"] for candidate in candidates)
    assert len(fake_client.calls) == 2


def test_validate_retry_reuses_operator_and_includes_reject_reason():
    records = load_jsonl(ROOT / "tests" / "fixtures" / "stage03_routing_input.jsonl")
    routed = route_records(records)
    item = {record["sample_id"]: record for record in routed}["stage03-o1"]
    fake_client = FakeValidateRetryClient(item["prompt"])
    processor = QuestionEvolutionProcessor(
        fake_client,
        model="mock-evolution-model",
        max_concurrent=1,
        max_retries=0,
        max_validation_retries=1,
    )

    evolved = asyncio.run(processor.process_item(item))
    metadata = evolved["meta_info"]["question_evolution_metadata"]

    assert len(fake_client.calls) == 2
    assert metadata["operator_used"] == "O1_gap_choice"
    assert metadata["validation_retry"]["attempts"] == 1
    assert "完全相同" in metadata["validation_retry"]["first_reject_reason"]


def test_llm_validation_result_can_reject_unanswerable_candidate():
    candidate = make_candidate(
        "stage04-llm",
        "cand_1",
        "请判断题干没有提供的外部行业惯例是否直接决定结论。",
    )
    validation = attach_validation_result(
        candidate,
        llm_validation={
            "main_axis_clear": True,
            "answerable": False,
            "external_knowledge_required": True,
            "repeated_pattern_with_previous_round": False,
            "format_difficulty_dominant": False,
            "reject_reason": "题干缺少行业惯例事实，无法仅凭题面回答。",
        },
    )["validation_result"]

    assert validation["llm_validation_used"] is True
    assert validation["passed"] is False
    assert validation["answerable"] is False
    assert "无法仅凭题面回答" in validation["reject_reason"]


def test_dynamic_candidate_budget_allocates_more_to_priority_samples():
    def item(sample_id, *, high=False, action="evolve_high_score_overscore", full_count=0):
        return {
            "sample_id": sample_id,
            "prompt": f"{sample_id} prompt",
            "meta_info": {"references": ["参考答案。"]},
            "rubric": [],
            "score_rate": 0.9,
            "scoring_result": {"candidate_answer": "候选答案。", "total_awarded": 9, "total_possible": 10},
            "evolution_action": action,
            "operator_route": {
                "primary_operator": "O1_gap_choice",
                "backup_operators": ["O2_subclaim_localization", "O4_near_level_ranking", "O8_double_threshold_claim"],
                "avoid_operators": [],
                "is_high_value_sample": high,
                "should_use_local_tree_search": False,
            },
            "evolution_state": {"consecutive_full_score_count": full_count},
        }

    items = [
        item("ordinary"),
        item("high", high=True),
        item("low", action="reconstruct_low_score_boundary"),
        item("full", full_count=2),
    ]
    processor = QuestionEvolutionProcessor(
        FakeEvolutionClient(),
        model="mock-evolution-model",
        max_concurrent=1,
        num_candidates=4,
        max_candidate_budget=8,
    )
    counts = processor.allocate_candidate_counts(items)

    assert counts["|||ordinary prompt"] == 1
    assert counts["|||high prompt"] == 2
    assert counts["|||low prompt"] == 2
    assert counts["|||full prompt"] == 3
    assert sum(counts.values()) <= 8


def test_validation_rejects_overlong_duplicate_multitask_and_accepts_clean_candidate():
    clean = make_candidate(
        "stage04-valid",
        "cand_1",
        "请比较 A 与 B 两个候选事实，判断哪一个才是支撑结论的最小关键事实，并说明另一个为什么不足。",
    )
    duplicate = make_candidate("stage04-dup", "cand_1", "原题：判断两个事实中哪一个能支撑结论。")
    overlong = make_candidate("stage04-long", "cand_1", "题目：" + "甲" * 1300)
    multitask = make_candidate(
        "stage04-multi",
        "cand_1",
        "请回答：1. 判断 A。2. 判断 B。3. 列表说明 C。最终再输出一个大表格。",
    )

    clean_result = attach_validation_result(clean)["validation_result"]
    duplicate_result = attach_validation_result(duplicate)["validation_result"]
    overlong_result = attach_validation_result(overlong)["validation_result"]
    multitask_result = attach_validation_result(multitask)["validation_result"]

    assert clean_result["passed"] is True
    assert duplicate_result["passed"] is False
    assert "完全相同" in duplicate_result["reject_reason"]
    assert overlong_result["passed"] is False
    assert "题长" in overlong_result["reject_reason"]
    assert multitask_result["passed"] is False
    assert "输出任务数" in multitask_result["reject_reason"] or "格式复杂度" in multitask_result["reject_reason"]


def test_candidate_selection_selects_valid_candidate_and_records_rejections():
    valid = make_candidate(
        "stage04-select",
        "cand_1",
        "请比较 A 与 B 两个候选事实，判断哪一个才是支撑结论的最小关键事实，并说明另一个为什么不足。",
        {
            "passed": True,
            "main_axis_count": 1,
            "new_facts_count": 1,
            "output_tasks_count": 1,
            "candidate_options_count": 2,
            "counterfactual_count": 0,
            "estimated_prompt_chars": 62,
            "external_knowledge_risk": "low",
            "format_difficulty_risk": "low",
            "repeat_pattern_risk": "low",
            "why_passed": "ok",
            "reject_reason": None,
        },
    )
    invalid = make_candidate(
        "stage04-select",
        "cand_2",
        "请输出复杂编号表格。",
        {
            "passed": False,
            "main_axis_count": 1,
            "new_facts_count": 1,
            "output_tasks_count": 3,
            "candidate_options_count": 0,
            "counterfactual_count": 0,
            "estimated_prompt_chars": 9,
            "external_knowledge_risk": "low",
            "format_difficulty_risk": "high",
            "repeat_pattern_risk": "low",
            "why_passed": "",
            "reject_reason": "存在格式复杂度压分风险",
            "invalid_type": "format_difficulty_dominant",
        },
    )

    selected, invalid_cases = select_candidates([invalid, valid])

    assert len(selected) == 1
    assert selected[0]["candidate_selection"]["selected_candidate_id"] == "stage04-select::cand_1"
    assert selected[0]["candidate_selection"]["rejected_candidates"]
    assert invalid_cases
    assert invalid_cases[0]["invalid_type"] == "format_difficulty_dominant"


if __name__ == "__main__":
    test_question_evolution_can_emit_primary_and_backup_candidates()
    test_validate_retry_reuses_operator_and_includes_reject_reason()
    test_llm_validation_result_can_reject_unanswerable_candidate()
    test_dynamic_candidate_budget_allocates_more_to_priority_samples()
    test_validation_rejects_overlong_duplicate_multitask_and_accepts_clean_candidate()
    test_candidate_selection_selects_valid_candidate_and_records_rejections()
    print("stage04 complexity validation and candidate selection checks passed")
