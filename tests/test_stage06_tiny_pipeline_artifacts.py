import asyncio
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from analyze_evolution_effect import analyze_records, build_effect_matrix
from candidate_selection import select_candidates
from operator_router import route_records
from profile_samples import ProfileProcessor
from question_evolution import QuestionEvolutionProcessor
from select_evolution_candidates import process_records as select_evolution_records
from update_sample_state import update_records
from validate_evolved_question import attach_validation_result


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeProfileClient:
    async def chat_completions_create(self, **kwargs):
        return FakeResponse(
            json.dumps(
                {
                    "sample_profile": {
                        "core_capability": "证据链补强",
                        "claim_level": "可疑线索",
                        "problem_shape": "候选项区分",
                        "reasoning_granularity": "两步链条",
                        "answer_mode_expected": "比较型",
                        "easy_judgment_risk": "low",
                        "external_knowledge_risk": "low",
                        "complexity_expansion_risk": "medium",
                    },
                    "overscore_diagnosis": {
                        "is_worth_evolving": True,
                        "candidate_overscore_cause": "漏最小关键事实",
                        "target_failure_mode": "选错最关键缺口",
                        "why_high_score_is_suspicious": "候选答案只泛泛说明证据不足，没有指出最卡结论的独立缺口。",
                    },
                },
                ensure_ascii=False,
            )
        )


class FakeEvolutionClient:
    async def chat_completions_create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "O2_subclaim_localization" in prompt:
            evolved_prompt = (
                "请只围绕原题结论中的一个子判断作答：现有事实支持哪一层，"
                "哪一层仍缺少独立必要事实？请说明最少还缺哪一类事实。"
            )
            strategy = "O2 子判断定位"
        else:
            evolved_prompt = (
                "请比较 A 与 B 两个候选补充事实，判断哪一个才是支撑结论的最小关键事实，"
                "并说明另一个为什么不足或已被吸收。"
            )
            strategy = "O1 候选缺口二选一"
        return FakeResponse(
            json.dumps(
                {
                    "evolved_prompt": evolved_prompt,
                    "evolution_strategy": strategy,
                    "complexity_budget": {
                        "main_axis_count": 1,
                        "new_facts_count": 1,
                        "output_tasks_count": 1,
                        "candidate_options_count": 2,
                        "counterfactual_count": 0,
                    },
                    "notes_for_reference": "需要围绕最小关键事实补充参考答案。",
                },
                ensure_ascii=False,
            )
        )


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_seed_record():
    return {
        "sample_id": "tiny-801",
        "index": 801,
        "prompt": "原题：根据现有证据判断结论是否成立，并说明还缺什么关键事实。",
        "meta_info": {"references": ["参考答案应指出最小关键事实，而不是泛泛说证据不足。"]},
        "rubric": [{"title": "核心判断", "description": "识别最小关键事实。", "weight": 10}],
        "score_prompt": "请按 rubric 对 <<<待评答案>> 评分。",
    }


def attach_scoring(record, *, awarded, possible, answer):
    scored = dict(record)
    scored["scoring_result"] = {
        "answer_mode": "llm",
        "answer_model": "mock-qwen",
        "candidate_answer": answer,
        "item_scores": [
            {
                "title": "核心判断",
                "weight": possible,
                "awarded": awarded,
                "brief_reason": "mock score",
            }
        ],
        "overall_comment": "mock scoring result",
        "total_awarded": awarded,
        "total_possible": possible,
        "judge_model": "mock-judge",
        "judge_raw_response": "{}",
    }
    scored["score_rate"] = awarded / possible if possible else 0.0
    return scored


def mock_collect_answers(records):
    collected = []
    for record in records:
        item = dict(record)
        meta_info = dict(item.get("meta_info") or {})
        meta_info["references"] = ["mock reference for evolved question"]
        item["meta_info"] = meta_info
        collected.append(item)
    return collected


def mock_gen_rubric(records):
    rubric_records = []
    for record in records:
        item = dict(record)
        item["rubric"] = [{"title": "核心判断", "description": "识别最小关键事实。", "weight": 10}]
        item["rubric_thought_process"] = "mock rubric thought"
        item["score_prompt"] = "请按 rubric 对 <<<待评答案>> 评分。"
        rubric_records.append(item)
    return rubric_records


async def run_tiny_pipeline(exp_dir: Path):
    memory_dir = exp_dir / "memory"
    memory_dir.mkdir(parents=True)
    operator_memory = memory_dir / "operator_memory_bank.jsonl"
    failure_memory = memory_dir / "failure_memory_bank.jsonl"
    invalid_memory = memory_dir / "invalid_generation_cases.jsonl"
    for path in (operator_memory, failure_memory, invalid_memory):
        path.write_text("", encoding="utf-8")

    seed_path = exp_dir / "admitted_seed_samples.jsonl"
    seed = make_seed_record()
    write_jsonl(seed_path, [seed])

    round0 = exp_dir / "round_0"
    round1 = exp_dir / "round_1"
    final_dir = exp_dir / "final"
    round0.mkdir()
    round1.mkdir()
    final_dir.mkdir()

    write_jsonl(round0 / "input.jsonl", [seed])
    previous_scored = [
        attach_scoring(
            seed,
            awarded=10,
            possible=10,
            answer="证据不足，还要更多材料。",
        )
    ]
    write_jsonl(round0 / "scored.jsonl", previous_scored)

    profile_processor = ProfileProcessor(FakeProfileClient(), model="mock-profile", max_concurrent=1)
    profiled = await profile_processor.process_records(previous_scored)
    write_jsonl(round1 / "profiled.jsonl", profiled)

    profiled_candidates = select_evolution_records(profiled, high_score_threshold=0.8)
    write_jsonl(round1 / "profiled_candidates.jsonl", profiled_candidates)

    routed = route_records(profiled_candidates)
    write_jsonl(round1 / "routed.jsonl", routed)

    evolution_processor = QuestionEvolutionProcessor(
        FakeEvolutionClient(),
        model="mock-evolution",
        max_concurrent=1,
        max_retries=0,
        num_candidates=2,
    )
    candidates = []
    for record in routed:
        candidates.extend(await evolution_processor.process_item_candidates(record))
    write_jsonl(round1 / "candidates.jsonl", candidates)

    validated = [attach_validation_result(candidate) for candidate in candidates]
    write_jsonl(round1 / "validated_candidates.jsonl", validated)

    evolved, invalid_cases = select_candidates(validated)
    write_jsonl(round1 / "evolved.jsonl", evolved)
    write_jsonl(round1 / "invalid_generation_cases.jsonl", invalid_cases)

    with_answers = mock_collect_answers(evolved)
    write_jsonl(round1 / "with_answers.jsonl", with_answers)

    rubric_records = mock_gen_rubric(with_answers)
    write_jsonl(round1 / "rubric.jsonl", rubric_records)

    current_scored = [
        attach_scoring(
            record,
            awarded=5,
            possible=10,
            answer="候选答案把 A 和 B 都当作最小关键事实，没有区分哪一个才是独立必要条件。",
        )
        for record in rubric_records
    ]
    write_jsonl(round1 / "scored.jsonl", current_scored)

    analyzed = analyze_records(current_scored, previous_records=previous_scored)
    write_jsonl(round1 / "effect_analysis.jsonl", analyzed)
    write_jsonl(round1 / "effect_matrix.jsonl", build_effect_matrix(analyzed))

    updated, operator_entries, failure_entries, invalid_entries = update_records(analyzed)
    write_jsonl(round1 / "state_updated.jsonl", updated)
    write_jsonl(operator_memory, operator_entries)
    write_jsonl(failure_memory, failure_entries)
    write_jsonl(invalid_memory, invalid_entries)
    write_jsonl(final_dir / "final_scored.jsonl", updated)

    return exp_dir


def test_tiny_pipeline_writes_stage06_artifacts_without_external_api():
    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = Path(tmp) / "exp"
        asyncio.run(run_tiny_pipeline(exp_dir))

        expected_files = [
            "admitted_seed_samples.jsonl",
            "round_0/input.jsonl",
            "round_0/scored.jsonl",
            "round_1/profiled.jsonl",
            "round_1/profiled_candidates.jsonl",
            "round_1/routed.jsonl",
            "round_1/candidates.jsonl",
            "round_1/validated_candidates.jsonl",
            "round_1/evolved.jsonl",
            "round_1/with_answers.jsonl",
            "round_1/rubric.jsonl",
            "round_1/scored.jsonl",
            "round_1/effect_analysis.jsonl",
            "round_1/effect_matrix.jsonl",
            "round_1/state_updated.jsonl",
            "memory/operator_memory_bank.jsonl",
            "memory/failure_memory_bank.jsonl",
            "memory/invalid_generation_cases.jsonl",
            "final/final_scored.jsonl",
        ]
        for relative_path in expected_files:
            path = exp_dir / relative_path
            assert path.exists(), f"missing artifact: {relative_path}"

        evolved = read_jsonl(exp_dir / "round_1" / "evolved.jsonl")
        assert evolved[0]["question_evolved"] is True
        assert evolved[0]["candidate_selection"]["selected_operator"]
        assert evolved[0]["validation_result"]["passed"] is True

        analyzed = read_jsonl(exp_dir / "round_1" / "effect_analysis.jsonl")
        effect = analyzed[0]["effect_analysis"]
        assert effect["lightweight_boundary_hit"] is True
        assert effect["effect_label"] == "effective_boundary_probe"

        final_scored = read_jsonl(exp_dir / "final" / "final_scored.jsonl")
        assert final_scored[0]["evolution_state"]["stop_status"] == "effective_boundary_sample"

        operator_memory = read_jsonl(exp_dir / "memory" / "operator_memory_bank.jsonl")
        assert operator_memory
        assert operator_memory[0]["sample_id"] == "tiny-801"


if __name__ == "__main__":
    test_tiny_pipeline_writes_stage06_artifacts_without_external_api()
    print("stage06 tiny pipeline artifact checks passed")
