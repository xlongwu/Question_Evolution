import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from operator_router import route_records
from prompts.operators import OPERATOR_SPECS
from question_evolution import QuestionEvolutionProcessor


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
        content = json.dumps(
            {
                "evolved_prompt": (
                    "请在原题基础上判断两个候选依据中哪一个才真正决定结论能否成立，"
                    "并说明另一个依据为什么不能单独支撑结论。"
                ),
                "evolution_strategy": "使用指定 operator 生成单主轴问题。",
            },
            ensure_ascii=False,
        )
        return FakeResponse(content)


def test_operator_registry_covers_o1_to_o9():
    assert len(OPERATOR_SPECS) == 9
    for index in range(1, 10):
        assert any(operator_id.startswith(f"O{index}_") for operator_id in OPERATOR_SPECS)


def test_router_covers_representative_stage03_paths():
    records = load_jsonl(ROOT / "tests" / "fixtures" / "stage03_routing_input.jsonl")
    routed = route_records(records)
    routes = {record["sample_id"]: record["operator_route"] for record in routed}

    assert routes["stage03-o1"]["primary_operator"] == "O1_gap_choice"
    assert "O2_subclaim_localization" in routes["stage03-o1"]["backup_operators"]

    assert routes["stage03-o2"]["primary_operator"] == "O2_subclaim_localization"
    assert "O1_gap_choice" in routes["stage03-o2"]["avoid_operators"]
    assert "O4_near_level_ranking" in routes["stage03-o2"]["backup_operators"]

    assert routes["stage03-o4"]["primary_operator"] == "O4_near_level_ranking"
    assert routes["stage03-o8"]["primary_operator"] == "O8_double_threshold_claim"
    assert routes["stage03-o9"]["primary_operator"] == "O9_abnormal_clue_mainline_switch"

    assert routes["stage03-pass"]["primary_operator"] is None


def test_question_evolution_uses_route_and_skips_passthrough():
    records = load_jsonl(ROOT / "tests" / "fixtures" / "stage03_routing_input.jsonl")
    routed = route_records(records)
    by_id = {record["sample_id"]: record for record in routed}
    fake_client = FakeEvolutionClient()
    processor = QuestionEvolutionProcessor(
        fake_client,
        model="mock-evolution-model",
        max_concurrent=1,
        max_retries=0,
    )

    evolved = asyncio.run(processor.process_item(by_id["stage03-o1"]))
    metadata = evolved["meta_info"]["question_evolution_metadata"]
    assert evolved["question_evolved"] is True
    assert metadata["operator_used"] == "O1_gap_choice"
    assert metadata["ability_axis"] == "独立必要条件识别"
    assert metadata["expected_qwen_failure"] == "选错最关键缺口"
    assert metadata["expected_evaluation_focus"]
    assert len(fake_client.calls) == 1
    assert "O1_gap_choice" in fake_client.calls[0]["messages"][0]["content"]

    passed = asyncio.run(processor.process_item(by_id["stage03-pass"]))
    assert passed["question_evolved"] is False
    assert len(fake_client.calls) == 1


if __name__ == "__main__":
    test_operator_registry_covers_o1_to_o9()
    test_router_covers_representative_stage03_paths()
    test_question_evolution_uses_route_and_skips_passthrough()
    print("stage03 operator routing checks passed")
