import sys
import types
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_dependency_stubs():
    if importlib.util.find_spec("openai") is None:
        openai_stub = types.ModuleType("openai")
        openai_stub.AsyncOpenAI = object
        sys.modules.setdefault("openai", openai_stub)

    if importlib.util.find_spec("aiofiles") is None:
        aiofiles_stub = types.ModuleType("aiofiles")
        sys.modules.setdefault("aiofiles", aiofiles_stub)

    if importlib.util.find_spec("tqdm") is None:
        tqdm_stub = types.ModuleType("tqdm")
        tqdm_asyncio_stub = types.ModuleType("tqdm.asyncio")
        tqdm_asyncio_stub.tqdm_asyncio = object
        sys.modules.setdefault("tqdm", tqdm_stub)
        sys.modules.setdefault("tqdm.asyncio", tqdm_asyncio_stub)


def test_score_prompt_placeholder_contract_is_shared():
    install_dependency_stubs()
    from gen_rubric import build_score_prompt
    from scoring import ANSWER_PLACEHOLDER, build_scoring_prompt

    rubric = [{"title": "核心判断", "description": "答对核心判断。", "weight": 10}]
    score_prompt = build_score_prompt({"prompt": "测试题目"}, rubric)
    rendered = build_scoring_prompt(score_prompt, "候选答案")

    assert ANSWER_PLACEHOLDER == "<<<待评答案>>"
    assert ANSWER_PLACEHOLDER in score_prompt
    assert "<<<待评答案>>>" not in score_prompt
    assert ANSWER_PLACEHOLDER not in rendered
    assert "候选答案" in rendered


if __name__ == "__main__":
    test_score_prompt_placeholder_contract_is_shared()
    print("score prompt placeholder contract checks passed")
