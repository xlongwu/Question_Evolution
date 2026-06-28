import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from local_api_config import get_config_list, get_config_value


def test_local_config_reads_plaintext_lists_and_scalars(tmp_path):
    config = tmp_path / "config.py"
    config.write_text(
        "\n".join(
            [
                "BASE_URL = 'https://example.test/v1'",
                "HIAPI_KEYS_BIG = ['key-a', 'key-b']",
                "QWEN_API_KEY = 'EMPTY_KEY'",
            ]
        ),
        encoding="utf-8",
    )

    assert get_config_value("BASE_URL", config_path=config) == "https://example.test/v1"
    assert get_config_list("HIAPI_KEYS_BIG", config_path=config) == ["key-a", "key-b"]
    assert get_config_list("QWEN_API_KEY", config_path=config) == ["EMPTY_KEY"]


def test_local_config_uses_first_non_empty_requested_name(tmp_path):
    config = tmp_path / "config.py"
    config.write_text(
        "OPENAI_API_KEY = ''\nHIAPI_KEYS_BIG = 'key-a,key-b'\n",
        encoding="utf-8",
    )

    assert get_config_list("OPENAI_API_KEY", "HIAPI_KEYS_BIG", config_path=config) == ["key-a", "key-b"]
    assert get_config_value("MISSING", default="fallback", config_path=config) == "fallback"


def test_local_config_ignores_template_placeholders(tmp_path):
    config = tmp_path / "config.py"
    config.write_text(
        "HIAPI_KEYS_BIG = ['在这里填入你的API_KEY', 'real-key']\n",
        encoding="utf-8",
    )

    assert get_config_list("HIAPI_KEYS_BIG", config_path=config) == ["real-key"]
