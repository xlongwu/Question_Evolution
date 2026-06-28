import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from local_api_config import get_config_list, get_config_value

REQUIRED_SEED_IDS = {"801", "13364", "14865", "8638", "7337", "18485", "6582", "5486"}
REQUIRED_IMPORTS = ("openai", "aiofiles", "tqdm", "pytest")
API_GROUPS = {
    "profile": ("PROFILE_API_KEYS", "EVOLVE_API_KEYS", "GPT_API_KEYS", "HIAPI_KEYS_BIG", "OPENAI_API_KEYS", "OPENAI_API_KEY"),
    "evolution": ("EVOLVE_API_KEYS", "GPT_API_KEYS", "HIAPI_KEYS_BIG", "OPENAI_API_KEYS", "OPENAI_API_KEY"),
    "answer": ("ANSWER_API_KEYS", "GPT_API_KEYS", "HIAPI_KEYS_BIG", "OPENAI_API_KEYS", "OPENAI_API_KEY"),
    "rubric": ("RUBRIC_API_KEYS", "GPT_API_KEYS", "HIAPI_KEYS_BIG", "OPENAI_API_KEYS", "OPENAI_API_KEY"),
}
JUDGE_KEY_NAMES = ("JUDGE_API_KEYS", "QWEN_API_KEYS", "QWEN_API_KEY", "OPENAI_API_KEYS", "OPENAI_API_KEY")


def discover_bash() -> Optional[str]:
    path_bash = shutil.which("bash")
    if path_bash:
        return path_bash

    candidates = [
        Path("C:/Program Files/Git/bin/bash.exe"),
        Path("C:/Program Files/Git/usr/bin/bash.exe"),
        Path("C:/Program Files (x86)/Git/bin/bash.exe"),
        Path("C:/msys64/usr/bin/bash.exe"),
        Path("C:/cygwin64/bin/bash.exe"),
        Path("D:/Git/bin/bash.exe"),
        Path("D:/Git/usr/bin/bash.exe"),
    ]

    git_path = shutil.which("git")
    if git_path:
        git_exe = Path(git_path).resolve()
        candidates.extend(
            [
                git_exe.parent.parent / "bin" / "bash.exe",
                git_exe.parent.parent / "usr" / "bin" / "bash.exe",
            ]
        )

    try:
        where_output = subprocess.run(
            ["where.exe", "git"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in where_output.stdout.splitlines():
            git_candidate = Path(line.strip())
            if git_candidate.name.lower() == "git.exe":
                candidates.extend(
                    [
                        git_candidate.parent.parent / "bin" / "bash.exe",
                        git_candidate.parent.parent / "usr" / "bin" / "bash.exe",
                    ]
                )
    except Exception:
        pass

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _has_value(env: Mapping[str, str], names: tuple[str, ...], config_path: Path) -> bool:
    if any(str(env.get(name, "")).strip() for name in names):
        return True
    return bool(get_config_list(*names, config_path=config_path))


def _get_value(env: Mapping[str, str], names: tuple[str, ...], config_path: Path, default: str = "") -> str:
    for name in names:
        value = str(env.get(name, "")).strip()
        if value:
            return value
    return get_config_value(*names, config_path=config_path, default=default).strip()


def _judge_config_ready(env: Mapping[str, str], config_path: Path) -> bool:
    if _has_value(env, JUDGE_KEY_NAMES, config_path):
        return True
    base_url = _get_value(
        env,
        ("JUDGE_BASE_URL", "QWEN_BASE_URL"),
        config_path,
        default="http://127.0.0.1:18011/v1",
    )
    model = _get_value(
        env,
        ("JUDGE_MODEL", "QWEN_MODEL"),
        config_path,
        default="hjl_Qwen3.6-27B",
    )
    return bool(base_url and model)


def check_seed_file(root: Path, input_file: str) -> Dict[str, Any]:
    path = (root / input_file).resolve()
    result: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "row_count": 0,
        "required_ids_present": False,
        "missing_required_ids": sorted(REQUIRED_SEED_IDS),
    }
    if not path.exists():
        return result

    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            result["row_count"] += 1
            item = json.loads(line)
            value = item.get("index", item.get("sample_id"))
            if value is not None:
                ids.add(str(value))
    missing = REQUIRED_SEED_IDS - ids
    result["required_ids_present"] = not missing
    result["missing_required_ids"] = sorted(missing)
    return result


def build_report(
    root: Path,
    *,
    input_file: str = "admitted_seed_samples.jsonl",
    env: Optional[Mapping[str, str]] = None,
    bash_path: Optional[str] = None,
) -> Dict[str, Any]:
    env = env or os.environ
    bash_path = bash_path if bash_path is not None else discover_bash()
    imports = {name: importlib.util.find_spec(name) is not None for name in REQUIRED_IMPORTS}
    config_path = root / "config.py"
    api_config = {group: _has_value(env, names, config_path) for group, names in API_GROUPS.items()}
    api_config["judge"] = _judge_config_ready(env, config_path)
    seed = check_seed_file(root, input_file)

    checks = {
        "bash_available": bool(bash_path),
        "admitted_seed_ready": bool(seed["exists"] and seed["required_ids_present"]),
        "python_dependencies_ready": all(imports.values()),
        "api_config_ready": all(api_config.values()),
    }
    bash_command = f'"{bash_path}"' if bash_path and " " in bash_path else (bash_path or "bash")
    ps_bash_command = f"& '{bash_path}'" if bash_path else "bash"
    return {
        "ready_for_real_stage06_e2e": all(checks.values()),
        "checks": checks,
        "bash_path": bash_path,
        "python_imports": imports,
        "api_config": api_config,
        "admitted_seed": seed,
        "next_commands": [
            f"{bash_command} -n run_loop.sh",
            f"INPUT_FILE={input_file} MAX_ROUNDS=1 {bash_command} run_loop.sh",
        ],
        "next_powershell_commands": [
            f"{ps_bash_command} -n run_loop.sh",
            f"$env:INPUT_FILE='{input_file}'; $env:MAX_ROUNDS='1'; {ps_bash_command} run_loop.sh",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check runtime prerequisites for real Stage 06 Bash/API E2E validation.")
    parser.add_argument("--input-file", default="admitted_seed_samples.jsonl")
    parser.add_argument("--json", action="store_true", help="Only print JSON report.")
    args = parser.parse_args()

    report = build_report(Path.cwd(), input_file=args.input_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.json and not report["ready_for_real_stage06_e2e"]:
        print("Runtime prerequisites are incomplete; do not claim real Bash/API E2E acceptance yet.", file=sys.stderr)
    return 0 if report["ready_for_real_stage06_e2e"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
