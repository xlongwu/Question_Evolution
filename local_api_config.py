"""Local plaintext API configuration loader.

The repository keeps real credentials out of tracked source files. For local
runs, users may create an ignored ``config.py`` next to this module and define
plain Python variables there.
"""

import importlib.util
from pathlib import Path
from typing import Any, List, Optional


_CONFIG_CACHE = {}
_PLACEHOLDER_MARKERS = (
    "REPLACE_WITH",
    "YOUR_LOCAL_KEY",
    "在这里填入",
    "填入你的",
)


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.py")


def _load_config_module(config_path: Optional[Path] = None):
    path = Path(config_path) if config_path is not None else _default_config_path()
    if not path.exists():
        return None

    resolved = path.resolve()
    mtime = resolved.stat().st_mtime
    cached = _CONFIG_CACHE.get(resolved)
    if cached and cached[0] == mtime:
        return cached[1]

    module_name = "_question_evolution_local_config"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load local API config: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CONFIG_CACHE[resolved] = (mtime, module)
    return module


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            text = str(item).strip()
            if text and not _is_placeholder(text):
                result.append(text)
        return result
    text = str(value).strip()
    if not text or _is_placeholder(text):
        return []
    return [
        part.strip()
        for part in text.replace("\n", ",").split(",")
        if part.strip() and not _is_placeholder(part.strip())
    ]


def _is_placeholder(text: str) -> bool:
    upper_text = text.upper()
    return any(marker.upper() in upper_text for marker in _PLACEHOLDER_MARKERS)


def get_config_list(*names: str, config_path: Optional[Path] = None) -> List[str]:
    """Return the first non-empty list-like config value for any requested name."""
    module = _load_config_module(config_path)
    if module is None:
        return []
    for name in names:
        if hasattr(module, name):
            values = _coerce_list(getattr(module, name))
            if values:
                return values
    return []


def get_config_value(*names: str, default: str = "", config_path: Optional[Path] = None) -> str:
    """Return the first non-empty scalar config value for any requested name."""
    module = _load_config_module(config_path)
    if module is None:
        return default
    for name in names:
        if hasattr(module, name):
            values = _coerce_list(getattr(module, name))
            if values:
                return values[0]
    return default
