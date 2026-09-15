"""YAML configuration loader with environment variable expansion and dot-access."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, MutableMapping
import yaml

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-(.*?))?\}")


def _interpolate_env_vars(raw: Any) -> Any:
    """Recursively expand ${VAR:-default} and ${VAR} patterns in string values."""
    if isinstance(raw, str):
        def _repl(match: re.Match) -> str:
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_val)

        return _ENV_VAR_PATTERN.sub(_repl, raw)
    elif isinstance(raw, dict):
        return {k: _interpolate_env_vars(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        return [_interpolate_env_vars(item) for item in raw]
    return raw


class Config(MutableMapping):
    """Dictionary wrapper providing attribute-style and nested key access with dot notation."""

    def __init__(self, data: Dict[str, Any] | None = None):
        self._data: Dict[str, Any] = {}
        if data:
            for k, v in data.items():
                self[k] = v

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            self._data[key] = Config(value)
        elif isinstance(value, list):
            self._data[key] = [
                Config(x) if isinstance(x, dict) and not isinstance(x, Config) else x
                for x in value
            ]
        else:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def get_nested(self, dot_key: str, default: Any = None) -> Any:
        """Access nested dictionary keys using dot notation (e.g. 'training.batch_size')."""
        curr: Any = self
        for part in dot_key.split("."):
            if isinstance(curr, (dict, Config)) and part in curr:
                curr = curr[part]
            else:
                return default
        return curr

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a standard Python dictionary."""
        out: Dict[str, Any] = {}
        for k, v in self._data.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                out[k] = v
        return out

    def update_nested(self, overrides: Dict[str, Any]) -> None:
        """Update configuration using a nested or dot-keyed dictionary."""
        for key, value in overrides.items():
            if "." in key:
                parts = key.split(".")
                target = self
                for p in parts[:-1]:
                    if p not in target or not isinstance(target[p], (dict, Config)):
                        target[p] = Config({})
                    target = target[p]
                target[parts[-1]] = value
            elif isinstance(value, dict) and key in self and isinstance(self[key], Config):
                self[key].update_nested(value)
            else:
                self[key] = value


def load_config(config_path: str | Path, overrides: Dict[str, Any] | None = None) -> Config:
    """Load and parse a YAML configuration file with environment variable expansion.

    Args:
        config_path: Path to YAML config file.
        overrides: Optional key-value dictionary to override config parameters.

    Returns:
        Config object.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    interpolated = _interpolate_env_vars(raw)
    cfg = Config(interpolated)

    if overrides:
        cfg.update_nested(overrides)

    return cfg


def save_config(config: Config | Dict[str, Any], output_path: str | Path) -> None:
    """Save configuration dictionary to a YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict() if isinstance(config, Config) else config
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
