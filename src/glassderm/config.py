"""Config loader with ``${a.b}`` interpolation and CLI-style overrides.

We keep the loader deliberately tiny (no hydra/omegaconf dependency):

* YAML is parsed with PyYAML.
* ``${key.subkey}`` strings are replaced by walking the tree.
* CLI overrides look like ``train.epochs=20`` and are applied last.

This means every experiment's knobs live in one small, auditable file.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

_INTERP = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")


class Config(dict):
    """Dict with attribute access and a ``get_path`` helper for dotted keys."""

    def __getattr__(self, item):
        try:
            val = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return _wrap(val)

    def __setattr__(self, key, value):
        self[key] = value

    # -- convenience -----------------------------------------------------------
    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set_path(self, dotted: str, value: Any) -> None:
        cur: dict = self
        parts = dotted.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value


def _wrap(value):
    if isinstance(value, dict) and not isinstance(value, Config):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = Config(_deep_to_config(data))
    if overrides:
        for ov in overrides:
            _apply_override(cfg, ov)
    _resolve_interpolation(cfg, cfg)
    return cfg


def _deep_to_config(d):
    if isinstance(d, dict):
        return {k: _deep_to_config(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deep_to_config(v) for v in d]
    return d


def _apply_override(cfg: Config, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Bad override (expected key=value): {override!r}")
    key, raw = override.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    cfg.set_path(key, value)


def _resolve_interpolation(root: dict, node: Any, _seen=None):
    _seen = _seen or set()
    if isinstance(node, dict):
        for k, v in list(node.items()):
            node[k] = _resolve_interpolation(root, v, _seen)
        return node
    if isinstance(node, list):
        return [_resolve_interpolation(root, v, _seen) for v in node]
    if isinstance(node, str) and "${" in node:
        return _interp_string(root, node, _seen)
    return node


def _interp_string(root, s: str, seen: set) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key in seen:
            raise ValueError(f"Cyclic interpolation on {key!r}")
        seen.add(key)
        cur: Any = root
        for part in key.split("."):
            cur = cur[part]
        if isinstance(cur, str) and "${" in cur:
            cur = _interp_string(root, cur, seen)
        seen.discard(key)
        return str(cur)

    prev, cur = None, s
    for _ in range(16):
        prev = cur
        cur = _INTERP.sub(repl, cur)
        if cur == prev:
            break
    return cur


def pick_device(cfg_device: str) -> str:
    """Return a concrete torch device string from the ``auto|cuda|cpu`` hint."""
    import torch

    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if cfg_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return cfg_device
