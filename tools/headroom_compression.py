"""Headroom tool-output compression — auto-compress large JSON tool results.

Sits between Layer 1 (per-tool truncation) and Layer 2 (sandbox persistence)
in the 3-layer tool result pipeline (see ``tools/tool_result_storage.py``).
Saves 50-70% tokens on JSON-shaped outputs (search results, KPI lists, API
responses) at the cost of a 0.4-0.6s SSH roundtrip to Mac Studio where
headroom-ai 0.26.0 is installed. Free-form text and multimodal results are
skipped by heuristic (text savings are only 3-5%, not worth the latency).

Architecture (graceful degradation, 3 tiers):
    Tier 2: SSH → Mac Studio → headroom-ai (best, 50-70% savings, 0.4-0.6s)
    Tier 1: (skipped — base install on MacBook 8GB doesn't have _core)
    Tier 0: return raw content unchanged (no error, no latency added past
            the threshold check)

Configuration (config.yaml):
    tool_compression:
      headroom:
        enabled: true            # master switch
        min_chars: 1500          # skip if content smaller than this
        min_savings_pct: 5       # only use compression if it saves >= N%
        json_only: true          # skip free-form text (3-5% savings not
                                 # worth SSH latency); auto-detect by shape
        circuit_breaker_threshold: 3   # skip after N consecutive failures
        circuit_breaker_cooldown_sec: 60  # try again after this many sec

Determinism: headroom 0.26.0's content-based transforms (router:protected,
router:mixed) are deterministic — same input → same output. The ML-based
Kompress compressor is NOT installed, so the determinism invariant is
preserved and prompt caching stays intact.

Cache safety: we modify only the content of `role=tool` messages AFTER the
assistant message that triggered them. The system-prompt prefix and prior
message history are untouched, so per-conversation prompt caching is
preserved.

Cache-invariance (config — tuned for MacBook 8GB + free model):
    tool_compression:
      headroom:
        min_chars: 2000
        min_savings_pct: 20
        json_only: true
        circuit_breaker_threshold: 5
        circuit_breaker_cooldown_sec: 120

Usage:
    from tools.headroom_compression import maybe_compress_tool_output

    def _on_tool_complete(result: str) -> str:
        return maybe_compress_tool_output(result)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────
# These match the recommended tuned values in config.yaml. They are the
# fallback when config.yaml is missing a key. The actual values come from
# config.tool_compression.headroom.* at runtime.
DEFAULTS = {
    "enabled": False,
    "min_chars": 2000,             # was 1500 — skip small outputs
    "min_savings_pct": 20,         # was 5 — only compress when worth latency
    "json_only": True,             # skip free-form text (3-5% savings not worth 0.4-0.6s SSH)
    "circuit_breaker_threshold": 5,  # was 3 — tolerate transient Mac Studio hiccups
    "circuit_breaker_cooldown_sec": 120,  # was 60 — longer cooldown for stability
}

# Adapter search paths (the adapter is a user-managed script in the Vault,
# not vendored into hermes-agent).  HEADROOM_ADAPTER_PATH is read live on
# every call so monkeypatch / env changes during tests take effect.
# Other paths are static fallbacks checked in order; first hit wins.
_ADAPTER_SEARCH_PATHS = [
    os.path.expanduser("~/Documents/Obsidian Vault/Shared/Scripts/headroom_hermes_adapter.py"),
    os.path.expanduser("~/.hermes/scripts/headroom_hermes_adapter.py"),
    os.path.expanduser("~/.hermes/headroom_hermes_adapter.py"),
]


def _candidate_adapter_paths() -> list[str]:
    """Return adapter search paths with live HEADROOM_ADAPTER_PATH prepended."""
    env_path = os.environ.get("HEADROOM_ADAPTER_PATH")
    if env_path:
        return [env_path] + list(_ADAPTER_SEARCH_PATHS)
    return list(_ADAPTER_SEARCH_PATHS)

# Module-level state — single shared state across the process.
# "uninit"   — never tried
# "disabled" — config flag off (terminal state, no retry)
# "no_adapter" — adapter script not found on disk (terminal-ish)
# "ready"    — adapter loaded and ready
# "error"    — adapter raised during load (transient — circuit breaker
#             may re-attempt after cooldown)
_HEADROOM_STATE: dict = {
    "status": "uninit",
    "fn": None,
    "config": None,
    "consecutive_failures": 0,
    "circuit_open_until": 0.0,
}


# ── Config resolution ─────────────────────────────────────────────────
def _resolve_config() -> dict:
    """Load headroom config from config.yaml with defaults.

    Falls back to defaults if config can't be loaded. Returns a fresh
    dict on each call (caller may mutate).
    """
    cfg = dict(DEFAULTS)
    try:
        from hermes_cli.config import load_config
        user_cfg = load_config()
        headroom_cfg = (
            user_cfg.get("tool_compression", {})
            .get("headroom", {})
        )
        for key, default in DEFAULTS.items():
            if key in headroom_cfg:
                cfg[key] = headroom_cfg[key]
    except Exception as exc:
        logger.debug("headroom config load failed, using defaults: %s", exc)
    return cfg


def is_headroom_enabled() -> bool:
    """Return True iff config flag is on. Cached after first call."""
    if _HEADROOM_STATE["config"] is not None:
        return _HEADROOM_STATE["config"].get("enabled", False)
    # Lazy one-shot: peek at config flag without loading the adapter.
    try:
        from hermes_cli.config import load_config
        user_cfg = load_config()
        enabled = (
            user_cfg.get("tool_compression", {})
            .get("headroom", {})
            .get("enabled", False)
        )
    except Exception:
        enabled = False
    return bool(enabled)


# ── Adapter loading ───────────────────────────────────────────────────
def _load_adapter() -> Optional[Any]:
    """Find and import the headroom adapter script. Returns the
    ``compress_tool_output`` callable or None on any failure.

    Searches a list of known paths (configurable via HEADROOM_ADAPTER_PATH
    env var, read live on every call). Once loaded, the adapter is cached
    in module state.
    """
    for path in _candidate_adapter_paths():
        if not path or not os.path.exists(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location("_hr_adapter", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "compress_tool_output"):
                logger.info("headroom adapter loaded from %s", path)
                return mod.compress_tool_output
            logger.warning("headroom adapter at %s missing compress_tool_output", path)
        except Exception as exc:
            logger.warning("headroom adapter load failed (%s): %s", path, exc)
    return None


def _init_state() -> None:
    """Lazy one-shot init: load config + adapter, update state.

    Idempotent and side-effect-only on first call. Safe to call from
    hot paths because subsequent calls are O(1) state checks.
    """
    if _HEADROOM_STATE["status"] != "uninit":
        return

    cfg = _resolve_config()
    _HEADROOM_STATE["config"] = cfg

    if not cfg.get("enabled", False):
        _HEADROOM_STATE["status"] = "disabled"
        return

    fn = _load_adapter()
    if fn is None:
        _HEADROOM_STATE["status"] = "no_adapter"
        return

    _HEADROOM_STATE["fn"] = fn
    _HEADROOM_STATE["status"] = "ready"


def reset_headroom_state() -> None:
    """Clear module-level state. For tests only."""
    _HEADROOM_STATE["status"] = "uninit"
    _HEADROOM_STATE["fn"] = None
    _HEADROOM_STATE["config"] = None
    _HEADROOM_STATE["consecutive_failures"] = 0
    _HEADROOM_STATE["circuit_open_until"] = 0.0


# ── Heuristics ────────────────────────────────────────────────────────
def is_json_like_content(content) -> bool:
    """Cheap heuristic: does this content look like a JSON document?

    Used to skip free-form text (logs, file dumps, terminal output)
    where headroom gives only 3-5% savings — not worth the 0.4-0.6s
    SSH roundtrip. Catches the common Hermes tool output shapes:
    - ``{"key": ...}`` — JSON object
    - ``[{...}, ...]`` — JSON array
    - ``[{"success": ...}]`` — wrapped results
    - Whitespace/newline prefix is OK.

    Returns True for any of those shapes; False for plain text or
    non-string inputs.
    """
    if not isinstance(content, str):
        return False
    stripped = content.lstrip()
    if not stripped:
        return False
    if stripped[0] not in ("{", "["):
        return False
    # Cheap sanity: ensure it parses as JSON. Cheap enough for our
    # size range (we're past min_chars=1500, so a quick parse is fine).
    try:
        json.loads(stripped)
        return True
    except (ValueError, TypeError):
        return False


# ── Main entry point ──────────────────────────────────────────────────
def maybe_compress_tool_output(result: Any) -> Any:
    """Compress tool output if it's a candidate. Pass-through otherwise.

    Decision tree:
        1. Not a string? → return as-is (multimodal results etc.)
        2. Below min_chars? → return as-is
        3. Config disabled / adapter missing? → return as-is
        4. Circuit breaker open? → return as-is
        5. json_only=True and content isn't JSON-shaped? → return as-is
        6. Try compress. If savings < min_savings_pct, return as-is.
        7. On any error, increment failure counter; return as-is.

    This function is the public hook for both tool executor paths.
    """
    # Fast path: only strings are compressible
    if not isinstance(result, str):
        return result

    # Lazy init
    _init_state()

    cfg = _HEADROOM_STATE["config"] or DEFAULTS
    status = _HEADROOM_STATE["status"]

    # Not enabled or not ready → no-op
    if status in ("disabled", "no_adapter"):
        return result
    if status != "ready":
        return result

    # Circuit breaker
    if time.monotonic() < _HEADROOM_STATE["circuit_open_until"]:
        return result

    min_chars = cfg.get("min_chars", DEFAULTS["min_chars"])
    if len(result) < min_chars:
        return result

    # JSON-only heuristic (skip free-form text)
    if cfg.get("json_only", DEFAULTS["json_only"]) and not is_json_like_content(result):
        return result

    # Try to compress
    fn = _HEADROOM_STATE["fn"]
    raw_len = len(result)
    try:
        compressed = fn(result)
    except Exception as exc:
        _HEADROOM_STATE["consecutive_failures"] += 1
        threshold = cfg.get(
            "circuit_breaker_threshold",
            DEFAULTS["circuit_breaker_threshold"],
        )
        if _HEADROOM_STATE["consecutive_failures"] >= threshold:
            cooldown = cfg.get(
                "circuit_breaker_cooldown_sec",
                DEFAULTS["circuit_breaker_cooldown_sec"],
            )
            _HEADROOM_STATE["circuit_open_until"] = time.monotonic() + cooldown
            logger.warning(
                "headroom circuit breaker OPEN after %d failures; "
                "cooldown %ds",
                _HEADROOM_STATE["consecutive_failures"],
                cooldown,
            )
        logger.debug("headroom compression error: %s", exc)
        return result

    # Did we actually save anything?
    if not isinstance(compressed, str):
        # Adapter returned non-string (e.g. an object) — treat as failure
        _HEADROOM_STATE["consecutive_failures"] += 1
        return result

    new_len = len(compressed)
    if new_len >= raw_len:
        # No savings — treat as success (don't penalize) but skip use.
        _HEADROOM_STATE["consecutive_failures"] = 0
        return result

    savings_pct = (1 - new_len / raw_len) * 100
    min_savings = cfg.get("min_savings_pct", DEFAULTS["min_savings_pct"])
    if savings_pct < min_savings:
        # Adapter shrunk it but not enough to justify the latency trade
        _HEADROOM_STATE["consecutive_failures"] = 0
        return result

    # Success!
    _HEADROOM_STATE["consecutive_failures"] = 0
    logger.debug(
        "headroom compressed %d→%d chars (%.1f%% savings)",
        raw_len,
        new_len,
        savings_pct,
    )
    return compressed


__all__ = [
    "is_headroom_enabled",
    "is_json_like_content",
    "maybe_compress_tool_output",
    "reset_headroom_state",
]