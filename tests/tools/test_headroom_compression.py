"""Tests for tools/headroom_compression.py — auto-compress large JSON tool results."""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from tools.headroom_compression import (
    DEFAULTS,
    _HEADROOM_STATE,
    is_headroom_enabled,
    is_json_like_content,
    maybe_compress_tool_output,
    reset_headroom_state,
)


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_state():
    """Always start tests with a clean module state."""
    reset_headroom_state()
    yield
    reset_headroom_state()


def _make_compressed(savings_ratio: float = 0.3):
    """Build a fake compress function that returns input shrunk by ratio."""
    def fake_compress(content: str) -> str:
        new_len = int(len(content) * (1 - savings_ratio))
        return content[:new_len]
    return fake_compress


def _install_ready_state(compress_fn=None, config_overrides: dict | None = None):
    """Force the module into 'ready' state with a fake compress fn."""
    cfg = dict(DEFAULTS)
    cfg["enabled"] = True
    if config_overrides:
        cfg.update(config_overrides)
    _HEADROOM_STATE["config"] = cfg
    _HEADROOM_STATE["fn"] = compress_fn or _make_compressed()
    _HEADROOM_STATE["status"] = "ready"
    _HEADROOM_STATE["consecutive_failures"] = 0
    _HEADROOM_STATE["circuit_open_until"] = 0.0


# ── is_json_like_content ─────────────────────────────────────────────
class TestIsJsonLikeContent:
    def test_object(self):
        assert is_json_like_content('{"a": 1, "b": [2, 3]}') is True

    def test_array(self):
        assert is_json_like_content('[{"a": 1}, {"a": 2}]') is True

    def test_whitespace_prefix(self):
        assert is_json_like_content('   \n  {"x": 1}') is True

    def test_plain_text(self):
        assert is_json_like_content("Hello world this is text") is False

    def test_terminal_log(self):
        assert is_json_like_content(
            "INFO 2026-06-19 12:00:00 server started on port 8080"
        ) is False

    def test_yaml_or_python_repr(self):
        # Not valid JSON
        assert is_json_like_content("{'a': 1, 'b': 2}") is False

    def test_malformed_json(self):
        assert is_json_like_content('{"unclosed": ') is False

    def test_empty_string(self):
        assert is_json_like_content("") is False

    def test_none(self):
        assert is_json_like_content(None) is False

    def test_non_string_types(self):
        assert is_json_like_content(42) is False
        assert is_json_like_content([1, 2, 3]) is False  # lists, not strings
        assert is_json_like_content({"a": 1}) is False


# ── maybe_compress_tool_output — fast-path / no-op cases ──────────────
class TestMaybeCompressFastPaths:
    def test_none_passthrough(self):
        assert maybe_compress_tool_output(None) is None

    def test_int_passthrough(self):
        assert maybe_compress_tool_output(42) == 42

    def test_list_passthrough(self):
        data = [1, 2, 3]
        assert maybe_compress_tool_output(data) is data

    def test_dict_passthrough(self):
        data = {"a": 1}
        assert maybe_compress_tool_output(data) is data

    def test_disabled_config_passthrough(self):
        # Default state is "uninit" + config flag missing → disabled
        # Already initialized via fixture, so set to disabled explicitly
        _HEADROOM_STATE["status"] = "disabled"
        large_json = json.dumps([{"k": i, "v": i * 2} for i in range(100)])
        out = maybe_compress_tool_output(large_json)
        assert out is large_json

    def test_no_adapter_passthrough(self):
        _HEADROOM_STATE["status"] = "no_adapter"
        large_json = json.dumps([{"k": i, "v": i * 2} for i in range(100)])
        out = maybe_compress_tool_output(large_json)
        assert out is large_json

    def test_below_min_chars_passthrough(self):
        _install_ready_state()
        small = json.dumps([{"k": i} for i in range(5)])  # < 1500 chars
        out = maybe_compress_tool_output(small)
        assert out is small

    def test_non_json_skipped_when_json_only(self):
        _install_ready_state()
        # 5000 chars of plain text
        text = ("This is just some free-form text with a few words. " * 100)
        out = maybe_compress_tool_output(text)
        assert out is text

    def test_json_allowed_when_json_only_disabled(self):
        # When json_only=False, non-JSON can also be compressed
        _install_ready_state()
        _HEADROOM_STATE["config"]["json_only"] = False
        text = ("This is just some free-form text with a few words. " * 100)
        out = maybe_compress_tool_output(text)
        assert out != text  # compressed
        assert len(out) < len(text)


# ── maybe_compress_tool_output — successful compression ───────────────
class TestMaybeCompressSuccess:
    def test_large_json_compressed(self):
        _install_ready_state()
        sample = [
            {"id": i, "name": f"item_{i}", "description": "verbose " * 20}
            for i in range(100)
        ]
        raw = json.dumps(sample)
        assert len(raw) > 1500
        out = maybe_compress_tool_output(raw)
        assert out != raw
        assert len(out) < len(raw)
        savings = (1 - len(out) / len(raw)) * 100
        assert savings >= 5  # min_savings_pct

    def test_below_min_savings_pct_rejected(self):
        """If compress_fn shrinks by < min_savings_pct, return original."""
        _install_ready_state(compress_fn=_make_compressed(savings_ratio=0.02))
        sample = [{"id": i, "k": f"v_{i}"} for i in range(100)]
        raw = json.dumps(sample)
        out = maybe_compress_tool_output(raw)
        # 2% savings is below 5% threshold → return original
        assert out is raw

    def test_compression_resets_failure_counter(self):
        _install_ready_state()
        _HEADROOM_STATE["consecutive_failures"] = 2
        sample = [{"id": i, "description": "x" * 100} for i in range(100)]
        raw = json.dumps(sample)
        maybe_compress_tool_output(raw)
        assert _HEADROOM_STATE["consecutive_failures"] == 0


# ── maybe_compress_tool_output — error handling ───────────────────────
class TestMaybeCompressErrors:
    def test_adapter_raises_returns_original(self):
        def boom(_):
            raise RuntimeError("ssh down")
        _install_ready_state(compress_fn=boom)
        sample = [{"id": i, "description": "x" * 100} for i in range(100)]
        raw = json.dumps(sample)
        out = maybe_compress_tool_output(raw)
        assert out is raw
        assert _HEADROOM_STATE["consecutive_failures"] == 1

    def test_adapter_returns_non_string(self):
        def returns_object(_):
            return {"unexpected": "dict"}
        _install_ready_state(compress_fn=returns_object)
        sample = [{"id": i, "description": "x" * 100} for i in range(100)]
        raw = json.dumps(sample)
        out = maybe_compress_tool_output(raw)
        assert out is raw
        assert _HEADROOM_STATE["consecutive_failures"] == 1

    def test_circuit_breaker_opens_after_threshold(self):
        def boom(_):
            raise RuntimeError("ssh down")
        _install_ready_state(
            compress_fn=boom,
            config_overrides={"circuit_breaker_threshold": 2, "circuit_breaker_cooldown_sec": 60},
        )
        sample = [{"id": i, "description": "x" * 100} for i in range(100)]
        raw = json.dumps(sample)
        # 1st call: increments to 1, still below threshold (2)
        maybe_compress_tool_output(raw)
        assert _HEADROOM_STATE["consecutive_failures"] == 1
        assert _HEADROOM_STATE["circuit_open_until"] == 0.0  # not yet open
        # 2nd call: hits threshold, opens circuit
        maybe_compress_tool_output(raw)
        assert _HEADROOM_STATE["consecutive_failures"] == 2
        assert _HEADROOM_STATE["circuit_open_until"] > time.monotonic()
        # 3rd call: circuit is open, return original without calling fn
        out = maybe_compress_tool_output(raw)
        assert out is raw

    def test_circuit_breaker_skips_while_open(self):
        # Simulate already-open circuit
        _install_ready_state()
        # Use a counting fn to verify it doesn't get called while open
        counter = {"calls": 0}
        def counting_boom(_):
            counter["calls"] += 1
            raise RuntimeError("boom")
        _HEADROOM_STATE["fn"] = counting_boom
        _HEADROOM_STATE["circuit_open_until"] = time.monotonic() + 60
        sample = [{"id": i, "description": "x" * 100} for i in range(100)]
        raw = json.dumps(sample)
        out = maybe_compress_tool_output(raw)
        assert out is raw
        assert counter["calls"] == 0  # circuit blocked the call


# ── is_headroom_enabled ───────────────────────────────────────────────
class TestIsHeadroomEnabled:
    def test_default_disabled(self):
        # load_config is imported inside the function so patch the source module
        with patch("hermes_cli.config.load_config", side_effect=Exception("no config")):
            assert is_headroom_enabled() is False

    def test_explicit_enabled(self):
        with patch("hermes_cli.config.load_config", return_value={
            "tool_compression": {"headroom": {"enabled": True}}
        }):
            assert is_headroom_enabled() is True

    def test_explicit_disabled(self):
        with patch("hermes_cli.config.load_config", return_value={
            "tool_compression": {"headroom": {"enabled": False}}
        }):
            assert is_headroom_enabled() is False

    def test_missing_key(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert is_headroom_enabled() is False


# ── _load_adapter (path resolution) ───────────────────────────────────
class TestLoadAdapter:
    def test_finds_adapter_at_env_path(self, tmp_path, monkeypatch):
        from tools.headroom_compression import _load_adapter, reset_headroom_state
        reset_headroom_state()
        adapter = tmp_path / "fake_adapter.py"
        adapter.write_text(
            "def compress_tool_output(content, model='x', min_chars=0, verbose=False):\n"
            "    return content[: len(content) // 2]\n"
        )
        monkeypatch.setenv("HEADROOM_ADAPTER_PATH", str(adapter))
        fn = _load_adapter()
        assert fn is not None
        assert fn("hello world") == "hello"

    def test_returns_none_when_no_adapter(self, tmp_path, monkeypatch):
        from tools.headroom_compression import _load_adapter, reset_headroom_state
        reset_headroom_state()
        # Point env at a non-existent path
        monkeypatch.setenv("HEADROOM_ADAPTER_PATH", str(tmp_path / "missing.py"))
        # Replace the other default paths with non-existent
        import tools.headroom_compression as hc
        original_paths = hc._ADAPTER_SEARCH_PATHS
        hc._ADAPTER_SEARCH_PATHS = [str(tmp_path / "missing.py")]
        try:
            assert _load_adapter() is None
        finally:
            hc._ADAPTER_SEARCH_PATHS = original_paths

    def test_skips_adapter_missing_compress_fn(self, tmp_path, monkeypatch):
        from tools.headroom_compression import _load_adapter, reset_headroom_state
        reset_headroom_state()
        adapter = tmp_path / "incomplete_adapter.py"
        adapter.write_text("# no compress_tool_output defined\n")
        monkeypatch.setenv("HEADROOM_ADAPTER_PATH", str(adapter))
        import tools.headroom_compression as hc
        original_paths = hc._ADAPTER_SEARCH_PATHS
        hc._ADAPTER_SEARCH_PATHS = [str(adapter)]
        try:
            assert _load_adapter() is None
        finally:
            hc._ADAPTER_SEARCH_PATHS = original_paths
