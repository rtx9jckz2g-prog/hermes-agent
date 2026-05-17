"""Tests for curator JSON/report CLI surfaces."""

from __future__ import annotations

import io
import json
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path


def test_status_json_outputs_machine_readable_payload(monkeypatch):
    import agent.curator as curator_state
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(curator_state, "load_state", lambda: {
        "paused": False,
        "last_run_at": "2026-05-01T00:00:00+00:00",
        "last_run_summary": "ok",
        "run_count": 2,
        "last_report_path": None,
    })
    monkeypatch.setattr(curator_state, "is_enabled", lambda: True)
    monkeypatch.setattr(curator_state, "get_interval_hours", lambda: 168)
    monkeypatch.setattr(curator_state, "get_stale_after_days", lambda: 30)
    monkeypatch.setattr(curator_state, "get_archive_after_days", lambda: 90)
    monkeypatch.setattr(skill_usage, "agent_created_report", lambda: [
        {"name": "alpha", "state": "active", "pinned": True, "activity_count": 1},
        {"name": "beta", "state": "stale", "pinned": False, "activity_count": 0},
    ])

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert curator_cli._cmd_status(Namespace(json=True)) == 0

    payload = json.loads(buf.getvalue())
    assert payload["status"] == "enabled"
    assert payload["runs"] == 2
    assert payload["skill_count"] == 2
    assert payload["by_state"]["active"] == 1
    assert payload["by_state"]["stale"] == 1
    assert payload["pinned"] == ["alpha"]


def test_report_json_aggregates_recent_run_json(tmp_path, monkeypatch):
    import agent.curator as curator_state
    import hermes_cli.curator as curator_cli

    root = tmp_path / "logs" / "curator"
    run_dir = root / "20260501-010203"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "started_at": "2026-05-01T01:02:03+00:00",
        "auto_transitions": {"reactivated": 1},
        "counts": {
            "archived_this_run": 2,
            "consolidated_this_run": 1,
            "pruned_this_run": 1,
            "added_this_run": 1,
            "state_transitions": 3,
            "cron_jobs_rewritten": 1,
            "tool_calls_total": 7,
        },
        "llm_error": None,
    }))
    monkeypatch.setattr(curator_state, "_reports_root", lambda: root)

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert curator_cli._cmd_report(Namespace(json=True, limit=5)) == 0

    payload = json.loads(buf.getvalue())
    assert payload["totals"]["runs"] == 1
    assert payload["totals"]["archived"] == 2
    assert payload["totals"]["consolidated"] == 1
    assert payload["totals"]["restored_or_reactivated"] == 1
    assert payload["runs"][0]["run_dir"] == str(run_dir)
