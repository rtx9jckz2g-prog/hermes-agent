#!/usr/bin/env python3
"""Token-saving worker router for Hermes Agent.

Runs one of a small allowlist of free Nous Portal models as a child Hermes
worker. Safety rule: paid models are blocked by default. Before each run, the
script fetches the Nous `/v1/models` catalog and verifies the selected model has
prompt=0 and completion=0. If the catalog cannot be checked, it refuses to run.

Examples:
  python scripts/worker_router.py --task code --prompt "Inspect foo.py and summarize risks"
  python scripts/worker_router.py --worker stepfun --prompt-file /tmp/input.txt
  python scripts/worker_router.py --list
  python scripts/worker_router.py --check-free
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("worker_models.free.json")
NOUS_MODELS_URL = "https://inference-api.nousresearch.com/v1/models"
CACHE_PATH = Path.home() / ".hermes" / "cache" / "worker_router_nous_models.json"
CACHE_TTL_SECONDS = 15 * 60


def _load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_worker(cfg: Dict[str, Any], *, worker: str = "", task: str = "") -> tuple[str, Dict[str, Any]]:
    workers = cfg.get("workers") or {}
    routes = cfg.get("routes") or {}
    key = (worker or "").strip().lower()
    if not key:
        key = str(routes.get((task or "").strip().lower()) or "stepfun")
    if key not in workers:
        raise SystemExit(f"Unknown worker '{key}'. Available: {', '.join(sorted(workers))}")
    return key, dict(workers[key])


def _read_nous_access_token() -> Optional[str]:
    """Use Hermes' existing Nous OAuth helper without printing secrets."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from tools.managed_tool_gateway import read_nous_access_token  # type: ignore

        token = read_nous_access_token()
        return token.strip() if isinstance(token, str) and token.strip() else None
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"ERROR: could not read Nous Portal auth token: {exc}", file=sys.stderr)
        return None


def _fetch_models(*, use_cache: bool = True) -> Dict[str, Any]:
    if use_cache and CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if time.time() - float(cached.get("fetched_at", 0)) < CACHE_TTL_SECONDS:
                return cached
        except Exception:
            pass

    token = _read_nous_access_token()
    if not token:
        raise SystemExit("BLOCKED: Nous Portal auth is missing. Run: hermes auth add nous")

    req = urllib.request.Request(
        NOUS_MODELS_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        raise SystemExit(f"BLOCKED: Nous model catalog check failed HTTP {exc.code}: {body}")
    except Exception as exc:
        raise SystemExit(f"BLOCKED: Nous model catalog check failed: {exc}")

    result = {"fetched_at": time.time(), "data": payload.get("data") or []}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _model_by_id(models_payload: Dict[str, Any], model_id: str) -> Optional[Dict[str, Any]]:
    for item in models_payload.get("data") or []:
        if item.get("id") == model_id:
            return item
        if model_id in (item.get("aliases") or []):
            return item
    return None


def _price_as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 999999.0


def _assert_free_model(model_id: str, *, use_cache: bool = True) -> Dict[str, Any]:
    models = _fetch_models(use_cache=use_cache)
    item = _model_by_id(models, model_id)
    if not item:
        raise SystemExit(f"BLOCKED: model not found in Nous catalog: {model_id}")
    pricing = item.get("pricing") or {}
    prompt = _price_as_float(pricing.get("prompt"))
    completion = _price_as_float(pricing.get("completion"))
    if prompt != 0.0 or completion != 0.0:
        raise SystemExit(
            f"BLOCKED: {model_id} is not free now "
            f"(prompt={pricing.get('prompt')}, completion={pricing.get('completion')})."
        )
    return item


def _compact_prompt(prompt: str, limit: int) -> str:
    prompt = prompt.strip()
    if len(prompt) <= limit:
        return prompt
    head = prompt[: int(limit * 0.65)]
    tail = prompt[-int(limit * 0.30) :]
    return (
        head
        + "\n\n[...TRUNCATED BY worker_router.py TO SAVE TOKENS...]\n\n"
        + tail
    )


def _build_worker_prompt(user_prompt: str, worker_key: str, worker_cfg: Dict[str, Any]) -> str:
    best_for = ", ".join(worker_cfg.get("best_for") or [])
    return textwrap.dedent(
        f"""
        You are a background worker for Mini/Hermes. Worker: {worker_key} ({worker_cfg.get('name')}).
        Best for: {best_for}.

        Rules:
        - Be concise. Do not paste long logs or full files unless explicitly asked.
        - Return only useful findings, risks, and recommended action.
        - If you used tools or edited files, include verifiable paths/status.
        - Never reveal secrets, tokens, API keys, passwords, or credentials.
        - Prefer this JSON-ish shape when possible:
          {{"summary":"...","findings":["..."],"risks":["..."],"recommended_action":"...","files_changed":[]}}

        Task:
        {user_prompt}
        """
    ).strip()


def _run_hermes(model: str, prompt: str, toolsets: str, timeout: int) -> int:
    cmd = [
        "hermes",
        "chat",
        "--provider",
        "nous",
        "-m",
        model,
        "-q",
        prompt,
        "--quiet",
    ]
    # Empty string deliberately disables tools for cheap summary workers.
    cmd += ["--toolsets", toolsets]
    print("WORKER_CMD:", " ".join(shlex.quote(x if x != prompt else "<prompt>") for x in cmd), file=sys.stderr)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, timeout=timeout)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Route work to free Nous Portal worker models only.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--worker", choices=["qwen", "stepfun", "owl"], default="")
    parser.add_argument("--task", default="summary", help="Route key, e.g. code/debug/summary/long")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--toolsets", default=None, help="Override worker default toolsets. Use '' for no tools.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--list", action="store_true", help="List configured workers.")
    parser.add_argument("--check-free", action="store_true", help="Check configured workers are still free, then exit.")
    parser.add_argument("--no-cache", action="store_true", help="Force live Nous catalog check.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve worker and free-check, but do not run Hermes.")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    if (cfg.get("policy") or {}).get("allow_paid") is not False:
        raise SystemExit("BLOCKED: config policy.allow_paid must be false.")

    if args.list:
        for key, worker in sorted((cfg.get("workers") or {}).items()):
            print(f"{key}: {worker.get('model')} — {worker.get('name')}")
        return 0

    if args.check_free:
        ok = True
        for key, worker in sorted((cfg.get("workers") or {}).items()):
            model = worker.get("model")
            try:
                item = _assert_free_model(model, use_cache=not args.no_cache)
                print(f"FREE: {key}: {model} ({item.get('name')})")
            except SystemExit as exc:
                ok = False
                print(str(exc), file=sys.stderr)
        return 0 if ok else 2

    key, worker = _resolve_worker(cfg, worker=args.worker, task=args.task)
    model = str(worker.get("model") or "")
    item = _assert_free_model(model, use_cache=not args.no_cache)
    print(f"FREE_OK: {key}: {model} ({item.get('name')})", file=sys.stderr)

    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        if not sys.stdin.isatty():
            prompt = sys.stdin.read()
    if not prompt.strip():
        raise SystemExit("Missing --prompt, --prompt-file, or stdin prompt.")

    limit = int(worker.get("max_prompt_chars") or 16000)
    worker_prompt = _build_worker_prompt(_compact_prompt(prompt, limit), key, worker)
    toolsets = worker.get("default_toolsets", "") if args.toolsets is None else args.toolsets

    if args.dry_run:
        print(json.dumps({"worker": key, "model": model, "toolsets": toolsets, "prompt_chars": len(worker_prompt)}, indent=2))
        return 0

    return _run_hermes(model, worker_prompt, str(toolsets), args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
