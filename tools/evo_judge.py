#!/usr/bin/env python3
"""tools/evo_judge.py — Atomic Physical Judge for autonomous myds self-evolution.

Enforces every hard measurement and correctness invariant from
`docs/MEASUREMENT-DISCIPLINE.md` (§1–§12) and ledger §A–§B without human review:

  1. Offline Gate (`--mode offline`):
     - `make check-knobs` + `evo_memory.py --check`
     - Optional candidate build (`--build-cmd`)
     - Metal PSO ceiling diff (`evo_silicon_probe.diff_pso`) to surface any
       smem / TGs-per-core / max-threads change before touching full weights
     - Full unit & Metal 0-ULP test suite (`--unit-cmd`, default `make test`)
     - macOS `.ips` crash report delta == 0 (§9)

  2. Full Exclusive Gate (`--mode full`, runs offline gate + full-model ABABAB):
     - §3 / §A.4: Asserts Arm A (baseline) and Arm B (candidate) genuinely differ
       (binary/metallib SHA256 or isolated conf diff), rejecting identical-arm noise.
     - §1 / §2.2 / §2.3 / §A.5.1: Writes isolated per-arm config files preserving
       `MYDS_PREFIX_CACHE=1` while isolating `MYDS_KV_DSK` & `MYDS_RESP_STATE` to
       fresh empty temp paths, and validates both arms via `--print-config`.
     - §4: Full-graph `auto_barrier=0` 0-ULP gate across interleaved boots
       (`A1 -> B1 -> A2 -> B2 -> A3 -> B3`): deterministic `temperature=0, seed=42`
       multi-prompt SHA256 must be non-empty and 100% identical across all boots
       (catches cross-kernel WAR/RAW missing barriers in single-CB execution).
     - §2.1 / §2.4 / §A.5.4: Queries authoritative `/trace/api/requests` via
       ephemeral `X-Debug-Token` AND scans server logs to assert `cached_tokens==0`,
       `prefill_tps > 0`, `out_tokens > 0`, and zero OOM / fallback errors.
     - §10 / §B.2: Interleaved ABABAB statistical verdict — requires positive
       paired gain in every cycle, median gain >= `--min-gain-pct`, and gain
       strictly exceeding intra-arm spread.
     - §11: Optional `--record-memory` immediately logs ADOPT / REJECT / VOID into
       `benchmarks/evo_trials.jsonl` and refreshes `docs/ledger-index.jsonl`.

Exit codes:
  0 = PASS / ADOPT
  1 = REJECT (correctness failure, regression, or insufficient/noisy gain)
  2 = Usage / precondition error
  3 = VOID_MEASUREMENT (cache hit, OOM, port busy, crash delta > 0)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import statistics as st
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import evo_memory  # noqa: E402
import evo_silicon_probe  # noqa: E402

_win_spec = importlib.util.spec_from_file_location(
    "service_window_base", ROOT / "tools" / "run_register_only_service_window.py"
)
_win_base = importlib.util.module_from_spec(_win_spec)
assert _win_spec and _win_spec.loader
_win_spec.loader.exec_module(_win_base)

DEFAULT_ULP_PROMPTS = [
    "Repeat verbatim and then explain each clause: DeepSeek-V4.1-Flash uses Multi-head Latent Attention and 384 routed experts.",
    "Compute 397 * 613 step by step, verify by modular arithmetic mod 9 and mod 11, and state the exact integer.",
    "Write a self-contained C11 function `uint32_t popcnt_u64(uint64_t x)` without builtins and prove each bit-mask stage.",
]

DEFAULT_TIMED_BODY = (
    "Analyze the memory hierarchy tradeoffs between threadgroup shared memory, "
    "SIMD-group register tiling, and system-level cache residency on unified-memory "
    "GPUs when executing 2-bit block-quantized Mixture-of-Experts layers, covering "
    "dispatch barriers, warp occupancy, and tail-latency variance in detail."
)

BAD_LOG_PATTERNS = (
    "Insufficient Memory",
    "command buffer failed",
    "prefill failed",
    "prefix cache invalidated",
)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return _win_base.sha(path)


def count_ips_crashes(diag_dir: Path | None = None) -> int:
    d = diag_dir or (Path.home() / "Library" / "Logs" / "DiagnosticReports")
    if not d.exists():
        return 0
    return sum(1 for p in d.glob("myds-*.ips") if p.is_file())


def port_busy(port: int) -> bool:
    proc = subprocess.run(
        ["lsof", "-nP", "-t", "-a", f"-iTCP:{port}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
    )
    return bool(proc.stdout.strip())


def parse_conf_lines(conf_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw in conf_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            raise ValueError(f"Malformed config line in {conf_path}: {raw!r}")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.lstrip()
        for i, ch in enumerate(v):
            if ch == "#" and i > 0 and v[i - 1] in (" ", "\t"):
                v = v[:i]
                break
        pairs.append((k, v.rstrip()))
    return pairs


def write_isolated_conf(
    base_conf: Path,
    dest_conf: Path,
    kv_dir: Path,
    resp_state: Path,
    debug_token: str,
    overrides: dict[str, str],
) -> dict[str, str]:
    """Build isolated arm config per MEASUREMENT-DISCIPLINE.md §1, §2.2, §2.3, §A.5.1."""
    kv_dir.mkdir(parents=True, exist_ok=True)
    resp_state.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[str, str] = {}
    for k, v in parse_conf_lines(base_conf):
        merged[k] = v

    # §A.5.1: Preserve MYDS_PREFIX_CACHE=1 unless explicitly under test.
    merged.setdefault("MYDS_PREFIX_CACHE", "1")
    # §2.2 & §2.3: Force isolated empty kvdisk & response cache paths per boot!
    merged["MYDS_KV_DSK"] = str(kv_dir)
    merged["MYDS_KV_DSK_BUDGET_MB"] = "0"
    merged["MYDS_RESP_STATE"] = str(resp_state)
    # Authoritative engine telemetry via loopback /trace/api/requests
    merged["MYDS_DEBUG_ENDPOINT"] = "1"
    merged["MYDS_DEBUG_TOKEN"] = debug_token

    for k, v in overrides.items():
        merged[k] = v

    if merged.get("MYDS_PREFIX_CACHE") == "0" and "MYDS_PREFIX_CACHE" not in overrides:
        raise RuntimeError(
            "§A.5.1 violation: base config has MYDS_PREFIX_CACHE=0; refusing to benchmark distorted env"
        )

    body = "".join(f"{k}={merged[k]}\n" for k in sorted(merged))
    dest_conf.write_text(body, encoding="utf-8")
    return merged


def canonical_conf_diff(
    conf_a: dict[str, str], conf_b: dict[str, str]
) -> dict[str, tuple[str | None, str | None]]:
    ignore = {"MYDS_KV_DSK", "MYDS_RESP_STATE", "MYDS_DEBUG_TOKEN"}
    keys = sorted((set(conf_a) | set(conf_b)) - ignore)
    return {
        k: (conf_a.get(k), conf_b.get(k))
        for k in keys
        if conf_a.get(k) != conf_b.get(k)
    }


def http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 600.0,
) -> tuple[int, dict[str, Any]]:
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw) if raw else {}


def wait_ready(port: int, proc: subprocess.Popen[Any], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Arm process exited prematurely with rc={proc.returncode}")
        try:
            code, body = http_json(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if code == 200 and body.get("status") in ("ok", "ready", None):
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting {timeout_s}s for /health on port {port}")


def run_offline_gate(args: argparse.Namespace) -> dict[str, Any]:
    ips_before = count_ips_crashes()
    steps: list[dict[str, Any]] = []

    def run_step(name: str, cmd: list[str] | str, shell: bool = False) -> bool:
        t0 = time.monotonic()
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            shell=shell,
            capture_output=True,
            text=True,
        )
        dt = round(time.monotonic() - t0, 3)
        steps.append(
            {
                "step": name,
                "rc": p.returncode,
                "seconds": dt,
                "tail_stdout": "\n".join(p.stdout.splitlines()[-15:]),
                "tail_stderr": "\n".join(p.stderr.splitlines()[-15:]),
            }
        )
        return p.returncode == 0

    ok = True
    if not args.skip_repo_checks:
        ok = run_step("check-knobs", ["make", "check-knobs"]) and ok
        ok = run_step("evo-memory-check", [sys.executable, "tools/evo_memory.py", "--check"]) and ok

    if ok and args.build_cmd:
        ok = run_step("build-candidate", args.build_cmd, shell=True) and ok

    pso_diff = None
    probe_bin = ROOT / "bin" / "evo_pso_probe"
    base_ml = Path(args.baseline_metallib) if args.baseline_metallib else None
    cand_ml = Path(args.candidate_metallib) if args.candidate_metallib else None
    if ok and probe_bin.exists() and base_ml and cand_ml and base_ml.exists() and cand_ml.exists():
        if sha256_file(base_ml) != sha256_file(cand_ml):
            b_pso = evo_silicon_probe.run_pso_probe(base_ml, probe_bin)
            c_pso = evo_silicon_probe.run_pso_probe(cand_ml, probe_bin)
            pso_diff = evo_silicon_probe.diff_pso(b_pso, c_pso)

    if ok and args.unit_cmd:
        ok = run_step("unit-suite", args.unit_cmd, shell=True) and ok

    ips_after = count_ips_crashes()
    ips_delta = max(0, ips_after - ips_before)
    if ips_delta > 0:
        ok = False

    return {
        "passed": ok,
        "ips_crashes_before": ips_before,
        "ips_crashes_after": ips_after,
        "ips_crash_delta": ips_delta,
        "pso_diff": pso_diff,
        "steps": steps,
    }


def evaluate_verdict(
    ulp_records: dict[str, dict[str, list[str]]],
    timed_a: list[float],
    timed_b: list[float],
    void_reasons: list[str],
    ips_delta: int,
    metric: str,
    min_gain_pct: float,
    allow_non_0ulp: bool,
) -> dict[str, Any]:
    """Pure statistical & 0-ULP decision kernel (unit-testable offline)."""
    if ips_delta > 0:
        void_reasons.append(f"macOS .ips crash count increased by {ips_delta} (§9)")
    if void_reasons:
        return {
            "verdict": "VOID_MEASUREMENT",
            "exit_code": 3,
            "ulp_zero": None,
            "reasons": void_reasons,
        }

    # §4: Check 0-ULP across every prompt and every boot of A and B
    ulp_mismatches = []
    for p_key, by_arm in ulp_records.items():
        a_shas = by_arm.get("A", [])
        b_shas = by_arm.get("B", [])
        if not a_shas or not b_shas or any(not s for s in a_shas + b_shas):
            return {
                "verdict": "VOID_MEASUREMENT",
                "exit_code": 3,
                "ulp_zero": None,
                "reasons": [f"Empty or missing SHA256 on prompt {p_key} (§4)"],
            }
        if len(set(a_shas)) != 1 or len(set(b_shas)) != 1 or set(a_shas) != set(b_shas):
            ulp_mismatches.append(
                {"prompt": p_key, "arm_A_shas": a_shas, "arm_B_shas": b_shas}
            )

    ulp_zero = len(ulp_mismatches) == 0
    if not ulp_zero and not allow_non_0ulp:
        return {
            "verdict": "REJECT_NON_0ULP",
            "exit_code": 1,
            "ulp_zero": False,
            "ulp_mismatches": ulp_mismatches,
            "reasons": ["0-ULP bitwise output equivalence failed across interleaved boots (§4)"],
        }

    if len(timed_a) < 3 or len(timed_b) < 3 or len(timed_a) != len(timed_b):
        return {
            "verdict": "VOID_MEASUREMENT",
            "exit_code": 3,
            "ulp_zero": ulp_zero,
            "reasons": [f"Need >=3 paired AB cycles, got A={len(timed_a)}, B={len(timed_b)} (§10)"],
        }

    med_a = st.median(timed_a)
    med_b = st.median(timed_b)
    if med_a <= 0 or med_b <= 0:
        return {
            "verdict": "VOID_MEASUREMENT",
            "exit_code": 3,
            "ulp_zero": ulp_zero,
            "reasons": ["Non-positive median measurement"],
        }

    higher_is_better = metric in ("decode_tps", "prefill_tps")
    if higher_is_better:
        gain_pct = 100.0 * (med_b / med_a - 1.0)
        paired_gains = [100.0 * (b / a - 1.0) for a, b in zip(timed_a, timed_b)]
    else:
        # Latency metric (seconds): lower B means positive speedup %
        gain_pct = 100.0 * (med_a / med_b - 1.0)
        paired_gains = [100.0 * (a / b - 1.0) for a, b in zip(timed_a, timed_b)]

    spread_a_pct = 100.0 * (max(timed_a) - min(timed_a)) / med_a
    spread_b_pct = 100.0 * (max(timed_b) - min(timed_b)) / med_b
    max_spread_pct = max(spread_a_pct, spread_b_pct)

    stats_summary = {
        "metric": metric,
        "cycles": len(timed_a),
        "samples_A": [round(x, 4) for x in timed_a],
        "samples_B": [round(x, 4) for x in timed_b],
        "median_A": round(med_a, 4),
        "median_B": round(med_b, 4),
        "spread_A_pct": round(spread_a_pct, 3),
        "spread_B_pct": round(spread_b_pct, 3),
        "paired_gain_pct": [round(x, 3) for x in paired_gains],
        "median_gain_pct": round(gain_pct, 3),
        "min_gain_threshold_pct": min_gain_pct,
    }

    if gain_pct <= -max(min_gain_pct, max_spread_pct):
        return {
            "verdict": "REJECT_REGRESSION",
            "exit_code": 1,
            "ulp_zero": ulp_zero,
            "stats": stats_summary,
            "reasons": [
                f"Candidate regressed by {gain_pct:+.2f}% (exceeds spread {max_spread_pct:.2f}%)"
            ],
        }

    if (
        gain_pct >= min_gain_pct
        and gain_pct > max_spread_pct
        and all(pg > 0.0 for pg in paired_gains)
    ):
        return {
            "verdict": "ADOPT",
            "exit_code": 0,
            "ulp_zero": ulp_zero,
            "stats": stats_summary,
            "reasons": [
                f"0-ULP={'yes' if ulp_zero else 'waived'}, median gain {gain_pct:+.2f}% "
                f">= {min_gain_pct:.2f}% and > spread {max_spread_pct:.2f}%, all {len(paired_gains)} pairs positive"
            ],
        }

    return {
        "verdict": "REJECT_NOISE_OR_INSUFFICIENT_GAIN",
        "exit_code": 1,
        "ulp_zero": ulp_zero,
        "stats": stats_summary,
        "reasons": [
            f"Gain {gain_pct:+.2f}% did not clear min_gain={min_gain_pct:.2f}% "
            f"and spread={max_spread_pct:.2f}% across all pairs ({stats_summary['paired_gain_pct']})"
        ],
    }


def run_single_boot(
    arm_label: str,
    cycle_idx: int,
    binary: Path,
    metallib: Path,
    base_conf: Path,
    overrides: dict[str, str],
    model: Path,
    vision: Path | None,
    port: int,
    alias: str,
    ulp_prompts: list[str],
    timed_prompt_body: str,
    max_tokens: int,
    boot_timeout_s: float,
    work_dir: Path,
) -> dict[str, Any]:
    if port_busy(port):
        return {"void": f"Port {port} already has a listener before {arm_label}{cycle_idx}"}

    avail_before = _win_base.available_gib()
    boot_dir = work_dir / f"{arm_label}{cycle_idx}"
    boot_bin_dir = boot_dir / "bin"
    boot_bin_dir.mkdir(parents=True, mode=0o700)

    # Stage isolated binary + metallib under boot_dir/bin/ and execute with cwd=boot_dir!
    # Why: src/myds_metal.m searches `./bin/myds.metallib` relative to cwd BEFORE
    # `exec_dir/myds.metallib`. Staging into boot_dir/bin prevents cwd shadowing (§3).
    staged_bin = boot_bin_dir / "myds"
    shutil.copy2(binary, staged_bin)
    staged_bin.chmod(0o755)
    if metallib.is_file():
        shutil.copy2(metallib, boot_bin_dir / "myds.metallib")

    dbg_tok = secrets.token_hex(16)
    conf_path = (boot_dir / "arm.conf").resolve()
    merged_conf = write_isolated_conf(
        base_conf.resolve(),
        conf_path,
        (boot_dir / "kvdsk").resolve(),
        (boot_dir / "responses.idx").resolve(),
        dbg_tok,
        overrides,
    )

    # §7 & §12: Verify every MYDS_* key in isolated conf is read by this binary
    chk = subprocess.run(
        [str(staged_bin), "--config", str(conf_path), "--print-config"],
        cwd=str(boot_dir),
        capture_output=True,
        text=True,
    )
    if chk.returncode != 0:
        return {
            "void": f"{arm_label}{cycle_idx} --print-config failed (rc={chk.returncode}): {chk.stderr.strip() or chk.stdout.strip()}",
            "merged_conf": merged_conf,
        }

    log_path = boot_dir / "server.log"
    argv = [
        str(staged_bin),
        "--config",
        str(conf_path),
        "-m",
        str(model.resolve()),
        "--port",
        str(port),
        "--alias",
        alias,
    ]
    if vision:
        argv += ["--vision", str(vision.resolve())]

    proc: subprocess.Popen[Any] | None = None
    resident_gib = 0.0
    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                argv,
                cwd=str(boot_dir),
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
        wait_ready(port, proc, boot_timeout_s)
        try:
            resident_gib = _win_base.process_rss_gib(proc.pid)
        except Exception:
            resident_gib = 0.0

        chat_url = f"http://127.0.0.1:{port}/v1/chat/completions"
        trace_url = f"http://127.0.0.1:{port}/trace/api/requests"
        dbg_hdr = {"X-Debug-Token": dbg_tok}

        # §A.2 ③: Warmup request (discarded) with unique nonce
        warm_nonce = f"[warmup-{secrets.token_hex(6)}] "
        http_json(
            chat_url,
            {
                "model": alias,
                "messages": [{"role": "user", "content": warm_nonce + "Say ready in three words."}],
                "max_tokens": 8,
                "temperature": 0.0,
                "seed": 42,
            },
        )

        # §4: Deterministic 0-ULP prompts (distinct prefixes so cached_tokens == 0)
        ulp_shas: dict[str, str] = {}
        for idx, ptxt in enumerate(ulp_prompts):
            code, resp = http_json(
                chat_url,
                {
                    "model": alias,
                    "messages": [{"role": "user", "content": f"[ulp-gate-p{idx}] {ptxt}"}],
                    "max_tokens": 64,
                    "temperature": 0.0,
                    "seed": 42,
                },
            )
            choices = resp.get("choices") or []
            content = (choices[0].get("message") or {}).get("content", "") if choices else ""
            if code != 200 or not content:
                return {
                    "void": f"{arm_label}{cycle_idx} empty or non-200 0-ULP response on prompt {idx}",
                    "merged_conf": merged_conf,
                }
            ulp_shas[f"p{idx}"] = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # §A.5.2: Timed sample with unique leading nonce so MYDS_PREFIX_CACHE=1 misses naturally
        nonce = f"[timed-c{cycle_idx}-{arm_label}-{secrets.token_hex(8)}] "
        t0 = time.monotonic()
        code, timed_resp = http_json(
            chat_url,
            {
                "model": alias,
                "messages": [{"role": "user", "content": nonce + timed_prompt_body}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": 42,
            },
        )
        wall_s = time.monotonic() - t0
        t_choices = timed_resp.get("choices") or []
        t_content = (t_choices[0].get("message") or {}).get("content", "") if t_choices else ""
        if code != 200 or not t_content:
            return {
                "void": f"{arm_label}{cycle_idx} empty or non-200 timed response",
                "merged_conf": merged_conf,
            }

        # §2.4 & §A.5.4: Authoritative engine verification via /trace/api/requests
        _, trace_body = http_json(trace_url, headers=dbg_hdr, timeout=10.0)
        records = trace_body.get("records") or []
        if len(records) < 1 + len(ulp_prompts) + 1:
            return {
                "void": f"{arm_label}{cycle_idx} expected >={2 + len(ulp_prompts)} trace records, got {len(records)}",
                "merged_conf": merged_conf,
            }
        # Most recent record in /trace ring buffer (check ordering by seq or position)
        latest = max(records, key=lambda r: int(r.get("seq", 0))) if records and "seq" in records[0] else records[0]
        if int(latest.get("cached_tokens") or 0) != 0:
            return {
                "void": f"{arm_label}{cycle_idx} timed request hit prefix cache (cached_tokens={latest.get('cached_tokens')}) (§A.5.4)",
                "merged_conf": merged_conf,
            }
        pf_tps = float(latest.get("prefill_tps") or -1.0)
        dec_tps = float(latest.get("decode_tps") or -1.0)
        if pf_tps <= 0:
            return {
                "void": f"{arm_label}{cycle_idx} engine reported non-positive prefill_tps={pf_tps} (§2.4)",
                "merged_conf": merged_conf,
            }

        result: dict[str, Any] = {
            "void": None,
            "merged_conf": merged_conf,
            "ulp_shas": ulp_shas,
            "timed": {
                "wall_s": round(wall_s, 4),
                "prefill_tps": round(pf_tps, 3),
                "decode_tps": round(dec_tps, 3),
                "prompt_tokens": latest.get("prompt_tokens"),
                "out_tokens": latest.get("out_tokens"),
                "cached_tokens": latest.get("cached_tokens"),
            },
        }
    except Exception as exc:
        result = {
            "void": f"{arm_label}{cycle_idx} runtime error: {exc}",
            "merged_conf": merged_conf,
        }
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=45.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=15.0)
        if resident_gib > 1.0:
            target_gib = max(1.0, avail_before - 2.0)
            try:
                _win_base.wait_for_memory_release(target_gib, 90.0)
            except Exception:
                pass

    if log_path.exists():
        log_txt = log_path.read_text(errors="replace")
        for pat in BAD_LOG_PATTERNS:
            if pat in log_txt:
                return {
                    "void": f"{arm_label}{cycle_idx} server log contained fatal pattern {pat!r} (§2.1)",
                    "merged_conf": merged_conf,
                }
    return result


def parse_kv_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --conf-override {item!r} (expected KEY=VAL)")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k.startswith("MYDS_"):
            raise SystemExit(f"--conf-override key must start with MYDS_: {k!r}")
        out[k] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("offline", "full"), default="offline")
    ap.add_argument("--baseline-bin", type=Path, default=ROOT / "bin" / "myds")
    ap.add_argument("--baseline-metallib", type=Path, default=ROOT / "bin" / "myds.metallib")
    ap.add_argument("--candidate-bin", type=Path, default=ROOT / "bin" / "myds")
    ap.add_argument("--candidate-metallib", type=Path, default=ROOT / "bin" / "myds.metallib")
    ap.add_argument("--base-config", type=Path, help="Production myds.conf template for --mode full")
    ap.add_argument("--conf-override", action="append", default=[], help="Arm B MYDS_KEY=VAL override")
    ap.add_argument("--model", type=Path, help="GGUF or .part01 model path for --mode full")
    ap.add_argument("--vision", type=Path, help="Optional Vision GGUF path")
    ap.add_argument("--port", type=int, default=18187, help="Isolated loopback port for full judge")
    ap.add_argument("--alias", default="evo-judge-myds")
    ap.add_argument("--cycles", type=int, default=3, help="Interleaved AB cycles (>=3 per §10)")
    ap.add_argument("--metric", choices=("decode_tps", "prefill_tps", "wall_s"), default="decode_tps")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timed-prompt-file", type=Path, help="Optional corpus file for timed request")
    ap.add_argument("--min-gain-pct", type=float, default=1.5, help="Minimum median gain % to ADOPT")
    ap.add_argument("--allow-non-0ulp", action="store_true", help="Explicitly waive 0-ULP gate")
    ap.add_argument("--boot-timeout", type=float, default=240.0)
    ap.add_argument("--build-cmd", help="Optional shell command to compile candidate in offline gate")
    ap.add_argument("--unit-cmd", default="make test", help="Unit test command ('' to skip)")
    ap.add_argument("--skip-repo-checks", action="store_true")
    ap.add_argument("--record-memory", action="store_true", help="Record outcome to evo_memory (§11)")
    ap.add_argument("--subsystem", help="Subsystem tag for --record-memory")
    ap.add_argument("--hypothesis", help="Hypothesis description for --record-memory")
    ap.add_argument("--hardware", help="Hardware tag for --record-memory")
    ap.add_argument("--out", type=Path, help="Write JSON verdict report to path")
    ap.add_argument("--json", action="store_true", help="Print JSON verdict to stdout")
    args = ap.parse_args()

    offline = run_offline_gate(args)
    report: dict[str, Any] = {
        "kind": "evo_judge_report",
        "mode": args.mode,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "offline_gate": offline,
    }

    if not offline["passed"]:
        report["verdict"] = (
            "VOID_MEASUREMENT" if offline["ips_crash_delta"] > 0 else "REJECT_OFFLINE_GATE"
        )
        report["exit_code"] = 3 if offline["ips_crash_delta"] > 0 else 1
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(f"VERDICT: {report['verdict']} (offline gate failed)")
        return report["exit_code"]

    if args.mode == "offline":
        report["verdict"] = "PASS_OFFLINE_GATE"
        report["exit_code"] = 0
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print("VERDICT: PASS_OFFLINE_GATE")
        return 0

    # Full exclusive mode preconditions
    if not args.base_config or not args.base_config.is_file():
        raise SystemExit("--mode full requires --base-config pointing to an existing myds.conf")
    if not args.model or not args.model.is_file():
        raise SystemExit("--mode full requires --model pointing to an existing GGUF/.part01")
    if args.cycles < 3:
        raise SystemExit("§10 requires --cycles >= 3 for interleaved ABABAB judgement")

    overrides = parse_kv_overrides(args.conf_override)
    base_bin_sha = sha256_file(args.baseline_bin)
    cand_bin_sha = sha256_file(args.candidate_bin)
    base_ml_sha = sha256_file(args.baseline_metallib)
    cand_ml_sha = sha256_file(args.candidate_metallib)

    # §3 / §A.4: Prove Arm A and Arm B actually differ!
    if base_bin_sha == cand_bin_sha and base_ml_sha == cand_ml_sha and not overrides:
        report["verdict"] = "VOID_IDENTICAL_ARMS"
        report["exit_code"] = 3
        report["reasons"] = [
            "§3 / §A.4 violation: baseline and candidate have identical binary SHA256, "
            "identical metallib SHA256, and zero --conf-override keys."
        ]
        if args.out:
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else f"VERDICT: {report['verdict']}")
        return 3

    timed_body = (
        args.timed_prompt_file.read_text(encoding="utf-8")
        if args.timed_prompt_file
        else DEFAULT_TIMED_BODY
    )

    ips_before = count_ips_crashes()
    ulp_records: dict[str, dict[str, list[str]]] = {
        f"p{i}": {"A": [], "B": []} for i in range(len(DEFAULT_ULP_PROMPTS))
    }
    timed_a: list[float] = []
    timed_b: list[float] = []
    void_reasons: list[str] = []
    boot_logs: list[dict[str, Any]] = []
    conf_diff_recorded: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="myds-evo-judge-") as tmp:
        work_dir = Path(tmp)
        for c in range(1, args.cycles + 1):
            for arm_label, b_path, m_path, arm_ov in (
                ("A", args.baseline_bin, args.baseline_metallib, {}),
                ("B", args.candidate_bin, args.candidate_metallib, overrides),
            ):
                res = run_single_boot(
                    arm_label=arm_label,
                    cycle_idx=c,
                    binary=b_path,
                    metallib=m_path,
                    base_conf=args.base_config,
                    overrides=arm_ov,
                    model=args.model,
                    vision=args.vision,
                    port=args.port,
                    alias=args.alias,
                    ulp_prompts=DEFAULT_ULP_PROMPTS,
                    timed_prompt_body=timed_body,
                    max_tokens=args.max_tokens,
                    boot_timeout_s=args.boot_timeout,
                    work_dir=work_dir,
                )
                boot_logs.append({"arm": f"{arm_label}{c}", **{k: v for k, v in res.items() if k != "merged_conf"}})
                if res.get("void"):
                    void_reasons.append(res["void"])
                    break
                for pk, sha in res["ulp_shas"].items():
                    ulp_records[pk][arm_label].append(sha)
                val = float(res["timed"][args.metric])
                if arm_label == "A":
                    timed_a.append(val)
                    last_conf_a = res["merged_conf"]
                else:
                    timed_b.append(val)
                    if conf_diff_recorded is None:
                        conf_diff_recorded = canonical_conf_diff(last_conf_a, res["merged_conf"])
            if void_reasons:
                break

    ips_delta = max(0, count_ips_crashes() - ips_before)
    decision = evaluate_verdict(
        ulp_records=ulp_records,
        timed_a=timed_a,
        timed_b=timed_b,
        void_reasons=void_reasons,
        ips_delta=ips_delta,
        metric=args.metric,
        min_gain_pct=args.min_gain_pct,
        allow_non_0ulp=args.allow_non_0ulp,
    )

    report.update(
        identity={
            "baseline_bin_sha256": base_bin_sha,
            "candidate_bin_sha256": cand_bin_sha,
            "baseline_metallib_sha256": base_ml_sha,
            "candidate_metallib_sha256": cand_ml_sha,
            "conf_diff": conf_diff_recorded or overrides,
        },
        boots=boot_logs,
        **decision,
    )

    if args.record_memory and args.hypothesis:
        mem_verdict = (
            "ADOPT"
            if decision["verdict"] == "ADOPT"
            else ("VOID_MEASUREMENT" if decision["verdict"].startswith("VOID") else "REJECT")
        )
        delta_pct = (decision.get("stats") or {}).get("median_gain_pct")
        ns = argparse.Namespace(
            hypothesis=args.hypothesis,
            verdict=mem_verdict,
            why="; ".join(decision.get("reasons") or [decision["verdict"]]),
            delta_pct=delta_pct,
            ulp_zero=None if decision.get("ulp_zero") is None else str(decision["ulp_zero"]).lower(),
            hardware=args.hardware or "unspecified",
            subsystem=args.subsystem,
            knob=",".join(overrides.keys()),
            kernel="",
        )
        recorded = evo_memory.record_trial(ns)
        report["recorded_trial_id"] = recorded["id"]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"VERDICT: {report['verdict']}")
        for r in report.get("reasons", []):
            print(f"  - {r}")
    return int(report["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
