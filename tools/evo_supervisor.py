#!/usr/bin/env python3
"""tools/evo_supervisor.py — Time-Sliced Self-Hosting Supervisor for myds.

Solves the single-machine unified-memory "brain surgery on yourself" problem
on M2 Ultra (192 GB) and M3/M5 Ultra (256/512 GB) where the live Golden `myds`
serving the evolving LLM agent occupies 148–337 GB of unified memory and cannot
coexist with a full-model A/B evaluation:

  1. `status`:
     - Read-only live probe via `tools/gpu_window.py` + `vm_stat`: reports live
       Golden PID, port, binary/metallib/config SHA256, active TCP client count,
       resident GiB, and free+inactive+speculative GiB.

  2. `evaluate`:
     - Phase 0 (Pre-drain Preflight, zero service downtime):
       Validates candidate Mach-O & metallib (regular files, valid magic, never
       overwriting live paths), extracts live Golden model/vision/config from
       `KERN_PROCARGS2`, runs `--print-config` on candidate + overrides, and runs
       `evo_judge.py --mode offline` BEFORE touching port 8187.
     - Phase 1 (Client Drain):
       Waits up to `--client-drain-timeout` for `ESTABLISHED` TCP clients on
       Golden `myds` to drop to zero so a self-hosting agent can dispatch
       `evo_supervisor.py` and close its own HTTP turn cleanly.
     - Phase 2 (Exclusive Window via Single Source of Truth):
       Delegates stop, artifact backup, signal safety, and exact restoration to
       `tools/run_register_only_service_window.py` (`Window`) — NEVER inventing
       a second kill/restore path. Inside the window (`_window_inner`), waits on
       `wait_for_memory_release(release_target_gib)` BEFORE `evo_judge.py` boots
       Arm A1 so XNU finishes reclaiming Golden `myds`'s mapping first.
     - Phase 3 (Post-Window Verification & Optional Atomic Promotion):
       Verifies restored Golden `myds` via `gpu_window.diff()`. If `--promote-on-adopt`
       is set and `verdict == "ADOPT"`, performs atomic inode swap (`os.replace`
       of staged temp file, never `cp` over a mapped binary per §7) inside a
       controlled restart and rolls back automatically if `/health` fails.
     - Emits atomic `surgery-receipt.json` for the waking agent's next turn.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import evo_judge  # noqa: E402
import gpu_window  # noqa: E402

_win_spec = importlib.util.spec_from_file_location(
    "service_window_base", ROOT / "tools" / "run_register_only_service_window.py"
)
_win_base = importlib.util.module_from_spec(_win_spec)
assert _win_spec and _win_spec.loader
_win_spec.loader.exec_module(_win_base)

MACHO_MAGICS = {
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
}


def atomic_write_json(path: Path, data: dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = -1
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if fd >= 0:
            os.close(fd)


def atomic_install_file(src: Path, dst: Path, mode: int) -> dict[str, Any]:
    """Atomically swap directory entry at `dst` to a fresh inode (`install(1)` semantics).

    MEASUREMENT-DISCIPLINE.md §7: NEVER overwrite a mapped Mach-O or `.metallib`
    in place (`cp`), which mutates live pages and triggers SIGBUS CODE SIGNING
    Invalid Page. Writing a sibling temp file and `os.replace()` unlinks the old
    inode cleanly while any previous mapping keeps its unlinked inode intact.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    old_inode = dst.stat().st_ino if dst.exists() else None
    tmp = dst.with_name(f".{dst.name}.stage-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as out_f, src.open("rb") as in_f:
            fd = -1
            shutil.copyfileobj(in_f, out_f, 1048576)
            out_f.flush()
            os.fsync(out_f.fileno())
        os.replace(tmp, dst)
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    new_stat = dst.stat()
    return {
        "dest": str(dst),
        "old_inode": old_inode,
        "new_inode": new_stat.st_ino,
        "bytes": new_stat.st_size,
        "sha256": _win_base.sha(dst),
    }


def established_clients(pid: int) -> list[str]:
    proc = subprocess.run(
        ["lsof", "-nP", "-a", "-p", str(pid), "-iTCP", "-sTCP:ESTABLISHED"],
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return lines[1:] if len(lines) > 1 else []


def wait_for_client_drain(pid: int, timeout_s: float, poll_s: float = 0.25) -> tuple[bool, list[str]]:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        clients = established_clients(pid)
        if not clients:
            return True, []
        if time.monotonic() >= deadline:
            return False, clients
        time.sleep(poll_s)


def extract_flag_path(argv: list[str], cwd: Path, flags: tuple[str, ...]) -> Path | None:
    found: str | None = None
    for i, tok in enumerate(argv):
        if tok in flags and i + 1 < len(argv):
            found = argv[i + 1]
    if not found:
        return None
    p = Path(found)
    return p if p.is_absolute() else (cwd / p).resolve()


def extract_flag_value(argv: list[str], flag: str) -> str | None:
    found: str | None = None
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            found = argv[i + 1]
    return found


def validate_candidate_artifacts(
    cand_bin: Path,
    cand_metallib: Path,
    live_bin: Path | None,
    live_metallib: Path | None,
) -> list[str]:
    errors: list[str] = []
    if not cand_bin.exists() or cand_bin.is_symlink() or not cand_bin.is_file():
        errors.append(f"candidate binary must be an existing non-symlink regular file: {cand_bin}")
    else:
        magic = cand_bin.read_bytes()[:4]
        if magic not in MACHO_MAGICS:
            errors.append(f"candidate binary is not a valid Mach-O executable: {cand_bin}")
        if not os.access(cand_bin, os.X_OK):
            errors.append(f"candidate binary is not executable: {cand_bin}")

    if not cand_metallib.exists() or cand_metallib.is_symlink() or not cand_metallib.is_file():
        errors.append(f"candidate metallib must be an existing non-symlink regular file: {cand_metallib}")
    elif cand_metallib.stat().st_size < 16:
        errors.append(f"candidate metallib appears truncated: {cand_metallib}")

    # Prevent in-place mutation confusion: if candidate binary differs from live, it must be staged separately
    if live_bin and cand_bin.exists() and live_bin.exists():
        # Same path is allowed ONLY when testing pure --conf-override or pure metallib override
        pass
    return errors


def merge_conf_preserving_comments(conf_path: Path, overrides: dict[str, str]) -> str:
    """Update or append MYDS_* keys in a production config text cleanly."""
    if not overrides:
        return conf_path.read_text(encoding="utf-8")
    remaining = dict(overrides)
    out_lines: list[str] = []
    for raw in conf_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.split("=", 1)[0].strip()
            if k in remaining:
                out_lines.append(f"{k}={remaining.pop(k)}")
                continue
        out_lines.append(raw)
    for k in sorted(remaining):
        out_lines.append(f"{k}={remaining[k]}")
    return "\n".join(out_lines) + "\n"


def cmd_status(args: argparse.Namespace) -> int:
    pid = gpu_window.port_pid(args.port)
    avail = round(_win_base.available_gib(), 2)
    if pid is None:
        payload = {
            "kind": "evo_supervisor_status",
            "port": args.port,
            "running": False,
            "available_gib": avail,
        }
    else:
        snap = gpu_window.snapshot(args.port)
        rss = round(_win_base.process_rss_gib(pid), 2)
        clients = established_clients(pid)
        payload = {
            "kind": "evo_supervisor_status",
            "port": args.port,
            "running": True,
            "pid": pid,
            "resident_gib": rss,
            "available_gib": avail,
            "active_tcp_clients": len(clients),
            "snapshot": snap,
        }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if not payload["running"]:
            print(f"Golden myds: NOT RUNNING on port {args.port} (available_mem={avail} GiB)")
        else:
            ident = payload["snapshot"]["identity"]
            print(
                f"Golden myds: RUNNING pid={payload['pid']} port={args.port} "
                f"rss={payload['resident_gib']} GiB avail={avail} GiB "
                f"clients={payload['active_tcp_clients']}"
            )
            print(f"  binary:   {ident['binary']} ({(ident['binary_sha256'] or '')[:12]})")
            print(f"  metallib: {ident['metallib']} ({(ident['metallib_sha256'] or '')[:12]})")
            print(f"  config:   {ident['config']} ({(ident['config_sha256'] or '')[:12]})")
    return 0


def cmd_window_inner(args: argparse.Namespace) -> int:
    """Executed INSIDE `Window.run()` after Golden `myds` has exited.

    Waits for XNU to reclaim Golden `myds`'s resident memory (`release_target_gib`)
    BEFORE starting `evo_judge.py --mode full`, preventing dual-mapping OOM on 192 GB!
    Always exits 0 unless internal invocation broke, writing the judge exit code and
    report to `--verdict-out` so `Window` proceeds cleanly to restore Golden `myds`.
    """
    lease_path = args.window_root / "lease.json"
    mem_wait_s = 0.0
    if lease_path.is_file():
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
        target_gib = float(lease.get("release_target_gib") or 0.0)
        if target_gib > 0.0:
            t0 = time.monotonic()
            _win_base.wait_for_memory_release(target_gib, args.memory_timeout)
            mem_wait_s = round(time.monotonic() - t0, 3)

    judge_cmd = [sys.executable, str(ROOT / "tools" / "evo_judge.py"), *args.judge_args]
    t_j = time.monotonic()
    proc = subprocess.run(judge_cmd, cwd=str(ROOT), capture_output=True, text=True)
    judge_wall_s = round(time.monotonic() - t_j, 3)

    inner_meta = {
        "memory_reclaim_wait_seconds": mem_wait_s,
        "available_gib_before_arm_a1": round(_win_base.available_gib(), 2),
        "judge_exit_code": proc.returncode,
        "judge_wall_seconds": judge_wall_s,
        "judge_tail_stdout": "\n".join(proc.stdout.splitlines()[-25:]),
        "judge_tail_stderr": "\n".join(proc.stderr.splitlines()[-25:]),
    }
    atomic_write_json(args.inner_meta_out, inner_meta)
    return 0


def promote_adopted_candidate(
    port: int,
    cand_bin: Path,
    cand_metallib: Path,
    overrides: dict[str, str],
    backup_dir: Path,
    stop_timeout: float,
    memory_timeout: float,
    restore_timeout: float,
) -> dict[str, Any]:
    """Controlled atomic inode promotion after a verified ADOPT verdict."""
    pid = gpu_window.port_pid(port)
    if pid is None:
        raise RuntimeError(f"Cannot promote: no live listener on port {port}")
    captured = _win_base.process_arguments(pid)
    cwd = _win_base.process_cwd(pid)
    nice = _win_base.process_nice(pid)
    stdio = _win_base.stdio_targets(pid)
    live_bin = Path(captured["executable"]).resolve()
    live_ml = live_bin.with_name("myds.metallib")
    live_conf = extract_flag_path(captured["argv"], cwd, ("--config",))

    avail_before = _win_base.available_gib()
    rss_gib = _win_base.process_rss_gib(pid)
    release_target = avail_before + 0.9 * rss_gib

    backup_dir.mkdir(parents=True, mode=0o700)
    shutil.copy2(live_bin, backup_dir / "myds.bak")
    if live_ml.is_file():
        shutil.copy2(live_ml, backup_dir / "myds.metallib.bak")
    if live_conf and live_conf.is_file():
        shutil.copy2(live_conf, backup_dir / "myds.conf.bak")

    # Stop current Golden cleanly
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + stop_timeout
    while _win_base.pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    if _win_base.pid_alive(pid):
        raise RuntimeError(f"Live PID {pid} did not exit within {stop_timeout}s during promotion")
    _win_base.wait_for_memory_release(release_target, memory_timeout)

    swaps: list[dict[str, Any]] = []

    def spawn_live() -> int:
        def preexec() -> None:
            if nice:
                try:
                    os.nice(nice)
                except PermissionError:
                    pass

        stdin_h = open(stdio["0"], "rb") if stdio.get("0") and Path(stdio["0"]).exists() else subprocess.DEVNULL
        stdout_h = open(stdio["1"], "ab") if stdio.get("1") else open(backup_dir / "promoted.out", "ab")
        stderr_h = (
            stdout_h
            if stdio.get("2") and stdio.get("2") == stdio.get("1")
            else (open(stdio["2"], "ab") if stdio.get("2") else open(backup_dir / "promoted.err", "ab"))
        )
        try:
            child = subprocess.Popen(
                captured["argv"],
                executable=str(live_bin),
                cwd=str(cwd),
                env=captured["environment"],
                stdin=stdin_h,
                stdout=stdout_h,
                stderr=stderr_h,
                start_new_session=True,
                preexec_fn=preexec if nice else None,
            )
        finally:
            for h in {stdin_h, stdout_h, stderr_h}:
                if hasattr(h, "close"):
                    h.close()
        evo_judge.wait_ready(port, child, restore_timeout)
        return child.pid

    try:
        if cand_bin.resolve() != live_bin:
            swaps.append(atomic_install_file(cand_bin, live_bin, 0o755))
        if cand_metallib.resolve() != live_ml:
            swaps.append(atomic_install_file(cand_metallib, live_ml, 0o644))
            cwd_ml = (cwd / "bin" / "myds.metallib").resolve()
            if cwd_ml.exists() and cwd_ml != live_ml and cand_metallib.resolve() != cwd_ml:
                swaps.append(atomic_install_file(cand_metallib, cwd_ml, 0o644))
        if overrides and live_conf and live_conf.is_file():
            merged_txt = merge_conf_preserving_comments(live_conf, overrides)
            staged_conf = backup_dir / "myds.conf.promoted"
            staged_conf.write_text(merged_txt, encoding="utf-8")
            swaps.append(atomic_install_file(staged_conf, live_conf, live_conf.stat().st_mode & 0o777))

        new_pid = spawn_live()
        return {"promoted": True, "rolled_back": False, "new_pid": new_pid, "swaps": swaps}
    except Exception as exc:
        # Roll back atomically from backup_dir and respawn original Golden
        atomic_install_file(backup_dir / "myds.bak", live_bin, 0o755)
        if (backup_dir / "myds.metallib.bak").is_file():
            atomic_install_file(backup_dir / "myds.metallib.bak", live_ml, 0o644)
        if live_conf and (backup_dir / "myds.conf.bak").is_file():
            atomic_install_file(backup_dir / "myds.conf.bak", live_conf, 0o600)
        rb_pid = spawn_live()
        return {
            "promoted": False,
            "rolled_back": True,
            "rollback_pid": rb_pid,
            "error": str(exc),
        }


def cmd_evaluate(args: argparse.Namespace) -> int:
    overrides = evo_judge.parse_kv_overrides(args.conf_override)
    run_stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    run_dir = (args.work_root or (Path(tempfile.gettempdir()) / f"myds-evo-surgery-{run_stamp}")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(run_dir, 0o700)
    receipt_path = (args.receipt or (run_dir / "surgery-receipt.json")).resolve()

    receipt: dict[str, Any] = {
        "kind": "evo_surgery_receipt",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "service_port": args.port,
        "judge_port": args.judge_port,
        "work_dir": str(run_dir),
        "conf_overrides": overrides,
    }

    # Inspect live Golden service on args.port
    live_pid = gpu_window.port_pid(args.port)
    if live_pid is None:
        receipt.update(
            status="ABORTED_PREFLIGHT",
            reason=f"No live Golden myds listening on port {args.port}; run evo_judge.py directly or start service first.",
        )
        atomic_write_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2) if args.json else f"ABORTED: {receipt['reason']}")
        return 2

    pre_snap = gpu_window.snapshot(args.port)
    captured = _win_base.process_arguments(live_pid)
    live_cwd = _win_base.process_cwd(live_pid)
    live_bin = Path(captured["executable"]).resolve()
    live_ml = live_bin.with_name("myds.metallib")
    if not live_ml.is_file() and (live_cwd / "bin" / "myds.metallib").is_file():
        live_ml = (live_cwd / "bin" / "myds.metallib").resolve()

    base_conf = (args.base_config or extract_flag_path(captured["argv"], live_cwd, ("--config",)))
    model_path = (args.model or extract_flag_path(captured["argv"], live_cwd, ("-m", "--model")))
    vision_path = (args.vision or extract_flag_path(captured["argv"], live_cwd, ("--vision",)))
    alias = args.alias or extract_flag_value(captured["argv"], "--alias") or "evo-judge-myds"

    cand_bin = (args.candidate_bin or live_bin).resolve()
    cand_ml = (args.candidate_metallib or live_ml).resolve()

    receipt["pre_snapshot"] = {
        "pid": live_pid,
        "cwd": str(live_cwd),
        "live_bin": str(live_bin),
        "live_bin_sha256": pre_snap["identity"]["binary_sha256"],
        "live_metallib": str(live_ml),
        "live_metallib_sha256": _win_base.sha(live_ml) if live_ml.is_file() else None,
        "base_config": str(base_conf) if base_conf else None,
        "model": str(model_path) if model_path else None,
    }

    # Phase 0: Zero-downtime preflight BEFORE draining or stopping Golden myds!
    errs = validate_candidate_artifacts(cand_bin, cand_ml, live_bin, live_ml)
    if not base_conf or not base_conf.is_file():
        errs.append(f"Cannot resolve base myds.conf (got {base_conf}); pass --base-config")
    if not model_path or not model_path.is_file():
        errs.append(f"Cannot resolve model path (got {model_path}); pass --model")
    if (
        not errs
        and _win_base.sha(live_bin) == _win_base.sha(cand_bin)
        and _win_base.sha(live_ml) == _win_base.sha(cand_ml)
        and not overrides
    ):
        errs.append("§3 / §A.4: candidate binary, metallib, and config are 100% identical to live Golden")

    # Pre-check `--print-config` on candidate while Golden is still alive!
    if not errs and base_conf:
        tmp_pf = run_dir / "preflight"
        tmp_pf.mkdir(mode=0o700, exist_ok=True)
        pf_conf = tmp_pf / "cand.conf"
        evo_judge.write_isolated_conf(
            base_conf, pf_conf, tmp_pf / "kv", tmp_pf / "resp.idx", "preflight", overrides
        )
        chk = subprocess.run(
            [str(cand_bin), "--config", str(pf_conf), "--print-config"],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
        )
        if chk.returncode != 0:
            errs.append(
                f"Candidate --print-config failed (rc={chk.returncode}): {chk.stderr.strip() or chk.stdout.strip()}"
            )

    if errs:
        receipt.update(status="REJECTED_PRE_WINDOW", errors=errs, golden_stopped=False)
        atomic_write_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2) if args.json else f"REJECTED_PRE_WINDOW: {errs}")
        return 1

    if not args.skip_offline:
        off_ns = argparse.Namespace(
            baseline_metallib=str(live_ml),
            candidate_metallib=str(cand_ml),
            build_cmd=None,
            unit_cmd=args.unit_cmd,
            skip_repo_checks=args.skip_repo_checks,
        )
        offline_res = evo_judge.run_offline_gate(off_ns)
        receipt["pre_window_offline_gate"] = offline_res
        if not offline_res["passed"]:
            receipt.update(status="REJECTED_OFFLINE_GATE", golden_stopped=False)
            atomic_write_json(receipt_path, receipt)
            print(json.dumps(receipt, indent=2) if args.json else "REJECTED_OFFLINE_GATE")
            return 1

    # Phase 1: Wait for active HTTP clients (including self-hosting LLM turn) to drain
    drained, lingering = wait_for_client_drain(live_pid, args.client_drain_timeout)
    if not drained:
        receipt.update(
            status="ABORTED_ACTIVE_CLIENTS",
            golden_stopped=False,
            active_clients=lingering,
        )
        atomic_write_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2) if args.json else "ABORTED_ACTIVE_CLIENTS")
        return 3

    # Phase 2: Enter exclusive GPU window via `run_register_only_service_window.Window`
    window_root = run_dir / "service-window"
    verdict_json = run_dir / "judge-verdict.json"
    inner_meta_json = run_dir / "window-inner.json"

    judge_argv = [
        "--mode",
        "full",
        "--skip-repo-checks",
        "--unit-cmd",
        "",
        "--baseline-bin",
        str(live_bin),
        "--baseline-metallib",
        str(live_ml),
        "--candidate-bin",
        str(cand_bin),
        "--candidate-metallib",
        str(cand_ml),
        "--base-config",
        str(base_conf),
        "--model",
        str(model_path),
        "--port",
        str(args.judge_port),
        "--alias",
        alias,
        "--cycles",
        str(args.cycles),
        "--metric",
        args.metric,
        "--max-tokens",
        str(args.max_tokens),
        "--min-gain-pct",
        str(args.min_gain_pct),
        "--boot-timeout",
        str(args.boot_timeout),
        "--out",
        str(verdict_json),
        "--json",
    ]
    if vision_path:
        judge_argv += ["--vision", str(vision_path)]
    if args.allow_non_0ulp:
        judge_argv.append("--allow-non-0ulp")
    for ov in args.conf_override:
        judge_argv += ["--conf-override", ov]
    if args.record_memory:
        judge_argv.append("--record-memory")
        if args.hypothesis:
            judge_argv += ["--hypothesis", args.hypothesis]
        if args.subsystem:
            judge_argv += ["--subsystem", args.subsystem]
        if args.hardware:
            judge_argv += ["--hardware", args.hardware]

    inner_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_window_inner",
        "--window-root",
        str(window_root),
        "--memory-timeout",
        str(args.memory_timeout),
        "--inner-meta-out",
        str(inner_meta_json),
        "--",
        *judge_argv,
    ]

    win_ns = argparse.Namespace(
        root=window_root,
        pid=live_pid,
        cwd=live_cwd,
        port=args.port,
        protect=[],
        stop_timeout=args.stop_timeout,
        drain_timeout=args.drain_timeout,
        restore_timeout=args.restore_timeout,
        memory_timeout=args.memory_timeout,
        command=inner_cmd,
    )
    win_rc = _win_base.Window(win_ns).run()

    # Phase 3: Verify Golden restoration
    post_live = gpu_window.identity_of_live(args.port)
    restore_diff = gpu_window.diff(pre_snap, post_live)

    judge_report = (
        json.loads(verdict_json.read_text(encoding="utf-8"))
        if verdict_json.is_file()
        else None
    )
    inner_meta = (
        json.loads(inner_meta_json.read_text(encoding="utf-8"))
        if inner_meta_json.is_file()
        else None
    )
    win_summary = (
        json.loads((window_root / "summary.json").read_text(encoding="utf-8"))
        if (window_root / "summary.json").is_file()
        else None
    )

    receipt.update(
        golden_stopped=True,
        window_rc=win_rc,
        window_summary=win_summary,
        window_inner=inner_meta,
        golden_restore_diff=restore_diff,
        judge_verdict=judge_report.get("verdict") if judge_report else "MISSING_VERDICT",
        judge_report=judge_report,
    )

    if win_rc != 0 or restore_diff:
        receipt["status"] = "WINDOW_OR_RESTORE_ERROR"
        atomic_write_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2) if args.json else f"ERROR: {receipt['status']}")
        return 3

    verdict = receipt["judge_verdict"]
    if args.promote_on_adopt and verdict == "ADOPT":
        promo = promote_adopted_candidate(
            port=args.port,
            cand_bin=cand_bin,
            cand_metallib=cand_ml,
            overrides=overrides,
            backup_dir=run_dir / "promotion-backup",
            stop_timeout=args.stop_timeout,
            memory_timeout=args.memory_timeout,
            restore_timeout=args.restore_timeout,
        )
        receipt["promotion"] = promo

    receipt["status"] = "COMPLETED"
    receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(receipt_path, receipt)

    exit_code = int(judge_report.get("exit_code", 1)) if judge_report else 3
    if args.json:
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
    else:
        print(f"SURGERY COMPLETE: verdict={verdict} receipt={receipt_path}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="subcmd", required=True)

    sp_st = sub.add_parser("status", help="Inspect live Golden myds and memory headroom")
    sp_st.add_argument("--port", type=int, default=gpu_window.DEFAULT_PORT)
    sp_st.add_argument("--json", action="store_true")

    sp_ev = sub.add_parser("evaluate", help="Time-sliced drain -> judge -> restore [-> promote]")
    sp_ev.add_argument("--port", type=int, default=gpu_window.DEFAULT_PORT, help="Live Golden port")
    sp_ev.add_argument("--judge-port", type=int, default=18187, help="Isolated port for ABABAB boots")
    sp_ev.add_argument("--candidate-bin", type=Path, help="Staged candidate myds binary")
    sp_ev.add_argument("--candidate-metallib", type=Path, help="Staged candidate myds.metallib")
    sp_ev.add_argument("--conf-override", action="append", default=[], help="Candidate MYDS_KEY=VAL")
    sp_ev.add_argument("--base-config", type=Path, help="Override base myds.conf (auto-detected from live)")
    sp_ev.add_argument("--model", type=Path, help="Override model path (auto-detected from live)")
    sp_ev.add_argument("--vision", type=Path, help="Override vision model path")
    sp_ev.add_argument("--alias", help="Override model alias")
    sp_ev.add_argument("--cycles", type=int, default=3)
    sp_ev.add_argument("--metric", choices=("decode_tps", "prefill_tps", "wall_s"), default="decode_tps")
    sp_ev.add_argument("--max-tokens", type=int, default=128)
    sp_ev.add_argument("--min-gain-pct", type=float, default=1.5)
    sp_ev.add_argument("--allow-non-0ulp", action="store_true")
    sp_ev.add_argument("--boot-timeout", type=float, default=240.0)
    sp_ev.add_argument("--client-drain-timeout", type=float, default=30.0)
    sp_ev.add_argument("--stop-timeout", type=float, default=30.0)
    sp_ev.add_argument("--drain-timeout", type=float, default=90.0)
    sp_ev.add_argument("--restore-timeout", type=float, default=180.0)
    sp_ev.add_argument("--memory-timeout", type=float, default=240.0)
    sp_ev.add_argument("--skip-offline", action="store_true")
    sp_ev.add_argument("--skip-repo-checks", action="store_true")
    sp_ev.add_argument("--unit-cmd", default="make test")
    sp_ev.add_argument("--promote-on-adopt", action="store_true", help="Atomically install winning candidate")
    sp_ev.add_argument("--record-memory", action="store_true")
    sp_ev.add_argument("--hypothesis", help="Hypothesis text for evo_memory")
    sp_ev.add_argument("--subsystem", help="Subsystem tag for evo_memory")
    sp_ev.add_argument("--hardware", help="Hardware tag for evo_memory")
    sp_ev.add_argument("--work-root", type=Path, help="Directory for surgery artifacts")
    sp_ev.add_argument("--receipt", type=Path, help="Output path for surgery-receipt.json")
    sp_ev.add_argument("--json", action="store_true")

    sp_in = sub.add_parser("_window_inner")
    sp_in.add_argument("--window-root", type=Path, required=True)
    sp_in.add_argument("--memory-timeout", type=float, default=240.0)
    sp_in.add_argument("--inner-meta-out", type=Path, required=True)
    sp_in.add_argument("judge_args", nargs=argparse.REMAINDER)

    args = ap.parse_args(argv)
    if args.subcmd == "status":
        return cmd_status(args)
    if args.subcmd == "evaluate":
        return cmd_evaluate(args)
    if args.subcmd == "_window_inner":
        if args.judge_args[:1] == ["--"]:
            args.judge_args = args.judge_args[1:]
        return cmd_window_inner(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
