#!/usr/bin/env python3
"""Exclusive measurement window for the myds service: stop, measure, restore, prove.

Why this exists
---------------
`tools/run_kprof_window.sh` stops the service and calls `start_fleet.sh` to bring
it back, but the process it restores is not the process it stopped. Production
runs at nice=5 with stdin inherited from the launching shell; start_fleet.sh sets
neither, so after a window the service came back at nice=0 with a different
stdin. Ledger §108 records that this made a real-model exclusive test
impossible. A window whose restore path changes the machine is not a
measurement, it is a side effect.

The contract here:
  1. capture the full launch context before touching anything
  2. stop only the listener on the given port, by PID (never `pkill -f`)
  3. restore through start_fleet.sh, re-applying what start_fleet.sh does not:
     nice level and stdin
  4. verify ALL of it, and exit non-zero when anything differs

Subcommands
    snapshot   read-only: capture the current state as JSON
    verify     compare the live process against a snapshot
    run        snapshot -> stop -> <command> -> restore -> verify

The `run` subcommand refuses to proceed unless MYDS_WINDOW_CONFIRM=1 is set.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PORT = int(os.environ.get("MYDS_WINDOW_PORT", "8187"))
FLEET_DIR = os.environ.get("MYDS_FLEET_DIR", "")
START_SCRIPT = os.environ.get("MYDS_FLEET_START", "")

STOP_GRACE_S = 60.0        # SIGTERM -> SIGKILL escalation window
RELEASE_GRACE_S = 120.0    # wait for the mapping teardown after exit
READY_TIMEOUT_S = 600.0    # cold start of a 362 GB model


class WindowError(RuntimeError):
    pass


# ------------------------------------------------------------------ primitives

def run(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, **kw)


def port_pid(port):
    """PID listening on `port`, or None. Never matches on a command line."""
    out = run(["lsof", "-nP", "-t", "-a", f"-iTCP:{port}", "-sTCP:LISTEN"]).stdout.split()
    pids = sorted({int(x) for x in out if x.isdigit()})
    if len(pids) > 1:
        raise WindowError(f"port {port} has several listeners: {pids}")
    return pids[0] if pids else None


def proc_field(pid, field):
    out = run(["ps", "-o", f"{field}=", "-p", str(pid)]).stdout.strip()
    return out


def proc_argv(pid):
    return proc_field(pid, "command")


def proc_nice(pid):
    raw = proc_field(pid, "nice")
    return int(raw) if raw.lstrip("-").isdigit() else None


def proc_env(pid):
    """MYDS_* entries in the process environment (macOS: ps eww)."""
    text = run(["ps", "eww", "-p", str(pid)]).stdout
    found = {}
    for token in text.split():
        if token.startswith("MYDS_") and "=" in token:
            name, value = token.split("=", 1)
            found[name] = value
    return dict(sorted(found.items()))


def fd_targets(pid):
    """Resolve fd 0/1/2 to their paths. Missing means the fd is not a file."""
    out = run(["lsof", "-p", str(pid), "-a", "-d", "0,1,2", "-Fn"]).stdout
    result, current = {}, None
    for line in out.splitlines():
        if line.startswith("f"):
            current = line[1:]
        elif line.startswith("n") and current is not None:
            result[current] = line[1:]
            current = None
    return {fd: result.get(fd) for fd in ("0", "1", "2")}


def sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_path_from_argv(argv_text):
    """--config is repeatable; the last one wins in myds main()."""
    tokens = argv_text.split()
    last = None
    for i, token in enumerate(tokens):
        if token == "--config" and i + 1 < len(tokens):
            last = tokens[i + 1]
    return last


def health(port, timeout=5.0):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.URLError as exc:
        return {"error": str(exc)}
    except ValueError:
        return {"error": "unparsable /health body"}


def cwd_of(pid):
    out = run(["lsof", "-p", str(pid), "-a", "-d", "cwd", "-Fn"]).stdout
    for line in out.splitlines():
        if line.startswith("n"):
            return line[1:]
    return None


# ------------------------------------------------------------------- snapshot

def snapshot(port=DEFAULT_PORT):
    pid = port_pid(port)
    if pid is None:
        raise WindowError(f"no listener on port {port}")
    argv_text = proc_argv(pid)
    binary = argv_text.split()[0] if argv_text else None
    metallib = str(Path(binary).with_name("myds.metallib")) if binary else None
    config = config_path_from_argv(argv_text)
    snap = {
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "port": port,
        "pid": pid,
        "nice": proc_nice(pid),
        "cwd": cwd_of(pid),
        "argv": argv_text,
        "stdio": fd_targets(pid),
        "inherited_env": proc_env(pid),
        "health": health(port),
        "identity": {
            "binary": binary,
            "binary_sha256": sha256(binary) if binary else None,
            "metallib": metallib,
            "metallib_sha256": sha256(metallib) if metallib else None,
            "config": config,
            "config_sha256": sha256(config) if config else None,
        },
    }
    return snap


def identity_of_live(port):
    pid = port_pid(port)
    if pid is None:
        return None
    argv_text = proc_argv(pid)
    binary = argv_text.split()[0] if argv_text else None
    metallib = str(Path(binary).with_name("myds.metallib")) if binary else None
    config = config_path_from_argv(argv_text)
    return {
        "pid": pid,
        "nice": proc_nice(pid),
        "cwd": cwd_of(pid),
        "stdio": fd_targets(pid),
        "inherited_env": proc_env(pid),
        "argv": argv_text,
        "binary_sha256": sha256(binary) if binary else None,
        "metallib_sha256": sha256(metallib) if metallib else None,
        "config_sha256": sha256(config) if config else None,
    }


# ------------------------------------------------------------------ stop/start

def stop(port, log):
    pid = port_pid(port)
    if pid is None:
        raise WindowError(f"no listener on port {port} to stop")
    log(f"stop: SIGTERM pid {pid}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + STOP_GRACE_S
    while time.time() < deadline:
        if port_pid(port) is None:
            break
        time.sleep(0.5)
    else:
        log(f"stop: still alive after {STOP_GRACE_S:.0f}s, SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.time() + STOP_GRACE_S
        while time.time() < deadline and port_pid(port) is not None:
            time.sleep(0.5)
    if port_pid(port) is not None:
        raise WindowError("service did not stop")
    # A 362 GB mapping is not released the instant the PID disappears; starting
    # a new one while the kernel is still tearing down the old one is how this
    # machine panicked twice (myfleet start_fleet.sh documents both).
    log("stop: waiting for the mapping to be released")
    deadline = time.time() + RELEASE_GRACE_S
    while time.time() < deadline:
        if not run(["pgrep", "-x", "myds"]).stdout.strip():
            break
        time.sleep(1.0)
    log("stop: port and process name are clear")


def start_and_restore(snap, log):
    if not FLEET_DIR or not START_SCRIPT:
        raise WindowError("MYDS_FLEET_DIR and MYDS_FLEET_START must be set to restore\n"
                          "  (the restore path is deliberately explicit: this tool will not guess)")
    script = Path(START_SCRIPT)
    if not script.exists():
        raise WindowError(f"start script not found: {script}")
    nice = snap.get("nice")
    stdin_target = (snap.get("stdio") or {}).get("0")

    argv = ["/usr/bin/nice", "-n", str(nice), str(script)] if nice is not None else [str(script)]
    stdin_handle = None
    if stdin_target and Path(stdin_target).exists() and os.access(stdin_target, os.R_OK):
        stdin_handle = open(stdin_target, "rb")
        log(f"restore: stdin <- {stdin_target}")
    else:
        log(f"restore: stdin target unavailable ({stdin_target}); inheriting /dev/null")
    try:
        result = subprocess.run(argv, cwd=FLEET_DIR, stdin=stdin_handle,
                                capture_output=True, text=True, timeout=READY_TIMEOUT_S + 120)
    finally:
        if stdin_handle:
            stdin_handle.close()
    log_path = Path(snap.get("cwd") or FLEET_DIR) / "logs" / "gpu_window_restore.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((result.stdout or "") + (result.stderr or ""))
    log(f"restore: start_fleet exit={result.returncode}, output {log_path}")
    if result.returncode != 0:
        raise WindowError(f"start_fleet.sh exited {result.returncode}; see {log_path}")

    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        if port_pid(snap["port"]) is not None:
            state = health(snap["port"])
            if state.get("status") == "ok":
                break
        time.sleep(2.0)
    else:
        raise WindowError("service did not become healthy after restore")

    # `nice -n N` is an increment, not an assignment: if this tool itself runs
    # at niceness > 0 the restored service lands at N + current. Force the
    # absolute value and verify it below rather than trusting the increment.
    pid = port_pid(snap["port"])
    if nice is not None and pid is not None:
        current = proc_nice(pid)
        if current != nice:
            log(f"restore: renice {current} -> {nice} (nice -n is relative)")
            subprocess.run(["renice", "-n", str(nice), "-p", str(pid)],
                           capture_output=True, text=True)
    log(f"restore: listener up, /health ok")


# --------------------------------------------------------------------- verify

def diff(snap, live):
    """Every field here was a real defect at some point in this project."""
    problems = []
    if live is None:
        return [f"no listener on port {snap['port']}"]
    if snap.get("nice") != live.get("nice"):
        problems.append(f"nice {snap.get('nice')} -> {live.get('nice')}")
    for fd in ("0", "1", "2"):
        before = (snap.get("stdio") or {}).get(fd)
        after = (live.get("stdio") or {}).get(fd)
        if before != after:
            problems.append(f"stdio[{fd}] {before} -> {after}")
    for key in ("binary_sha256", "metallib_sha256", "config_sha256"):
        before = (snap.get("identity") or {}).get(key)
        after = live.get(key)
        if before != after:
            problems.append(f"{key} {before} -> {after}")
    if snap.get("cwd") != live.get("cwd"):
        problems.append(f"cwd {snap.get('cwd')} -> {live.get('cwd')}")
    before_env = {k: v for k, v in (snap.get("inherited_env") or {}).items() if k not in CONTROL_ENV}
    after_env = {k: v for k, v in (live.get("inherited_env") or {}).items() if k not in CONTROL_ENV}
    for key in sorted(set(before_env) | set(after_env)):
        if before_env.get(key) != after_env.get(key):
            problems.append(f"env {key} {before_env.get(key)} -> {after_env.get(key)}")
    state = health(snap["port"])
    if state.get("status") != "ok":
        problems.append(f"/health is {state}")
    return problems


def unverified(snap):
    """Fields a PASS does not cover. Reporting them is the point: a silent
    None==None comparison would look like a passed check."""
    gaps = []
    identity = snap.get("identity") or {}
    if not identity.get("config"):
        gaps.append("config identity: no --config in argv, nothing to compare")
    if not identity.get("binary_sha256"):
        gaps.append("binary identity: argv[0] is not a readable file")
    if not identity.get("metallib_sha256"):
        gaps.append("metallib identity: myds.metallib not found next to argv[0]")
    return gaps


def residency_samples(path):
    """Parse `powermetrics --samplers gpu_power` output for GPU active residency.

    Ledger §106.3 requires the window to prove the GPU was actually occupied;
    a benchmark that finishes while the GPU is idle measures nothing.
    """
    values = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if "GPU HW active residency" in line:
            for token in line.split(":", 1)[1].split():
                if token.endswith("%"):
                    try:
                        values.append(float(token.rstrip("%")))
                    except ValueError:
                        pass
                    break
    return values


def start_powermetrics(path):
    """Best-effort sampler. Returns a Popen or None with a reason."""
    if not shutil.which("powermetrics"):
        return None, "powermetrics not found"
    handle = open(path, "wb")
    proc = subprocess.Popen(["sudo", "-n", "powermetrics", "--samplers", "gpu_power",
                             "-i", "1000"], stdout=handle, stderr=subprocess.STDOUT)
    time.sleep(2.0)
    if proc.poll() is not None:
        handle.close()
        return None, "powermetrics exited immediately (sudo -n needs a cached credential)"
    return proc, None


# ------------------------------------------------------------------------ main

# Environment the tool itself injects. The service inherits these through
# start_fleet.sh, so comparing them would report the tool's own presence as a
# restore defect. Everything else is compared.
CONTROL_ENV = {"MYDS_FLEET_DIR", "MYDS_FLEET_START", "MYDS_WINDOW_CONFIRM",
               "MYDS_WINDOW_PORT"}


def _split_dashdash(argv):
    """Split at the first `--`: options before it, the inner command after.

    The tail is kept as a list so quoting inside the command survives; joining
    and re-splitting silently rewrote `sh -c 'a && b'` into `sh -c a && b`.
    """
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], True, argv[i + 1:]
    return argv, False, []



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["snapshot", "verify", "run"])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--json", type=Path, help="write the snapshot/verification JSON here")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--powermetrics", type=Path,
                    help="sample GPU residency during the window (needs sudo -n)")
    # The command to run inside the window is split off manually: REMAINDER
    # would swallow the options that follow the subcommand.
    argv, separator, inner = _split_dashdash(sys.argv[1:])
    args = ap.parse_args(argv)
    args.rest = inner if separator else []

    stamps = []

    def log(msg):
        stamps.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if not args.quiet:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)

    try:
        if args.command == "snapshot":
            snap = snapshot(args.port)
            text = json.dumps(snap, indent=2, ensure_ascii=False)
            if args.json:
                args.json.write_text(text + "\n")
            print(text)
            return 0

        if args.command == "verify":
            if not args.json or not args.json.exists():
                raise WindowError("verify needs --json pointing at an existing snapshot")
            snap = json.loads(args.json.read_text())
            problems = diff(snap, identity_of_live(args.port))
            for item in problems:
                print(f"MISMATCH {item}")
            for item in unverified(snap):
                print(f"UNVERIFIED {item}")
            print("VERDICT: PASS" if not problems else f"VERDICT: FAIL ({len(problems)} mismatches)")
            return 0 if not problems else 1

        # run
        command = args.rest
        if not command:
            raise WindowError("run needs a command after --")
        if os.environ.get("MYDS_WINDOW_CONFIRM") != "1":
            raise WindowError("refusing to stop the service without MYDS_WINDOW_CONFIRM=1")

        snap = snapshot(args.port)
        log(f"snapshot: pid={snap['pid']} nice={snap['nice']} stdio={snap['stdio']}")
        out_path = args.json or Path(f"/tmp/myds-window-{int(time.time())}.json")
        out_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n")

        problems, exit_code = [], 1
        residency = {"samples": [], "unavailable": None}
        sampler = None
        try:
            stop(args.port, log)
            if args.powermetrics:
                sampler, reason = start_powermetrics(args.powermetrics)
                if sampler is None:
                    residency["unavailable"] = reason
                    log(f"powermetrics: unavailable -- {reason}")
                else:
                    log(f"powermetrics: sampling to {args.powermetrics}")
            log(f"run: {' '.join(command)}")
            completed = subprocess.run(command)
            exit_code = completed.returncode
            log(f"run: exit={exit_code}")
        finally:
            if sampler is not None:
                sampler.terminate()
                try:
                    sampler.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    sampler.kill()
                values = residency_samples(args.powermetrics)
                residency["samples"] = values
                if values:
                    values_sorted = sorted(values)
                    residency["min"] = values_sorted[0]
                    residency["median"] = values_sorted[len(values_sorted) // 2]
                    log(f"powermetrics: GPU active residency median {residency['median']:.2f}% "
                        f"min {residency['min']:.2f}% over {len(values)} samples")
                    if residency["median"] < 95.0:
                        problems.append(
                            f"GPU residency median {residency['median']:.2f}% < 95%: "
                            "this window does not prove exclusive GPU occupancy (§106.3)")
                else:
                    residency["unavailable"] = "no GPU residency lines parsed"
                    log("powermetrics: no residency lines parsed")
            try:
                start_and_restore(snap, log)
            except WindowError as exc:
                problems.append(f"restore failed: {exc}")
            live = identity_of_live(args.port)
            problems += diff(snap, live)
            report = {"snapshot": str(out_path), "command": command,
                      "command_exit": exit_code, "mismatches": problems,
                      "unverified": unverified(snap), "gpu_residency": residency,
                      "log": stamps}
            args_verbose = out_path.with_suffix(".verify.json")
            args_verbose.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            log(f"verification written to {args_verbose}")
            for item in problems:
                log(f"MISMATCH {item}")

        if problems:
            log(f"VERDICT: FAIL -- {len(problems)} mismatch(es); the machine is not as it was")
            return 1
        log("VERDICT: PASS -- restored identically")
        return exit_code
    except WindowError as exc:
        print(f"gpu_window: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
