#!/usr/bin/env python3
"""Read-only inspection of the live myds service, plus the shared residency gate.

This tool deliberately does NOT stop or restore anything. `tools/
run_register_only_service_window.py` already does that far more rigorously: it
captures the exact argv/environment/cwd, backs up and re-verifies every
protected artifact by SHA256, refuses to run while a client is connected, never
SIGKILLs the original, and now also preserves the original niceness and stdio.

Two independent stop/restore paths would be the exact defect this project keeps
paying for: two sources of truth for one fact. So the window lives there, and
this file provides what that base does not:

  snapshot   record the live process as JSON (pid, nice, argv, cwd, stdio,
             inherited MYDS_* env, binary/metallib/config SHA256, /health)
  verify     compare the live process against such a snapshot
  residency  parse `powermetrics --samplers gpu_power` output into GPU active
             residency samples, and say whether a window actually proved
             exclusive occupancy (§106.3 requires median >= 95%)

Typical use around a window run by the base:

    python3 tools/gpu_window.py snapshot --port 8187 --json /tmp/before.json
    #   ... run the window (see run_register_only_service_window.py) ...
    python3 tools/gpu_window.py verify --port 8187 --json /tmp/before.json --residency /tmp/window.pm.txt

Exit codes: 0 = consistent, 1 = mismatch, 2 = precondition/usage error.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_PORT = int(os.environ.get("MYDS_WINDOW_PORT", "8187"))
RESIDENCY_MIN_PERCENT = 95.0

# Environment this inspector itself may be invoked with; it is not something the
# service was launched with, so comparing it would report the tool's own
# presence as a restore defect.
CONTROL_ENV = {"MYDS_FLEET_DIR", "MYDS_FLEET_START", "MYDS_WINDOW_CONFIRM",
               "MYDS_WINDOW_PORT"}


class WindowError(RuntimeError):
    pass


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
    return run(["ps", "-o", f"{field}=", "-p", str(pid)]).stdout.strip()


def proc_argv(pid):
    return proc_field(pid, "command")


def proc_nice(pid):
    raw = proc_field(pid, "nice")
    return int(raw) if raw.lstrip("-").isdigit() else None


def proc_env(pid):
    """MYDS_* entries in the process environment (macOS: ps eww)."""
    found = {}
    for token in run(["ps", "eww", "-p", str(pid)]).stdout.split():
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
    found = None
    for i, token in enumerate(tokens):
        if token == "--config" and i + 1 < len(tokens):
            found = tokens[i + 1]
    return found


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


def snapshot(port=DEFAULT_PORT):
    pid = port_pid(port)
    if pid is None:
        raise WindowError(f"no listener on port {port}")
    argv_text = proc_argv(pid)
    binary = argv_text.split()[0] if argv_text else None
    metallib = str(Path(binary).with_name("myds.metallib")) if binary else None
    config = config_path_from_argv(argv_text)
    return {
        "captured_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    __import__("time").gmtime()),
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
    before_env = {k: v for k, v in (snap.get("inherited_env") or {}).items()
                  if k not in CONTROL_ENV}
    after_env = {k: v for k, v in (live.get("inherited_env") or {}).items()
                 if k not in CONTROL_ENV}
    for key in sorted(set(before_env) | set(after_env)):
        if before_env.get(key) != after_env.get(key):
            problems.append(f"env {key} {before_env.get(key)} -> {after_env.get(key)}")
    state = health(snap["port"])
    if state.get("status") != "ok":
        problems.append(f"/health is {state}")
    return problems


def unverified(snap):
    """Fields a PASS does not cover.

    Reporting them is the point: a silent None == None comparison would look
    like a passed check while verifying nothing.
    """
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

    Ledger §106.3 requires a window to prove the GPU was actually occupied; a
    benchmark that finishes while the GPU is idle has measured nothing.
    """
    values = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if "GPU HW active residency" not in line:
            continue
        for token in line.split(":", 1)[1].split():
            if token.endswith("%"):
                try:
                    values.append(float(token.rstrip("%")))
                except ValueError:
                    pass
                break
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["snapshot", "verify", "residency"])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--json", type=Path, help="snapshot file to write (snapshot) or read (verify)")
    ap.add_argument("--residency", type=Path, help="powermetrics output to include in the verdict")
    args = ap.parse_args()

    try:
        if args.command == "snapshot":
            snap = snapshot(args.port)
            text = json.dumps(snap, indent=2, ensure_ascii=False)
            if args.json:
                args.json.write_text(text + "\n")
            print(text)
            return 0

        if args.command == "residency":
            if not args.residency:
                raise WindowError("residency needs --residency <powermetrics output>")
            values = residency_samples(args.residency)
            if not values:
                print("VERDICT: UNKNOWN -- no GPU residency lines parsed")
                return 2
            ordered = sorted(values)
            median = ordered[len(ordered) // 2]
            ok = median >= RESIDENCY_MIN_PERCENT
            print(f"samples={len(values)} min={ordered[0]:.2f}% median={median:.2f}% "
                  f"max={ordered[-1]:.2f}%")
            print(f"VERDICT: {'PASS' if ok else 'FAIL'} -- median residency "
                  f"{median:.2f}% {'>=' if ok else '<'} {RESIDENCY_MIN_PERCENT:.0f}%")
            return 0 if ok else 1

        if not args.json or not args.json.exists():
            raise WindowError("verify needs --json pointing at an existing snapshot")
        snap = json.loads(args.json.read_text())
        problems = diff(snap, identity_of_live(args.port))
        for item in problems:
            print(f"MISMATCH {item}")
        for item in unverified(snap):
            print(f"UNVERIFIED {item}")
        if args.residency:
            values = residency_samples(args.residency)
            if values:
                ordered = sorted(values)
                median = ordered[len(ordered) // 2]
                print(f"residency median {median:.2f}% over {len(values)} samples")
                if median < RESIDENCY_MIN_PERCENT:
                    problems.append(
                        f"GPU residency median {median:.2f}% < {RESIDENCY_MIN_PERCENT:.0f}%: "
                        "window does not prove exclusive occupancy (§106.3)")
            else:
                print("residency: no samples parsed (unverified)")
        print("VERDICT: PASS" if not problems else f"VERDICT: FAIL ({len(problems)} mismatches)")
        return 0 if not problems else 1
    except WindowError as exc:
        print(f"gpu_window: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
