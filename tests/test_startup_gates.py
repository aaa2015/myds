#!/usr/bin/env python3
"""Startup gates: unknown CLI args, unknown MYDS_* keys, busy port, instance lock.

These all fail *before* the model is loaded, which is the point: a config typo or
a second instance must not cost 148 GB of wired memory before it is noticed.
Each case runs the real binary with a bogus model path, so a gate that fails to
fire shows up as exit 1 ("failed to open GGUF") instead of the expected code.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin" / "myds"
CONF = os.environ.get("MYDS_GATE_TEST_CONF", "")

# A model path that never exists: reaching model load is itself the failure signal.
BAD_MODEL = "/nonexistent/startup-gate-test.gguf"


def run(args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS_")}
    env.pop("DS4_DSPARK", None)
    # Empty config: the gate must judge the environment, not a stray dev file.
    env["MYDS_CONFIG"] = "/dev/null"
    if env_extra:
        env.update(env_extra)
    p = subprocess.run([str(BIN)] + args, capture_output=True, text=True, env=env, timeout=60)
    return p.returncode, p.stdout + p.stderr


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    if not BIN.exists():
        print(f"SKIP: {BIN} not built")
        return 0
    failures = [0]

    def check(name, cond, detail=""):
        if cond:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name} {detail}")
            failures[0] += 1

    # 1) Unknown CLI argument must be rejected, not silently ignored.
    #    Regression: `--temperature 0` used to start the server with the code
    #    default in force, so the flag looked applied and was not.
    rc, out = run(["--temperature", "0", "-m", BAD_MODEL])
    check("unknown CLI arg rejected (rc=2)", rc == 2, f"rc={rc}")
    check("unknown CLI arg named in the message", "--temperature" in out, out[-200:])

    # 2) Unknown MYDS_* key must abort startup.
    rc, out = run(["-m", BAD_MODEL], {"MYDS_SOME_TYPO_KEY": "1"})
    check("unknown MYDS_* key rejected (rc=2)", rc == 2, f"rc={rc}")
    check("unknown key named in the message", "MYDS_SOME_TYPO_KEY" in out, out[-200:])

    # 3) MYDS_STRICT_CONFIG=0 downgrades to a warning and lets startup continue.
    rc, out = run(["-m", BAD_MODEL], {"MYDS_SOME_TYPO_KEY": "1", "MYDS_STRICT_CONFIG": "0"})
    check("strict gate can be downgraded", rc == 1, f"rc={rc} (1 = reached model load)")
    check("downgrade still warns", "MYDS_SOME_TYPO_KEY" in out, out[-200:])

    # 4) Busy port must be refused before the model is mapped.
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    busy = listener.getsockname()[1]
    try:
        rc, out = run(["--port", str(busy), "-m", BAD_MODEL])
        check("busy port refused (rc=3)", rc == 3, f"rc={rc}")
        check("busy port named in the message", str(busy) in out, out[-200:])
    finally:
        listener.close()

    # 5) Instance lock: a second process must not get past it even if nothing
    #    is listening yet (the window before bind is exactly the dangerous one).
    port = free_port()
    lock_path = f"/tmp/myds-instance.lock-{port}"
    holder = open(lock_path, "w")
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc, out = run(["--port", str(port), "-m", BAD_MODEL])
        check("instance lock held elsewhere refused (rc=3)", rc == 3, f"rc={rc}")
        check("lock message mentions the port", str(port) in out, out[-200:])
    finally:
        holder.close()
        try:
            os.unlink(lock_path)
        except OSError:
            pass

    # 6) MYDS_ALLOW_MULTI_INSTANCE=1 clears both instance gates.
    holder = open(lock_path, "w")
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc, out = run(["--port", str(port), "-m", BAD_MODEL],
                      {"MYDS_ALLOW_MULTI_INSTANCE": "1"})
        check("multi-instance override works", rc == 1, f"rc={rc} (1 = reached model load)")
    finally:
        holder.close()
        try:
            os.unlink(lock_path)
        except OSError:
            pass

    if failures[0]:
        print(f"\n{failures[0]} startup-gate case(s) FAILED")
        return 1
    print("\nAll startup gates behave as specified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
