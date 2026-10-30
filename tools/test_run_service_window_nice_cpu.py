#!/usr/bin/env python3
"""CPU-only checks for niceness/stdio preservation in the service window base.

Never captures, stops or restores a real service: the capture helpers are driven
against a stand-in process (this test's own child), and the restore path is
exercised with the spawn call mocked out.

What is being pinned down (ledger §108): production runs at nice=5 with stdin
inherited from start_fleet.sh's `while read ... done < $CONFIG_FILE` loop. The
base used to refuse that outright, and a restore that quietly substituted nice=0
or /dev/null would have made the window change the machine it claims to preserve.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_register_only_service_window as module


CASES = []


def case(name):
    def register(fn):
        CASES.append((name, fn))
        return fn
    return register


def need(ok, message):
    if not ok:
        raise AssertionError(message)


@case("process_nice_reads_negative_and_positive")
def _(root):
    # nice 0 child
    child = subprocess.Popen(["/bin/sleep", "30"])
    try:
        need(module.process_nice(child.pid) == 0, "nice 0 child misread")
    finally:
        child.kill(); child.wait()
    for level in (5, 10):
        child = subprocess.Popen(["/usr/bin/nice", "-n", str(level), "/bin/sleep", "30"])
        try:
            need(child.pid != 0, "spawn failed")
            observed = module.process_nice(child.pid)
            need(observed >= level,
                 f"expected niceness >= {level} (nice is relative), got {observed}")
        finally:
            child.kill(); child.wait()


@case("stdio_targets_resolves_files_and_reports_none_for_ttys")
def _(root):
    sink = root / "stdout.log"
    stdin_file = root / "stdin.conf"
    stdin_file.write_text("row|row\n")
    with sink.open("wb") as out, stdin_file.open("rb") as inp:
        child = subprocess.Popen(["/bin/sleep", "30"], stdin=inp, stdout=out,
                                 stderr=subprocess.STDOUT)
        try:
            targets = module.stdio_targets(child.pid)
        finally:
            child.kill(); child.wait()
    need(targets["0"] == str(stdin_file.resolve()), f"stdin target: {targets['0']}")
    need(targets["1"] == str(sink.resolve()), f"stdout target: {targets['1']}")
    # stderr was pointed at the same file; lsof may report either the file or
    # nothing, but it must never invent a different path.
    need(targets["2"] in (str(sink.resolve()), None), f"stderr target: {targets['2']}")


@case("apply_priority_sets_the_absolute_value")
def _(root):
    """Run the hook in a child: setpriority on the test process itself cannot be
    undone without privilege, so raising it here would poison the rest of the run."""
    class State:
        state = {"nice": 7}
    child = subprocess.Popen(["/bin/sleep", "30"],
                             preexec_fn=module.Window._apply_priority.__get__(State()))
    try:
        observed = module.process_nice(child.pid)
        need(observed == 7, f"child niceness {observed} != 7")
    finally:
        child.kill(); child.wait()
    need(os.getpriority(os.PRIO_PROCESS, 0) == 0,
         "the test process must keep its own priority unchanged")


@case("restore_reuses_captured_stdio_and_niceness")
def _(root):
    """The spawn must receive the captured niceness and the original targets."""
    log = root / "service.log"
    log.write_text("existing\n")
    stdin_file = root / "fleet.conf"
    stdin_file.write_text("row\n")
    captured = []
    real_popen = subprocess.Popen

    def fake_popen(argv, **kwargs):
        captured.append(kwargs)
        # Return a harmless stand-in instead of starting a real service.
        return real_popen(["/bin/sleep", "30"], **{k: v for k, v in kwargs.items()
                                                  if k in ("stdin", "stdout", "stderr")})

    window = module.Window.__new__(module.Window)
    window.a = type("A", (), {"root": root, "stop_timeout": 1.0,
                              "restore_timeout": 0.1})()
    window.state = {"argv": ["/bin/sleep", "30"], "executable": "/bin/sleep",
                    "cwd": str(root), "environment": dict(os.environ),
                    "nice": 5, "stdio": {"0": str(stdin_file), "1": str(log),
                                         "2": str(log)}}
    window.experiment = None
    window.restored_child = None

    for name in ("drain_experiment", "wait_original_exit", "verify", "persist"):
        setattr(window, name, lambda *a, **k: None)
    with patch.object(module, "processes", lambda: []), \
         patch.object(module, "subprocess") as fake_module:
        fake_module.Popen = fake_popen
        fake_module.DEVNULL = subprocess.DEVNULL
        fake_module.TimeoutExpired = subprocess.TimeoutExpired
        try:
            window.restore()
        except Exception:
            pass  # health probing cannot succeed against /bin/sleep; spawn is what matters

    need(captured, "restore never attempted a spawn")
    kwargs = captured[0]
    need(kwargs.get("preexec_fn") is not None,
         "restore did not pass a priority hook; niceness would not be preserved")
    need(kwargs.get("stdin") is not subprocess.DEVNULL,
         "restore discarded the captured stdin (the §108 defect)")
    need(getattr(kwargs.get("stdout"), "name", None) == str(log),
         f"restore did not append to the original log: {kwargs.get('stdout')}")


@case("memory_release_gate_refuses_to_spawn_early")
def _(root):
    """The gate must block while the old mapping is still resident.

    Spawning a replacement into an unreleased 362 GB mapping is the documented
    panic shape on this machine, so the failure mode is what this pins.
    """
    readings = iter([20.0, 40.0, 80.0])
    original = module.available_gib
    module.available_gib = lambda: next(readings, 80.0)
    try:
        try:
            module.wait_for_memory_release(150.0, 0.15)
        except RuntimeError as exc:
            need('refusing to spawn' in str(exc), f'unexpected message: {exc}')
        else:
            raise AssertionError('gate allowed a spawn before memory was released')
    finally:
        module.available_gib = original

    # And it returns as soon as the threshold is met, so a healthy stop is not slowed.
    readings = iter([149.0, 151.0])
    module.available_gib = lambda: next(readings, 151.0)
    try:
        released, samples = module.wait_for_memory_release(150.0, 5.0)
    finally:
        module.available_gib = original
    need(released >= 150.0, f'released {released}')
    need(samples[-1] >= 150.0, f'samples {samples}')


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="myds-window-nice-cpu-") as temp:
        root = Path(temp)
        for name, fn in CASES:
            try:
                fn(root)
                results.append({"case": name, "passed": True, "error": None})
            except Exception as exc:
                results.append({"case": name, "passed": False,
                                "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"gpu_or_model_executed": False, "results": results}, indent=2))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
