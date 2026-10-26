#!/usr/bin/env python3
"""Frozen Q0 collection only; never calls a performance reader or deploys.

Importing this module is CPU-only. Root must separately authorize main() on
the target. Each child uses the ordinary native driver with no GPU observer.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def snapshot(path, full_sha=True):
    path = Path(path)
    resolved = path.resolve(strict=True)
    st = resolved.stat()
    result = dict(path=str(path), canonical_path=str(resolved), bytes=st.st_size,
                  device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns)
    if full_sha:
        result["sha256"] = sha(resolved)
    return result


def verify_descriptor(item):
    actual = snapshot(item["path"])
    require(actual["bytes"] == item["bytes"] and actual["sha256"] == item["sha256"],
            "frozen artifact bytes/SHA differ: " + item["path"])
    return actual


def canonical_state(protocol):
    return {name: verify_descriptor(item) for name, item in protocol["canonical_files"].items()}


def model_state(protocol):
    model = protocol["model"]
    result = snapshot(model["path"], False)
    with Path(model["path"]).open("rb") as stream:
        prefix = stream.read(model["prefix_bytes"])
    require(len(prefix) == model["prefix_bytes"], "complete model identity prefix")
    result.update(prefix_bytes=len(prefix), prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                  whole_model_payload_sha256_verified=False)
    require(all(result[name] == model[name] for name in
                ("bytes", "inode", "mtime_ns", "prefix_bytes", "prefix_sha256")), "model stat/prefix identity drift")
    return result, prefix


def processes():
    result = subprocess.run(["ps", "-axo", "pid=,comm="], capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        name = Path(fields[1]).name.lower()
        if name in {"myds", "myglm", "ds4", "llama-server", "ollama", "capture_indexer_k"} or name.startswith("bench_"):
            rows.append(dict(pid=int(fields[0]), comm=fields[1]))
    return rows


def child_environment(inherited, mode, logits, state):
    env = {key: value for key, value in inherited.items() if not key.startswith("MYDS_")}
    env.update(MYDS_INDEXER_K_RESIDENT=str(mode), MYDS_BENCH_LOGITS_DIR=str(logits), MYDS_BENCH_STATE_DIR=str(state))
    env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
    return env


def command(protocol, provider):
    c = protocol["request_contract"]
    return [provider["binary"]["path"], "--config", protocol["artifacts"]["config"]["path"], "--run",
            protocol["model"]["path"], protocol["artifacts"]["request"]["path"], str(c["repeats"]),
            str(c["output_tokens"]), str(c["max_seq_len"])]


def raw_inventory(directory):
    directory = Path(directory)
    items = {}
    paths = [directory / "stdout.jsonl", directory / "stderr.txt"]
    for name in ("logits", "state"):
        paths.extend((directory / name).rglob("*"))
    for path in sorted(paths):
        require(not path.is_symlink(), "raw evidence symlink refused")
        if path.is_dir():
            continue
        require(path.is_file(), "retained regular raw evidence")
        items[str(path.relative_to(directory))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    return dict(schema_version=1, files=items, file_count=len(items), logical_bytes=sum(x["bytes"] for x in items.values()),
                tree_sha256=hashlib.sha256(canonical).hexdigest(), whole_payload_files_rehashed=True)


def terminate_owned(proc):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def interrupted(signum, frame):
    del frame
    raise InterruptedError("window interrupted by signal " + str(signum))


def collect_provider(protocol, provider, out, protocol_sha, frozen_inputs):
    out.mkdir(exist_ok=False)
    for name in ("logits", "state", "cwd"):
        (out / name).mkdir()
    before = canonical_state(protocol)
    model_before, _ = model_state(protocol)
    initial = processes()
    require(not initial, "recognized inference competitor before " + provider["name"])
    env = child_environment(os.environ, provider["mode"], out / "logits", out / "state")
    argv = command(protocol, provider)
    started_epoch, started = time.time(), time.monotonic()
    plan = dict(schema_version=1, stage="Q0", provider=provider["name"], mode=provider["mode"], argv=argv,
                cwd=str(out / "cwd"), protocol_sha256=protocol_sha, wrapper_sha256=sha(__file__),
                reader_sha256=protocol["artifacts"]["reader"]["sha256"],
                expected_loaded_library_path=provider["expected_loaded_library_path"],
                binary_sha256=provider["binary"]["sha256"], library_sha256=provider["library"]["sha256"],
                target_build_manifest_sha256=protocol["artifacts"]["target_build_manifest"]["sha256"],
                private_build_manifest_sha256=protocol["artifacts"]["private_build_manifest"]["sha256"],
                request_sha256=protocol["artifacts"]["request"]["sha256"],
                config_sha256=protocol["artifacts"]["config"]["sha256"],
                MYDS_inherited={key: value for key, value in env.items() if key.startswith("MYDS_")},
                config_loaded_in_child_override_env=True, effective_MYDS_count_expected=65,
                timeout_s=protocol["timeout_s"], monitor_interval_s=protocol["monitor_interval_s"],
                shared_lock_path=protocol["lock_path"], exclusive_recognized_process_check=True,
                adoption_cohort=False, performance_acceptance=False, started_epoch=started_epoch)
    write_json(out / "execution-plan.json", plan)
    monitor, proc, reason, raw, after, model_after, final = [], None, None, None, None, None, []
    try:
        with (out / "stdout.jsonl").open("xb") as sout, (out / "stderr.txt").open("xb") as serr:
            proc = subprocess.Popen(argv, cwd=out / "cwd", env=env, stdout=sout, stderr=serr, start_new_session=True)
            while proc.poll() is None:
                current = processes()
                monitor.append(dict(elapsed_s=time.monotonic() - started, recognized=current))
                if any(row["pid"] != proc.pid for row in current):
                    reason = "recognized inference competitor"
                    terminate_owned(proc)
                    break
                if time.monotonic() - started > protocol["timeout_s"]:
                    reason = "timeout"
                    terminate_owned(proc)
                    break
                time.sleep(protocol["monitor_interval_s"])
            proc.wait(timeout=10)
        if proc.returncode != 0:
            reason = reason or "native nonzero exit"
        else:
            lines = (out / "stdout.jsonl").read_text().splitlines()
            require(len(lines) == 5 and all(line.startswith("{") for line in lines), "five complete native JSONL records")
            records = [json.loads(line) for line in lines]
            require([row["kind"] for row in records] == ["indexer_metadata", "metadata", "load", "sample", "sample"],
                    "native metadata/load/warmup/verification order")
    except BaseException as error:
        reason = reason or type(error).__name__ + ": " + str(error)
        terminate_owned(proc)
    finally:
        try:
            after = canonical_state(protocol)
            model_after, _ = model_state(protocol)
            final = processes()
            require(before == after and model_before == model_after, "canonical/model identity changed across child")
            require(not final, "recognized inference process remains after child")
            for item in frozen_inputs:
                verify_descriptor(item)
        except BaseException as error:
            reason = reason or "post-window identity: " + str(error)
        try:
            raw = raw_inventory(out)
            write_json(out / "raw-inventory.json", raw)
        except BaseException as error:
            reason = reason or "raw inventory: " + str(error)
        write_json(out / "monitor.json", monitor)
        result = dict(schema_version=1, stage="Q0", provider=provider["name"], mode=provider["mode"],
                      child_pid=proc.pid if proc else None, exit_code=proc.returncode if proc else None,
                      abort_reason=reason, started_epoch=started_epoch, finished_epoch=time.time(),
                      elapsed_s=time.monotonic() - started, canonical_before=before, canonical_after=after,
                      canonical_unchanged=before == after, model_before=model_before, model_after=model_after,
                      model_unchanged=model_before == model_after, recognized_before=initial, recognized_after=final,
                      protocol_sha256=protocol_sha, wrapper_sha256=plan["wrapper_sha256"],
                      reader_sha256=plan["reader_sha256"],
                      expected_loaded_library_path=provider["expected_loaded_library_path"],
                      raw_inventory_sha256=sha(out / "raw-inventory.json") if raw else None,
                      raw_tree_sha256=raw["tree_sha256"] if raw else None,
                      external_reader_required=True, qualification_passed=None, performance_acceptance=False)
        write_json(out / "window.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("/private/tmp/myds-indexer-r3-20261025/fullmodel-q-protocol.json"))
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text())
    require(protocol["kind"] == "indexer_fullmodel_q0_qualification_protocol" and protocol["schema_version"] == 1 and
            protocol["stage"] == "Q0" and protocol["performance_acceptance"] is False and
            [p["name"] for p in protocol["providers"]] == ["G_lib", "off", "on"] and
            [p["mode"] for p in protocol["providers"]] == [0, 0, 1], "frozen three-provider Q0 scope")
    require(protocol["wrapper_sha256"] == sha(__file__), "frozen wrapper source identity")
    out = Path(protocol["window_root"])
    require(not out.exists(), "new exclusive Q0 evidence root required")
    # A terminated wrapper must unwind through the same owned-child cleanup
    # and retained evidence path as a timeout, rather than orphaning its GPU job.
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(protocol["lock_path"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen_inputs = list(protocol["artifacts"].values()) + [dict(path=str(protocol_path), bytes=protocol_path.stat().st_size, sha256=sha(protocol_path))]
        frozen_inputs += [p[kind] for p in protocol["providers"] for kind in ("binary", "library")]
        for item in frozen_inputs:
            verify_descriptor(item)
        before = canonical_state(protocol)
        model_before, prefix = model_state(protocol)
        initial = processes()
        require(not initial, "recognized inference competitor before Q0")
        out.mkdir(exist_ok=False)
        (out / "model-prefix.bin").write_bytes(prefix)
        protocol_sha = sha(protocol_path)
        write_json(out / "execution-plan.json", dict(stage="Q0", protocol_sha256=protocol_sha,
            wrapper_sha256=sha(__file__), run_order=["G_lib", "off", "on"], timeout_s=protocol["timeout_s"],
            reader_sha256=protocol["artifacts"]["reader"]["sha256"],
            shared_lock_path=protocol["lock_path"], started_epoch=time.time(), performance_acceptance=False))
        results, reason = [], None
        try:
            for provider in protocol["providers"]:
                result = collect_provider(protocol, provider, out / provider["name"], protocol_sha, frozen_inputs)
                results.append(result)
                if result["exit_code"] != 0 or result["abort_reason"]:
                    reason = "provider collection stopped: " + provider["name"]
                    break
        except BaseException as error:
            reason = type(error).__name__ + ": " + str(error)
        finally:
            after, model_after, final = None, None, []
            try:
                after = canonical_state(protocol)
                model_after, _ = model_state(protocol)
                final = processes()
                require(before == after and model_before == model_after and not final, "final Q0 identity/exclusivity")
                require(sha(protocol_path) == protocol_sha and sha(__file__) == protocol["wrapper_sha256"], "protocol/wrapper unchanged")
            except BaseException as error:
                reason = reason or "final-window identity: " + str(error)
            complete = len(results) == 3 and all(r["exit_code"] == 0 and r["abort_reason"] is None for r in results) and reason is None
            result = dict(schema_version=1, stage="Q0", run_order=["G_lib", "off", "on"], providers=results,
                abort_reason=reason, execution_complete=complete, protocol_sha256=protocol_sha,
                wrapper_sha256=protocol["wrapper_sha256"], canonical_before=before, canonical_after=after,
                reader_sha256=protocol["artifacts"]["reader"]["sha256"],
                canonical_unchanged=before == after, model_before=model_before, model_after=model_after,
                model_unchanged=model_before == model_after, recognized_before=initial, recognized_after=final,
                model_prefix=dict(path=str(out / "model-prefix.bin"), bytes=len(prefix), sha256=hashlib.sha256(prefix).hexdigest()),
                lock_held_through_final_checks=True, service_or_production_mutation_performed=False,
                external_reader_required=True, qualification_passed=None, fullmodel_speedup_proven=False,
                production_acceptance=False, performance_acceptance=False, finished_epoch=time.time())
            write_json(out / "window.json", result)
        print(json.dumps(result), flush=True)
        return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
