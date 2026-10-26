#!/usr/bin/env python3
"""Serial HTTP functional bridge for the default-off register-only scorer.

The caller owns the exclusive GPU window and stops its original service first.
This tool only starts and terminates its own three temporary servers. No retry,
production deployment, native timing reconstruction, or HTTP speed gate occurs.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
WORKLOADS = ("business96", "code38404")
N = 256
PORT = 8198
HEALTH_TIMEOUT = 120
INFERENCE_TIMEOUT = 900
MODE = "MYDS_INDEXER_REGISTER_ONLY"


class BridgeError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise BridgeError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def descriptor(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def clean_environment(mode):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS_")}
    if mode is not None:
        need(mode in (0, 1), "Candidate mode must be zero or one")
        env[MODE] = str(mode)
    return env


def inspect_config(path):
    values = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        need("=" in line, f"Malformed configuration line {number}")
        key, value = (part.strip() for part in line.split("=", 1))
        value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        need(key.startswith("MYDS_"), f"Non-MYDS configuration line {number}")
        need(not key.startswith("MYDS_KV_DSK"), "Disk-cache settings are forbidden in this HTTP bridge")
        need(key != MODE, "Config must not override the single candidate intervention")
        need(key != "MYDS_RESP_STATE", "Config must not override isolated response-state storage")
        need(key not in values, f"Duplicate configuration key: {key}")
        values[key] = value
    need(values, "Empty configuration")
    return values


def inference_processes():
    result = subprocess.run(["ps", "-axo", "pid=,comm="], capture_output=True, text=True, check=True)
    found = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        pid, command = int(fields[0]), fields[1]
        name = Path(command).name
        if name == "myds" or name.startswith(("myds-", "myds_", "bench_")):
            found.append({"pid": pid, "command": command})
    return found


def exclusive(owned_pid=None):
    processes = inference_processes()
    unexpected = [p for p in processes if p["pid"] != owned_pid]
    need(not unexpected, f"Unexpected inference processes; no signals sent: {unexpected}")
    return processes


def port_free(port):
    need(port == PORT, "This frozen bridge uses only port 8198")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise BridgeError(f"Port {port} is not free; no process was stopped") from exc


def loaded_library(log_text, expected):
    matches = re.findall(r"loaded precompiled library from (.*?) \(all required V4\.1 kernels verified\)", log_text)
    need(len(matches) == 1, "Startup log does not identify exactly one loaded precompiled library path")
    actual = Path(matches[0])
    need(actual.is_absolute(), "Loaded library log must identify an absolute path in this isolated cwd")
    need(actual.resolve() == Path(expected).resolve(), "Actual loaded library differs from executable-adjacent library")
    return str(actual.resolve())


def prepare_request(path):
    value = json.loads(Path(path).read_text())
    need(isinstance(value, dict) and isinstance(value.get("messages"), list) and value["messages"], "Invalid retained request")
    need(value.get("max_tokens") == N and value.get("temperature") == 0 and value.get("thinking") is False,
         "Retained request must explicitly specify N256, temperature zero, and thinking false")
    need(not value.get("tools") and not value.get("previous_response_id"), "Expected standalone retained text request")
    request = dict(value)
    request["stream"] = True
    request["stream_options"] = {"include_usage": True}
    return json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()


def counts(usage):
    need(isinstance(usage, dict), "Missing final usage")
    values = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        need(type(value) is int and value >= 0, "Invalid usage." + key)
        values.append(value)
    need(values[0] > 0 and values[1] == N and values[2] == values[0] + values[1], "Incomplete token usage")
    return tuple(values)


class StreamState:
    def __init__(self, started):
        self.started = started
        self.first_text = None
        self.last_text = None
        self.usage = None
        self.finish = None
        self.done = False
        self.pieces = []
        self.content = []
        self.reasoning = []
        self.visible_events = 0

    def event(self, data, arrival):
        need(not self.done, "Unexpected SSE data after DONE")
        text = data.decode("utf-8")
        if text.strip() == "[DONE]":
            self.done = True
            return
        event = json.loads(text)
        need(isinstance(event, dict) and not event.get("error"), "Invalid/error SSE event")
        if event.get("usage") is not None:
            need(isinstance(event["usage"], dict), "Invalid usage object")
            need(self.usage is None or self.usage == event["usage"], "Conflicting usage objects")
            self.usage = event["usage"]
        choices = event.get("choices", [])
        need(isinstance(choices, list) and len(choices) <= 1, "Exactly one output choice is supported")
        if not choices:
            return
        choice = choices[0]
        need(isinstance(choice, dict), "Invalid output choice")
        finish = choice.get("finish_reason")
        if finish is not None:
            need(self.finish is None or self.finish == finish, "Conflicting finish reasons")
            self.finish = finish
        delta = choice.get("delta", {})
        need(isinstance(delta, dict) and not delta.get("tool_calls"), "Unexpected structured/tool delta")
        visible = False
        for key, collection in (("reasoning_content", self.reasoning), ("content", self.content)):
            piece = delta.get(key)
            need(piece is None or isinstance(piece, str), "Non-text SSE delta")
            if piece:
                collection.append(piece)
                self.pieces.append(piece)
                visible = True
        if visible:
            if self.first_text is None:
                self.first_text = arrival
            self.last_text = arrival
            self.visible_events += 1

    def finish_check(self):
        need(self.done and self.finish == "length", "Missing DONE or non-length completion")
        counts(self.usage)
        need(self.first_text is not None and self.pieces, "Empty output cannot prove equivalence")

    def record(self, ended):
        return {"started_monotonic_s": self.started, "first_visible_text_monotonic_s": self.first_text,
                "last_visible_text_monotonic_s": self.last_text, "ended_monotonic_s": ended,
                "http_visible_ttft_s": None if self.first_text is None else self.first_text - self.started,
                "http_total_s": ended - self.started, "visible_events_not_tokens": self.visible_events,
                "usage": self.usage, "finish_reason": self.finish, "done": self.done,
                "native_prefill_timing_available": False, "native_decode_timing_available": False}


def exchange(port, request, output, interrupt_after_events=None):
    output.mkdir()
    (output / "request.json").write_bytes(request)
    started = time.monotonic()
    state = StreamState(started)
    result = {"started_epoch_ns": time.time_ns(), "interruption_requested": interrupt_after_events is not None,
              "intentionally_interrupted": False, "success": False}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=INFERENCE_TIMEOUT)
    response = None
    sock = None
    try:
        connection.connect()
        sock = connection.sock
        result["client_port"] = sock.getsockname()[1]
        connection.request("POST", "/v1/chat/completions", body=request,
                           headers={"Content-Type": "application/json", "Connection": "close"})
        response = connection.getresponse()
        write_json(output / "response-headers.json", {"status": response.status, "headers": response.getheaders()})
        if response.status != 200:
            (output / "response.sse.bin").write_bytes(response.read())
        need(response.status == 200, f"HTTP status {response.status}")
        need("text/event-stream" in response.getheader("Content-Type", ""), "Response is not SSE")
        data = []
        with (output / "response.sse.bin").open("xb") as raw, (output / "arrivals.jsonl").open("x") as arrivals:
            while True:
                remaining = INFERENCE_TIMEOUT - (time.monotonic() - started)
                need(remaining > 0, "Total inference timeout exceeded")
                sock.settimeout(remaining)
                line = response.readline()
                arrival = time.monotonic()
                if not line:
                    break
                offset = raw.tell()
                raw.write(line)
                raw.flush()
                arrivals.write(json.dumps({"offset": offset, "bytes": len(line), "monotonic_s": arrival}) + "\n")
                arrivals.flush()
                clean = line.rstrip(b"\r\n")
                if clean.startswith(b"data:"):
                    data.append(clean[5:].lstrip(b" "))
                elif not clean and data:
                    state.event(b"\n".join(data), arrival)
                    data.clear()
                    if interrupt_after_events is not None and state.visible_events >= interrupt_after_events:
                        need(not state.done and state.finish is None, "Response completed before interruption")
                        sock.shutdown(socket.SHUT_RDWR)
                        result["intentionally_interrupted"] = True
                        break
                    if state.done:
                        break
            if data and not result["intentionally_interrupted"]:
                state.event(b"\n".join(data), time.monotonic())
        if interrupt_after_events is None:
            state.finish_check()
        else:
            need(result["intentionally_interrupted"], "No decode stream interruption occurred")
        result["success"] = True
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if response is not None:
            response.close()
        connection.close()
        if not (output / "response.sse.bin").exists():
            (output / "response.sse.bin").write_bytes(b"")
        result.update(state.record(time.monotonic()))
        for name, pieces in (("output.utf8.bin", state.pieces), ("content.utf8.bin", state.content),
                             ("reasoning.utf8.bin", state.reasoning)):
            (output / name).write_bytes("".join(pieces).encode())
        write_json(output / "observation.json", result)


def completed_record(data, client_port):
    text = data.decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines()
             if f"[127.0.0.1:{client_port}] Completed ChatCompletions /v1/chat/completions - " in line]
    need(len(lines) <= 1, "Ambiguous request completion log")
    if not lines:
        return None
    line = lines[0]
    prompt = re.search(r"prompt: (\d+) tok, prefill: (.*?), ttft:", line)
    decode = re.search(r"decode: (\d+) tok in .*finish_reason: (\w+)", line)
    need(prompt is not None and decode is not None, "Incomplete completion record")
    stage = prompt.group(2)
    cached = re.search(r"\(cached (\d+)\)", stage)
    all_cached = re.fullmatch(r"none \(all (\d+) cached\)", stage)
    need(cached is not None or all_cached is not None, "Missing actual request cached count")
    need(not stage.startswith("FAILED"), "Server reported prefill failure")
    return {"line": line, "prompt_tokens": int(prompt.group(1)),
            "cached_tokens": int((cached or all_cached).group(1)),
            "decode_tokens": int(decode.group(1)), "finish_reason": decode.group(2),
            "log_times_are_rounded_diagnostics_not_native_samples": True}


def wait_completed(log_path, offset, client_port, process, output):
    deadline = time.monotonic() + 120
    while True:
        with log_path.open("rb") as f:
            f.seek(offset)
            data = f.read()
        (output / "server-request-interval.log").write_bytes(data)
        record = completed_record(data, client_port)
        if record is not None:
            record.update({"log_start_offset": offset, "log_end_offset": offset + len(data), "client_port": client_port})
            write_json(output / "server-completion.json", record)
            return record
        need(process.poll() is None, "Temporary service exited before its completion record")
        need(time.monotonic() < deadline, "No matching server completion record within 120 seconds")
        time.sleep(.1)


def check_completed(record, observation, cache):
    expected = counts(observation["usage"])
    need(record["prompt_tokens"] == expected[0] and record["decode_tokens"] == N and record["finish_reason"] == "length",
         "SSE usage and matching server completion differ")
    if cache == "cold":
        need(record["cached_tokens"] == 0, "Cold request actually reused cache; retain failure without changing input")
    elif cache == "warm":
        need(record["cached_tokens"] > 0, "Warm request did not actually reuse cache")


def compare_outputs(left, right):
    for name in ("output.utf8.bin", "content.utf8.bin", "reasoning.utf8.bin"):
        need((left / name).read_bytes() == (right / name).read_bytes(), f"HTTP output mismatch: {left} vs {right} ({name})")
    a, b = (json.loads((p / "observation.json").read_text()) for p in (left, right))
    need(counts(a["usage"]) == counts(b["usage"]), "HTTP token usage mismatch")


def health(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/health", headers={"Connection": "close"})
        response = connection.getresponse()
        body = response.read()
        return response.status == 200 and json.loads(body).get("status") == "ok"
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def terminate_owned(process):
    # Popen retains ownership of this exact child; never discover or signal a
    # PID from a port, process-name match, stale file, or an external service.
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise BridgeError(f"Owned child PID {process.pid} did not stop after SIGTERM; no SIGKILL sent") from exc
    return {"pid": process.pid, "returncode": process.returncode, "terminated": process.poll() is not None}


def run_arm(name, executable, mode, args, requests, summary):
    root = args.stage_root / name
    root.mkdir()
    cwd = root / "cwd"
    cwd.mkdir()
    log_path = root / "service.log"
    library = executable.parent / "myds.metallib"
    identities = {"executable": descriptor(executable), "library": descriptor(library), "config": descriptor(args.config)}
    env = clean_environment(mode)
    # Cancellation invalidates response IDs even for ChatCompletions. Its
    # default store is under HOME, so all three arms must isolate that write.
    env["MYDS_RESP_STATE"] = str(root / "responses.idx")
    command = [str(executable), "--config", str(args.config), "--model", str(args.model),
               "--ctx", "1048576", "--max-tokens", str(N), "--temp", "0", "--top-p", "1",
               "--no-thinking", "--host", "127.0.0.1", "--port", str(args.port), "--alias", "myds-http-bridge"]
    write_json(root / "launch.json", {"argv": command, "cwd": str(cwd), "inherited_MYDS_removed": True,
                                     "MYDS_child_additions": {k: v for k, v in env.items() if k.startswith("MYDS_")},
                                     "identities": identities, "prelaunch_processes": exclusive()})
    port_free(args.port)
    process = None
    receipt = {"arm": name, "passed": False}
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt["pid"] = process.pid
            write_json(root / "process.json", {"pid": process.pid, "owned_child": True})
            deadline = time.monotonic() + HEALTH_TIMEOUT
            while not health(args.port):
                need(process.poll() is None, "Temporary service exited during startup")
                need(time.monotonic() < deadline, "Service health timeout after 120 seconds")
                time.sleep(.25)
            receipt["loaded_library"] = loaded_library(log_path.read_text(errors="replace"), library)
            for workload in WORKLOADS:
                for cache in ("cold", "warm"):
                    write_json(root / (workload + "-" + cache + ".preflight.json"),
                               {"owned_pid": process.pid, "processes": exclusive(process.pid)})
                    output = root / (workload + "-" + cache)
                    offset = log_path.stat().st_size
                    observation = exchange(args.port, requests[workload], output)
                    summary["completed_standard_generations"] += 1
                    record = wait_completed(log_path, offset, observation["client_port"], process, output)
                    check_completed(record, observation, cache)
                    write_json(output / "processes-after.json", {"owned_pid": process.pid, "processes": exclusive(process.pid)})
                    if cache == "warm":
                        compare_outputs(root / (workload + "-cold"), output)
                    if name != "gold" and cache == "cold":
                        compare_outputs(args.stage_root / "gold" / (workload + "-cold"), output)
            if name == "on":
                write_json(root / "decode-interruption.preflight.json", {"owned_pid": process.pid, "processes": exclusive(process.pid)})
                output = root / "decode-interruption"
                offset = log_path.stat().st_size
                interrupted = exchange(args.port, requests["code38404"], output, interrupt_after_events=8)
                record = wait_completed(log_path, offset, interrupted["client_port"], process, output)
                need(record["finish_reason"] == "cancelled" and 0 < record["decode_tokens"] < N,
                     "Server did not confirm cancellation during decode before N256")
                need(process.poll() is None and health(args.port), "Service not healthy after cancellation")
                output = root / "code38404-recovery"
                offset = log_path.stat().st_size
                recovered = exchange(args.port, requests["code38404"], output)
                summary["completed_recovery_generations"] += 1
                record = wait_completed(log_path, offset, recovered["client_port"], process, output)
                check_completed(record, recovered, None)
                compare_outputs(root / "code38404-cold", output)
                need(process.poll() is None and health(args.port), "Service not healthy after recovery")
                summary["decode_cancellation_and_recovery_passed"] = True
            exclusive(process.pid)
            need(identities == {"executable": descriptor(executable), "library": descriptor(library), "config": descriptor(args.config)},
                 "Executable, library or configuration changed during this arm")
            receipt["passed"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if process is not None:
                receipt["cleanup"] = terminate_owned(process)
        except Exception as exc:
            receipt["passed"] = False
            receipt["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_json(root / "arm-result.json", receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage-root", "gold-exe", "candidate-exe", "config", "model", "inputs-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--port", default=PORT, type=int)
    args = parser.parse_args(argv)
    need(args.stage_root.is_absolute() and not args.stage_root.exists(), "Stage root must be a new absolute directory; no retries")
    args.stage_root.mkdir(parents=True)
    summary = {"kind": "register_only_HTTP_functional_bridge", "http_functional_bridge_passed": False,
               "completed_standard_generations": 0, "required_standard_generations": 12,
               "completed_recovery_generations": 0, "decode_cancellation_and_recovery_passed": False,
               "full_state_equivalence_validated": False, "per_token_logits_validated": False,
               "native_performance_gate_evaluated": False, "HTTP_speedup_gate_evaluated": False,
               "production_deployed": False, "no_retry_or_replacement": True,
               "cold_cross_arm_outputs_usage_equal": False, "warm_repeat_outputs_usage_equal": False,
               "cold_cache_zero_verified": False, "warm_cache_reuse_verified": False,
               "candidate_dispatch_counters_available": False,
               "disk_cache_access_authorized": False, "request_timeout_s": INFERENCE_TIMEOUT,
               "health_timeout_s": HEALTH_TIMEOUT, "started_epoch_ns": time.time_ns()}
    prior_signals = {}
    def interrupted(signum, frame):
        raise BridgeError(f"Bridge interrupted by signal {signum}; cleaning up owned service")
    for signum in (signal.SIGTERM, signal.SIGINT):
        prior_signals[signum] = signal.signal(signum, interrupted)
    try:
        for key in ("gold_exe", "candidate_exe", "config", "model", "inputs_root"):
            setattr(args, key, getattr(args, key).resolve(strict=True))
        inspect_config(args.config)
        for executable in (args.gold_exe, args.candidate_exe):
            need(executable.is_file() and os.access(executable, os.X_OK), "Executable unavailable")
            need((executable.parent / "myds.metallib").is_file(), "Executable-adjacent metallib unavailable")
        need(args.gold_exe != args.candidate_exe, "Gold and candidate must be distinct executable files")
        requests = {name: prepare_request(args.inputs_root / (name + ".request.json")) for name in WORKLOADS}
        write_json(args.stage_root / "protocol.json", {"argv": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "requests": {name: {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()} for name, body in requests.items()},
                   "config": descriptor(args.config), "arms": ["gold", "off", "on"],
                   "model_stat": {"path": str(args.model), "bytes": args.model.stat().st_size,
                                  "inode": args.model.stat().st_ino, "mtime_ns": args.model.stat().st_mtime_ns,
                                  "whole_model_SHA_verified": False},
                   "response_state_isolated_equally_in_all_arms": True,
                   "order_per_arm": [name + "-" + cache for name in WORKLOADS for cache in ("cold", "warm")],
                   "source_request_files": {name: descriptor(args.inputs_root / (name + ".request.json")) for name in WORKLOADS}})
        for name, executable, mode in (("gold", args.gold_exe, None), ("off", args.candidate_exe, 0), ("on", args.candidate_exe, 1)):
            run_arm(name, executable, mode, args, requests, summary)
        need(summary["completed_standard_generations"] == 12 and summary["decode_cancellation_and_recovery_passed"], "Incomplete HTTP bridge")
        summary["http_functional_bridge_passed"] = True
        for key in ("cold_cross_arm_outputs_usage_equal", "warm_repeat_outputs_usage_equal",
                    "cold_cache_zero_verified", "warm_cache_reuse_verified"):
            summary[key] = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        summary["finished_epoch_ns"] = time.time_ns()
        write_json(args.stage_root / "summary.json", summary)
        for signum, previous in prior_signals.items():
            signal.signal(signum, previous)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["http_functional_bridge_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
