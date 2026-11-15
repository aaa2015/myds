#!/usr/bin/env python3
"""tests/test_evo_loop.py — Deterministic verification of Silicon Self-Evolution.

Tests all 4 autonomous evolution components without requiring the 337 GiB GGUF:
  1. `bin/evo_pso_probe` + `tools/evo_silicon_probe.py` (PSO ceilings, diff,
     §6.1 `cbs/layers` roofline math, §5.1 `gpu_gap`, §6.2 `kprof` guard)
  2. `tools/evo_memory.py` + `docs/ledger-index.jsonl` (freshness, query, trial log)
  3. `tools/evo_judge.py` (`--config` isolation, `--print-config` real `bin/myds`
     gate, verdict matrix, and end-to-end `--mode full` ABABAB over a compiled
     Mach-O `myds` HTTP harness covering ADOPT, VOID_IDENTICAL_ARMS, cache-hit
     VOID, and non-0-ULP REJECT)
  4. `tools/evo_supervisor.py` (atomic inode `install` without mutating mapped
     pages per §7, live `status`, zero-downtime preflight rejection, and active
     TCP client drain protection)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import evo_judge  # noqa: E402
import evo_memory  # noqa: E402
import evo_silicon_probe  # noqa: E402
import evo_supervisor  # noqa: E402


MOCK_MYDS_C = r"""
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;
static bool g_fast_arm = false;
static bool g_corrupt_ulp = false;
static int g_fake_cached = 0;
static char g_dbg_tok[128] = {0};
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_seq = 0;

static void on_sig(int sig) { (void)sig; g_stop = 1; }

static void *handle_client(void *arg) {
    int c = (int)(intptr_t)arg;
    char req[16384];
    ssize_t n = 0;
    while (n < (ssize_t)sizeof(req) - 1) {
        ssize_t got = recv(c, req + n, sizeof(req) - 1 - (size_t)n, 0);
        if (got <= 0) break;
        n += got;
        req[n] = '\0';
        char *hdr_end = strstr(req, "\r\n\r\n");
        if (hdr_end) {
            size_t body_have = (size_t)(n - (hdr_end + 4 - req));
            size_t content_len = 0;
            char *cl = strstr(req, "Content-Length:");
            if (!cl) cl = strstr(req, "content-length:");
            if (cl && cl < hdr_end) content_len = (size_t)atoi(cl + 15);
            if (body_have >= content_len) break;
        }
    }
    if (n <= 0) { close(c); return NULL; }
    req[n] = '\0';

    char body[4096];
    int status = 200;
    pthread_mutex_lock(&g_mu);
    if (strncmp(req, "GET /health", 11) == 0) {
        snprintf(body, sizeof(body), "{\"status\":\"ok\"}");
    } else if (strncmp(req, "POST /v1/chat/completions", 25) == 0) {
        g_seq++;
        const char *ptag = "generic";
        if (strstr(req, "[ulp-gate-p0]")) ptag = "p0";
        else if (strstr(req, "[ulp-gate-p1]")) ptag = "p1";
        else if (strstr(req, "[ulp-gate-p2]")) ptag = "p2";
        snprintf(body, sizeof(body),
                 "{\"choices\":[{\"message\":{\"role\":\"assistant\","
                 "\"content\":\"det-ans-%s-%s\"}}]}",
                 ptag, g_corrupt_ulp ? "DRIFT" : "EXACT");
    } else if (strncmp(req, "GET /trace/api/requests", 23) == 0) {
        if (g_dbg_tok[0] && !strstr(req, g_dbg_tok)) {
            status = 403;
            snprintf(body, sizeof(body), "{\"error\":\"forbidden\"}");
        } else {
            double dtps = g_fast_arm ? 49.6 : 45.0;
            double ptps = g_fast_arm ? 735.0 : 714.0;
            int ctok = g_fake_cached;
            int off = snprintf(body, sizeof(body), "{\"count\":%d,\"capacity\":64,\"records\":[", g_seq);
            for (int k = g_seq; k >= 1 && off < (int)sizeof(body) - 256; --k) {
                off += snprintf(body + off, sizeof(body) - (size_t)off,
                    "%s{\"seq\":%d,\"prompt_tokens\":512,\"cached_tokens\":%d,"
                    "\"out_tokens\":64,\"prefill_tps\":%.2f,\"decode_tps\":%.2f}",
                    (k == g_seq) ? "" : ",", k, (k == g_seq) ? ctok : 0, ptps, dtps);
            }
            snprintf(body + off, sizeof(body) - (size_t)off, "]}");
        }
    } else {
        status = 404;
        snprintf(body, sizeof(body), "{\"error\":\"not found\"}");
    }
    pthread_mutex_unlock(&g_mu);

    char resp[8192];
    int rlen = snprintf(resp, sizeof(resp),
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n%s",
        status, (status == 200) ? "OK" : "ERR", strlen(body), body);
    (void)send(c, resp, (size_t)rlen, 0);
    close(c);
    return NULL;
}

int main(int argc, char **argv) {
    const char *conf_path = NULL;
    int port = 18187;
    bool print_cfg = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) conf_path = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--print-config") == 0) print_cfg = true;
    }

    if (conf_path) {
        FILE *f = fopen(conf_path, "r");
        if (!f) { fprintf(stderr, "cannot open conf\n"); return 2; }
        char line[1024];
        while (fgets(line, sizeof(line), f)) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = '\0';
            if (line[0] == '#' || line[0] == '\0') continue;
            if (strncmp(line, "MYDS_BAD_UNKNOWN_KEY=", 21) == 0) {
                fprintf(stderr, "unknown config key MYDS_BAD_UNKNOWN_KEY\n");
                fclose(f);
                return 2;
            }
            if (strcmp(line, "MYDS_INDEXER_TOPK_RADIX_CAUSAL=1") == 0) g_fast_arm = true;
            if (strcmp(line, "MYDS_INDEXER_RO_SMEM_Q=0") == 0) g_corrupt_ulp = true;
            if (strcmp(line, "MYDS_INDEXER_PREFILL_COMPACT=0") == 0) g_fake_cached = 128;
            if (strncmp(line, "MYDS_DEBUG_TOKEN=", 17) == 0) {
                strncpy(g_dbg_tok, line + 17, sizeof(g_dbg_tok) - 1);
            }
        }
        fclose(f);
    }

    if (print_cfg) {
        printf("OK\n");
        return 0;
    }
    FILE *ml = fopen("./bin/myds.metallib", "rb");
    if (!ml) {
        fprintf(stderr, "missing staged ./bin/myds.metallib in cwd\n");
        return 1;
    }
    fclose(ml);

    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(srv, 16) != 0) {
        perror("bind/listen");
        return 1;
    }

    while (!g_stop) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(srv, &rfds);
        struct timeval tv = {0, 100000};
        int r = select(srv + 1, &rfds, NULL, NULL, &tv);
        if (r <= 0) continue;
        int c = accept(srv, NULL, NULL);
        if (c < 0) continue;
        pthread_t th;
        if (pthread_create(&th, NULL, handle_client, (void *)(intptr_t)c) == 0) {
            pthread_detach(th);
        } else {
            close(c);
        }
    }
    close(srv);
    return 0;
}
"""


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_silicon_probe() -> None:
    probe_bin = ROOT / "bin" / "evo_pso_probe"
    metallib = ROOT / "bin" / "myds.metallib"
    assert probe_bin.is_file(), f"Missing {probe_bin}"
    assert metallib.is_file(), f"Missing {metallib}"

    pso = evo_silicon_probe.run_pso_probe(metallib, probe_bin)
    ml = pso["metallib"]
    hw = pso["device"]
    assert ml["compiled_count"] == ml["matched_functions"]
    assert ml["compiled_count"] > 500
    assert ml["failed_count"] == 0
    assert hw["max_tg_mem_bytes"] == 32768
    assert hw["apple_gpu_family_max"] >= 8

    kmap = {k["name"]: k for k in pso["kernels"]}
    # Verify [P-T1] §176 radix Top-K invariant: 4096 B smem -> 8 TGs/core!
    radix_k = kmap["kernel_myds_topk_radix_pass1"]
    assert radix_k["static_tg_mem_bytes"] == 4096
    assert radix_k["max_tgs_per_core_by_smem"] == 8
    assert radix_k["category"] == "indexer"

    # Test PSO diff regression detection
    cand_fake = {
        "kernels": [
            {
                **radix_k,
                "static_tg_mem_bytes": 16384,
                "max_tgs_per_core_by_smem": 2,
            }
        ]
    }
    base_sub = {"kernels": [radix_k]}
    d = evo_silicon_probe.diff_pso(base_sub, cand_fake)
    assert len(d["occupancy_or_register_regressions"]) == 1
    assert d["occupancy_or_register_regressions"][0]["max_tgs_per_core_by_smem"] == [8, 2]

    # Test §6.1 cbs/layers normalization (1000 CBs / 40 layers = 25 sampled tokens, NOT out_tokens=200!)
    rec = {
        "prompt_tokens": 32768,
        "out_tokens": 200,
        "phases": {"prefill": 44.5, "decode": 4.40},
        "detail": {
            "decode_gpu": 4.28,
            "dtags": [
                [25, 0.250, 1000],  # DEC_FFN: 40 layers/tok -> 25 sampled tokens -> 10,000 us/tok
            ],
        },
    }
    tr = evo_silicon_probe.analyze_trace_record(rec, peak_gbps=800.0, decode_topk=3)
    assert tr["gpu_gap"]["overlappable_before_first_cb"] is False
    assert len(tr["dtags_per_token"]) == 1
    row = tr["dtags_per_token"][0]
    assert row["sampled_tokens"] == 25.0
    assert abs(row["us_per_token"] - 10000.0) < 0.01
    assert "§6.1" in row["normalization"]
    assert 35.0 < tr["decode_roofline"]["roof_utilization_pct"] < 55.0

    # Test peak bandwidth table for the 5 target Mac Studio models
    assert evo_silicon_probe.infer_peak_gbps("Apple M2 Ultra", None)[0] == 800.0
    assert evo_silicon_probe.infer_peak_gbps("Apple M3 Ultra", None)[0] == 819.0
    assert evo_silicon_probe.infer_peak_gbps("Apple M5 Ultra", None)[0] == 1024.0

    # Test §6.2 kprof join guard
    with tempfile.TemporaryDirectory() as td:
        kp = Path(td) / "sample.kprof.jsonl"
        kp.write_text(
            json.dumps({"kind": "kprof_run", "mode": "decode"}) + "\n"
            + json.dumps(
                {
                    "kind": "kprof_kernel",
                    "name": "kernel_myds_topk_radix_pass1",
                    "calls": 10,
                    "sum_us": 5000.0,
                    "min_us": 480.0,
                    "max_us": 520.0,
                    "grid": [8, 1, 1],
                    "threads": [256, 1, 1],
                    "merged_encoders": 0,
                }
            ) + "\n"
            + json.dumps({"kind": "kprof_summary", "tokens": 10, "floor_us": 2.0}) + "\n",
            encoding="utf-8",
        )
        joined = evo_silicon_probe.join_kprof_with_pso(kp, pso, top_n=5)
        assert joined["serialized_only_no_bandwidth_claim"] is True
        assert joined["top_kernels"][0]["pso"]["static_tg_mem_bytes"] == 4096


def test_evo_memory() -> None:
    # 1. Verify tracked docs/ledger-index.jsonl matches ledger byte-for-byte
    rc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "evo_memory.py"), "--check"],
        capture_output=True,
        text=True,
    ).returncode
    assert rc == 0, "docs/ledger-index.jsonl out of date"

    records = [
        json.loads(ln)
        for ln in (ROOT / "docs" / "ledger-index.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(records) >= 240
    # Must include §176 [P-T1] and disproven MoE entries
    assert any("MYDS_INDEXER_TOPK_RADIX_CAUSAL" in json.dumps(r) for r in records)
    q_moe = evo_memory.query_index(
        argparse.Namespace(
            subsystem="moe",
            verdict=None,
            knob=None,
            kernel=None,
            text="float4",
            limit=25,
        )
    )
    assert q_moe["counts"]["total_matched"] >= 1
    assert len(q_moe["disproven_rules"]) >= 1

    # 2. Test sandboxed trial recording without mutating repo files
    with tempfile.TemporaryDirectory() as td:
        orig_trials = evo_memory.TRIALS_PATH
        orig_index = evo_memory.INDEX_PATH
        try:
            evo_memory.TRIALS_PATH = Path(td) / "evo_trials.jsonl"
            evo_memory.INDEX_PATH = Path(td) / "ledger-index.jsonl"
            ns = argparse.Namespace(
                hypothesis="Split Q2_K down into 2-row simdgroups on M2 Ultra",
                verdict="REJECT",
                why="Regressed +18% due to halved reuse",
                delta_pct=-18.0,
                ulp_zero="true",
                hardware="M2 Ultra 192GB",
                subsystem="moe",
                knob="",
                kernel="kernel_mul_mv_q2_K_f32",
            )
            entry = evo_memory.record_trial(ns)
            assert entry["id"].startswith("trial:")
            hits = evo_memory.query_index(
                argparse.Namespace(
                    subsystem="moe",
                    verdict="REJECT",
                    knob=None,
                    kernel="kernel_mul_mv_q2_K_f32",
                    text="2-row",
                    limit=10,
                )
            )
            assert len(hits["autonomous_trials"]) == 1
            assert hits["autonomous_trials"][0]["id"] == entry["id"]
        finally:
            evo_memory.TRIALS_PATH = orig_trials
            evo_memory.INDEX_PATH = orig_index


def test_evo_judge_and_supervisor() -> None:
    with tempfile.TemporaryDirectory(prefix="test-evo-") as td:
        tdir = Path(td).resolve()

        # 1. Test write_isolated_conf & real bin/myds --print-config enforcement
        bad_base = tdir / "bad.conf"
        bad_base.write_text("MYDS_PREFIX_CACHE=0\n", encoding="utf-8")
        try:
            evo_judge.write_isolated_conf(
                bad_base, tdir / "out.conf", tdir / "kv", tdir / "r.idx", "tok", {}
            )
            raise AssertionError("Expected §A.5.1 RuntimeError on MYDS_PREFIX_CACHE=0")
        except RuntimeError as exc:
            assert "§A.5.1" in str(exc)

        good_base = tdir / "base.conf"
        good_base.write_text(
            "# Production baseline\nMYDS_PREFIX_CACHE=1\nMYDS_MOE_DECODE_TOPK=3\n",
            encoding="utf-8",
        )
        iso_conf = tdir / "iso.conf"
        merged = evo_judge.write_isolated_conf(
            good_base,
            iso_conf,
            tdir / "kv_iso",
            tdir / "resp_iso.idx",
            "secret123",
            {"MYDS_INDEXER_TOPK_RADIX_CAUSAL": "1"},
        )
        assert merged["MYDS_PREFIX_CACHE"] == "1"
        assert merged["MYDS_KV_DSK_BUDGET_MB"] == "0"

        real_myds = ROOT / "bin" / "myds"
        if real_myds.is_file():
            ok_chk = subprocess.run(
                [str(real_myds), "--config", str(iso_conf), "--print-config"],
                capture_output=True,
                text=True,
            )
            assert ok_chk.returncode == 0, ok_chk.stderr

            typo_conf = tdir / "typo.conf"
            evo_judge.write_isolated_conf(
                good_base,
                typo_conf,
                tdir / "kv_typo",
                tdir / "resp_typo.idx",
                "secret123",
                {"MYDS_DEFINITELY_NOT_A_REAL_KNOB": "1"},
            )
            bad_chk = subprocess.run(
                [str(real_myds), "--config", str(typo_conf), "--print-config"],
                capture_output=True,
                text=True,
            )
            assert bad_chk.returncode == 2

        # 2. Test pure statistical & 0-ULP decision matrix
        good_ulp = {"p0": {"A": ["h1", "h1", "h1"], "B": ["h1", "h1", "h1"]}}
        bad_ulp = {"p0": {"A": ["h1", "h1", "h1"], "B": ["h1", "DIFF", "h1"]}}

        v_adopt = evo_judge.evaluate_verdict(
            good_ulp, [45.0, 45.1, 44.9], [48.5, 48.6, 48.4], [], 0, "decode_tps", 1.5, False
        )
        assert v_adopt["verdict"] == "ADOPT" and v_adopt["exit_code"] == 0

        v_non0 = evo_judge.evaluate_verdict(
            bad_ulp, [45.0, 45.0, 45.0], [90.0, 90.0, 90.0], [], 0, "decode_tps", 1.5, False
        )
        assert v_non0["verdict"] == "REJECT_NON_0ULP" and v_non0["exit_code"] == 1

        v_noise = evo_judge.evaluate_verdict(
            good_ulp, [44.0, 47.0, 45.0], [45.8, 46.0, 45.9], [], 0, "decode_tps", 1.5, False
        )
        assert v_noise["verdict"] == "REJECT_NOISE_OR_INSUFFICIENT_GAIN"

        v_reg = evo_judge.evaluate_verdict(
            good_ulp, [48.0, 48.1, 47.9], [44.0, 44.1, 43.9], [], 0, "decode_tps", 1.5, False
        )
        assert v_reg["verdict"] == "REJECT_REGRESSION"

        v_ips = evo_judge.evaluate_verdict(
            good_ulp, [45.0, 45.0, 45.0], [49.0, 49.0, 49.0], [], 2, "decode_tps", 1.5, False
        )
        assert v_ips["verdict"] == "VOID_MEASUREMENT" and v_ips["exit_code"] == 3

        # 3. Compile genuine Mach-O mock `myds` binary for end-to-end ABABAB tests
        c_src = tdir / "mock_myds.c"
        c_src.write_text(MOCK_MYDS_C, encoding="utf-8")
        mock_bin_dir = tdir / "live" / "bin"
        mock_bin_dir.mkdir(parents=True)
        mock_bin = mock_bin_dir / "myds"
        mock_ml = mock_bin_dir / "myds.metallib"
        mock_ml.write_bytes(b"MTLB_MOCK_BASELINE_METALLIB_BYTES")
        fake_model = tdir / "fake.gguf"
        fake_model.write_bytes(b"GGUF\x00\x00\x00\x03")

        cc_res = subprocess.run(
            ["clang", "-O2", "-Wall", "-Wextra", "-lpthread", str(c_src), "-o", str(mock_bin)],
            capture_output=True,
            text=True,
        )
        assert cc_res.returncode == 0, cc_res.stderr

        jport = free_tcp_port()
        base_judge_cmd = [
            sys.executable,
            str(ROOT / "tools" / "evo_judge.py"),
            "--mode",
            "full",
            "--skip-repo-checks",
            "--unit-cmd",
            "",
            "--baseline-bin",
            str(mock_bin),
            "--baseline-metallib",
            str(mock_ml),
            "--candidate-bin",
            str(mock_bin),
            "--candidate-metallib",
            str(mock_ml),
            "--base-config",
            str(good_base),
            "--model",
            str(fake_model),
            "--port",
            str(jport),
            "--cycles",
            "3",
            "--json",
        ]

        # 3a. §3 Identical arms -> VOID_IDENTICAL_ARMS (exit 3)
        p_id = subprocess.run(base_judge_cmd, capture_output=True, text=True)
        assert p_id.returncode == 3, p_id.stdout + p_id.stderr
        assert json.loads(p_id.stdout)["verdict"] == "VOID_IDENTICAL_ARMS"

        # 3b. End-to-end ABABAB ADOPT (0-ULP + +10.2% decode_tps, cached_tokens==0)
        p_ok = subprocess.run(
            [*base_judge_cmd, "--conf-override", "MYDS_INDEXER_TOPK_RADIX_CAUSAL=1"],
            capture_output=True,
            text=True,
        )
        assert p_ok.returncode == 0, p_ok.stdout + p_ok.stderr
        rep_ok = json.loads(p_ok.stdout)
        assert rep_ok["verdict"] == "ADOPT"
        assert rep_ok["ulp_zero"] is True
        assert len(rep_ok["boots"]) == 6  # A1,B1,A2,B2,A3,B3
        assert [b["arm"] for b in rep_ok["boots"]] == ["A1", "B1", "A2", "B2", "A3", "B3"]

        # 3c. End-to-end 0-ULP drift -> REJECT_NON_0ULP (exit 1)
        p_drift = subprocess.run(
            [*base_judge_cmd, "--conf-override", "MYDS_INDEXER_RO_SMEM_Q=0"],
            capture_output=True,
            text=True,
        )
        assert p_drift.returncode == 1
        assert json.loads(p_drift.stdout)["verdict"] == "REJECT_NON_0ULP"

        # 3d. End-to-end silent cache hit (cached_tokens=128) -> VOID_MEASUREMENT (exit 3)
        p_cache = subprocess.run(
            [*base_judge_cmd, "--conf-override", "MYDS_INDEXER_PREFILL_COMPACT=0"],
            capture_output=True,
            text=True,
        )
        assert p_cache.returncode == 3
        rep_cache = json.loads(p_cache.stdout)
        assert rep_cache["verdict"] == "VOID_MEASUREMENT"
        assert any("cached_tokens=128" in r for r in rep_cache["reasons"])

        # 4. Test Supervisor atomic_install_file (§7: never mutate mapped inode in place!)
        mapped_dst = tdir / "mapped_target"
        mapped_dst.write_bytes(b"OLD_MAPPED_CODE_SIGN_PAGE_BYTES")
        old_ino = mapped_dst.stat().st_ino
        new_src = tdir / "new_candidate"
        new_src.write_bytes(b"NEW_CANDIDATE_BINARY_BYTES_V2!!")

        with mapped_dst.open("rb") as live_fd:
            swap = evo_supervisor.atomic_install_file(new_src, mapped_dst, 0o755)
            assert swap["old_inode"] == old_ino
            assert swap["new_inode"] != old_ino
            # Old open descriptor MUST still read unmutated old bytes (zero SIGBUS risk!)
            assert live_fd.read() == b"OLD_MAPPED_CODE_SIGN_PAGE_BYTES"
        assert mapped_dst.read_bytes() == b"NEW_CANDIDATE_BINARY_BYTES_V2!!"

        # 5. Test merge_conf_preserving_comments
        merged_txt = evo_supervisor.merge_conf_preserving_comments(
            good_base,
            {"MYDS_MOE_DECODE_TOPK": "6", "MYDS_INDEXER_TOPK_RADIX_CAUSAL": "1"},
        )
        assert "# Production baseline" in merged_txt
        assert "MYDS_MOE_DECODE_TOPK=6" in merged_txt
        assert "MYDS_INDEXER_TOPK_RADIX_CAUSAL=1" in merged_txt

        # 6. Test Supervisor live status, zero-downtime preflight rejection & client-drain guard
        live_port = free_tcp_port()
        st_off = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "evo_supervisor.py"), "status", "--port", str(live_port), "--json"],
            capture_output=True,
            text=True,
        )
        assert st_off.returncode == 0 and json.loads(st_off.stdout)["running"] is False

        live_srv = subprocess.Popen(
            [
                str(mock_bin),
                "--config",
                str(good_base),
                "-m",
                str(fake_model),
                "--port",
                str(live_port),
            ],
            cwd=str(tdir / "live"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            evo_judge.wait_ready(live_port, live_srv, 5.0)
            st_on = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "evo_supervisor.py"), "status", "--port", str(live_port), "--json"],
                capture_output=True,
                text=True,
            )
            st_body = json.loads(st_on.stdout)
            assert st_body["running"] is True and st_body["pid"] == live_srv.pid

            # 6a. Preflight rejects bad config override WITHOUT stopping live Golden!
            rc_pf = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "evo_supervisor.py"),
                    "evaluate",
                    "--port",
                    str(live_port),
                    "--conf-override",
                    "MYDS_BAD_UNKNOWN_KEY=1",
                    "--skip-offline",
                    "--work-root",
                    str(tdir / "pf-reject"),
                    "--json",
                ],
                capture_output=True,
                text=True,
            )
            assert rc_pf.returncode == 1, rc_pf.stderr + rc_pf.stdout
            assert rc_pf.stdout, rc_pf.stderr
            pf_rec = json.loads(rc_pf.stdout)
            assert pf_rec["status"] == "REJECTED_PRE_WINDOW"
            assert pf_rec["golden_stopped"] is False
            assert live_srv.poll() is None  # Golden never touched!

            # 6b. Active client drain timeout protects in-flight self-hosting agent HTTP turn!
            with socket.create_connection(("127.0.0.1", live_port), timeout=2.0) as active_sock:
                time.sleep(0.05)  # allow accept() to register fd on live_srv
                rc_dr = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools" / "evo_supervisor.py"),
                        "evaluate",
                        "--port",
                        str(live_port),
                        "--conf-override",
                        "MYDS_INDEXER_TOPK_RADIX_CAUSAL=1",
                        "--skip-offline",
                        "--client-drain-timeout",
                        "0.35",
                        "--work-root",
                        str(tdir / "drain-abort"),
                        "--json",
                    ],
                    capture_output=True,
                    text=True,
                )
                assert active_sock.fileno() >= 0
            assert rc_dr.returncode == 3, f"rc={rc_dr.returncode}\nOUT:{rc_dr.stdout}\nERR:{rc_dr.stderr}"
            dr_rec = json.loads(rc_dr.stdout)
            assert dr_rec["status"] == "ABORTED_ACTIVE_CLIENTS"
            assert dr_rec["golden_stopped"] is False
            assert live_srv.poll() is None
        finally:
            live_srv.terminate()
            live_srv.wait(timeout=5.0)


def main() -> int:
    t0 = time.monotonic()
    test_silicon_probe()
    test_evo_memory()
    test_evo_judge_and_supervisor()
    dt = time.monotonic() - t0
    print(f"PASS: test_evo_loop (all 4 Silicon Self-Evolution components verified in {dt:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
