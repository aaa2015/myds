#!/bin/bash
#
# 部署目录由 MYDS_FLEET_DIR 指定（本仓库不含该目录）。
# 未设置时这些脚本会明确报错退出，而不是猜一个路径。

# Exclusive-window per-kernel GPU timing on macstudio (stops 8181, ALWAYS restores it).
#   tools/run_kprof_window.sh /abs/NEW/output_dir [records_short_ctx=0 8k=14 32k=56 ...]
# Uses the staged bench_kprof (never touches bin/myds). Run only after deploy.
set -uo pipefail
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
repo=~/Projects/myds
out=${1:?provide a NEW absolute output directory}
# 端口 / 模型 / alias 参数化（2026-10-20）：原值保持为默认，可覆盖以测 1017 与 8187。
PORT=${KPROF_PORT:-8181}
model=${KPROF_MODEL:-$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-Q2-dense-q4.gguf}
alias_name=${KPROF_ALIAS:-local-myds-deepseek-v4.1-flash-q2}
stage=/tmp/myds-kprof-stage
mkdir "$out" || exit 1
cd "$repo"
git rev-parse HEAD > "$out/source-head.txt"
git status --porcelain > "$out/source-status.txt"
[[ -s "$out/source-status.txt" ]] && { echo "dirty tree" >&2; exit 1; }
rm -rf "$stage"; mkdir -p "$stage"
make BIN_DIR="$stage" -j8 "$stage/bench_kprof" > "$out/build.log" 2>&1 || { echo "build failed" >&2; exit 1; }
cmp bin/myds.metallib "$stage/myds.metallib" || { echo "metallib differs from deployed" >&2; exit 1; }
shasum -a 256 "$stage/bench_kprof" "$stage/myds.metallib" > "$out/binary-identities.txt"

# Requests: short hand-written, plus deterministic long inputs.
cat > "$out/req-short.json" <<'JSON'
{"model":"'"$alias_name"'","messages":[{"role":"user","content":"请用中文详细介绍一下快速排序算法，包括原理、时间复杂度、Python 实现和优缺点。写得尽量长。"}],"max_tokens":64,"temperature":0}
JSON
python3 tools/make_decode_business_request.py --records 14 --output "$out/req-8k.json" > "$out/gen-8k.log" 2>&1
python3 tools/make_decode_business_request.py --records 56 --output "$out/req-32k.json" > "$out/gen-32k.log" 2>&1

restore=0
cleanup() {
  if [[ $restore = 1 ]]; then
    (cd "${MYDS_FLEET_DIR:?set MYDS_FLEET_DIR}" && ./start_fleet.sh) > "$out/restore-service.log" 2>&1
    sleep 5
    curl -fsS http://127.0.0.1:$PORT/health > "$out/restored-health.json" 2>"$out/restored-health.err"
    lsof -nP -a -iTCP:$PORT -sTCP:LISTEN > "$out/restored-listener.txt" 2>&1
    echo "RESTORED $(cat "$out/restored-health.json" 2>/dev/null)" > "$out/DONE"
  fi
}
trap cleanup EXIT
pid=$(lsof -nP -t -a -iTCP:$PORT -sTCP:LISTEN)
[[ -n "$pid" ]] || { echo "no 8181 listener" >&2; exit 1; }
ps -p "$pid" -o pid=,command= > "$out/service-before.txt"
restore=1
kill -TERM "$pid"
for i in {1..30}; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
kill -0 "$pid" 2>/dev/null && { echo "service did not stop" >&2; exit 1; }
python3 -c "import sys; sys.path.insert(0,'tools'); import bench_reproduce; bench_reproduce.check_idle($PORT)" || { echo "not idle" >&2; exit 1; }
vm_stat > "$out/vm-before.txt"; sysctl vm.swapusage > "$out/swap-before.txt"

# Production decode settings minus speculation / KV disk / prefix cache.
# NOTE 2026-10-30: this list used to set MYDS_TOPK_THRESHOLD=1, which no code
# reads (the real key is MYDS_INDEXER_TOPK_THRESHOLD, default false, non-0-ULP).
# Past kprof decode breakdowns therefore ran with the threshold path OFF.
# Verify any config mirror with: tools/knob_inventory.py --check --conf <file>
common=(MYDS_PREFIX_CACHE=0 MYDS_PLD=0 MYDS_KV_DSK=0
  MYDS_ATTN_INDEXER_EARLY=1 MYDS_BATCH2_AUTO_BARRIER=0 MYDS_DECODE_SPLIT=20
  MYDS_F16_GEMV_BF16_FUSE=1 MYDS_INDEXER_HEAD_SLOTS=1 MYDS_INDEXER_ROPE_QUANT_FUSE=1
  MYDS_INDEXER_TOPK_NAN_FUSE=1 MYDS_INDEXER_TOPK_PREFIX_MERGE=1 MYDS_LM_HEAD_Q8_NR0=4
  MYDS_METAL_CONCURRENT=1 MYDS_MOE_CONCURRENT=1 MYDS_MOE_NSG=4 MYDS_MOE_CHUNKS=9
  MYDS_PREFILL_CHUNK_SIZE=512 MYDS_RESIDENCY_KEEP_ALIVE_S=86400
  MYDS_SUFFIX_SHRINK_SHIFT=512 MYDS_DENSE_Q4_NR0=2)
printf '%s\n' "${common[@]}" > "$out/env.txt"

run_case() { # name request max_seq
  echo "== $1 $(date +%T)"
  env "${common[@]}" "$stage/bench_kprof" "$model" "$2" 8 24 "$3" > "$out/$1.jsonl" 2> "$out/$1.log" \
    && python3 tools/kprof_report.py "$out/$1.jsonl" --top 40 > "$out/$1.txt" 2>&1 \
    || echo "case $1 FAILED (see $out/$1.log)"
}
if [[ -n "${MOE_SWEEP:-}" ]]; then
  # e.g. MOE_SWEEP="4:4 2:2 2:4 4:2" (MYDS_MOE_NSG:MYDS_MOE_DOWN_NSG), short context only.
  for cfg in $MOE_SWEEP; do
    g=${cfg%%:*}; d=${cfg##*:}
    echo "== sweep nsg=$g down=$d $(date +%T)"
    env "${common[@]}" MYDS_MOE_NSG=$g MYDS_MOE_DOWN_NSG=$d "$stage/bench_kprof" "$model" "$out/req-short.json" 8 24 4096 \
      > "$out/sweep-$g-$d.jsonl" 2> "$out/sweep-$g-$d.log" \
      && python3 tools/kprof_report.py "$out/sweep-$g-$d.jsonl" --top 40 > "$out/sweep-$g-$d.txt" 2>&1 \
      || echo "sweep $g:$d FAILED"
  done
  echo "cases finished $(date +%T)"
  exit 0
fi
run_case short "$out/req-short.json" 4096
run_case ctx8k "$out/req-8k.json" 12288
run_case ctx32k "$out/req-32k.json" 40960
echo "cases finished $(date +%T)"
