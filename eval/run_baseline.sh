#!/bin/zsh
#
# 部署目录由 MYDS_FLEET_DIR 指定（本仓库不含该目录）。
# 未设置时这些脚本会明确报错退出，而不是猜一个路径。

# =================================================================
# run_baseline.sh — 在 macstudio 上跑一次完整基线度量
#
#   停生产 → 逐条 score（ppl + logits dump）→ 【无论成败】恢复生产
#
# 为什么必须停生产：模型 337 GB，192 GB 机器上再加载一份会触发交换，
# 测量结果和被测进程都会被拖垮。
#
# 铁律（AGENTS.md）：
#   · 长任务用会话托管，不用 nohup & 甩给会立刻结束的 shell；
#   · 停生产后【必须】恢复，脚本用 trap 兜住一切退出路径；
#   · 取 pid 做数字校验，停后 kill -0 复检。
#
# 用法: ./run_baseline.sh <tag> <model.gguf> <corpus_dir>
#   tag 例如 base-q2 / imatrix-q2，dump 落在 $OUT/<tag>/<stem>/
# =================================================================
set -u

TAG="${1:?usage: run_baseline.sh <tag> <model.gguf> <corpus_dir>}"
MODEL="${2:?missing model}"
CORPUS="${3:?missing corpus dir}"
MYDS_DIR="$HOME/Projects/myds"
FLEET_DIR="${MYDS_FLEET_DIR:?set MYDS_FLEET_DIR to your deployment directory}"
OUT="$HOME/myds-eval/$TAG"
MAXTOK="${MAXTOK:-1024}"

PORT=8181
PID=""
RESTORED=0

restore_production() {
    (( RESTORED )) && return
    RESTORED=1
    echo "── 恢复生产 ──"
    cd "$FLEET_DIR" || return
    env -u MYDS_PROFILE_FILE -u MYDS_PROF ./start_fleet.sh 2>&1 | tail -3
    for i in $(seq 1 150); do
        if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
            echo "HEALTH_OK $(date +%H:%M:%S)"
            return
        fi
        sleep 2
    done
    echo "❌ 生产未在 300 s 内恢复健康 —— 需要人工介入"
}

trap 'restore_production' EXIT INT TERM

mkdir -p "$OUT"

# ── 停生产（数字校验 + 停后复检）──────────────────────────────
PID="$(lsof -nP -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | head -1)"
if [[ -n "$PID" && "$PID" == <-> ]]; then
    echo "── 停生产 pid=$PID ──"
    kill "$PID"
    for i in $(seq 1 60); do
        kill -0 "$PID" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "  仍在运行，强制终止"
        kill -9 "$PID"
        sleep 1
    fi
else
    echo "── :$PORT 无监听（视为已停）──"
fi
sleep 5
echo "  内存空闲: $(memory_pressure 2>/dev/null | grep -i 'free percentage' | grep -oE '[0-9]+%')"

# ── 逐条打分 ────────────────────────────────────────────────
cd "$MYDS_DIR" || { echo "❌ 找不到 $MYDS_DIR"; exit 1; }
RES="$OUT/ppl.jsonl"
: > "$RES"

n=0
for f in "$CORPUS"/*.txt; do
    [[ -f "$f" ]] || continue
    stem="$(basename "$f" .txt)"
    d="$OUT/$stem"
    rm -rf "$d"
    mkdir -p "$d"
    n=$((n + 1))
    printf "  [%2d] %-34s " "$n" "$stem"
    if line="$(./bin/score "$MODEL" --text "$f" --max-tokens "$MAXTOK" --dump "$d" 2>"$OUT/$stem.score.err")"; then
        echo "$line" >> "$RES"
        echo "$line" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("ppl=%7.4f top1=%.3f n=%d" % (d["ppl"], d["top1_rate"], d["scored"]))'
    else
        echo "  [FAIL] $stem  ← $(tail -1 "$OUT/$stem.score.err" 2>/dev/null | cut -c1-90)"
        echo "{\"kind\":\"score\",\"text\":\"$stem\",\"error\":true}" >> "$RES"
    fi
done

echo "── 汇总 ──"
python3 - "$RES" <<'PY'
import json, sys, math
rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    if d.get("error"):
        continue
    rows.append(d)
if not rows:
    print("  no successful runs")
    raise SystemExit(1)
tot = sum(r["nll_sum"] for r in rows)
cnt = sum(r["scored"] for r in rows)
hits = sum(r["top1_hits"] for r in rows)
print("  texts=%d  tokens=%d  corpus_ppl=%.4f  top1=%.4f"
      % (len(rows), cnt, math.exp(tot / cnt), hits / cnt))
PY

echo "DONE tag=$TAG"
