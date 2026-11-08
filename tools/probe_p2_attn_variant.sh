#!/bin/bash
# P-2: MYDS_PREFILL_ATTN=matrix vs vec 端到端 prefill 对比
#
# 目的（账本 §160.6 第 2 项）：给「放弃 0-ULP 换 attention tiling」这个选项
# 补一个真实数字。§149.3 已证明 §89 的 +11~14% 上界只适用于 heads 模板，
# 生产跑的是 vec，所以必须实测 matrix 到底快不快。
#
# 纪律（§26.10.1 的教训）：
#   1. 两臂必须证明【真的不同】—— 必须 MYDS_ATTN_WHICH=1 且看到 matrix_encoded=1。
#      matrix 编码失败会【静默回退 vec】（myds_v41_gpu.c:2027/2046），
#      那样两臂就是同一个东西，测出来的"持平"是假的。
#   2. conf 用副本，不用环境变量（MYDS_PREFILL_ATTN 虽不在 conf 里，
#      但副本能留档、能 diff）。
#   3. 语料必须是 benchmarks/register_only_2026-10-27/prompts/p32k_ab__v{0,1,2}.txt
#      三个变体各自 cold prefill（不同内容 ⇒ token 0 就分叉 ⇒ 全量 prefill）。
#
# 内存约束：macstudio 192 GB，单实例 wired ≈ 150 GB ⇒ 两臂必须【串行】，
# 且跑之前必须停生产 8187。
set -u
REPO=$HOME/Projects/myds
FLEET=$HOME/Projects/myfleet
OUT=$HOME/bench/ab/p2_matrix
PORT=8188
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf
PROMPTS=$REPO/benchmarks/register_only_2026-10-27/prompts

rm -rf "$OUT"; mkdir -p "$OUT" || exit 1
cd "$OUT" || exit 1

echo "=== 产物身份（两臂共用同一个二进制与 metallib）==="
shasum -a 256 "$REPO/bin/myds" "$REPO/bin/myds.metallib" | tee identity.txt
python3 -c "import json;d=json.load(open('$REPO/bin/BUILD-INFO.json'));print('commit',d['commit_short'],'dirty',d.get('dirty_build_paths'))" | tee -a identity.txt

echo
echo "=== 造 conf 副本（只差 MYDS_PREFILL_ATTN 一行）==="
cp "$FLEET/myds.conf" myds-p2-vec.conf
cp "$FLEET/myds.conf" myds-p2-matrix.conf
# 去掉可能已存在的该键，再分别写入
grep -vE '^MYDS_PREFILL_ATTN=' myds-p2-vec.conf > t && mv t myds-p2-vec.conf
grep -vE '^MYDS_PREFILL_ATTN=' myds-p2-matrix.conf > t && mv t myds-p2-matrix.conf
echo 'MYDS_PREFILL_ATTN=vec'    >> myds-p2-vec.conf
echo 'MYDS_PREFILL_ATTN=matrix' >> myds-p2-matrix.conf
echo "--- diff（必须只有这一行不同）---"
diff myds-p2-vec.conf myds-p2-matrix.conf || true
echo

# 把语料包成 chat completions 请求体，max_tokens=8（只要 prefill）
for v in 0 1 2; do
  python3 - "$PROMPTS/p32k_ab__v$v.txt" "pf_v$v.json" <<'PY'
import json, sys
txt = open(sys.argv[1], encoding="utf-8").read()
body = {"model": "local-deepseek-flash",
        "messages": [{"role": "user", "content": txt}],
        "max_tokens": 8, "temperature": 0}
json.dump(body, open(sys.argv[2], "w"), ensure_ascii=False)
PY
done

launch() {  # $1 = tag, $2 = conf
  local tag=$1 conf=$2
  MYDS_ATTN_WHICH=1 nohup "$REPO/bin/myds" --config "$conf" -m "$MODEL" \
    --port $PORT --host 127.0.0.1 --ctx-size 1048576 --alias p2probe \
    --temp 1.0 --top-p 1.0 --max-tokens 262144 --thinking off --vision "$VISION" \
    >> "server_$tag.log" 2>&1 &
  echo $! > "pid_$tag"
}

wait_ready() {  # $1 = tag
  local tag=$1 i=0
  while (( i < 180 )); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
    sleep 1; ((i++))
  done
  return 1
}

stop_inst() {  # $1 = tag
  local tag=$1 pid
  pid=$(cat "pid_$tag" 2>/dev/null)
  [[ -z "$pid" ]] && return
  kill -TERM "$pid" 2>/dev/null
  local i=0
  while (( i < 180 )); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1; ((i++))
  done
  kill -KILL "$pid" 2>/dev/null
}

run_arm() {  # $1 = tag, $2 = conf
  local tag=$1 conf=$2
  echo "=== 臂 $tag 启动 ==="
  launch "$tag" "$conf"
  if ! wait_ready "$tag"; then echo "!! $tag 启动失败"; tail -20 "server_$tag.log"; stop_inst "$tag"; return 1; fi
  echo "--- 派发身份确认（必须是 matrix_encoded=1 才算真 matrix）---"
  grep -iE "flash_attn|matrix_encoded|attn_which|MYDS_PREFILL_ATTN" "server_$tag.log" | tail -5

  for v in 0 1 2; do
    curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" --data-binary @"pf_v$v.json" -o /dev/null \
      -w "$tag v$v %{time_total}\n" | tee -a "times_$tag.txt"
  done
  echo "--- prefill 行 ---"
  grep -oE "prefill: [0-9]+ tok in [0-9.]+s = [0-9.]+ t/s" "server_$tag.log" | tee -a "times_$tag.txt"
  stop_inst "$tag"
  echo "=== 臂 $tag 已停 ==="
  sleep 3
}

run_arm vec    myds-p2-vec.conf
run_arm matrix myds-p2-matrix.conf

echo
echo "=== 汇总 ==="
for t in vec matrix; do
  echo "--- $t ---"
  cat "times_$t.txt" 2>/dev/null
done
echo P2_DONE
