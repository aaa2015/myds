#!/bin/bash
# §162 验收：MYDS_ENGRAM_ASYNC_DECODE=0 vs 1
#   门 1：输出逐字节相同（0-ULP）
#   门 2：decode 吞吐不退化，且应上升
#   门 3：证明两臂确实是不同配置（打印 knob 值 + 日志行）
# 纪律：同一二进制；conf 副本只差一行；两臂串行（内存只够一个实例）。
set -u
REPO=$HOME/Projects/myds
FLEET=$HOME/Projects/myfleet
OUT=$HOME/bench/ab/engram_async
PORT=8188
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf

rm -rf "$OUT"; mkdir -p "$OUT" || exit 1
cd "$OUT" || exit 1

echo "=== 产物身份 ==="
shasum -a 256 "$REPO/bin/myds" "$REPO/bin/myds.metallib" | tee identity.txt
python3 -c "import json;d=json.load(open('$REPO/bin/BUILD-INFO.json'));print('commit',d['commit_short'],'dirty',d.get('dirty_build_paths'))" | tee -a identity.txt

# 固定 prompt（短，只为 decode）；temp=0
python3 - <<'PY'
import json
body = {"model": "probe",
        "messages": [{"role": "user", "content": "用中文解释快速排序的原理，并给出一个例子。"}],
        "max_tokens": 256, "temperature": 0}
json.dump(body, open("req.json", "w"), ensure_ascii=False)
PY

for arm in off on; do
  cp "$FLEET/myds.conf" "myds-$arm.conf"
  grep -vE '^MYDS_ENGRAM_ASYNC_DECODE=' "myds-$arm.conf" > t && mv t "myds-$arm.conf"
  if [ "$arm" = "on" ]; then echo 'MYDS_ENGRAM_ASYNC_DECODE=1' >> "myds-$arm.conf"; fi
done
echo "--- conf diff（应只有 ASYNC 一行）---"
diff myds-off.conf myds-on.conf || true
echo "--- 两臂 IO 线程（必须是 16，否则 reader 为 NULL，异步不生效）---"
grep -E '^MYDS_ENGRAM_IO_THREADS=' myds-off.conf

launch() { local tag=$1
  nohup "$REPO/bin/myds" --config "myds-$tag.conf" -m "$MODEL" \
    --port $PORT --host 127.0.0.1 --ctx-size 262144 --alias probe \
    --temp 1.0 --top-p 1.0 --max-tokens 512 --thinking off --vision "$VISION" \
    >> "server_$tag.log" 2>&1 &
  echo $! > "pid_$tag"; }
wait_ready() { local i=0
  while (( i < 200 )); do curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 1; ((i++)); done; return 1; }
stop_it() { local pid; pid=$(cat "pid_$1" 2>/dev/null); [ -z "$pid" ] && return
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 120); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
  kill -KILL "$pid" 2>/dev/null; }

run_arm() { local tag=$1
  echo "=== 臂 $tag ==="
  launch "$tag"
  wait_ready "$tag" || { echo "!! $tag 启动失败"; tail -20 "server_$tag.log"; stop_it "$tag"; return 1; }
  echo "--- 配置生效确认 ---"
  grep -E "Engram async decode" "server_$tag.log" || echo "(无 async 日志行 ⇒ 走同步路径，符合 off 臂预期)"
  # 预热一次（建 prefix cache），不记入测量
  curl -sf -m 600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" --data-binary @req.json -o warm.json -w "warm %{time_total}\n"
  for r in 1 2 3; do
    curl -sf -m 600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" --data-binary @req.json -o "out_$tag$r.json" \
      -w "$tag run$r %{time_total}\n" | tee -a "times_$tag.txt"
  done
  echo "--- 输出内容 sha（0-ULP 判据）---"
  for r in 1 2 3; do
    python3 -c "
import json,hashlib,sys
d=json.load(open('out_$tag$r.json'))
c=d['choices'][0]['message']['content']
print('$tag run$r len=%d sha=%s' % (len(c), hashlib.sha256(c.encode()).hexdigest()[:16]))
" | tee -a "sha_$tag.txt"
  done
  echo "--- decode t/s ---"
  grep -oE "decode: [0-9]+ tok in [0-9.]+s = [0-9.]+ t/s" "server_$tag.log" | tail -3 | tee -a "tps_$tag.txt"
  stop_it "$tag"; sleep 3
}

run_arm off
run_arm on

echo
echo "=== 汇总 ==="
for a in off on; do echo "--- $a ---"; cat "sha_$a.txt" 2>/dev/null; cat "tps_$a.txt" 2>/dev/null; done
echo ENGRAM_ASYNC_DONE
