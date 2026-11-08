#!/bin/bash
# P-1 验收：flash_attn vec kernel 的 so4 进寄存器（只换 metallib，二进制不变）
#   门 1：0-ULP —— 同 prompt、temp 0，两臂输出逐字节相同
#   门 2：32K 冷 prefill 吞吐提升（目标 >= 2%）
#   门 3：decode 不退化（±0.5%）
# 纪律：两臂串行（macstudio 只装得下一个实例）；cwd 必须在没有 ./bin/myds.metallib 的地方。
set -u
REPO=$HOME/Projects/myds
FLEET=$HOME/Projects/myfleet
OUT=$HOME/bench/ab/p1_attn_reg
PORT=8188
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf
PROMPTS=$REPO/benchmarks/register_only_2026-10-27/prompts

rm -rf "$OUT"; mkdir -p "$OUT" || exit 1
cd "$OUT" || exit 1

echo "=== 两臂身份 ==="
shasum -a 256 /tmp/attn_old/myds /tmp/attn_old/myds.metallib /tmp/attn_new/myds /tmp/attn_new/myds.metallib | tee identity.txt
echo "--- 二进制必须相同、metallib 必须不同 ---"

for v in 0 1 2; do
  python3 - "$PROMPTS/p32k_ab__v$v.txt" "pf_v$v.json" <<'PY'
import json, sys
txt = open(sys.argv[1], encoding="utf-8").read()
tag = "[p1%s] " % sys.argv[2].replace("pf_v", "v").replace(".json", "")
json.dump({"model": "probe", "messages": [{"role": "user", "content": tag + txt}],
           "max_tokens": 8, "temperature": 0}, open(sys.argv[2], "w"), ensure_ascii=False)
PY
done
# decode + 0-ULP 用的固定 prompt
python3 - <<'PY'
import json
json.dump({"model": "probe",
           "messages": [{"role": "user", "content": "用中文解释快速排序的原理，并给出一个例子。"}],
           "max_tokens": 128, "temperature": 0},
          open("dec.json", "w"), ensure_ascii=False)
PY

wait_ready() { local i=0
  while (( i < 240 )); do curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 1; ((i++)); done; return 1; }

run_arm() { local tag=$1 dir=$2
  echo "=== 臂 $tag ($dir) ==="
  # ★ 响应重放缓存必须每臂独立：myds 会把 previous_response 存到
  #   $HOME/.cache/myds/responses.idx，另一个臂用【完全相同的请求】时会直接重放，
  #   得到一个 0.13 秒的假 prefill（2026-11-08 第一次跑就是这么被骗的）。
  rm -rf "$HOME/.cache/myds"
  mkdir -p "$HOME/.cache/myds"
  # ★ MYDS_KV_DSK 在 myfleet/myds.conf 里是【开着】的（/Users/fangqingyuan/myds-kv，
  #   100 GB 预算），而且 conf 的优先级高于环境变量 ⇒ 只能改 conf 副本。
  #   不清空它会直接命中磁盘快照："prefill: none (all 32219 cached)"，
  #   2026-11-08 连续三次跑出 0.137 秒的假 prefill 就是这么来的。
  rm -rf "$OUT/kv_$tag"; mkdir -p "$OUT/kv_$tag"
  cp "$FLEET/myds.conf" "$OUT/conf_$tag"
  grep -vE '^MYDS_KV_DSK=' "$OUT/conf_$tag" > "$OUT/conf_$tag.t" && mv "$OUT/conf_$tag.t" "$OUT/conf_$tag"
  echo "MYDS_KV_DSK=$OUT/kv_$tag" >> "$OUT/conf_$tag"
  echo "--- conf_$tag 与生产 conf 的差异（应只有 KV_DSK 一行）---"
  diff "$FLEET/myds.conf" "$OUT/conf_$tag" || true
  ( cd /tmp && exec env MYDS_RESP_STATE="$HOME/.cache/myds/responses.idx.$tag" \
      nohup "$dir/myds" --config "$OUT/conf_$tag" -m "$MODEL" \
      --port $PORT --host 127.0.0.1 --ctx-size 262144 --alias probe \
      --temp 1.0 --top-p 1.0 --max-tokens 512 --thinking off --vision "$VISION" \
      ) >> "$OUT/server_$tag.log" 2>&1 &
  echo $! > "pid_$tag"
  sleep 1
  if ! wait_ready; then echo "!! $tag 启动失败"; tail -15 "server_$tag.log" | tr -d '\000'; return 1; fi
  echo "--- 实际加载的 metallib ---"
  grep -a "loaded precompiled library" "server_$tag.log" | tail -1

  echo "--- decode 先跑（短上下文，不受 32K 之后的状态影响）---"
  for r in 1 2 3; do
    if ! curl -sf -m 900 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" --data-binary @dec.json -o "out_$tag$r.json" \
      -w "$tag decode$r %{time_total}\n" | tee -a "times_$tag.txt"; then
      echo "!! $tag decode$r 失败"; tail -8 "server_$tag.log" | tr -d '\000'; return 1
    fi
  done

  echo "--- 32K 冷 prefill ×3（每个文件开头加唯一标签，保证 token 3 就分叉）---"
  for v in 0 1 2; do
    if ! curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" --data-binary @"pf_v$v.json" -o /dev/null \
      -w "$tag prefill_v$v %{time_total}\n" | tee -a "times_$tag.txt"; then
      echo "!! $tag prefill_v$v 失败"; tail -8 "server_$tag.log" | tr -d '\000'; return 1
    fi
  done
  local n_real
  n_real=$(grep -ac "prefill: 322[0-9][0-9] tok in" "server_$tag.log" || true)
  echo "[assert] $tag 真实 32K prefill 次数 = $n_real（必须是 3）"
  if [ "${n_real:-0}" -lt 3 ]; then
    echo "!! $tag 出现缓存重放（prefill: none / all cached），本轮数据无效"
    grep -a "Completed ChatCompletions" "server_$tag.log" | tail -4 | tr -d '\000'
    return 1
  fi
  grep -aoE "prefill: 32[0-9][0-9] tok in [0-9.]+s = [0-9.]+ t/s" "server_$tag.log" | tee -a "tps_$tag.txt"
  grep -aoE "decode: [0-9]+ tok in [0-9.]+s = [0-9.]+ t/s" "server_$tag.log" | tee -a "tps_$tag.txt"

  echo "--- 输出 sha（0-ULP 判据）---"
  for r in 1 2 3; do
    python3 -c "
import json,hashlib
c=json.load(open('out_$tag$r.json'))['choices'][0]['message']['content']
print('$tag r$r len=%d sha=%s' % (len(c), hashlib.sha256(c.encode()).hexdigest()[:16]))" | tee -a "sha_$tag.txt"
  done

  local p; p=$(cat "pid_$tag" 2>/dev/null)
  [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 3
}

run_arm old /tmp/attn_old
run_arm new /tmp/attn_new

echo
echo "=== 汇总 ==="
for a in old new; do echo "--- $a ---"; cat "sha_$a.txt" 2>/dev/null; cat "tps_$a.txt" 2>/dev/null; done
echo P1_DONE
