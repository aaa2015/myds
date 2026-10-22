#!/bin/bash
# =================================================================
# rss_sampler.sh — 独立于 SSH 会话的内存采样器 + 硬护栏
#
# 为什么必须独立：2026-10-22 的 500K 实验里，看门狗是
# `ssh macstudio 'bash test.sh'` 的子进程。SSH 一断，脚本和看门狗
# 一起收到 SIGHUP 退出，被测进程继续分配内存到 205 GB，整机 panic。
# 教训：保命的进程不能依附于会断的会话。
#
# 因此本脚本必须这样启动（nohup + disown，脱离会话）：
#   nohup ./rss_sampler.sh --port 8199 --limit 175 --out /tmp/s.log > /dev/null 2>&1 &
#   disown
#
# 或者用 launchd。启动后靠心跳文件确认它还活着：
#   test -f /tmp/s.log.hb && [ $(( $(date +%s) - $(cat /tmp/s.log.hb) )) -lt 5 ]
#
# 用法：
#   ./rss_sampler.sh --port 8199 --limit 175 --out /tmp/s.log [--interval 1]
#
# 产物：
#   $OUT        每行 "epoch rss_gb" 采样
#   $OUT.peak   峰值 GB（单个整数）
#   $OUT.hb     心跳（最后采样时刻的 epoch）
#   $OUT.trip   "OVER" 表示触发阈值并已 kill
# =================================================================

set -u
PORT=8187
LIMIT=175
OUT=/tmp/rss_sampler.log
INTERVAL=1

while [ $# -gt 0 ]; do
    case "$1" in
        --port)     PORT="$2"; shift 2 ;;
        --limit)    LIMIT="$2"; shift 2 ;;
        --out)      OUT="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "bad arg: $1" >&2; exit 2 ;;
    esac
done

rm -f "$OUT.peak" "$OUT.peakwired" "$OUT.trip" "$OUT.hb"
echo 0 > "$OUT.peak"
echo 0 > "$OUT.peakwired"
: > "$OUT"

PID_OF_PORT() {
    lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1
}

echo "start $(date +%s) port=$PORT limit=$LIMIT" >> "$OUT"

while true; do
    P=$(PID_OF_PORT)
    if [ -z "$P" ]; then
        echo "no-listener $(date +%s)" >> "$OUT"
        # 端口没监听不立刻退出：进程可能在 prefill 中途（单飞锁时 /health 也不回）
        # 只有连续 30 秒都没有监听进程才认定结束
        sleep 2
        continue
    fi
    RSS=$(ps -o rss= -p "$P" 2>/dev/null | awk '{printf "%d", $1/1048576}')
    if [ -z "$RSS" ]; then
        echo "gone $(date +%s)" >> "$OUT"
        break
    fi
    # wired 才是打死机器的东西：wired 页不可回收，打满后内核一页都拿不回，
    # watchdogd 饿死 → 内核 panic。RSS 只说明进程占用，看它不够。
    read -r WIRED FREE INACT <<<"$(vm_stat | awk '
        /Pages wired down/ {w=$4} /Pages free/ {f=$3} /Pages inactive/ {i=$3}
        END {printf "%d %d %d", w*16384/1e9, f*16384/1e9, i*16384/1e9}')"
    NOW=$(date +%s)
    echo "$NOW rss=$RSS wired=$WIRED free=$FREE inact=$INACT" >> "$OUT"
    echo "$NOW" > "$OUT.hb"

    OLD=$(cat "$OUT.peak" 2>/dev/null || echo 0)
    if [ "$RSS" -gt "$OLD" ]; then echo "$RSS" > "$OUT.peak"; fi
    OLDW=$(cat "$OUT.peakwired" 2>/dev/null || echo 0)
    if [ "$WIRED" -gt "$OLDW" ]; then echo "$WIRED" > "$OUT.peakwired"; fi

    # 硬护栏盯 wired，不盯 RSS —— 这才是会 panic 的量
    LIMW="${WIRED_LIMIT:-190}"
    if [ "$WIRED" -ge "$LIMW" ]; then
        echo "OVER wired=$WIRED rss=$RSS" > "$OUT.trip"
        echo "TRIP $NOW wired=$WIRED rss=$RSS limit=$LIMW" >> "$OUT"
        kill -TERM "$P" 2>/dev/null
        sleep 5
        kill -KILL "$P" 2>/dev/null
        break
    fi

    if [ "$RSS" -ge "$LIMIT" ]; then
        echo "OVER $RSS" > "$OUT.trip"
        echo "TRIP $NOW rss=$RSS limit=$LIMIT" >> "$OUT"
        kill -TERM "$P" 2>/dev/null
        sleep 5
        kill -KILL "$P" 2>/dev/null
        break
    fi
    sleep "$INTERVAL"
done

echo "end $(date +%s) peak=$(cat "$OUT.peak")" >> "$OUT"
