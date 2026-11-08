#!/bin/sh
# prod_guard.sh — 「会杀掉生产实例」的脚本必须先 source 这个。
#
# 2026-11-08 事故（账本 §164）：生产 8187 在 07:10–07:25 被反复起停 4 次，
# 最后一次停止后【没有任何脚本把它拉回来】，停摆 6 分 32 秒（07:25:04 → 07:31:36）。
# 根因是结构性的：仓库里 6 个验收/基准脚本都会 kill 生产，没有一个负责恢复。
# 其中 tools/bench_register_only_ab.py 的 stop() 更狠 —— 它用
# `pgrep -x myds` 杀掉【所有】myds 进程，与端口无关。
#
# 用法（bash/zsh）：
#   . "$(dirname "$0")/prod_guard.sh"
# 效果：
#   1. source 的那一刻记下生产进程的【完整命令行】与端口占用者；
#   2. 脚本以任何方式退出（正常结束 / 出错退出 / SIGINT / SIGTERM / SIGHUP）
#      都会检查端口，若空则按原来那一条命令行重新拉起；
#   3. 已经有人在监听该端口时什么都不做（不会覆盖别人的实例）。
#
# 环境变量：
#   MYDS_GUARD_PORT    默认 8187（生产）
#   MYDS_GUARD_LOG     重新拉起时的追加日志，默认 myfleet/logs/local-deepseek-flash.log
#   MYDS_GUARD_DRYRUN  非空 ⇒ 只打印将要执行的命令，不真的拉起（用于自检）
#
# 设计取舍：
#   · 用「记下完整命令行再原样重放」而不是「调 start_fleet.sh」——
#     start_fleet.sh 会截断日志、按 fleet.conf 重建参数，可能和当时跑的实例不同。
#     事故里那 4 次启动都不是 start_fleet.sh 起的（日志里没有 FLEET_CMD 行）。
#   · trap 挂在 EXIT，所以 `set -e` 提前退出、kill -9 之外的所有路径都会恢复。
#   · 不检查「是不是同一次运行」——宁可多拉起一次，也不要留下死掉的生产。

MYDS_GUARD_PORT="${MYDS_GUARD_PORT:-8187}"
MYDS_GUARD_LOG="${MYDS_GUARD_LOG:-$HOME/Projects/myfleet/logs/local-deepseek-flash.log}"
PROD_GUARD_CMD=""
PROD_GUARD_PID=""
PROD_GUARD_CWD=""
PROD_GUARD_RESTORE=""

prod_guard_owner() {
    lsof -nP -iTCP:"$MYDS_GUARD_PORT" -sTCP:LISTEN -t 2>/dev/null | head -1
}

prod_guard_capture() {
    PROD_GUARD_PID="$(prod_guard_owner)"
    [ -z "$PROD_GUARD_PID" ] && return 0

    local raw cmd argv0 rest cwd abs0
    raw="$(ps -o command= -p "$PROD_GUARD_PID" 2>/dev/null)"
    # 去掉前导空白
    cmd="${raw#"${raw%%[![:space:]]*}"}"
    [ -z "$cmd" ] && return 0

    # cwd：生产的命令行用的是相对路径 `../myds/bin/myds`
    # （start_fleet.sh 在 $HOME/Projects/myfleet 下启动）。
    # 2026-11-08 第一次自检就栽在这里：命令捕获到了，但恢复时 cwd 不同。
    cwd="$(lsof -a -p "$PROD_GUARD_PID" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)"
    [ -z "$cwd" ] && cwd="$HOME/Projects/myfleet"
    PROD_GUARD_CWD="$cwd"

    # argv0 规范化成【真实绝对路径】，并保留其余参数原样。
    # 不这么做的话，重放时 `$HOME/../myds/...` 这种串会解析到错误位置。
    argv0="${cmd%% *}"
    rest="${cmd#"$argv0"}"
    case "$argv0" in
        /*) abs0="$argv0" ;;
        *)  abs0="$cwd/$argv0" ;;
    esac
    abs0="$(cd "$(dirname "$abs0")" 2>/dev/null && printf '%s/%s' "$(pwd -P)" "$(basename "$abs0")")"
    [ -z "$abs0" ] && abs0="${cmd%% *}"

    PROD_GUARD_CMD="$abs0$rest"
    # 拼好整条恢复命令 —— 恢复时不再依赖任何变量展开
    PROD_GUARD_RESTORE="cd '$PROD_GUARD_CWD' && exec $PROD_GUARD_CMD"
    echo "[prod_guard] 捕获 pid=$PROD_GUARD_PID cwd=$PROD_GUARD_CWD"
    echo "[prod_guard]   argv0 -> $abs0"
}

prod_guard_restore() {
    [ -z "$PROD_GUARD_CMD" ] && return 0
    if [ -n "$(prod_guard_owner)" ]; then
        echo "[prod_guard] 端口 $MYDS_GUARD_PORT 已被占用，跳过恢复"
        return 0
    fi
    if [ -n "${MYDS_GUARD_DRYRUN:-}" ]; then
        echo "[prod_guard][dryrun] 将执行：$PROD_GUARD_RESTORE"
        return 0
    fi
    echo "[prod_guard] 恢复生产：$PROD_GUARD_RESTORE"
    nohup sh -c "$PROD_GUARD_RESTORE" >>"$MYDS_GUARD_LOG" 2>&1 &
    echo "[prod_guard] 已拉起 pid=$!（日志 $MYDS_GUARD_LOG）"
}

trap 'prod_guard_restore' EXIT
trap 'prod_guard_restore; exit 130' INT
trap 'prod_guard_restore; exit 143' TERM HUP

prod_guard_capture
