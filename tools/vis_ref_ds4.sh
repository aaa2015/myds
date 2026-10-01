#!/bin/zsh
# ─────────────────────────────────────────────────────────────────────
#  vis_ref_ds4.sh —— 跑 ds4 的【视觉编码参照】，产出逐阶段中间量供对拍
#
#  用法（★ 在装 ds4 + 主模型的机器上执行，本仓库默认是 macstudio）:
#     zsh tools/vis_ref_ds4.sh <图片> [ctx] [超时秒] ["额外ds4参数"]
#
#  ★★ 重要前提：ds4 启动时要把 ~152 GiB 映射进 Metal 并申请【驻留】，
#     所以如果目标机上已有别的推理服务占着内存，必须先停掉它 —— 实测：
#       ds4: Metal model warmup failed: Insufficient Memory
#       ds4: mapped 155401.05 MiB ... metal failed to map model views
#     停生产前先记录 argv/env 以便原样恢复（本仓 tmp/myds_restore_recipe.txt 有样例）。
#
#  参照侧张量名（与本仓 myds_visfwd.c 的转储名【已对齐】，可直接逐阶段比）：
#     patch_embed / block0_norm1 / block0_qkv / block0_q / block0_k / block0_v /
#     block0_attention / block0_attn_residual / block0_norm2 / block0_mlp_w1 /
#     block0_mlp_mid / block0 / block15 / block31 / vision_norm / aligned
#
#  实测数据（2026-10-01，square_512，macstudio M2 Ultra）：
#     ds4 侧 prefill 147.62 t/s、generation 24.06 t/s，16 个转储
#     与 myds 对拍：早期阶段 3.6e-5，随深度平滑增长到 aligned 的 2.9e-2
#     F64 真值验证：12/12 采样位置两侧完全相同（差 1 个 BF16 ULP 以内）
#       ⇒ 残差是 BF16 量化边界的敏感性，不是实现错误
# ─────────────────────────────────────────────────────────────────────
#  历史名字：run_vision_ref.sh
#
#  ★ 本脚本【在 macstudio 上执行】（不是 macmini）。跨机脚本必须标明执行主机，
#    否则「本地追加」会写错地方。
#
#  目的：拿到 ds4 编码器各阶段的中间量，与 myds 逐层对拍。
#  做法：ds4 走 mmap 惰性加载（不带 --warm-weights）⇒ 不必把 365GB 全驻留；
#        只需 --vision 指向编码器，然后 REPL 里 /read <图片>。
#        /read 会先【编码并转储】（DS4_DEEPSEEK4_VISION_DEBUG_PREFIX），
#        之后才进聊天回合 —— 所以哪怕聊天回合很慢，转储已经落地。
#
#  坑（第一版踩的，两条都是 zsh/ macOS 特有）：
#    ① zsh 默认【glob 无匹配就报错并中止】（nomatch）⇒ 用 null_glob
#    ② macOS 没有 GNU timeout / gtimeout ⇒ 自己起看门狗
# ─────────────────────────────────────────────────────────────────────
setopt null_glob no_nomatch       # ← 坑①：让无匹配的 glob 展开成空
set -u

DS4_DIR="$HOME/Projects/ds4"
M="$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-Q2.gguf"
E="$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-Vision.gguf"
OUT=/tmp/ds4vis
IMG="${1:-/tmp/visrun/square_512.png}"
CTX="${2:-4096}"
BUDGET="${3:-150}"
EXTRA="${4:-}"          # 额外 ds4 参数（zsh 里用 ${=EXTRA} 显式分词）

mkdir -p "$OUT"
rm -f "$OUT"/ref.* "$OUT"/err.log "$OUT"/out.log

mem() {
    vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} \
        END{printf "  内存: 可用≈%.2fGB (free %.2f + inactive %.2f)\n", \
        (f+i)*ps/1073741824, f*ps/1073741824, i*ps/1073741824}'
}

echo "══════ ds4 视觉参照（在 macstudio 上跑）══════"
echo "主模型: ${M:t}"
echo "编码器: ${E:t}"
echo "图片  : $IMG"
echo "ctx   : $CTX    超时: ${BUDGET}s"
echo "额外参数: ${EXTRA:-（无）}"
echo
echo "── 启动前 ──"; mem

cd "$DS4_DIR" || exit 1
export DS4_DEEPSEEK4_VISION_DEBUG_PREFIX="$OUT/ref"

echo
echo "── 启动 ds4（不带 --warm-weights，走 mmap 惰性）──"
# 坑②：macOS 无 timeout ⇒ 自起看门狗
# ★ zsh 里字符串不会自动分词 ⇒ 用 ${=EXTRA} 显式要求按空格拆（这正是我反复踩的那条坑，
#   这次用【显式的】分词语法，而不是依赖默认行为）
if [[ -n "$EXTRA" ]]; then
    printf '/read %s\n' "$IMG" | ./ds4 -m "$M" --vision "$E" -c "$CTX" ${=EXTRA} \
        > "$OUT/out.log" 2> "$OUT/err.log" &
else
    printf '/read %s\n' "$IMG" | ./ds4 -m "$M" --vision "$E" -c "$CTX" \
        > "$OUT/out.log" 2> "$OUT/err.log" &
fi
ds4_pid=$!
( sleep "$BUDGET"; kill -9 "$ds4_pid" 2>/dev/null; echo "WATCHDOG: 超时 ${BUDGET}s，已杀 $ds4_pid" >> "$OUT/err.log" ) &
wd_pid=$!
wait "$ds4_pid" 2>/dev/null
rc=$?
kill "$wd_pid" 2>/dev/null

echo "  ds4 退出码: $rc （137=被看门狗杀掉，属预期：转储在此之前已完成）"
echo
echo "── 退出后 ──"; mem

echo
echo "── 转储文件 ──"
ls -la "$OUT" | grep -E "ref\.|out\.log|err\.log" | awk '{printf "  %12s  %s\n", $5, $9}'
n=$(ls "$OUT" | grep -c '^ref\.')
echo "  合计 $n 个转储"

echo
echo "── stderr（前 30 行）──"
head -30 "$OUT/err.log" 2>/dev/null | sed 's/^/  /'
echo
echo "── stdout 尾（前 15 行）──"
tail -15 "$OUT/out.log" 2>/dev/null | sed 's/^/  /'
