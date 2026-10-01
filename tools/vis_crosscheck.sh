#!/bin/zsh
# ─────────────────────────────────────────────────────────────────────
#  vis_crosscheck.sh — 图片预处理/布局 与 ds4 的【逐字节】对拍门禁
#
#  为什么用两个可执行文件而不是一个测试程序：
#    ds4_image.c 与 myds_vispre.c 都 #define JPEG_IMPLEMENTATION /
#    PNG_IMPLEMENTATION，链进同一二进制会符号冲突。既然要求算法完全一致，
#    最干净的判据就是「两边写出的文件必须一模一样」。
#
#  参照实现【直接编译 ds4 的源码】（不改 ds4 一个字节）——
#    这样参照永远是真的 ds4，不会因为我抄错而一起错。
#
#  用法: tools/vis_crosscheck.sh [图片目录] [ds4 源码目录]
# ─────────────────────────────────────────────────────────────────────
set -u

ROOT="$(cd "$(dirname "${(%):-%x}")/.." && pwd)"
IMG_DIR="${1:-$ROOT/tests/images}"
DS4_DIR="${2:-$HOME/Projects/ds4}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

CC=${CC:-cc}
# ★ zsh 不对未加引号的变量做词分割 ⇒ 用数组（写成 $CFLAGS 会被当成一个参数）
CFLAGS=(-O2 -std=c11)

if [[ ! -f "$DS4_DIR/ds4_image.c" ]]; then
    echo "⚠️  找不到 ds4 参照源码: $DS4_DIR/ds4_image.c"
    echo "    跳过对拍（这是【开发期门禁】，不是构建依赖）"
    exit 0
fi

echo "🔬 编译两个版本 ..."
$CC $CFLAGS -I"$ROOT/include" -I"$ROOT/third_party" \
    -o "$TMP/myds" "$ROOT/tools/vis_pre_dump.c" "$ROOT/src/myds_vispre.c" -lm || exit 1
$CC $CFLAGS -DUSE_DS4_REF -I"$DS4_DIR" \
    -o "$TMP/ds4" "$ROOT/tools/vis_pre_dump.c" "$DS4_DIR/ds4_image.c" -lm || exit 1

shopt -s nullglob 2>/dev/null || setopt null_glob 2>/dev/null
imgs=("$IMG_DIR"/*.png "$IMG_DIR"/*.jpg "$IMG_DIR"/*.jpeg)

if (( ${#imgs} == 0 )); then
    echo "⚠️  $IMG_DIR 里没有图片 —— 生成一批 ..."
    mkdir -p "$IMG_DIR"
    python3 - "$IMG_DIR" <<'PY'
import sys, os
from PIL import Image
out = sys.argv[1]
cases = [("tiny_1x1",(1,1)),("wide_800x100",(800,100)),("tall_100x800",(100,800)),
         ("square_512",(512,512)),("huge_4000x3000",(4000,3000)),
         ("ratio8_1600x200",(1600,200)),("odd_333x777",(333,777)),
         ("photo_1024x768",(1024,768))]
for name,(w,h) in cases:
    im = Image.new("RGB",(w,h)); px = im.load()
    for y in range(h):
        for x in range(w):
            px[x,y] = ((x*7+y*3)%256, (x*3+y*11)%256, (x*13+y*5)%256)
    im.save(os.path.join(out, name + ".png"))
Image.new("L",(300,200),128).save(os.path.join(out,"gray_300x200.png"))
Image.new("RGBA",(200,300),(10,20,30,128)).save(os.path.join(out,"rgba_200x300.png"))
for n in ("photo_1024x768","square_512","wide_800x100"):
    Image.open(os.path.join(out,n+".png")).save(os.path.join(out,n+".jpg"), quality=92)
PY
    imgs=("$IMG_DIR"/*.png "$IMG_DIR"/*.jpg)
fi

pass=0; fail=0
echo "🔬 逐字节对拍（${#imgs} 张）..."
for img in "${imgs[@]}"; do
    r1=$("$TMP/myds" "$img" "$TMP/a.bin" 2>/dev/null | head -1)
    r2=$("$TMP/ds4"  "$img" "$TMP/b.bin" 2>/dev/null | head -1)
    name="${img:t}"
    if [[ "$r1" != "$r2" ]]; then
        echo "  ✗ $name  摘要不同: myds[$r1] ds4[$r2]"; fail=$((fail+1))
    elif cmp -s "$TMP/a.bin" "$TMP/b.bin"; then
        echo "  ✓ $name  $r1"; pass=$((pass+1))
    else
        echo "  ✗ $name  字节不同 ($(stat -f%z "$TMP/a.bin") vs $(stat -f%z "$TMP/b.bin"))"
        echo "      $(cmp "$TMP/a.bin" "$TMP/b.bin" 2>&1 | head -1)"
        fail=$((fail+1))
    fi
done

echo "──────────────────────────────────────────"
echo "  逐字节一致: $pass   不一致: $fail"
if (( fail > 0 )); then echo "❌ 对拍失败"; exit 1; fi
echo "✅ 预处理与布局和 ds4 完全一致（含浮点 patch 逐位）"
exit 0
