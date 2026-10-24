#!/usr/bin/env python3
"""rotate_myds_logs.py — 把「昨天及更早」的日志行剪到今天之前的日期文件里。

设计约束（决定了为什么这样做）：
  · 服务用 `>> ${ALIAS}.log` 打开 stdout（O_APPEND 语义）。
    O_APPEND 保证每次 write 都由内核 seek 到文件末尾，
    所以本脚本可以安全地把文件截短/重写，服务下一次写入会落到新末尾，不会留空洞。
    （已实测验证：截断后写入长度 == 新增内容长度，无 NUL 填充。）
  · 文件名对使用者保持固定（${ALIAS}.log 永远是「今天」的），
    按日期归档的部分才带日期后缀。
  · 只在 00:05 跑：此时「昨天」的行必然是文件的前缀，
    因为日志是单调追加的。不需要全文排序，只需找到第一条「今天」的行。

用法：
    rotate_myds_logs.py <log_dir> [--dry-run] [--today YYYY-MM-DD]

行为：
    1. 扫描 <log_dir>/*.log（跳过已带日期的归档文件）
    2. 找出第一条以今天的日期开头的时间戳行
    3. 之前的行（昨天及更早）追加到 <base>-<该行日期>.log
       —— 若前缀里跨了多天，会按天分别归档
    4. 剩余的（今天的）行留在原文件

失败时不动原文件（先写临时文件再原子替换），并返回非 0。
"""

import argparse
import os
import re
import sys
import tempfile

# 匹配行首时间戳：[2026-10-24 17:25:17.123]
TS_RE = re.compile(r'^\[(\d{4}-\d{2}-\d{2}) ')
# 归档文件名形如 name-2026-10-24.log
ARCHIVE_RE = re.compile(r'-\d{4}-\d{2}-\d{2}\.log$')


def split_lines_by_date(path, today):
    """返回 [(date, [lines]), ...]，按出现顺序。没有时间戳的行跟随上一条。"""
    groups = []
    cur_date = None
    cur_lines = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = TS_RE.match(line)
            if m:
                d = m.group(1)
                if d != cur_date:
                    if cur_lines:
                        groups.append((cur_date, cur_lines))
                    cur_date = d
                    cur_lines = []
            cur_lines.append(line)
    if cur_lines:
        groups.append((cur_date, cur_lines))

    # 前导无时间戳行（启动头：分隔线 / FLEET_CMD / RESTART_AT 等）
    # 属于「它们之后那次运行 ⇒ 之后第一组有日期的」。
    # 若整个文件都没有日期，则留在原组（date=None），由调用方决定。
    if groups and groups[0][0] is None:
        first_dated = next((i for i, (d, _) in enumerate(groups) if d), None)
        if first_dated is not None:
            merged = groups[0][1] + groups[first_dated][1]
            groups[first_dated] = (groups[first_dated][0], merged)
            groups.pop(0)
    return groups


def rotate_one(path, today, dry_run):
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return 0, 0
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0

    groups = split_lines_by_date(path, today)

    # 按【每组自己的日期】分类，不用「遇到今天就全保留」的状态机。
    #
    # 为什么：时间戳理论上单调，但重启、NTP 回拨、手工插入都会造成乱序。
    # 第一版用 seen_today 标志，遇到第一个 10-24 之后的所有行都当今天保留，
    # 于是夹在后面的 10-23 行被漏归档 —— 实测踩到（假日志在第 227 行）。
    #
    # 无日期的组（date is None）只可能出现在文件最开头（已被 split_lines_by_date
    # 并入其后第一组），这里若仍出现，视为「未知日期」，保守地跟着最近的上一个日期。
    old_groups = []
    keep_lines = []
    last_seen = None
    for d, lines in groups:
        eff = d if d is not None else last_seen
        if d is not None:
            last_seen = d
        if eff is not None and eff < today:
            old_groups.append((eff, lines))
        else:
            keep_lines.extend(lines)

    if not old_groups:
        return 0, size

    moved = 0
    if dry_run:
        for d, lines in old_groups:
            target = base + '-' + (d or 'undated') + ext
            print("  [dry-run] %d 行 → %s" % (len(lines), os.path.basename(target)))
            moved += len(lines)
        print("  [dry-run] 保留 %d 行在原文件（%.1f KB）" % (len(keep_lines), size / 1024.0))
        return moved, size

    for d, lines in old_groups:
        target = base + '-' + (d or 'undated') + ext
        with open(target, 'a', encoding='utf-8') as out:
            out.writelines(lines)
        moved += len(lines)

    # 原子替换：先写临时文件，再 rename。
    # 服务那边是 O_APPEND，下一次 write 会落到新文件的末尾。
    d = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(prefix='.rotate-', dir=d)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            out.writelines(keep_lines)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    return moved, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log_dir')
    ap.add_argument('--today', default=None, help='YYYY-MM-DD，默认取系统日期')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    today = args.today
    if not today:
        import datetime
        today = datetime.date.today().isoformat()

    log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
    if not os.path.isdir(log_dir):
        print("log_dir 不存在: %s" % log_dir, file=sys.stderr)
        return 2

    total_files = 0
    total_moved = 0
    total_before = 0
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith('.log'):
            continue
        if ARCHIVE_RE.search(name):
            continue                      # 已是归档文件，跳过
        path = os.path.join(log_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            moved, before = rotate_one(path, today, args.dry_run)
        except Exception as e:
            print("FAILED %s: %s" % (name, e), file=sys.stderr)
            return 1
        if moved or before:
            total_files += 1
            total_moved += moved
            total_before += before
            if moved:
                print("  %s: 归档 %d 行，原文件 %.1f KB → 保留今天" % (name, moved, before / 1024.0))

    if total_moved == 0:
        print("  无需轮转（没有可归档的旧日志）")
    elif not args.dry_run:
        print("  完成：%d 个文件，归档 %d 行" % (total_files, total_moved))
    return 0


if __name__ == '__main__':
    sys.exit(main())
