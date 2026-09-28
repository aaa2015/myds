#!/usr/bin/env python3
"""Regenerate src/myds_tokenizer_unicode.h using Python's Unicode 15.1 data.

Run with Python 3.13 (unicodedata 15.1.0). Pinning the version makes pre-tokenizer
classification deterministic across hosts and independent of the C locale.
"""
from pathlib import Path
import unicodedata

if unicodedata.unidata_version != "15.1.0":
    raise SystemExit("Generation requires unicodedata 15.1.0 (Python 3.13)")

whitespace = {9, 10, 11, 12, 13, 32, 0x85, 0xA0, 0x1680, 0x2028, 0x2029,
              0x202F, 0x205F, 0x3000, *range(0x2000, 0x200B)}
ranges = []
for cp in range(0x110000):
    category = unicodedata.category(chr(cp))[0]
    mask = {"L": 1, "M": 2, "N": 4, "P": 8, "S": 8}.get(category, 0)
    mask |= 16 if cp in whitespace else 0
    if not mask:
        continue
    if ranges and ranges[-1][1] + 1 == cp and ranges[-1][2] == mask:
        ranges[-1][1] = cp
    else:
        ranges.append([cp, cp, mask])

source = '''/* Generated from Python unicodedata 15.1.0 General_Category L/M/N/P/S and
 * Unicode White_Space. No locale-dependent classification. */
#ifndef MYDS_TOKENIZER_UNICODE_H
#define MYDS_TOKENIZER_UNICODE_H
#include <stdint.h>
#include <stddef.h>
enum { U_LETTER=1, U_MARK=2, U_NUMBER=4, U_PUNCT=8, U_SPACE=16 };
static const struct { uint32_t lo, hi; unsigned cls; } myds_unicode_ranges[] = {
'''
source += "".join("    {0x%x, 0x%x, %d},\n" % tuple(row) for row in ranges)
source += '''};
static unsigned myds_unicode_class(uint32_t cp) {
    size_t lo = 0, hi = sizeof(myds_unicode_ranges)/sizeof(myds_unicode_ranges[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo)/2;
        if (cp < myds_unicode_ranges[mid].lo) hi = mid;
        else if (cp > myds_unicode_ranges[mid].hi) lo = mid + 1;
        else return myds_unicode_ranges[mid].cls;
    }
    return 0;
}
#endif
'''
(Path(__file__).resolve().parents[1] / "src/myds_tokenizer_unicode.h").write_text(source)
