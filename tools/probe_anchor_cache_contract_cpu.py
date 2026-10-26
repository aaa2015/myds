#!/usr/bin/env python3
"""CPU diagnostic of source-extracted anchor routing; never loads Metal/model.

This executes unchanged current source bodies with checked CPU memcpy buffers.
It is not a full-engine, GPU, numerical, timing, or deployment validation.
Expected output describes current defects, so a zero exit code means the probe
ran and matched its explicit observations, not that the cache is correct.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def extract_braced(text, signature):
    if text.count(signature) != 1:
        raise ValueError(f"expected one source anchor: {signature!r}")
    start = text.index(signature)
    opening = text.index("{", start)
    level, state, escaped = 0, "code", False
    i = opening
    while i < len(text):
        c, pair = text[i], text[i:i + 2]
        if state == "line":
            if c == "\n":
                state = "code"
        elif state == "block":
            if pair == "*/":
                state = "code"
                i += 1
        elif state in ("string", "char"):
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == ('"' if state == "string" else "'"):
                state = "code"
        elif pair == "//":
            state = "line"
            i += 1
        elif pair == "/*":
            state = "block"
            i += 1
        elif c == '"':
            state = "string"
        elif c == "'":
            state = "char"
        elif c == "{":
            level += 1
        elif c == "}":
            level -= 1
            if level == 0:
                return text[start:i + 1]
        i += 1
    raise ValueError("unclosed source body")


PREAMBLE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "myds_snapshot_slots.h"
#define V41_LAYERS 40u
#define V41_WINDOW 128u
#define V41_HEAD_DIM 512u
#define V41_SNAP_SLOTS MYDS_SNAP_SLOTS
#define V41_SLOT_CAPACITY MYDS_SLOT_CAPACITY
#define MYDS_LOG_I(...) ((void)0)
typedef struct { unsigned char *data; size_t size; } myds_metal_buffer_t;
typedef struct { uint64_t history; uint32_t position; } engram_t;
typedef struct {
    bool snap_enabled, valid;
    uint32_t snap_pos[MYDS_SLOT_CAPACITY], snap_staged_pos[MYDS_SLOT_CAPACITY];
    uint32_t next_position;
    myds_metal_buffer_t *window[40], *previous_kv[4], *previous_score[4];
    myds_metal_buffer_t *snap_window[MYDS_SLOT_CAPACITY];
    myds_metal_buffer_t *snap_prev_kv[MYDS_SLOT_CAPACITY][4];
    myds_metal_buffer_t *snap_prev_score[MYDS_SLOT_CAPACITY][4];
    uint64_t snap_engram[MYDS_SLOT_CAPACITY];
    engram_t *engram;
    myds_metal_buffer_t *compressed[4], *index_cache[4];
} myds_v41_context_t;
typedef struct {
    myds_v41_context_t *v41;
    int32_t *anchor_tokens[MYDS_ANCHOR_SLOTS_MAX];
    uint32_t anchor_len[MYDS_ANCHOR_SLOTS_MAX], anchor_slots;
} myds_context_t;
static void lowkv_join(myds_v41_context_t *g) { (void)g; }
static bool myds_metal_commit_batch(void) { return true; }
static bool myds_metal_begin_batch(void) { return true; }
static myds_metal_buffer_t *buf(size_t size) {
    myds_metal_buffer_t *b=calloc(1,sizeof(*b)); assert(b);
    b->data=calloc(1,size); b->size=size; assert(b->data); return b;
}
static bool myds_metal_buffer_copy(myds_metal_buffer_t *dst, size_t off_dst,
    myds_metal_buffer_t *src, size_t off_src, size_t bytes) {
    assert(dst && src && off_dst<=dst->size && bytes<=dst->size-off_dst);
    assert(off_src<=src->size && bytes<=src->size-off_src);
    memcpy(dst->data+off_dst,src->data+off_src,bytes); return true;
}
static bool snapshot_ensure_slot(myds_v41_context_t *g,uint32_t s) {
    if(!g->snap_window[s])g->snap_window[s]=buf(40u*128u*512u*4u);
    for(uint32_t o=0;o<4;o++) {
        if(!g->snap_prev_kv[s][o])g->snap_prev_kv[s][o]=buf(512u*4u);
        if(!g->snap_prev_score[s][o])g->snap_prev_score[s][o]=buf(512u*4u);
    }
    return true;
}
static void fill_live(myds_v41_context_t *g,unsigned char mark) {
    for(uint32_t l=0;l<40;l++)memset(g->window[l]->data,mark,g->window[l]->size);
    for(uint32_t o=0;o<4;o++) {
        memset(g->previous_kv[o]->data,mark,g->previous_kv[o]->size);
        memset(g->previous_score[o]->data,mark,g->previous_score[o]->size);
    }
    g->engram->history=mark;
    for(uint32_t o=0;o<4;o++) {
        memset(g->compressed[o]->data,mark,g->compressed[o]->size);
        memset(g->index_cache[o]->data,mark,g->index_cache[o]->size);
    }
}
'''

MAIN = r'''
int main(void) {
    myds_v41_context_t g={.snap_enabled=true}; engram_t engram={0};g.engram=&engram;
    for(uint32_t l=0;l<40;l++)g.window[l]=buf(128u*512u*4u);
    for(uint32_t o=0;o<4;o++) {
        g.previous_kv[o]=buf(512u*4u);g.previous_score[o]=buf(512u*4u);
        g.compressed[o]=buf(2048u*512u*2u);g.index_cache[o]=buf(2048u*128u*2u);
    }
    enum{N=2048};int32_t a[N],b[N],prompt[N+1];
    for(uint32_t i=0;i<N;i++){a[i]=(int32_t)(1000+i);b[i]=(int32_t)(2000+i);prompt[i]=b[i];}
    prompt[N]=99;
    myds_context_t ctx={.v41=&g,.anchor_tokens={a,b},.anchor_len={N,N},.anchor_slots=2};
    fill_live(&g,0xAA);assert(myds_v41_snapshot_store_slot(&g,myds_anchor_slot(0),N));
    fill_live(&g,0xBB);assert(myds_v41_snapshot_store_slot(&g,myds_anchor_slot(1),N));
    uint32_t seat=UINT32_MAX;
    assert(myds_v41_anchor_pick(prompt,N+1,ctx.anchor_tokens,ctx.anchor_len,2,&seat)==N && seat==1);
    uint32_t gated=source_engine_anchor_select(&ctx,prompt,N+1,1024);
    assert(gated==1024);
    uint32_t selected=source_engine_anchor_select(&ctx,prompt,N+1,0);
    assert(selected==N);
    assert(myds_v41_snapshot_restore(&g,selected));
    unsigned char collision=g.window[0]->data[0];
    assert(collision==0xAA && g.engram->history==0xAA);
    /* Remove positional collision: even the intended slot does not restore
       the full per-prefix compressed/index cache, which is shared live state. */
    g.snap_pos[myds_anchor_slot(0)]=0;
    fill_live(&g,0xCC);
    assert(myds_v41_snapshot_restore(&g,N));
    assert(g.window[0]->data[0]==0xBB && g.engram->history==0xBB);
    for(uint32_t o=0;o<4;o++) {
        for(size_t j=0;j<g.compressed[o]->size;j++)assert(g.compressed[o]->data[j]==0xCC);
        for(size_t j=0;j<g.index_cache[o]->size;j++)assert(g.index_cache[o]->data[j]==0xCC);
    }
    printf("{\"shallow_snapshot\":{\"selected\":%u,\"matching_anchor\":%u,\"skipped\":true},"
           "\"same_length_anchor_collision\":{\"token_matched_seat\":%u,\"expected_window_marker\":187,\"actual_window_marker\":%u,\"wrong_slot_restored\":true},"
           "\"cross_prefix_state\":{\"window_marker\":%u,\"live_compressed_marker\":%u,\"live_index_marker\":%u,\"prefix_payload_not_restored\":true}}\n",
           gated,N,seat,collision,g.window[0]->data[0],g.compressed[0]->data[0],g.index_cache[0]->data[0]);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [root / p for p in ("src/myds_v41.c", "src/myds_engine.c", "include/myds_internal.h", "include/myds_snapshot_slots.h")]
    texts = {p.name: p.read_text() for p in paths}
    v41 = texts["myds_v41.c"]
    bodies = {
        "store_slot": extract_braced(v41, "bool myds_v41_snapshot_store_slot("),
        "restore": extract_braced(v41, "bool myds_v41_snapshot_restore("),
        "anchor_pick": extract_braced(texts["myds_internal.h"], "static inline uint32_t myds_v41_anchor_pick("),
        "engine_anchor_selection": extract_braced(texts["myds_engine.c"], "if (pcache_on && ctx->v41 && resume_from == 0) {"),
    }
    select = "\nstatic uint32_t source_engine_anchor_select(myds_context_t *ctx, const int32_t *prompt_tokens, int32_t n_prompt, uint32_t resume_from) {\n const bool pcache_on=true; bool snap_hit=false;\n" + bodies["engine_anchor_selection"] + "\n (void)snap_hit; return resume_from;\n}\n"
    generated = PREAMBLE + bodies["store_slot"] + "\n" + bodies["restore"] + "\n" + bodies["anchor_pick"] + select + MAIN
    with tempfile.TemporaryDirectory(prefix="myds-anchor-contract-cpu-") as tmp:
        cfile, exe = Path(tmp) / "probe.c", Path(tmp) / "probe"
        cfile.write_text(generated)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(root / "include"), str(cfile), "-o", str(exe)], check=True, capture_output=True, text=True)
        run = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    for p in paths:
        if p.read_text() != texts[p.name]:
            raise RuntimeError(f"source changed while probing: {p}")
    result = {
        "scope": "source-extracted actual C functions; CPU memcpy Metal stubs; no full-engine/GPU/numerical/performance validation",
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "extracted_body_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in bodies.items()},
        "generated_c_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "observations": json.loads(run.stdout),
        "production_changed": False,
        "adoptable_speedup_proved": False,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
