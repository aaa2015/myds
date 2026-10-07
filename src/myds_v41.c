/* Native DeepSeek V4.1 text graph. Numerical ordering is checked against the
 * separately built, pinned ds4 oracle (MIT); no reference engine is linked.
 * Keep this graph diagnostic until the recorded whole-model gates pass. */
#include "myds_internal.h"
#include "myds_engram.h"
#include "myds_v41_gpu.h"
#include "myds_v41_moe.h"
#include "myds_v41_decode.h"
#include "myds_cpu.h"
#include "myds_log.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sys/qos.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define V41_LAYERS 40u
#define V41_DIM 5120u
#define V41_HEADS 64u
#define V41_HEAD_DIM 512u
#define V41_INDEX_HEADS 32u
#define V41_INDEX_DIM 128u
#define V41_WINDOW 128u
#define V41_TOPK 512u
/* Per-token workspace arrays below are sized by this, NOT by 512.  They used to
 * be hardcoded 512-wide while the scalar fields (cur_res_view, cur_pre_view,
 * hc_flat_norm_buf, hc_mix_buf) sat after them in the struct, so any capacity
 * above 512 made the assembly loop write arr[t] past the arrays and silently
 * overwrite those fields -- hc_flat_norm_buf came back as size 0 and prefill
 * died at layer 0 with "HC attention input".  Raising the cap therefore needs
 * this sizing, not just a relaxed clamp. */
/* 单个 chunk 的 token 上界。用 myds_v41_gpu.h 里的【单一真源】：以前这里钳到 4096、
 * 而 chunk attention 只接受 2048 —— 于是 MYDS_PREFILL_CHUNK_SIZE=3000 这种「配置合法」
 * 的值会在运行时让整个 prefill 失败。现在两边同源，host 侧统一钳到 2048。 */
#define V41_MAX_CHUNK MYDS_V41_CHUNK_MAX
/* MoE 批的 token 上限（内存护栏）：批缓冲约 0.44 MB/token。8192 ⇒ 约 3.6 GB。 */
#define MYDS_MOE_BATCH_MAX_TOKENS 8192u

typedef struct {
    myds_metal_buffer_t *attn_fn, *attn_scale, *attn_base;
    myds_metal_buffer_t *ffn_fn, *ffn_scale, *ffn_base;
    myds_metal_buffer_t *compressor_kv, *compressor_gate, *compressor_norm;
    myds_metal_buffer_t *index_k, *index_k_norm, *index_q, *index_proj;
} v41_layer_t;

struct myds_v41_model {
    v41_layer_t layers[V41_LAYERS];
    myds_engram_t *engram;
    myds_metal_buffer_t *engram_proj[2], *engram_q[2], *engram_k[2];
    float hc_eps;
};

/* ── 前缀快照 ──
 * 引擎里唯一【无法从「按位置索引」重建】的状态有两样：
 *   · 窗口环（每层 128 槽）—— raw KV 只存在这里，而它只保留【最后 128 个位置】；
 *   · engram 的滚动 tail[3] —— 且它在【层循环之前】就已被推到 prompt 末尾。
 * 压缩 KV 与 index cache 都是按位置累积的，续跑时直接用，因此不入快照。
 * 窗口环事后无法回溯（已被后续 token 覆盖）⇒ 必须在 prefill【经过】该位置时就存下来。
 * 槽位用确定性映射（P/STRIDE % SLOTS）：engram 循环与层循环各自独立写入，
 * 但映射与覆盖顺序一致 ⇒ 同一槽里的 engram 与各层窗口环必然对应同一个 P。
 * ds4 的对应机制见 ds4_kvstore.c: boundary_trim/align + find_text_prefix（默认 trim=32/align=2048）。 */
/* 槽位分配抽到 include/myds_snapshot_slots.h（可单测）。 */
#include "myds_snapshot_slots.h"
#include "myds_kvdsk.h"
#include "myds_lowkv_policy.h"
#define V41_SNAP_SLOTS   MYDS_SNAP_SLOTS
/* 快照点必须是 chunk 边界；512 同时满足 ratio(2)/32 键块的对齐要求。
 * ★ 它还是【偶数】这一点是承重的：压缩器的 pool2 跨位置有状态（previous_kv/previous_score），
 *   但只在【奇数】位置被读取 —— 从偶数位置续跑时它先被本位置覆盖，因此无需进快照。
 *   若将来把 STRIDE 改成奇数，就必须把这些 previous_* 一并纳入快照。 */
#define V41_SNAP_STRIDE  MYDS_SNAP_STRIDE
#define V41_SNAP_MIN     MYDS_SNAP_MIN  /* 太短的点不值一份窗口环 */

typedef struct v41_chunk_workspace_s v41_chunk_workspace_t;
static void v41_chunk_workspace_free(v41_chunk_workspace_t *cw);

/* 一条待落盘的低位区快照：槽号、位置、前缀哈希（哈希在主线程算好，线程不碰 token 数组）。 */
typedef struct {
    uint32_t s, p;
    uint64_t hash;
    uint64_t segment_hashes[MYDS_SNAP_PINNED];
} v41_lowkv_job_t;

struct myds_v41_context {
    myds_engram_t *engram;
    myds_v41_gpu_scratch_t *gpu;
    uint32_t capacity, next_position;
    bool valid;
    myds_metal_buffer_t *residual, *after_attn, *flat_norm, *mix;
    myds_metal_buffer_t *attn_split, *ffn_split, *pre;
    myds_metal_buffer_t *engram_rows[2], *engram_kv;
    myds_metal_buffer_t *window[V41_LAYERS];
    /* 快照：每槽把 V41_LAYERS 层的窗口环连续存放（40 x 256 KB = 10.5 MB/槽） */
    myds_metal_buffer_t *snap_window[V41_SNAP_SLOTS];
    uint32_t              snap_pos[V41_SNAP_SLOTS];      /* 0 = 空槽，仅全部层与压缩器状态写入后才对外发布 */
    uint32_t              snap_staged_pos[V41_SNAP_SLOTS]; /* 准备中尚未发布的快照位置 */
    myds_engram_history_t snap_engram[V41_SNAP_SLOTS];
    myds_metal_buffer_t *snap_prev_kv[V41_SNAP_SLOTS][4];
    myds_metal_buffer_t *snap_prev_score[V41_SNAP_SLOTS][4];
    bool                  snap_enabled;
    /* 低位区 KV 的磁盘存储（NULL = 未启用）。由引擎在创建后显式打开。 */
    myds_kvdsk_t         *low_store;
    myds_kvdsk_t         *low_segments;
    myds_kvdsk_t         *low_ends;
    uint64_t              low_segment_writes, low_segment_bytes, low_snapshot_bytes;
    uint32_t              low_fail_segment_end; /* fixture-only partial-write injection */
    /* 身份戳的来源（arch / 形状 / 模型大小）。不持有所有权。 */
    const myds_model_t   *model;
    /* 每组位置的 way 数（1..V41_LOWKV_WAYS_MAX，现为 32）。见 V41_LOWKV_WAYS_DEFAULT 的说明。 */
    uint32_t              low_ways;
    /* 低位区【后台】写盘（MYDS_KV_DSK_ASYNC=1，默认关）。见 myds_v41_low_kv_save 的契约说明。
     * 线程只读 low_jobs[] + snap_window/snap_engram[s] + compressed/index_cache 的 [0, p/r) 行，
     * 只写 low_store；任何可能碰到这些的入口都先 lowkv_join()。 */
    bool                  low_async;
    bool                  low_thr_live;
    bool                  low_thr_ok;
    pthread_t             low_thr;
    uint32_t              low_job_n;
    v41_lowkv_job_t       low_jobs[MYDS_SNAP_PINNED];
    myds_metal_buffer_t *compressed[4], *index_cache[4];
    myds_metal_buffer_t *previous_kv[4], *previous_score[4];
    myds_metal_buffer_t *pool_kv, *pool_score, *latent;
    myds_metal_buffer_t *index_k, *index_q, *index_weights, *index_scores;
    myds_metal_buffer_t *selected, *selected_kv;
    myds_metal_buffer_t *block_scores, *block_selected, *block_mask;
    /* 预填充块级工作区（长短请求复用，避免反复分配/释放抖动） */
    v41_chunk_workspace_t *chunk_ws;
    /* 全量 prefill 临时缓冲区复用（避免每次请求反复分配释放数吉字节） */
    myds_metal_buffer_t *prompt_res;
    myds_metal_buffer_t *prompt_pre;
    myds_metal_buffer_t *prompt_selected;
    myds_metal_buffer_t *prompt_block_mask;
    float *prompt_engram[2];
    uint32_t prompt_tokens_cap;
    size_t prompt_mask_bytes_cap;
};

/* 每次请求开始前清掉【本次会失效】的点：只保留位置 <= keep_up_to 的。
 *
 * 判据是「本次请求从哪里开始写」：本次只写 [keep_up_to, ...)，因此比 keep_up_to 更早的位置
 * 其窗口环（[S-128, S-1] ⊆ [.., keep_up_to-1]）与压缩 KV 都【不会被本次触碰】⇒ 那些点仍然有效。
 * 反过来 keep_up_to 之后的点必须清掉：它们的窗口环已被后续 token 覆盖。
 * keep_up_to == 0（全量 prefill）时等于全清，与之前的行为一致。
 *
 * 清空全部是错的：实测第 2 次命中 6144 之后，第 3 次的公共前缀只到 6153（不到末尾 6162），
 * 唯一能用的就是被清掉的那个 6144，于是又退回全量 20 秒。 */
static void lowkv_join(myds_v41_context_t *g);   /* 低位区后台写盘的汇合点，定义见「低位区 KV 的 SSD 落盘」 */

/* Allocate a slot only when it is first captured. Disabled snapshots own no
 * slot buffers; published slots always have the complete set of resources. */
static bool snapshot_ensure_slot(myds_v41_context_t *g, uint32_t s) {
    if (!g || !g->snap_enabled || s >= V41_SNAP_SLOTS) return false;
    if (!g->snap_window[s])
        g->snap_window[s] = myds_metal_buffer_alloc((size_t)V41_LAYERS * V41_WINDOW * V41_HEAD_DIM * 4u);
    if (!g->snap_window[s]) return false;
    for (uint32_t o = 0; o < 4u; ++o) {
        if (!g->snap_prev_kv[s][o]) g->snap_prev_kv[s][o] = myds_metal_buffer_alloc(512u * 4u);
        if (!g->snap_prev_score[s][o]) g->snap_prev_score[s][o] = myds_metal_buffer_alloc(512u * 4u);
        if (!g->snap_prev_kv[s][o] || !g->snap_prev_score[s][o]) return false;
    }
    return true;
}

void myds_v41_snapshot_clear(myds_v41_context_t *g, uint32_t keep_up_to) {
    if (!g) return;
    lowkv_join(g);   /* 每个请求开头都会走到这里（engine 在 prefill/decode 之前调） */
    for (uint32_t s = 0; s < V41_SNAP_SLOTS; ++s) {
        if (g->snap_pos[s] > keep_up_to) g->snap_pos[s] = 0;
        if (g->snap_staged_pos[s] > keep_up_to) g->snap_staged_pos[s] = 0;
    }
}

/* 槽 0 专供「本次序列的末尾」。理由：续跑常常只推进很少的 token（比如兄弟请求之间），
 * 那种情况下序列末尾【不会】落在 512 的对齐点上，于是本次请求结束后槽里没有有效点，
 * 下一个兄弟请求又要全量重跑 —— 实测就是这样退化的（第 2 次命中 40x，第 3 次全量）。
 * 末尾点恰好是「下一个兄弟请求最可能用到的」那一个，所以给它一个固定槽、每次覆盖。 */
static uint32_t snapshot_slot_for(uint32_t p, bool is_last) {
    return myds_snap_slot_for(p, is_last ? 1 : 0);
}
/* 处理完位置 p 之后，记下 engram 的滚动状态。在 engram 循环里调用（层循环之前）。
 * ★ 事务契约：这里仅暂存 engram 状态与目标位置到 staged 槽，绝不提前置 snap_pos，
 * 避免在各层窗口环拷贝完成前被 snapshot_best 抢先选出半状态。
 * 若该槽此前存有旧快照，在此处先使旧发布标签失效，避免覆盖失败时旧标签仍有效。 */
static void snapshot_store_engram(myds_v41_context_t *g, uint32_t p, bool is_last) {
    if (!g->snap_enabled || p < V41_SNAP_MIN) return;
    if (!is_last && (p % V41_SNAP_STRIDE)) return;
    const uint32_t s = snapshot_slot_for(p, is_last);
    g->snap_pos[s] = 0;
    g->snap_staged_pos[s] = p;
    g->snap_engram[s] = g->engram->history;
}
/* 在层循环里、每个 chunk 结束时调用：把【本层】在 p 处的窗口环存进对应槽。
 * 只有当最后一层（il == V41_LAYERS-1）也保存完成，并且同步保存压缩器的 previous_kv/score
 * 之后，才正式发布 g->snap_pos[s] = p。任一步失败都清除 staged/published 标记。 */
static bool snapshot_store_window(myds_v41_context_t *g, uint32_t il, uint32_t p, bool is_last) {
    if (!g->snap_enabled || p < V41_SNAP_MIN) return true;
    if (!is_last && (p % V41_SNAP_STRIDE)) return true;
    const uint32_t s = snapshot_slot_for(p, is_last);
    if (g->snap_staged_pos[s] != p) return true;
    if (!snapshot_ensure_slot(g, s)) {
        g->snap_pos[s] = g->snap_staged_pos[s] = 0;
        return false;
    }
    const size_t bytes = (size_t)V41_WINDOW * V41_HEAD_DIM * 4u;
    if (!myds_metal_buffer_copy(g->snap_window[s], (size_t)il * bytes, g->window[il], 0, bytes)) {
        g->snap_pos[s] = 0;
        g->snap_staged_pos[s] = 0;
        return false;
    }
    if (il == V41_LAYERS - 1u) {
        for (uint32_t o = 0; o < 4u; ++o) {
            if (!myds_metal_buffer_copy(g->snap_prev_kv[s][o], 0, g->previous_kv[o], 0, 512 * 4u)) {
                g->snap_pos[s] = 0;
                g->snap_staged_pos[s] = 0;
                return false;
            }
            if (!myds_metal_buffer_copy(g->snap_prev_score[s][o], 0, g->previous_score[o], 0, 512 * 4u)) {
                g->snap_pos[s] = 0;
                g->snap_staged_pos[s] = 0;
                return false;
            }
        }
        g->snap_pos[s] = p;
        g->snap_staged_pos[s] = 0;
    }
    return true;
}
/* prefill 结束后由引擎调用：把【当前状态】存进末尾槽（0 号）。
 * ★ 批量提交：40 层窗口 + 4 previous KV + 4 previous score = 48 次复制，
 *   在同一个 command buffer 中编码，一次提交/等待。逐次提交路径（每次复制
 *   单独创建 CB → commit → waitUntilCompleted）实测 ~9.5 ms，批量 ~1.3 ms。
 * ★ 事务一致性：覆盖 slot 0 前先清除其 snap_pos 与 snap_staged_pos，确保若复制
 *   或 GPU 提交失败，旧快照标签不会残留指向新旧层混合的数据。 */
bool myds_v41_snapshot_store_now(myds_v41_context_t *g, uint32_t p) {
    if (!g || !g->snap_enabled || !p || p < V41_SNAP_MIN) return true;
    lowkv_join(g);
    /* 首次覆盖 slot 0 前使原发布标签和 staged 标记失效。 */
    g->snap_pos[0] = 0;
    g->snap_staged_pos[0] = 0;
    if (!snapshot_ensure_slot(g, 0)) return false;
    /* 先把调用方可能遗留的 in-flight 工作提交完成，保证窗口环/previous 数据已落定。 */
    if (!myds_metal_commit_batch()) return false;
    /* 开一个新 batch，把全部 48 次复制编进同一 command buffer。 */
    if (!myds_metal_begin_batch()) return false;
    const size_t bytes = (size_t)V41_WINDOW * V41_HEAD_DIM * 4u;
    for (uint32_t il = 0; il < V41_LAYERS; ++il)
        if (!myds_metal_buffer_copy(g->snap_window[0], (size_t)il * bytes, g->window[il], 0, bytes))
            goto store_fail;
    for (uint32_t o = 0; o < 4u; ++o) {
        if (!myds_metal_buffer_copy(g->snap_prev_kv[0][o], 0, g->previous_kv[o], 0, 512 * 4u))
            goto store_fail;
        if (!myds_metal_buffer_copy(g->snap_prev_score[0][o], 0, g->previous_score[o], 0, 512 * 4u))
            goto store_fail;
    }
    if (!myds_metal_commit_batch()) {
        g->snap_pos[0] = 0;
        g->snap_staged_pos[0] = 0;
        return false;
    }
    /* GPU 操作成功后才发布快照位置和有效状态。 */
    g->snap_engram[0] = g->engram->history;   /* engram 的 position 恒等于 p（每 token 推进一次） */
    g->snap_pos[0] = p;
    return true;
store_fail:
    /* 复制失败路径：提交残余 batch（可能部分编码），确保不发布快照。
     * 快照缺失的后果只是下一个请求无法从此点续跑（退回全量 prefill），不会静默出错。 */
    g->snap_pos[0] = 0;
    g->snap_staged_pos[0] = 0;
    (void)myds_metal_commit_batch();
    return false;
}

/* 在 [0, common] 内选最大的快照点 */
uint32_t myds_v41_snapshot_best(const myds_v41_context_t *g, uint32_t common) {
    if (!g->snap_enabled) return 0;
    uint32_t best = 0;
    for (uint32_t s = 0; s < V41_SNAP_SLOTS; ++s)
        if (g->snap_pos[s] && g->snap_pos[s] <= common && g->snap_pos[s] > best) best = g->snap_pos[s];
    return best;
}
/* 把某点的【全部】状态装回去：engram tail + 各层窗口环 + 压缩器跨步状态
 * ★ 批量提交：与 store_now 同理，48 次复制编进同一 command buffer。 */
bool myds_v41_snapshot_restore(myds_v41_context_t *g, uint32_t p) {
    lowkv_join(g);   /* restore 会改活窗口环；与线程读的 snap_window 同源 */
    if (!myds_metal_commit_batch()) return false;
    for (uint32_t s = 0; s < V41_SNAP_SLOTS; ++s) {
        if (g->snap_pos[s] != p) continue;
        /* 开一个新 batch，把全部复制编进同一 command buffer。 */
        if (!myds_metal_begin_batch()) return false;
        const size_t bytes = (size_t)V41_WINDOW * V41_HEAD_DIM * 4u;
        for (uint32_t il = 0; il < V41_LAYERS; ++il)
            if (!myds_metal_buffer_copy(g->window[il], 0, g->snap_window[s], (size_t)il * bytes, bytes))
                goto restore_fail;
        for (uint32_t o = 0; o < 4u; ++o) {
            if (!myds_metal_buffer_copy(g->previous_kv[o], 0, g->snap_prev_kv[s][o], 0, 512 * 4u))
                goto restore_fail;
            if (!myds_metal_buffer_copy(g->previous_score[o], 0, g->snap_prev_score[s][o], 0, 512 * 4u))
                goto restore_fail;
        }
        if (!myds_metal_commit_batch()) goto restore_fail;
        /* GPU 操作成功后才更新 CPU 状态，保证不会在 GPU 复制失败时留下半初始化的状态。 */
        g->engram->history = g->snap_engram[s];
        g->engram->position = p;
        g->next_position = p;
        g->valid = true;
        return true;
restore_fail:
        (void)myds_metal_commit_batch();
        return false;
    }
    return false;
}

/* ── 快照诊断与单测夹具 ── */
myds_v41_context_t *myds_v41_context_create_snapshot_fixture(void) {
    myds_v41_context_t *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->engram = calloc(1, sizeof(*g->engram));
    if (!g->engram) { free(g); return NULL; }
    for (uint32_t o = 0; o < 2; o++) g->engram->table[o].fd = -1;
    g->snap_enabled = getenv("MYDS_NO_PREFIX_SNAPSHOT") == NULL;
    for (uint32_t l = 0; l < V41_LAYERS; l++) {
        g->window[l] = myds_metal_buffer_alloc(V41_WINDOW * V41_HEAD_DIM * 4u);
        if (!g->window[l]) goto fail;
    }
    for (uint32_t o = 0; o < 4; o++) {
        g->previous_kv[o] = myds_metal_buffer_alloc(512 * 4u);
        g->previous_score[o] = myds_metal_buffer_alloc(512 * 4u);
        if (!g->previous_kv[o] || !g->previous_score[o]) goto fail;
    }
    g->valid = true;
    return g;
fail:
    myds_v41_context_free(g);
    return NULL;
}

uint32_t myds_v41_snapshot_get_pos(const myds_v41_context_t *g, uint32_t slot) {
    return (g && slot < V41_SNAP_SLOTS) ? g->snap_pos[slot] : 0;
}

uint32_t myds_v41_snapshot_get_staged_pos(const myds_v41_context_t *g, uint32_t slot) {
    return (g && slot < V41_SNAP_SLOTS) ? g->snap_staged_pos[slot] : 0;
}

myds_metal_buffer_t *myds_v41_snapshot_get_window(myds_v41_context_t *g, uint32_t layer) {
    return (g && layer < V41_LAYERS) ? g->window[layer] : NULL;
}

myds_metal_buffer_t *myds_v41_snapshot_get_snap_window(myds_v41_context_t *g, uint32_t slot) {
    return snapshot_ensure_slot(g, slot) ? g->snap_window[slot] : NULL;
}

myds_metal_buffer_t *myds_v41_snapshot_get_snap_prev_kv(myds_v41_context_t *g, uint32_t slot, uint32_t o) {
    return (snapshot_ensure_slot(g, slot) && o < 4u) ? g->snap_prev_kv[slot][o] : NULL;
}

myds_metal_buffer_t *myds_v41_snapshot_get_snap_prev_score(myds_v41_context_t *g, uint32_t slot, uint32_t o) {
    return (snapshot_ensure_slot(g, slot) && o < 4u) ? g->snap_prev_score[slot][o] : NULL;
}

uint64_t myds_v41_snapshot_allocated_bytes(const myds_v41_context_t *g) {
    uint64_t bytes = 0;
    if (!g) return 0;
    for (uint32_t s = 0; s < V41_SNAP_SLOTS; ++s) {
        bytes += myds_metal_buffer_size(g->snap_window[s]);
        for (uint32_t o = 0; o < 4u; ++o) {
            bytes += myds_metal_buffer_size(g->snap_prev_kv[s][o]);
            bytes += myds_metal_buffer_size(g->snap_prev_score[s][o]);
        }
    }
    return bytes;
}

myds_metal_buffer_t *myds_v41_snapshot_get_previous_kv(myds_v41_context_t *g, uint32_t o) {
    return (g && o < 4) ? g->previous_kv[o] : NULL;
}

myds_metal_buffer_t *myds_v41_snapshot_get_previous_score(myds_v41_context_t *g, uint32_t o) {
    return (g && o < 4) ? g->previous_score[o] : NULL;
}

void myds_v41_snapshot_set_engram_history(myds_v41_context_t *g, uint32_t t0, uint32_t t1, uint32_t t2) {
    if (g && g->engram) {
        g->engram->history.tail[0] = t0;
        g->engram->history.tail[1] = t1;
        g->engram->history.tail[2] = t2;
    }
}

void myds_v41_snapshot_get_engram_history(const myds_v41_context_t *g, uint32_t out[3]) {
    if (g && g->engram && out) {
        out[0] = g->engram->history.tail[0];
        out[1] = g->engram->history.tail[1];
        out[2] = g->engram->history.tail[2];
    }
}

void myds_v41_snapshot_set_positions(myds_v41_context_t *g, uint32_t engram_pos, uint32_t next_pos) {
    if (g) {
        if (g->engram) g->engram->position = engram_pos;
        g->next_position = next_pos;
    }
}

bool myds_v41_snapshot_get_valid(const myds_v41_context_t *g) {
    return g ? g->valid : false;
}


/* ══════════════════════════════════════════════════════════════════════════
 * 低位区 KV 的 SSD 落盘
 *
 * 内存快照只活在【进程内】，且只在与上一条序列共享前缀时才安全（见 snapshot_clear）。
 * 重启之后、或者客户端每次当新会话（只带系统提示词 + 工具定义）时，那一段常量前缀
 * 就要重新 prefill。这里把【低位区】那批快照及其引用的 KV 段落盘，让它跨进程复用。
 *
 * 快照描述符保存一个快照点的：
 *   · engram 滚动历史（决定后续 token 的 n-gram 行）
 *   · 40 层的窗口环（局部注意力读的 KV）
 *   · 压缩 KV/index 段的引用（slot、区间、完整前缀哈希、字节数）
 * 4 个 owner 的 packed BF16 行放在独立段库：首段 [0,1024)，之后每段 512 token。
 * 段的 key 是末位置及该位置的完整 token 前缀哈希，所以同一前缀的多个快照共用早期段。
 * 每个 owner 的行区间按其 ratio 推导。恢复描述符和全部引用段的校验均成功后才发布快照；
 * 被淘汰或损坏的引用使本次加载 miss。
 *
 * ── 关键不变量（与内存快照共用，改动前务必先读）──
 * 快照点必须是 512 的倍数（偶数）。压缩器的 previous_kv/previous_score 在【偶数】位置
 * 被本位置覆盖、在【奇数】位置才被读取 ⇒ 从偶数点续跑时它们先被覆盖，因此不必进快照。
 * 若把 MYDS_SNAP_STRIDE 改成奇数，这里也必须把 previous_* 一并存，否则会静默接错状态。
 * ══════════════════════════════════════════════════════════════════════════ */
#define V41_LOWKV_ABI    3u /* windows/history plus shared-prefix segment references */
#define V41_LOWKV_SEG_ABI 1u
#define V41_LOWKV_END_ABI 0x34314531u /* '41E1' prompt-end snapshot */
#define V41_LOWKV_ENDS_MAX 32u
#define V41_LOWKV_END_STRIDE 14680064ull /* 14 MiB */
#define V41_LOWKV_ENTRY_IS_END 0x80000000u
#define V41_LOWKV_REF_BYTES 32u
#define V41_CACHE_BYTES  2u
#define V41_LOWKV_HEADER 64u
#define V41_LOWKV_STAGE  (1u << 20)  /* 1 MiB 搬运分块：峰值内存只多这 1 MiB */
#define V41_LOWKV_LATENT 512u        /* 压缩 KV 每行宽度（须与 publish_compressor 一致） */
/* 每组位置开几个条目（way）。★ 为什么需要它：
 * MYDS_SNAP_PINNED 个固定位置槽【不是 P 个前缀】，而是【同一个前缀的 P 个深度层】——
 * 所以「一个位置一条」等于「一个前缀独占全部 P 槽」，第二个前缀进来就把第一个整个顶掉，
 * 两个前缀交替出现时互相挤没（每个请求都全量 + 重写，且日志只报 miss）。
 * 开 way 之后：同一位置可以同时容纳 W 个不同前缀（按前缀哈希区分），
 * 默认 3 ⇒ 可容纳三套「system + tools」互不干扰。
 * 两个库各有 MYDS_SNAP_PINNED × ways 个条目：描述符约 10 MiB 窗口加历史/refs，
 * 段库固定 stride 取首段容量，后续短段仅写自己的有效字节。总预算覆盖两个文件的
 * 数据区及对齐后的 backend 头/条目表；MYDS_KV_DSK_BUDGET_MB 显式启用预算限制。 */
#define V41_LOWKV_WAYS_DEFAULT 3u
/* 调大 MYDS_KV_DSK_WAYS 可让各位置保存更多分叉。
 * 槽数不进身份戳，但 open 时会比对槽数 ⇒
 *   自动重建文件（旧内容作废，代价是一次冷 prefill），不会读写错位。
 * ★ 同时须把 MYDS_KV_DSK_BUDGET_MB 配足：预算不够 open 会【直接失败】，不会降级为更少槽位。 */
#define V41_LOWKV_WAYS_MAX     32u

/* 后台写盘与 decode 的「行区间不相交」只排除【值】冲突；Metal 对 shared 资源另要求 CPU/GPU
 * 不同时访问同一区域。这里再钉一层结构性保证：行边界必须落在 128 B 缓存行上，否则一次
 * memcpy 可能跨进 GPU 正在写的行。下面按 packed BF16 的实际行字节数检查对齐。 */
_Static_assert((V41_LOWKV_LATENT * V41_CACHE_BYTES) % 128u == 0u, "LOWKV latent 行宽必须对齐 128 B 缓存行");
_Static_assert((V41_INDEX_DIM * V41_CACHE_BYTES) % 128u == 0u, "LOWKV index 行宽必须对齐 128 B 缓存行");
#define V41_LOWKV_NO_ENTRY     0xFFFFFFFFu

static void lk_put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xFFu);          p[1] = (unsigned char)((v >> 8) & 0xFFu);
    p[2] = (unsigned char)((v >> 16) & 0xFFu);  p[3] = (unsigned char)((v >> 24) & 0xFFu);
}
static uint32_t lk_get_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* 层配置的三个判定函数定义在文件后面（压缩器那一段），这里先声明：
 * 低位区的 payload 布局依赖它们，而写死常量会在改层配置时悄悄分叉。 */
static uint32_t ratio_for(uint32_t l);
static uint32_t owner_for(uint32_t l);
static bool     kv_source(uint32_t l);

/* 每个 owner 的压缩比 —— 从 kv_source/owner_for/ratio_for 推导，不写死：
 * 将来改层配置时，写死的常量会与真实布局悄悄分叉（那正好是静默错误）。 */
static uint32_t lowkv_owner_ratio(uint32_t owner) {
    for (uint32_t il = 0; il < V41_LAYERS; ++il)
        if (kv_source(il) && owner_for(il) == owner) return ratio_for(il);
    return 0;
}

/* 条目编号 = 位置组 × way 数 + way。位置组分 1..MYDS_SNAP_PINNED，
 * 对应的位置由 myds_snap_slot_for 决定（1024 + 512k，k = 0..MYDS_SNAP_PINNED-1；现为 1024..11264）。
 * 注意快照点只有 512 的倍数 ⇒ p 与位置组是一一对应的，不会歧义。 */
static uint32_t lowkv_entry_index(const myds_v41_context_t *g, uint32_t group, uint32_t way) {
    return group * g->low_ways + way;
}

static uint32_t lowkv_segment_count(uint32_t p) {
    if (p < MYDS_SNAP_MIN || (p - MYDS_SNAP_MIN) % MYDS_SNAP_STRIDE) return 0;
    const uint32_t count = 1u + (p - MYDS_SNAP_MIN) / MYDS_SNAP_STRIDE;
    return count <= MYDS_SNAP_PINNED ? count : 0;
}

static uint32_t lowkv_segment_end(uint32_t group) {
    return MYDS_SNAP_MIN + group * MYDS_SNAP_STRIDE;
}

static uint32_t lowkv_segment_start(uint32_t group) {
    return group ? lowkv_segment_end(group - 1u) : 0u;
}

static uint64_t lowkv_payload_bytes(uint32_t p) {
    return (uint64_t)V41_LOWKV_HEADER + sizeof(myds_engram_history_t)
        + (uint64_t)V41_LAYERS * V41_WINDOW * V41_HEAD_DIM * 4u
        + (uint64_t)lowkv_segment_count(p) * V41_LOWKV_REF_BYTES;
}

static uint64_t lowkv_segment_bytes(uint32_t start, uint32_t end) {
    uint64_t bytes = V41_LOWKV_HEADER;
    for (uint32_t o = 0; o < 4u; ++o) {
        const uint32_t ratio = lowkv_owner_ratio(o);
        if (!ratio || start % ratio || end % ratio || end <= start) return 0;
        bytes += (uint64_t)((end - start) / ratio) *
            (V41_LOWKV_LATENT + V41_INDEX_DIM) * V41_CACHE_BYTES;
    }
    return bytes;
}

static uint32_t lowkv_base_segment_count(uint32_t p) {
    if (p < MYDS_SNAP_MIN) return 0;
    uint32_t count = (p - MYDS_SNAP_MIN) / MYDS_SNAP_STRIDE + 1u;
    return (count > MYDS_SNAP_PINNED) ? MYDS_SNAP_PINNED : count;
}

static uint32_t lowkv_base_segment_end(uint32_t base_count) {
    return base_count ? lowkv_segment_end(base_count - 1u) : 0u;
}

static uint64_t lowkv_end_tail_bytes(uint32_t base_end, uint32_t p) {
    if (p <= base_end) return 0;
    uint64_t bytes = 0;
    for (uint32_t o = 0; o < 4u; ++o) {
        const uint32_t ratio = lowkv_owner_ratio(o);
        if (!ratio) return 0;
        const uint32_t start_row = base_end / ratio;
        const uint32_t end_row = p / ratio;
        if (end_row > start_row) {
            bytes += (uint64_t)(end_row - start_row) *
                (V41_LOWKV_LATENT + V41_INDEX_DIM) * V41_CACHE_BYTES;
        }
    }
    return bytes;
}

static uint64_t lowkv_end_payload_bytes(uint32_t p, uint32_t base_count, uint32_t base_end) {
    return (uint64_t)V41_LOWKV_HEADER
        + sizeof(myds_engram_history_t)
        + 16384u /* 4*2048 prev_kv + 4*2048 prev_score */
        + (uint64_t)V41_LAYERS * V41_WINDOW * V41_HEAD_DIM * 4u
        + (uint64_t)base_count * V41_LOWKV_REF_BYTES
        + lowkv_end_tail_bytes(base_end, p);
}

static void lk_put_u64(unsigned char *p, uint64_t v) {
    lk_put_u32(p, (uint32_t)v); lk_put_u32(p + 4, (uint32_t)(v >> 32));
}
static uint64_t lk_get_u64(const unsigned char *p) {
    return (uint64_t)lk_get_u32(p) | ((uint64_t)lk_get_u32(p + 4) << 32);
}

static inline void digest_update(uint64_t *d, const void *data, size_t len) {
    if (!d || !data || len == 0) return;
    const uint8_t *p = (const uint8_t *)data;
    const size_t u64_count = len / 8;
    for (size_t k = 0; k < u64_count; k++) {
        uint64_t val;
        memcpy(&val, p + k * 8, 8);
        *d ^= val;
        *d *= 1099511628211ull;
    }
    for (size_t k = u64_count * 8; k < len; k++) {
        *d ^= (uint64_t)p[k];
        *d *= 1099511628211ull;
    }
}

static inline void digest_update_u64(uint64_t *d, uint64_t val) {
    if (!d) return;
    *d ^= val;
    *d *= 1099511628211ull;
}

static inline void digest_update_str(uint64_t *d, const char *s) {
    if (!d || !s) return;
    digest_update(d, s, strlen(s));
}

static void digest_update_metadata_kv(uint64_t *d, const gguf_kv_t *kv) {
    if (!d || !kv) return;
    if (kv->key) digest_update_str(d, kv->key);
    digest_update_u64(d, (uint64_t)kv->type);
    switch (kv->type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            digest_update(d, &kv->value.val_u8, 1);
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            digest_update(d, &kv->value.val_u16, 2);
            break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            digest_update(d, &kv->value.val_u32, 4);
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            digest_update(d, &kv->value.val_u64, 8);
            break;
        case GGUF_TYPE_STRING:
            if (kv->value.val_str) digest_update_str(d, kv->value.val_str);
            break;
        case GGUF_TYPE_ARRAY: {
            digest_update_u64(d, (uint64_t)kv->value.val_arr.elem_type);
            digest_update_u64(d, kv->value.val_arr.count);
            if (kv->value.val_arr.elem_type == GGUF_TYPE_STRING) {
                char **strs = (char **)kv->value.val_arr.data;
                if (strs) {
                    for (uint64_t j = 0; j < kv->value.val_arr.count; j++) {
                        if (strs[j]) digest_update_str(d, strs[j]);
                    }
                }
            } else if (kv->value.val_arr.data) {
                size_t elem_sz = 1;
                switch (kv->value.val_arr.elem_type) {
                    case GGUF_TYPE_UINT8:
                    case GGUF_TYPE_INT8:
                    case GGUF_TYPE_BOOL: elem_sz = 1; break;
                    case GGUF_TYPE_UINT16:
                    case GGUF_TYPE_INT16: elem_sz = 2; break;
                    case GGUF_TYPE_UINT32:
                    case GGUF_TYPE_INT32:
                    case GGUF_TYPE_FLOAT32: elem_sz = 4; break;
                    case GGUF_TYPE_UINT64:
                    case GGUF_TYPE_INT64:
                    case GGUF_TYPE_FLOAT64: elem_sz = 8; break;
                    default: elem_sz = 0; break;
                }
                if (elem_sz > 0 && kv->value.val_arr.count <= (SIZE_MAX / elem_sz)) {
                    digest_update(d, kv->value.val_arr.data, kv->value.val_arr.count * elem_sz);
                }
            }
            break;
        }
        default:
            break;
    }
}

static void digest_update_tensor_descriptor(uint64_t *d, const gguf_tensor_t *t) {
    if (!d || !t) return;
    const size_t name_len = strlen(t->name);
    digest_update_u64(d, (uint64_t)name_len);
    digest_update(d, t->name, name_len);
    digest_update_u64(d, (uint64_t)t->type);
    digest_update_u64(d, (uint64_t)t->n_dims);
    for (uint32_t k = 0; k < t->n_dims && k < GGUF_MAX_DIMS; k++) {
        digest_update_u64(d, t->dims[k]);
    }
    digest_update_u64(d, t->offset);
    digest_update_u64(d, (uint64_t)t->size_bytes);
}

static bool lowkv_read_companion_digest(const char *model_path, char *buf, size_t buf_sz) {
    if (!model_path || !model_path[0] || !buf || buf_sz < 32) return false;
    char path[1200];
    FILE *fp = NULL;

    /* Candidate 1: <model_path>.digest */
    snprintf(path, sizeof(path), "%s.digest", model_path);
    fp = fopen(path, "r");
    /* Candidate 2: <model_path>.sha256 */
    if (!fp) {
        snprintf(path, sizeof(path), "%s.sha256", model_path);
        fp = fopen(path, "r");
    }
    /* Candidate 3 & 4: replace .gguf with .digest / .sha256 */
    if (!fp) {
        size_t len = strlen(model_path);
        if (len > 5 && strcmp(model_path + len - 5, ".gguf") == 0) {
            char base[1100];
            if (len - 5 < sizeof(base)) {
                memcpy(base, model_path, len - 5);
                base[len - 5] = '\0';
                snprintf(path, sizeof(path), "%s.digest", base);
                fp = fopen(path, "r");
                if (!fp) {
                    snprintf(path, sizeof(path), "%s.sha256", base);
                    fp = fopen(path, "r");
                }
            }
        }
    }
    if (!fp) return false;

    char line[512];
    bool found = false;
    while (fgets(line, sizeof(line), fp)) {
        char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '#' || *p == '\0') continue;
        if (strncmp(p, "SHA256", 6) == 0) {
            char *eq = strchr(p, '=');
            if (eq) p = eq + 1;
            while (*p == ' ' || *p == '\t') p++;
        }
        char *start = p;
        while ((*p >= '0' && *p <= '9') ||
               (*p >= 'a' && *p <= 'f') ||
               (*p >= 'A' && *p <= 'F')) {
            p++;
        }
        size_t hlen = (size_t)(p - start);
        if (hlen >= 32 && hlen < buf_sz) {
            memcpy(buf, start, hlen);
            buf[hlen] = '\0';
            found = true;
            break;
        }
    }
    fclose(fp);
    return found;
}

/* 身份戳：决定这些字节含义的【全部】参数。任何一项变了，旧文件都必须作废 ——
 * 宁可重新付一次 prefill，也不能把上一版模型的 KV 接上去。 */
static uint64_t lowkv_identity(const myds_model_t *m) {
    if (!m || !m->gf) return 0;

    uint64_t v[32];
    uint32_t n = 0;
    v[n++] = V41_LOWKV_ABI;
    v[n++] = V41_LOWKV_SEG_ABI;
    v[n++] = V41_LOWKV_END_ABI;
    v[n++] = V41_LOWKV_REF_BYTES;
    v[n++] = V41_CACHE_BYTES;
    v[n++] = V41_LAYERS;
    v[n++] = V41_DIM;
    v[n++] = V41_HEAD_DIM;
    v[n++] = V41_WINDOW;
    v[n++] = V41_SNAP_STRIDE;
    v[n++] = MYDS_SNAP_SLOTS;
    v[n++] = MYDS_SNAP_PINNED;
    v[n++] = MYDS_SNAP_MIN;
    v[n++] = (uint64_t)V41_LOWKV_LATENT;
    v[n++] = (uint64_t)V41_INDEX_DIM;
    for (uint32_t o = 0; o < 4u; ++o) v[n++] = lowkv_owner_ratio(o);
    v[n++] = (uint64_t)m->gf->tensor_count;
    v[n++] = (uint64_t)m->gf->metadata_kv_count;
    v[n++] = (uint64_t)m->gf->tensor_data_offset;
    v[n++] = (uint64_t)m->gf->mmap_size;   /* 模型文件大小：换模型立刻作废 */

    /* 权重内容指纹：区分同结构、同大小但权重不同的 checkpoint */
    uint64_t weight_digest = 1469598103934665603ull;
    const char *manifest = NULL;
    char companion_digest[128] = {0};
    manifest = gguf_get_str(m->gf, "models.manifest.digest", NULL);
    if (!manifest) manifest = gguf_get_str(m->gf, "general.sha256", NULL);
    if (!manifest) manifest = gguf_get_str(m->gf, "general.source.sha256", NULL);
    if (!manifest) manifest = gguf_get_str(m->gf, "models.digest", NULL);
    if (!manifest) manifest = gguf_get_str(m->gf, "model.digest", NULL);
    if (!manifest) manifest = gguf_get_str(m->gf, "general.digest", NULL);
    if (!manifest) manifest = gguf_get_str(m->gf, "general.file_sha256", NULL);
    if (!manifest) manifest = gguf_get_str(m->gf, "models.file_digest", NULL);
    if (!manifest) manifest = getenv("MYDS_MODEL_DIGEST");
    if (!manifest && m->model_path[0]) {
        if (lowkv_read_companion_digest(m->model_path, companion_digest, sizeof(companion_digest))) {
            manifest = companion_digest;
            MYDS_LOG_I("kvdisk: 发现模型伴侣指纹文件: %s", companion_digest);
        }
    }

    /* ① 全量哈希所有影响查表与计算语义的 GGUF 元数据（如 engram.token_map, primes, multipliers 等） */
    if (m->gf->kvs) {
        for (uint64_t i = 0; i < m->gf->metadata_kv_count; i++) {
            digest_update_metadata_kv(&weight_digest, &m->gf->kvs[i]);
        }
    }

    /* ② 全量哈希所有张量描述符（名称及长度、类型、维度、偏移、大小），将张量元信息牢固绑定到模型身份 */
    if (m->gf->tensors) {
        for (uint64_t i = 0; i < m->gf->tensor_count; i++) {
            digest_update_tensor_descriptor(&weight_digest, &m->gf->tensors[i]);
        }
    }

    if (manifest && *manifest) {
        for (const unsigned char *p = (const unsigned char *)manifest; *p; p++) {
            weight_digest ^= (uint64_t)*p;
            weight_digest *= 1099511628211ull;
        }
    } else {

        uint64_t total_tensor_bytes = 0;
        if (m->gf->tensors) {
            for (uint64_t i = 0; i < m->gf->tensor_count; i++) {
                total_tensor_bytes += m->gf->tensors[i].size_bytes;
            }
        }
        const bool force_full = (getenv("MYDS_FORCE_FULL_DIGEST") != NULL);
        const bool is_small_model = (total_tensor_bytes <= 256ull * 1024ull * 1024ull);

        if (is_small_model || force_full) {
            bool digest_error = false;
            if (m->gf->tensors) {
                for (uint64_t i = 0; i < m->gf->tensor_count; i++) {
                    const gguf_tensor_t *t = &m->gf->tensors[i];
                    /* 显式将该张量描述符与该张量的数据强绑定 */
                    digest_update_tensor_descriptor(&weight_digest, t);
                    if (t->size_bytes == 0) continue;
                    if (t->data) {
                        digest_update(&weight_digest, t->data, t->size_bytes);
                    } else if (m->gf->fd >= 0) {
                        /* 解除映射后的张量（例如 Engram unmap 后的表）：通过文件描述符流式读取校验 */
                        uint8_t chunk[65536];
                        uint64_t file_offset = m->gf->tensor_data_offset + t->offset;
                        size_t remaining = t->size_bytes;
                        while (remaining > 0) {
                            size_t to_read = remaining < sizeof(chunk) ? remaining : sizeof(chunk);
                            ssize_t nread = pread(m->gf->fd, chunk, to_read, (off_t)file_offset);
                            if (nread <= 0 || (size_t)nread != to_read) {
                                digest_error = true;
                                break;
                            }
                            digest_update(&weight_digest, chunk, (size_t)nread);
                            file_offset += (uint64_t)nread;
                            remaining -= (size_t)nread;
                        }
                        if (digest_error) break;
                    } else {
                        digest_error = true;
                        break;
                    }
                }
            }
            if (digest_error) {
                MYDS_LOG_W("kvdisk: 模型权重或文件读取失败，无法计算完整指纹");
                return 0;
            }
            if (m->tok_embeddings && myds_metal_buffer_data(m->tok_embeddings)) {
                digest_update_str(&weight_digest, "tok_embeddings");
                digest_update_u64(&weight_digest, (uint64_t)m->tok_embeddings_type);
                digest_update(&weight_digest, myds_metal_buffer_data(m->tok_embeddings),
                              myds_metal_buffer_size(m->tok_embeddings));
            }
            if (m->output && myds_metal_buffer_data(m->output)) {
                digest_update_str(&weight_digest, "output");
                digest_update_u64(&weight_digest, (uint64_t)m->output_type);
                digest_update(&weight_digest, myds_metal_buffer_data(m->output),
                              myds_metal_buffer_size(m->output));
            }
        } else {
            /* 模型 > 256MB 且缺少 manifest digest：
             * 使用该 model 实例专有的 load_generation（同 model 实例各 context 共享；
             * 同进程不同加载实例或独立 model 结构不同，跨进程 pid 亦不同），彻底杜绝误复用。 */
            uint64_t gen = m->load_generation ? m->load_generation : (((uint64_t)(uintptr_t)m << 13) ^ 0x517cc1b727220a95ull);
            if (!gen) {
                static atomic_uint_fast64_t s_boot_counter = 0;
                gen = ((uint64_t)getpid() << 32) ^ (uint64_t)time(NULL) ^ (uint64_t)atomic_fetch_add(&s_boot_counter, 1);
                if (!gen) gen = 1;
            }
            weight_digest ^= gen;
            weight_digest *= 1099511628211ull;

            /* 保留多点抽样作为辅助检查（覆盖内存映射张量与已 unmap 的 fd 张量） */
            if (m->gf->tensors && m->gf->tensor_count > 0) {
                const uint64_t tc = m->gf->tensor_count;
                const uint64_t step = (tc > 16u) ? (tc / 16u) : 1u;
                for (uint64_t i = 0; i < tc; i += step) {
                    const gguf_tensor_t *t = &m->gf->tensors[i];
                    digest_update_tensor_descriptor(&weight_digest, t);
                    if (t->size_bytes == 0) continue;
                    uint64_t w1 = 0, w2 = 0;
                    size_t s1 = t->size_bytes < 8 ? t->size_bytes : 8;
                    size_t mid = t->size_bytes / 2;
                    size_t s2 = (t->size_bytes - mid) < 8 ? (t->size_bytes - mid) : 8;
                    if (t->data) {
                        const uint8_t *b = (const uint8_t *)t->data;
                        memcpy(&w1, b, s1);
                        memcpy(&w2, b + mid, s2);
                    } else if (m->gf->fd >= 0) {
                        uint64_t file_offset = m->gf->tensor_data_offset + t->offset;
                        ssize_t r1 = pread(m->gf->fd, &w1, s1, (off_t)file_offset);
                        ssize_t r2 = pread(m->gf->fd, &w2, s2, (off_t)(file_offset + mid));
                        if (r1 <= 0 || (size_t)r1 != s1 || r2 <= 0 || (size_t)r2 != s2) {
                            MYDS_LOG_W("kvdisk: 大模型抽样权重读取失败");
                            return 0;
                        }
                    } else {
                        MYDS_LOG_W("kvdisk: 大模型存在未映射且无有效文件描述符的张量");
                        return 0;
                    }
                    weight_digest ^= w1; weight_digest *= 1099511628211ull;
                    weight_digest ^= w2; weight_digest *= 1099511628211ull;
                }
            }
        }
    }
    v[n++] = weight_digest;

    uint64_t id = myds_kvdsk_identity(m ? m->arch : "", v, n);
    return id ? id : 1ull;
}

uint64_t myds_v41_model_identity(const myds_model_t *m) {
    return lowkv_identity(m);
}

/* 为「位置 p、前缀哈希 hash」挑一个条目来写。策略本体在 myds_lowkv_policy.h
 * （纯函数，有单测）：同哈希 > 空位 > 淘汰命中最少的。
 * 这里只负责把磁盘上的 way 状态读出来喂给它、再把 way 换算回条目编号。 */
static uint32_t lowkv_pick_entry(myds_kvdsk_t *store, myds_v41_context_t *g, uint32_t group,
                                uint32_t p, uint64_t hash, bool *already_here) {
    if (already_here) *already_here = false;
    if (g->low_ways > V41_LOWKV_WAYS_MAX) {
        if (already_here) *already_here = false;
        return V41_LOWKV_NO_ENTRY;
    }
    myds_lowkv_way_t ways[V41_LOWKV_WAYS_MAX];
    for (uint32_t w = 0; w < g->low_ways; ++w) {
        const uint32_t e = lowkv_entry_index(g, group, w);
        uint32_t ep = 0; uint64_t eh = 0, eu = 0;
        ways[w].occupied = myds_kvdsk_slot_meta(store, e, &ep, &eh, &eu);
        ways[w].position = ep;
        ways[w].hash = eh;
        ways[w].hits = 0;
        if (ways[w].occupied) (void)myds_kvdsk_slot_hits(store, e, &ways[w].hits);
    }
    const uint32_t way = myds_lowkv_pick_way(ways, g->low_ways, p, hash, already_here);
    if (way == MYDS_LOWKV_NO_WAY) return V41_LOWKV_NO_ENTRY;
    return lowkv_entry_index(g, group, way);
}

/* 把 Metal buffer 的一段搬到磁盘（分块，避免一次性申请几十 MB）。 */
static bool lowkv_emit(myds_kvdsk_t *k, const myds_metal_buffer_t *buf, uint64_t bytes,
                       unsigned char *stage, uint64_t *file_off, uint64_t src) {
    while (bytes) {
        const size_t n = (bytes < (uint64_t)V41_LOWKV_STAGE) ? (size_t)bytes : (size_t)V41_LOWKV_STAGE;
        if (!myds_metal_buffer_read(buf, (size_t)src, stage, n)) return false;
        if (!myds_kvdsk_write_chunk(k, *file_off, stage, n)) return false;
        src += n; *file_off += n; bytes -= n;
    }
    return true;
}

typedef struct {
    uint32_t slot, start, end;
    uint64_t hash, bytes;
} lowkv_segment_ref_t;

static void lowkv_segment_header(unsigned char hdr[V41_LOWKV_HEADER], uint32_t start, uint32_t end) {
    memset(hdr, 0, V41_LOWKV_HEADER);
    lk_put_u32(hdr, V41_LOWKV_SEG_ABI);
    lk_put_u32(hdr + 4, start); lk_put_u32(hdr + 8, end);
    lk_put_u32(hdr + 12, 4u); lk_put_u32(hdr + 16, V41_LOWKV_LATENT);
    lk_put_u32(hdr + 20, V41_INDEX_DIM); lk_put_u32(hdr + 24, V41_CACHE_BYTES);
    for (uint32_t o = 0; o < 4u; ++o) {
        const uint32_t ratio = lowkv_owner_ratio(o);
        lk_put_u32(hdr + 32 + o * 4u, start / ratio);
        lk_put_u32(hdr + 48 + o * 4u, (end - start) / ratio);
    }
}

/* Save-time validation before reusing an already committed segment. Restore
 * instead uses private checked staging, so it cannot partially overwrite KV. */
static bool lowkv_read_segment(myds_v41_context_t *g, const lowkv_segment_ref_t *ref,
    unsigned char *stage) {
    uint32_t pos = 0; uint64_t hash = 0, used = 0;
    if (!myds_kvdsk_slot_meta(g->low_segments, ref->slot, &pos, &hash, &used) ||
        pos != ref->end || hash != ref->hash || used != ref->bytes) return false;
    if (!myds_kvdsk_read_start(g->low_segments, ref->slot, &used)) return false;
    unsigned char hdr[V41_LOWKV_HEADER], expected[V41_LOWKV_HEADER];
    lowkv_segment_header(expected, ref->start, ref->end);
    if (!myds_kvdsk_read_chunk(g->low_segments, 0, hdr, sizeof(hdr)) ||
        memcmp(hdr, expected, sizeof(hdr))) goto bad;
    uint64_t off = V41_LOWKV_HEADER;
    for (uint32_t kind = 0; kind < 2u; ++kind) {
        const uint32_t width = kind ? V41_INDEX_DIM : V41_LOWKV_LATENT;
        for (uint32_t o = 0; o < 4u; ++o) {
            const uint32_t ratio = lowkv_owner_ratio(o);
            const uint64_t bytes = (uint64_t)((ref->end - ref->start) / ratio) * width * V41_CACHE_BYTES;
            uint64_t left = bytes;
            while (left) {
                const uint64_t n = left < V41_LOWKV_STAGE ? left : V41_LOWKV_STAGE;
                if (!myds_kvdsk_read_chunk(g->low_segments, off, stage, n)) goto bad;
                off += n; left -= n;
            }
        }
    }
    if (off != used) goto bad;
    return myds_kvdsk_read_verify(g->low_segments);
bad:
    myds_kvdsk_read_abort(g->low_segments);
    (void)myds_kvdsk_slot_clear(g->low_segments, ref->slot);
    return false;
}

static bool lowkv_ensure_segment(myds_v41_context_t *g, uint32_t group, uint64_t hash,
    unsigned char *stage, lowkv_segment_ref_t *ref) {
    ref->start = lowkv_segment_start(group); ref->end = lowkv_segment_end(group);
    ref->hash = hash; ref->bytes = lowkv_segment_bytes(ref->start, ref->end);
    if (!ref->bytes) return false;
    bool already = false;
    ref->slot = lowkv_pick_entry(g->low_segments, g, group, ref->end, hash, &already);
    if (ref->slot == V41_LOWKV_NO_ENTRY) return false;
    if (already && lowkv_read_segment(g, ref, stage)) return true;
    unsigned char hdr[V41_LOWKV_HEADER];
    lowkv_segment_header(hdr, ref->start, ref->end);
    if (!myds_kvdsk_write_start(g->low_segments, ref->slot) ||
        !myds_kvdsk_write_chunk(g->low_segments, 0, hdr, sizeof(hdr))) goto bad;
    if (g->low_fail_segment_end == ref->end) goto bad;
    uint64_t off = V41_LOWKV_HEADER;
    for (uint32_t kind = 0; kind < 2u; ++kind) {
        const uint32_t width = kind ? V41_INDEX_DIM : V41_LOWKV_LATENT;
        for (uint32_t o = 0; o < 4u; ++o) {
            const uint32_t ratio = lowkv_owner_ratio(o);
            const uint64_t bytes = (uint64_t)((ref->end - ref->start) / ratio) * width * V41_CACHE_BYTES;
            const uint64_t src = (uint64_t)(ref->start / ratio) * width * V41_CACHE_BYTES;
            if (!lowkv_emit(g->low_segments, kind ? g->index_cache[o] : g->compressed[o],
                bytes, stage, &off, src)) goto bad;
        }
    }
    if (off != ref->bytes || !myds_kvdsk_write_finish(g->low_segments, ref->slot,
        ref->end, ref->hash, off)) goto bad;
    ++g->low_segment_writes; g->low_segment_bytes += off;
    return true;
bad:
    myds_kvdsk_write_abort(g->low_segments);
    return false;
}

static bool lowkv_write_slot(myds_v41_context_t *g, uint32_t entry,
    const v41_lowkv_job_t *job, const lowkv_segment_ref_t *refs, unsigned char *stage) {
    const uint32_t window_bytes = V41_WINDOW * V41_HEAD_DIM * 4u;
    const uint32_t nrefs = lowkv_segment_count(job->p);
    unsigned char hdr[V41_LOWKV_HEADER] = {0};
    lk_put_u32(hdr, V41_LOWKV_ABI); lk_put_u32(hdr + 4, job->p);
    lk_put_u32(hdr + 8, V41_LAYERS); lk_put_u32(hdr + 12, window_bytes);
    lk_put_u32(hdr + 16, sizeof(myds_engram_history_t)); lk_put_u32(hdr + 20, nrefs);
    lk_put_u32(hdr + 40, V41_LOWKV_LATENT); lk_put_u32(hdr + 44, V41_INDEX_DIM);
    lk_put_u32(hdr + 48, V41_CACHE_BYTES);
    if (!myds_kvdsk_write_start(g->low_store, entry) ||
        !myds_kvdsk_write_chunk(g->low_store, 0, hdr, sizeof(hdr))) goto bad;
    uint64_t off = V41_LOWKV_HEADER;
    if (!myds_kvdsk_write_chunk(g->low_store, off, &g->snap_engram[job->s], sizeof(myds_engram_history_t))) goto bad;
    off += sizeof(myds_engram_history_t);
    if (!lowkv_emit(g->low_store, g->snap_window[job->s],
        (uint64_t)V41_LAYERS * window_bytes, stage, &off, 0)) goto bad;
    for (uint32_t i = 0; i < nrefs; ++i) {
        unsigned char bytes[V41_LOWKV_REF_BYTES] = {0};
        lk_put_u32(bytes, refs[i].slot); lk_put_u32(bytes + 4, refs[i].start);
        lk_put_u32(bytes + 8, refs[i].end); lk_put_u64(bytes + 16, refs[i].hash);
        lk_put_u64(bytes + 24, refs[i].bytes);
        if (!myds_kvdsk_write_chunk(g->low_store, off, bytes, sizeof(bytes))) goto bad;
        off += sizeof(bytes);
    }
    if (off != lowkv_payload_bytes(job->p) ||
        !myds_kvdsk_write_finish(g->low_store, entry, job->p, job->hash, off)) goto bad;
    g->low_snapshot_bytes += off;
    return true;
bad:
    myds_kvdsk_write_abort(g->low_store);
    return false;
}

/* Descriptor equality needs only its small refs range. This peek is not a
 * restore: readers must still validate the entire payload and every segment. */
static bool lowkv_snapshot_refs_match(myds_v41_context_t *g, uint32_t entry,
    const v41_lowkv_job_t *job, const lowkv_segment_ref_t *refs) {
    unsigned char expected[MYDS_SNAP_PINNED * V41_LOWKV_REF_BYTES] = {0};
    unsigned char actual[sizeof(expected)];
    const uint32_t count = lowkv_segment_count(job->p);
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char *bytes = expected + i * V41_LOWKV_REF_BYTES;
        lk_put_u32(bytes, refs[i].slot); lk_put_u32(bytes + 4, refs[i].start);
        lk_put_u32(bytes + 8, refs[i].end); lk_put_u64(bytes + 16, refs[i].hash);
        lk_put_u64(bytes + 24, refs[i].bytes);
    }
    const uint64_t off = V41_LOWKV_HEADER + sizeof(myds_engram_history_t) +
        (uint64_t)V41_LAYERS * V41_WINDOW * V41_HEAD_DIM * 4u;
    return myds_kvdsk_peek_committed(g->low_store, entry, off, actual,
        (uint64_t)count * V41_LOWKV_REF_BYTES) &&
        memcmp(expected, actual, (size_t)count * V41_LOWKV_REF_BYTES) == 0;
}

static uint32_t lowkv_pick_end_entry(myds_v41_context_t *g, uint32_t p, uint64_t hash,
                                     uint64_t total_bytes, bool *already_here) {
    if (already_here) *already_here = false;
    if (!g || !g->low_ends || total_bytes > V41_LOWKV_END_STRIDE) return V41_LOWKV_NO_ENTRY;

    uint32_t empty_slot = V41_LOWKV_NO_ENTRY;
    uint32_t victim_slot = 0;
    uint32_t min_hits = UINT32_MAX;

    for (uint32_t s = 0; s < V41_LOWKV_ENDS_MAX; ++s) {
        uint32_t pos = 0; uint64_t h = 0, used = 0;
        if (myds_kvdsk_slot_meta(g->low_ends, s, &pos, &h, &used)) {
            if (pos == p && h == hash && used == total_bytes) {
                if (already_here) *already_here = true;
                return s;
            }
            uint32_t hits = 0;
            (void)myds_kvdsk_slot_hits(g->low_ends, s, &hits);
            if (hits < min_hits) {
                min_hits = hits;
                victim_slot = s;
            }
        } else {
            if (empty_slot == V41_LOWKV_NO_ENTRY) empty_slot = s;
        }
    }
    return (empty_slot != V41_LOWKV_NO_ENTRY) ? empty_slot : victim_slot;
}

static bool lowkv_end_refs_match(myds_v41_context_t *g, uint32_t entry,
                                 uint32_t base_count, const lowkv_segment_ref_t *refs) {
    if (!g || !g->low_ends || !refs) return false;
    if (base_count == 0) return true;
    if (base_count > MYDS_SNAP_PINNED) return false;

    unsigned char expected[MYDS_SNAP_PINNED * V41_LOWKV_REF_BYTES] = {0};
    unsigned char actual[sizeof(expected)];
    for (uint32_t i = 0; i < base_count; ++i) {
        unsigned char *bytes = expected + i * V41_LOWKV_REF_BYTES;
        lk_put_u32(bytes, refs[i].slot);
        lk_put_u32(bytes + 4, refs[i].start);
        lk_put_u32(bytes + 8, refs[i].end);
        lk_put_u64(bytes + 16, refs[i].hash);
        lk_put_u64(bytes + 24, refs[i].bytes);
    }
    const uint64_t off = V41_LOWKV_HEADER + sizeof(myds_engram_history_t) + 16384u +
        (uint64_t)V41_LAYERS * V41_WINDOW * V41_HEAD_DIM * 4u;
    return myds_kvdsk_peek_committed(g->low_ends, entry, off, actual,
        (uint64_t)base_count * V41_LOWKV_REF_BYTES) &&
        memcmp(expected, actual, (size_t)base_count * V41_LOWKV_REF_BYTES) == 0;
}

static bool lowkv_write_end_slot(myds_v41_context_t *g, uint32_t entry,
    uint32_t p, uint64_t hash, uint32_t base_count, uint32_t base_end,
    const lowkv_segment_ref_t *refs, unsigned char *stage) {
    const uint32_t window_bytes = V41_WINDOW * V41_HEAD_DIM * 4u;
    const uint64_t total_bytes = lowkv_end_payload_bytes(p, base_count, base_end);

    unsigned char hdr[V41_LOWKV_HEADER] = {0};
    lk_put_u32(hdr, V41_LOWKV_END_ABI);
    lk_put_u32(hdr + 4, p);
    lk_put_u32(hdr + 8, V41_LAYERS);
    lk_put_u32(hdr + 12, window_bytes);
    lk_put_u32(hdr + 16, sizeof(myds_engram_history_t));
    lk_put_u32(hdr + 20, base_count);
    lk_put_u32(hdr + 24, base_end);
    lk_put_u32(hdr + 28, p - base_end);
    lk_put_u32(hdr + 40, V41_LOWKV_LATENT);
    lk_put_u32(hdr + 44, V41_INDEX_DIM);
    lk_put_u32(hdr + 48, V41_CACHE_BYTES);

    if (!myds_kvdsk_write_start(g->low_ends, entry) ||
        !myds_kvdsk_write_chunk(g->low_ends, 0, hdr, sizeof(hdr))) goto bad;
    uint64_t off = V41_LOWKV_HEADER;

    /* 1. Engram history (12 bytes) */
    if (!myds_kvdsk_write_chunk(g->low_ends, off, &g->snap_engram[0], sizeof(myds_engram_history_t))) goto bad;
    off += sizeof(myds_engram_history_t);

    /* 2. Prev KV (4 * 2048) & Score (4 * 2048) = 16384 bytes */
    for (uint32_t o = 0; o < 4u; ++o) {
        if (!lowkv_emit(g->low_ends, g->snap_prev_kv[0][o], 512u * 4u, stage, &off, 0)) goto bad;
    }
    for (uint32_t o = 0; o < 4u; ++o) {
        if (!lowkv_emit(g->low_ends, g->snap_prev_score[0][o], 512u * 4u, stage, &off, 0)) goto bad;
    }

    /* 3. Window (40 layers * window_bytes) */
    if (!lowkv_emit(g->low_ends, g->snap_window[0],
        (uint64_t)V41_LAYERS * window_bytes, stage, &off, 0)) goto bad;

    /* 4. Base segment refs */
    for (uint32_t i = 0; i < base_count; ++i) {
        unsigned char bytes[V41_LOWKV_REF_BYTES] = {0};
        lk_put_u32(bytes, refs[i].slot);
        lk_put_u32(bytes + 4, refs[i].start);
        lk_put_u32(bytes + 8, refs[i].end);
        lk_put_u64(bytes + 16, refs[i].hash);
        lk_put_u64(bytes + 24, refs[i].bytes);
        if (!myds_kvdsk_write_chunk(g->low_ends, off, bytes, sizeof(bytes))) goto bad;
        off += sizeof(bytes);
    }

    /* 5. Tail packed rows [base_end, p) */
    for (uint32_t kind = 0; kind < 2u; ++kind) {
        const uint32_t width = kind ? V41_INDEX_DIM : V41_LOWKV_LATENT;
        for (uint32_t o = 0; o < 4u; ++o) {
            const uint32_t ratio = lowkv_owner_ratio(o);
            const uint32_t start_row = base_end / ratio;
            const uint32_t end_row = p / ratio;
            if (end_row > start_row) {
                const uint64_t row_bytes = (uint64_t)(end_row - start_row) * width * V41_CACHE_BYTES;
                const uint64_t src = (uint64_t)start_row * width * V41_CACHE_BYTES;
                if (!lowkv_emit(g->low_ends, kind ? g->index_cache[o] : g->compressed[o],
                    row_bytes, stage, &off, src)) goto bad;
            }
        }
    }

    if (off != total_bytes ||
        !myds_kvdsk_write_finish(g->low_ends, entry, p, hash, off)) goto bad;
    return true;
bad:
    myds_kvdsk_write_abort(g->low_ends);
    return false;
}

/* Read once into private memory, then validate those very bytes. A separate
 * verification pass followed by a second disk read would admit a changed
 * payload between verification and publication. No live Metal buffers or
 * RAM snapshot contents are touched by this operation. */
static bool lowkv_read_stage_range(myds_kvdsk_t *store, unsigned char *dst,
    uint64_t *offset, uint64_t bytes) {
    while (bytes) {
        const uint64_t n = bytes < V41_LOWKV_STAGE ? bytes : V41_LOWKV_STAGE;
        if (!myds_kvdsk_read_chunk(store, *offset, dst + (size_t)*offset, n)) return false;
        *offset += n;
        bytes -= n;
    }
    return true;
}

static bool lowkv_read_checked_end(myds_v41_context_t *g, uint32_t entry, uint32_t position,
    uint64_t hash, uint64_t bytes, unsigned char *dst, uint32_t *out_base_count, uint32_t *out_base_end) {
    uint32_t actual_pos = 0;
    uint64_t actual_hash = 0, used = 0;
    if (!g || !g->low_ends || !dst || !myds_kvdsk_read_start(g->low_ends, entry, &used)) return false;
    if (used != bytes || !myds_kvdsk_slot_meta(g->low_ends, entry, &actual_pos, &actual_hash, NULL) ||
        actual_pos != position || actual_hash != hash) goto bad;

    uint64_t off = 0;
    if (!lowkv_read_stage_range(g->low_ends, dst, &off, V41_LOWKV_HEADER)) goto bad;

    const unsigned char *hdr = dst;
    const uint32_t window_bytes = V41_WINDOW * V41_HEAD_DIM * 4u;
    if (lk_get_u32(hdr) != V41_LOWKV_END_ABI || lk_get_u32(hdr + 4) != position ||
        lk_get_u32(hdr + 8) != V41_LAYERS || lk_get_u32(hdr + 12) != window_bytes ||
        lk_get_u32(hdr + 16) != sizeof(myds_engram_history_t) ||
        lk_get_u32(hdr + 40) != V41_LOWKV_LATENT || lk_get_u32(hdr + 44) != V41_INDEX_DIM ||
        lk_get_u32(hdr + 48) != V41_CACHE_BYTES) goto bad;

    const uint32_t base_count = lk_get_u32(hdr + 20);
    const uint32_t base_end = lk_get_u32(hdr + 24);
    if (base_count > MYDS_SNAP_PINNED || base_end > position) goto bad;
    if (out_base_count) *out_base_count = base_count;
    if (out_base_end) *out_base_end = base_end;

    /* 1. Engram history */
    if (!lowkv_read_stage_range(g->low_ends, dst, &off, sizeof(myds_engram_history_t))) goto bad;

    /* 2. Prev KV & Score (4 * 2048 each) */
    for (uint32_t o = 0; o < 4u; ++o) {
        if (!lowkv_read_stage_range(g->low_ends, dst, &off, 512u * 4u)) goto bad;
    }
    for (uint32_t o = 0; o < 4u; ++o) {
        if (!lowkv_read_stage_range(g->low_ends, dst, &off, 512u * 4u)) goto bad;
    }

    /* 3. Window */
    if (!lowkv_read_stage_range(g->low_ends, dst, &off, (uint64_t)V41_LAYERS * window_bytes)) goto bad;

    /* 4. Base segment refs */
    for (uint32_t i = 0; i < base_count; ++i) {
        if (!lowkv_read_stage_range(g->low_ends, dst, &off, V41_LOWKV_REF_BYTES)) goto bad;
    }

    /* 5. Tail packed rows */
    for (uint32_t kind = 0; kind < 2u; ++kind) {
        const uint32_t width = kind ? V41_INDEX_DIM : V41_LOWKV_LATENT;
        for (uint32_t o = 0; o < 4u; ++o) {
            const uint32_t ratio = lowkv_owner_ratio(o);
            const uint32_t start_row = base_end / ratio;
            const uint32_t end_row = position / ratio;
            if (end_row > start_row) {
                const uint64_t row_bytes = (uint64_t)(end_row - start_row) * width * V41_CACHE_BYTES;
                if (!lowkv_read_stage_range(g->low_ends, dst, &off, row_bytes)) goto bad;
            }
        }
    }

    if (off != bytes) goto bad;
    return myds_kvdsk_read_verify(g->low_ends);
bad:
    myds_kvdsk_read_abort(g->low_ends);
    (void)myds_kvdsk_slot_clear(g->low_ends, entry);
    return false;
}

static bool lowkv_read_checked(myds_kvdsk_t *store, uint32_t entry, uint32_t position,
    uint64_t hash, uint64_t bytes, unsigned char *dst, bool snapshot, uint32_t start) {
    uint32_t actual_pos = 0;
    uint64_t actual_hash = 0, used = 0;
    if (!dst || !myds_kvdsk_read_start(store, entry, &used)) return false;
    if (used != bytes || !myds_kvdsk_slot_meta(store, entry, &actual_pos, &actual_hash, NULL) ||
        actual_pos != position || actual_hash != hash) goto bad;
    /* The existing word-wise checksum has call-boundary semantics whenever
     * a write ends with a partial 8-byte word. Preserve the writer's exact
     * ranges, especially the 12-byte Engram history, without changing the
     * on-disk ABI or rereading any bytes after verification. */
    uint64_t off = 0;
    if (!lowkv_read_stage_range(store, dst, &off, V41_LOWKV_HEADER)) goto bad;
    if (snapshot) {
        if (!lowkv_read_stage_range(store, dst, &off, sizeof(myds_engram_history_t)) ||
            !lowkv_read_stage_range(store, dst, &off,
                (uint64_t)V41_LAYERS * V41_WINDOW * V41_HEAD_DIM * 4u)) goto bad;
        const uint32_t count = lowkv_segment_count(position);
        for (uint32_t i = 0; i < count; ++i)
            if (!lowkv_read_stage_range(store, dst, &off, V41_LOWKV_REF_BYTES)) goto bad;
    } else {
        for (uint32_t kind = 0; kind < 2u; ++kind) {
            const uint32_t width = kind ? V41_INDEX_DIM : V41_LOWKV_LATENT;
            for (uint32_t o = 0; o < 4u; ++o) {
                const uint64_t owner_bytes = (uint64_t)((position - start) / lowkv_owner_ratio(o)) *
                    width * V41_CACHE_BYTES;
                if (!lowkv_read_stage_range(store, dst, &off, owner_bytes)) goto bad;
            }
        }
    }
    if (off != bytes) goto bad;
    return myds_kvdsk_read_verify(store);
bad:
    myds_kvdsk_read_abort(store);
    (void)myds_kvdsk_slot_clear(store, entry);
    return false;
}

typedef struct {
    unsigned char *snapshot;
    unsigned char *segment[MYDS_SNAP_PINNED];
    lowkv_segment_ref_t ref[MYDS_SNAP_PINNED];
    uint32_t count;
    bool is_end;
    uint32_t base_end;
} lowkv_restore_stage_t;

static void lowkv_restore_stage_free(lowkv_restore_stage_t *st) {
    free(st->snapshot);
    for (uint32_t i = 0; i < MYDS_SNAP_PINNED; ++i) free(st->segment[i]);
}

static bool lowkv_stage_slot(myds_v41_context_t *g, uint32_t entry, uint32_t p,
    const int32_t *tokens, lowkv_restore_stage_t *st) {
    if (entry & V41_LOWKV_ENTRY_IS_END) {
        const uint32_t raw_entry = entry & ~V41_LOWKV_ENTRY_IS_END;
        if (!g->low_ends || raw_entry >= V41_LOWKV_ENDS_MAX) return false;
        uint32_t pos = 0; uint64_t hash = 0, used = 0;
        if (!myds_kvdsk_slot_meta(g->low_ends, raw_entry, &pos, &hash, &used) || pos != p) return false;
        st->is_end = true;
        st->snapshot = (unsigned char *)malloc((size_t)used);
        if (!st->snapshot) return false;
        uint32_t base_count = 0, base_end = 0;
        if (!lowkv_read_checked_end(g, raw_entry, p, myds_kvdsk_hash_tokens(tokens, p),
            used, st->snapshot, &base_count, &base_end)) return false;
        st->count = base_count;
        st->base_end = base_end;
        const size_t window_bytes = (size_t)V41_LAYERS * V41_WINDOW * V41_HEAD_DIM * 4u;
        size_t off = V41_LOWKV_HEADER + sizeof(myds_engram_history_t) + 16384u + window_bytes;
        for (uint32_t i = 0; i < st->count; ++i) {
            const unsigned char *ref = st->snapshot + off;
            off += V41_LOWKV_REF_BYTES;
            st->ref[i] = (lowkv_segment_ref_t){lk_get_u32(ref), lk_get_u32(ref + 4),
                lk_get_u32(ref + 8), lk_get_u64(ref + 16), lk_get_u64(ref + 24)};
            const lowkv_segment_ref_t *r = &st->ref[i];
            if (r->slot / g->low_ways != i || r->start != lowkv_segment_start(i) ||
                r->end != lowkv_segment_end(i) || r->hash != myds_kvdsk_hash_tokens(tokens, r->end) ||
                r->bytes != lowkv_segment_bytes(r->start, r->end) || r->bytes > SIZE_MAX) goto bad_end;
            st->segment[i] = (unsigned char *)malloc((size_t)r->bytes);
            if (!st->segment[i]) return false;
            if (!lowkv_read_checked(g->low_segments, r->slot, r->end, r->hash, r->bytes,
                st->segment[i], false, r->start)) goto bad_end;
            unsigned char expected[V41_LOWKV_HEADER];
            lowkv_segment_header(expected, r->start, r->end);
            if (memcmp(st->segment[i], expected, sizeof(expected))) {
                (void)myds_kvdsk_slot_clear(g->low_segments, r->slot);
                goto bad_end;
            }
        }
        return true;
    bad_end:
        (void)myds_kvdsk_slot_clear(g->low_ends, raw_entry);
        return false;
    }

    st->is_end = false;
    st->count = lowkv_segment_count(p);
    const uint64_t bytes = lowkv_payload_bytes(p);
    if (!st->count || st->count > MYDS_SNAP_PINNED || !bytes || bytes > SIZE_MAX) return false;
    st->snapshot = (unsigned char *)malloc((size_t)bytes);
    if (!st->snapshot) return false;
    if (!lowkv_read_checked(g->low_store, entry, p, myds_kvdsk_hash_tokens(tokens, p),
        bytes, st->snapshot, true, 0)) return false;
    const unsigned char *hdr = st->snapshot;
    if (lk_get_u32(hdr) != V41_LOWKV_ABI || lk_get_u32(hdr + 4) != p ||
        lk_get_u32(hdr + 8) != V41_LAYERS || lk_get_u32(hdr + 12) != V41_WINDOW * V41_HEAD_DIM * 4u ||
        lk_get_u32(hdr + 16) != sizeof(myds_engram_history_t) || lk_get_u32(hdr + 20) != st->count ||
        lk_get_u32(hdr + 40) != V41_LOWKV_LATENT || lk_get_u32(hdr + 44) != V41_INDEX_DIM ||
        lk_get_u32(hdr + 48) != V41_CACHE_BYTES) goto bad;
    size_t off = V41_LOWKV_HEADER + sizeof(myds_engram_history_t) +
        (size_t)V41_LAYERS * V41_WINDOW * V41_HEAD_DIM * 4u;
    for (uint32_t i = 0; i < st->count; ++i) {
        const unsigned char *ref = st->snapshot + off;
        off += V41_LOWKV_REF_BYTES;
        st->ref[i] = (lowkv_segment_ref_t){lk_get_u32(ref), lk_get_u32(ref + 4),
            lk_get_u32(ref + 8), lk_get_u64(ref + 16), lk_get_u64(ref + 24)};
        const lowkv_segment_ref_t *r = &st->ref[i];
        if (r->slot / g->low_ways != i || r->start != lowkv_segment_start(i) ||
            r->end != lowkv_segment_end(i) || r->hash != myds_kvdsk_hash_tokens(tokens, r->end) ||
            r->bytes != lowkv_segment_bytes(r->start, r->end) || r->bytes > SIZE_MAX) goto bad;
        st->segment[i] = (unsigned char *)malloc((size_t)r->bytes);
        if (!st->segment[i]) return false;  /* Allocation pressure is a miss, not disk corruption. */
        if (!lowkv_read_checked(g->low_segments, r->slot, r->end, r->hash, r->bytes,
            st->segment[i], false, r->start)) goto bad;
        unsigned char expected[V41_LOWKV_HEADER];
        lowkv_segment_header(expected, r->start, r->end);
        if (memcmp(st->segment[i], expected, sizeof(expected))) {
            (void)myds_kvdsk_slot_clear(g->low_segments, r->slot);
            goto bad;
        }
    }
    return off == bytes;
bad:
    (void)myds_kvdsk_slot_clear(g->low_store, entry);
    return false;
}

static unsigned char *lowkv_restore_destination(myds_metal_buffer_t *buf,
    uint64_t offset, uint64_t bytes) {
    const uint64_t size = myds_metal_buffer_size(buf);
    if (!buf || offset > size || bytes > size - offset || offset > SIZE_MAX) return NULL;
    unsigned char *data = (unsigned char *)myds_metal_buffer_data(buf);
    return data ? data + (size_t)offset : NULL;
}

/* All allocations, disk reads, checksum checks and destination checks finish
 * before the first memcpy. v41 KV/snapshot buffers use shared storage; after
 * commit_batch has drained GPU work, these in-memory copies cannot fail.
 * Maximum staging is one 10 MiB window plus packed rows through 11264 (about
 * 34.4 MiB), headers/refs/history. No full context/weight copy is required. */
static bool lowkv_publish_stage(myds_v41_context_t *g, uint32_t s, uint32_t p,
    const lowkv_restore_stage_t *st) {
    const size_t window_bytes = (size_t)V41_LAYERS * V41_WINDOW * V41_HEAD_DIM * 4u;
    if (!snapshot_ensure_slot(g, s)) return false;
    unsigned char *window = lowkv_restore_destination(g->snap_window[s], 0, window_bytes);
    unsigned char *compressed[4], *index[4], *prev_kv[4], *prev_score[4];
    if (!window) return false;
    for (uint32_t o = 0; o < 4u; ++o) {
        const uint64_t rows = p / lowkv_owner_ratio(o);
        compressed[o] = lowkv_restore_destination(g->compressed[o], 0,
            rows * V41_LOWKV_LATENT * V41_CACHE_BYTES);
        index[o] = lowkv_restore_destination(g->index_cache[o], 0,
            rows * V41_INDEX_DIM * V41_CACHE_BYTES);
        prev_kv[o] = lowkv_restore_destination(g->snap_prev_kv[s][o], 0, 512u * 4u);
        prev_score[o] = lowkv_restore_destination(g->snap_prev_score[s][o], 0, 512u * 4u);
        if (!compressed[o] || !index[o] || !prev_kv[o] || !prev_score[o]) return false;
    }

    if (st->is_end) {
        const size_t win_off = V41_LOWKV_HEADER + sizeof(myds_engram_history_t) + 16384u;
        memcpy(window, st->snapshot + win_off, window_bytes);

        size_t prev_off = V41_LOWKV_HEADER + sizeof(myds_engram_history_t);
        for (uint32_t o = 0; o < 4u; ++o) {
            memcpy(prev_kv[o], st->snapshot + prev_off, 512u * 4u);
            prev_off += 512u * 4u;
        }
        for (uint32_t o = 0; o < 4u; ++o) {
            memcpy(prev_score[o], st->snapshot + prev_off, 512u * 4u);
            prev_off += 512u * 4u;
        }

        for (uint32_t i = 0; i < st->count; ++i) {
            size_t src = V41_LOWKV_HEADER;
            for (uint32_t kind = 0; kind < 2u; ++kind) {
                const uint32_t width = kind ? V41_INDEX_DIM : V41_LOWKV_LATENT;
                for (uint32_t o = 0; o < 4u; ++o) {
                    const uint32_t ratio = lowkv_owner_ratio(o);
                    const size_t n = (size_t)((st->ref[i].end - st->ref[i].start) / ratio) * width * V41_CACHE_BYTES;
                    const size_t dst = (size_t)(st->ref[i].start / ratio) * width * V41_CACHE_BYTES;
                    memcpy((kind ? index[o] : compressed[o]) + dst, st->segment[i] + src, n);
                    src += n;
                }
            }
        }

        size_t tail_off = win_off + window_bytes + (size_t)st->count * V41_LOWKV_REF_BYTES;
        for (uint32_t kind = 0; kind < 2u; ++kind) {
            const uint32_t width = kind ? V41_INDEX_DIM : V41_LOWKV_LATENT;
            for (uint32_t o = 0; o < 4u; ++o) {
                const uint32_t ratio = lowkv_owner_ratio(o);
                const uint32_t start_row = st->base_end / ratio;
                const uint32_t end_row = p / ratio;
                if (end_row > start_row) {
                    const size_t n = (size_t)(end_row - start_row) * width * V41_CACHE_BYTES;
                    const size_t dst = (size_t)start_row * width * V41_CACHE_BYTES;
                    memcpy((kind ? index[o] : compressed[o]) + dst, st->snapshot + tail_off, n);
                    tail_off += n;
                }
            }
        }
    } else {
        memcpy(window, st->snapshot + V41_LOWKV_HEADER + sizeof(myds_engram_history_t), window_bytes);
        for (uint32_t i = 0; i < st->count; ++i) {
            size_t src = V41_LOWKV_HEADER;
            for (uint32_t kind = 0; kind < 2u; ++kind) {
                const uint32_t width = kind ? V41_INDEX_DIM : V41_LOWKV_LATENT;
                for (uint32_t o = 0; o < 4u; ++o) {
                    const uint32_t ratio = lowkv_owner_ratio(o);
                    const size_t n = (size_t)((st->ref[i].end - st->ref[i].start) / ratio) * width * V41_CACHE_BYTES;
                    const size_t dst = (size_t)(st->ref[i].start / ratio) * width * V41_CACHE_BYTES;
                    memcpy((kind ? index[o] : compressed[o]) + dst, st->segment[i] + src, n);
                    src += n;
                }
            }
        }
        for (uint32_t o = 0; o < 4u; ++o) {
            memset(prev_kv[o], 0, 512u * 4u);
            memset(prev_score[o], 0, 512u * 4u);
        }
    }

    memcpy(&g->snap_engram[s], st->snapshot + V41_LOWKV_HEADER, sizeof(myds_engram_history_t));
    for (uint32_t i = 0; i < V41_SNAP_SLOTS; ++i) {
        g->snap_pos[i] = i == s ? p : 0;
        g->snap_staged_pos[i] = 0;
    }
    return true;
}

static uint64_t lowkv_store_file_bytes(uint32_t entries, uint64_t stride) {
    const uint64_t table = 4096u + (uint64_t)entries * 64u;
    return (table + 4095u) / 4096u * 4096u + (uint64_t)entries * stride;
}

static bool lowkv_open_stores(myds_v41_context_t *g, const char *snap_path,
    const char *segment_path, const char *end_path, uint64_t identity, uint32_t ways, uint64_t budget) {
    const uint32_t entries = MYDS_SNAP_PINNED * ways;
    const uint64_t snapshot_stride = lowkv_payload_bytes(lowkv_segment_end(MYDS_SNAP_PINNED - 1u));
    const uint64_t segment_stride = lowkv_segment_bytes(0, MYDS_SNAP_MIN);
    const uint64_t end_stride = V41_LOWKV_END_STRIDE;
    const uint64_t total = lowkv_store_file_bytes(entries, snapshot_stride) +
        lowkv_store_file_bytes(entries, segment_stride) +
        lowkv_store_file_bytes(V41_LOWKV_ENDS_MAX, end_stride);
    if (!segment_stride || (budget && total > budget)) {
        MYDS_LOG_E("kvdisk: snapshot + segments + ends require %.1f MiB, budget %.1f MiB",
            (double)total / 1048576.0, (double)budget / 1048576.0);
        return false;
    }
    const uint64_t snapshot_id = myds_kvdsk_identity("v41-lowkv-snapshot-v3", &identity, 1);
    const uint64_t segment_id = myds_kvdsk_identity("v41-lowkv-segments-v1", &identity, 1);
    const uint64_t end_id = myds_kvdsk_identity("v41-lowkv-ends-v1", &identity, 1);
    g->low_segments = myds_kvdsk_open(segment_path, segment_id, entries, segment_stride, 0);
    if (!g->low_segments) return false;
    g->low_store = myds_kvdsk_open(snap_path, snapshot_id, entries, snapshot_stride, 0);
    if (!g->low_store) { myds_kvdsk_close(g->low_segments); g->low_segments = NULL; return false; }
    g->low_ends = myds_kvdsk_open(end_path, end_id, V41_LOWKV_ENDS_MAX, end_stride, 0);
    if (!g->low_ends) {
        myds_kvdsk_close(g->low_store); g->low_store = NULL;
        myds_kvdsk_close(g->low_segments); g->low_segments = NULL;
        return false;
    }
    g->low_ways = ways;
    return true;
}

bool myds_v41_low_kv_enable(myds_v41_context_t *g, const char *dir) {
    if (!g) return false;
    if (g->low_store) return true;
    if (!dir || !*dir || strcmp(dir, "0") == 0) return false;
    if (!g->snap_enabled) {
        MYDS_LOG_W("kvdisk: 快照被 MYDS_NO_PREFIX_SNAPSHOT 关掉了 ⇒ 低位区落盘不启用");
        return false;
    }
    const uint32_t deepest = MYDS_SNAP_MIN + (MYDS_SNAP_PINNED - 1u) * MYDS_SNAP_STRIDE;
    const uint64_t stride = lowkv_payload_bytes(deepest);

    /* way 数：同一位置能同时容纳几套不同的「system + tools」。见 V41_LOWKV_WAYS_DEFAULT。 */
    uint32_t ways = V41_LOWKV_WAYS_DEFAULT;
    const char *we = getenv("MYDS_KV_DSK_WAYS");
    if (we && *we) {
        const long v = atol(we);
        if (v >= 1 && v <= (long)V41_LOWKV_WAYS_MAX) ways = (uint32_t)v;
        else MYDS_LOG_W("kvdisk: MYDS_KV_DSK_WAYS=%s 超出 1..%u，改用默认 %u",
                        we, (unsigned)V41_LOWKV_WAYS_MAX, (unsigned)V41_LOWKV_WAYS_DEFAULT);
    }
    g->low_ways = ways;
    {   /* 后台写盘开关：只在启用时读一次（不进热路径）。默认关 = 与以前完全一致。 */
        const char *ae = getenv("MYDS_KV_DSK_ASYNC");
        g->low_async = (ae && atoi(ae) != 0);
    }
    const uint32_t entries = MYDS_SNAP_PINNED * ways;
    const uint64_t identity = lowkv_identity(g->model);
    if (!identity) {
        MYDS_LOG_W("kvdisk: 模型指纹计算失败 ⇒ 禁用低位区磁盘复用");
        return false;
    }

    /* 默认预算不限。显式预算检查三个固定 stride 文件的虚拟大小之和，包含头/表对齐。
     * 预算不足时在打开任一文件前失败，不会减少 ways 或只启用一个库。 */
    uint64_t budget = 0;   /* 0 = 不限 */
    const char *be = getenv("MYDS_KV_DSK_BUDGET_MB");
    if (be && *be) {
        const long mb = atol(be);
        if (mb > 0) budget = (uint64_t)mb << 20;
        else MYDS_LOG_W("kvdisk: MYDS_KV_DSK_BUDGET_MB=%s 非法（需 > 0）⇒ 按【不限】处理", be);
    }

    char path[1200], segment_path[1200], end_path[1200];
    const int n = snprintf(path, sizeof(path), "%s/kvlow-v3-%016llx.bin",
                           dir, (unsigned long long)identity);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        MYDS_LOG_E("kvdisk: 目录过长，无法组成文件名: %s", dir);
        return false;
    }
    const int sn = snprintf(segment_path, sizeof(segment_path), "%s/kvsegments-v1-%016llx.bin",
                            dir, (unsigned long long)identity);
    if (sn <= 0 || (size_t)sn >= sizeof(segment_path)) return false;
    const int en = snprintf(end_path, sizeof(end_path), "%s/kvends-v1-%016llx.bin",
                            dir, (unsigned long long)identity);
    if (en <= 0 || (size_t)en >= sizeof(end_path)) return false;
    if (!lowkv_open_stores(g, path, segment_path, end_path, identity, ways, budget)) {
        MYDS_LOG_E("kvdisk: 打开失败: %s（%s）", path, myds_kvdsk_open_error());
        return false;
    }
    MYDS_LOG_I("kvdisk: 低位区落盘已启用 · %u 个位置组 × %u way = %u 条 · 每条 %.1f MiB "
               "· 位置 %u..%u · %s",
               (unsigned)MYDS_SNAP_PINNED, (unsigned)ways, (unsigned)entries,
               (double)stride / 1048576.0,
               (unsigned)MYDS_SNAP_MIN, (unsigned)deepest, path);
    MYDS_LOG_I("kvdisk: 同一位置可容纳 %u 套不同的「system + tools」—— 它们的哈希不同，"
               "互不顶替；只有全部占满才按命中数淘汰", (unsigned)ways);
    return true;
}

bool myds_v41_low_kv_enabled(const myds_v41_context_t *g) {
    return g && g->low_store != NULL && g->low_segments != NULL && g->low_ends != NULL;
}

/* ── 后台写盘 ──
 * 首轮保存需要写各快照的窗口/历史/refs，以及尚未存在的共享段。
 *
 * 为什么可以（与 decode 并行的安全论证，改动前务必读）：
 *   线程读的只有 ① snap_window[s] / snap_engram[s]（s >= 1），② compressed/index_cache 的 [0, p/r) 行。
 *   ① 只被 prefill 的存点、store_now（槽 0，线程不读）、restore、low_kv_load 改 ⇒ 它们都先 join；
 *     decode（myds_v41_forward）不碰快照槽。
 *   ② decode 从位置 q0 = n_tokens-1 起写压缩行：ratio 1 写行 q，ratio 2 只在奇数 q 写行 (q-1)/2。
 *     当 p <= q0 时两者的行号都 >= p/r ⇒ 与线程读的半开区间 [0, p/r) 不相交。
 *     ★ 这里【不】需要「p 是 512 的倍数」：pinned 槽恰好是 1024+512k，但不相交对任意 p 成立。
 *     ★ p == n_tokens 时 q0 = p-1，ratio 1 会重写第 p-1 行（decode 路径的数值可能与 prefill 不同位）
 *       ⇒ 这种条目【必须】在返回前同步写完。下一个请求在碰任何状态前都会 join。
 *   线程独占两个磁盘库。每个 job 的各段前缀哈希在调用线程冻结，后台不访问 tokens。
 *   所有段提交后才写描述符；write_start 先置空，元信息最后写。保存时同一批 jobs
 *   对每个 group/hash 仅校验一次，命中描述符只 peek 小 refs 区间，不重复读取窗口。 */
static bool lowkv_run_jobs(myds_v41_context_t *g, const v41_lowkv_job_t *jobs, uint32_t n,
                           uint32_t *written, uint64_t *bytes) {
    if (written) *written = 0;
    if (bytes) *bytes = 0;
    if (!n) return true;
    unsigned char *stage = (unsigned char *)malloc(V41_LOWKV_STAGE);
    if (!stage) { MYDS_LOG_E("kvdisk: 分块缓冲分配失败"); return false; }
    bool ok = true;
    lowkv_segment_ref_t verified[MYDS_SNAP_PINNED] = {0};
    bool have_verified[MYDS_SNAP_PINNED] = {false};
    for (uint32_t i = 0; i < n && ok; ++i) {
        const uint32_t s = jobs[i].s, p = jobs[i].p;
        bool already = false;
        const uint32_t nrefs = lowkv_segment_count(p);
        lowkv_segment_ref_t refs[MYDS_SNAP_PINNED];
        if (!nrefs || s != nrefs) { ok = false; break; }
        const uint64_t seg_before = g->low_segment_bytes;
        for (uint32_t j = 0; j < nrefs && ok; ++j) {
            if (!have_verified[j] || verified[j].hash != jobs[i].segment_hashes[j]) {
                ok = lowkv_ensure_segment(g, j, jobs[i].segment_hashes[j], stage, &verified[j]);
                have_verified[j] = ok;
            }
            refs[j] = verified[j];
        }
        if (bytes) *bytes += g->low_segment_bytes - seg_before;
        if (!ok) break;
        const uint32_t entry = lowkv_pick_entry(g->low_store, g, s - 1u, p, jobs[i].hash, &already);
        if (entry == V41_LOWKV_NO_ENTRY) { ok = false; break; }
        if (already && lowkv_snapshot_refs_match(g, entry, &jobs[i], refs)) continue;
        /* Rewrite refs when a backing segment was evicted and reinserted into
         * a different way. Windows/history still belong to this frozen job. */
        ok = lowkv_write_slot(g, entry, &jobs[i], refs, stage);
        if (!ok) { MYDS_LOG_W("kvdisk: 位置 %u 写盘失败", p); break; }
        if (written) ++*written;
        if (bytes) *bytes += lowkv_payload_bytes(p);
    }
    free(stage);
    return ok;
}

static double lowkv_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void *lowkv_thread_main(void *arg) {
    myds_v41_context_t *g = (myds_v41_context_t *)arg;
    /* 不与 decode 线程抢 CPU：落盘是「做完就行」的活。 */
    (void)pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    const double t0 = lowkv_now_ms();
    uint32_t written = 0; uint64_t bytes = 0;
    g->low_thr_ok = lowkv_run_jobs(g, g->low_jobs, g->low_job_n, &written, &bytes);
    if (written || !g->low_thr_ok)
        MYDS_LOG_I("kvdisk: 后台写盘 %u/%u 条 · %.1f MiB · %.1f ms%s", written, g->low_job_n,
                   (double)bytes / 1048576.0, lowkv_now_ms() - t0, g->low_thr_ok ? "" : " · 有失败");
    return NULL;
}

static void lowkv_join(myds_v41_context_t *g) {
    if (!g || !g->low_thr_live) return;
    const double t0 = lowkv_now_ms();
    (void)pthread_join(g->low_thr, NULL);
    g->low_thr_live = false;
    const double waited = lowkv_now_ms() - t0;
    /* 等待超过 1 ms 就说出来：这是「后台写盘拖住了下一个请求」的唯一信号。 */
    if (waited > 1.0) MYDS_LOG_I("kvdisk: 等待上一轮后台写盘 %.1f ms", waited);
}

void myds_v41_low_kv_wait(myds_v41_context_t *g) { lowkv_join(g); }

bool myds_v41_low_kv_save(myds_v41_context_t *g, const int32_t *tokens, uint32_t n_tokens) {
    if (!g || !g->low_store) return true;              /* 未启用不是错误 */
    if (!g->snap_enabled || !tokens || n_tokens == 0) return true;
    lowkv_join(g);
    /* 哈希在这里算完：调用方返回后就会释放 tokens。 */
    v41_lowkv_job_t sync_jobs[MYDS_SNAP_PINNED];
    uint64_t prefix_hashes[MYDS_SNAP_PINNED] = {0};
    bool have_hash[MYDS_SNAP_PINNED] = {false};
    uint32_t n_sync = 0, n_async = 0;
    for (uint32_t s = 1; s <= MYDS_SNAP_PINNED; ++s) {
        const uint32_t p = g->snap_pos[s];
        if (!p || p < V41_SNAP_MIN || p > n_tokens) continue;
        v41_lowkv_job_t job = { .s = s, .p = p };
        const uint32_t nrefs = lowkv_segment_count(p);
        if (!nrefs || nrefs != s) continue;
        for (uint32_t j = 0; j < nrefs; ++j) {
            if (!have_hash[j]) {
                prefix_hashes[j] = myds_kvdsk_hash_tokens(tokens, lowkv_segment_end(j));
                have_hash[j] = true;
            }
            job.segment_hashes[j] = prefix_hashes[j];
        }
        job.hash = prefix_hashes[nrefs - 1u];
        if (g->low_async && p < n_tokens) g->low_jobs[n_async++] = job;   /* 见上方论证 ② */
        else sync_jobs[n_sync++] = job;
    }
    const double t0 = lowkv_now_ms();
    uint32_t written = 0; uint64_t bytes = 0;
    bool ok = lowkv_run_jobs(g, sync_jobs, n_sync, &written, &bytes);
    if (written)   /* 这段时间全部挂在首 token 之前 */
        MYDS_LOG_I("kvdisk: 同步写盘 %u 条 · %.1f MiB · %.1f ms（首 token 前）",
                   written, (double)bytes / 1048576.0, lowkv_now_ms() - t0);

    /* Prompt-end 快照落盘（Slot 0）：位置由 snapshot_store_now 记下（通常为 n_tokens - 1）。
     * 仅当该点未被 pinned 槽覆盖且达到 MYDS_SNAP_MIN 时落盘。必须同步写完以避免与后续 decode 竞争。 */
    const uint32_t p_end = g->snap_pos[0];
    if (g->low_ends && p_end >= MYDS_SNAP_MIN && p_end < n_tokens) {
        const bool on_grid = ((p_end - MYDS_SNAP_MIN) % MYDS_SNAP_STRIDE == 0) &&
            (p_end <= MYDS_SNAP_MIN + (MYDS_SNAP_PINNED - 1u) * MYDS_SNAP_STRIDE);
        const uint32_t pinned_s = on_grid ? snapshot_slot_for(p_end, false) : 0;
        const bool pinned_active = (pinned_s > 0 && g->snap_pos[pinned_s] == p_end);
        if (!pinned_active) {
            const uint32_t bc = lowkv_base_segment_count(p_end);
            const uint32_t be = lowkv_base_segment_end(bc);
            const uint64_t end_hash = myds_kvdsk_hash_tokens(tokens, p_end);
            const uint64_t end_bytes = lowkv_end_payload_bytes(p_end, bc, be);
            if (end_bytes <= V41_LOWKV_END_STRIDE) {
                bool already = false;
                const uint32_t end_slot = lowkv_pick_end_entry(g, p_end, end_hash, end_bytes, &already);
                if (end_slot != V41_LOWKV_NO_ENTRY) {
                    lowkv_segment_ref_t refs[MYDS_SNAP_PINNED];
                    unsigned char *stage = (unsigned char *)malloc(V41_LOWKV_STAGE);
                    bool end_ok = (stage != NULL);
                    for (uint32_t j = 0; j < bc && end_ok; ++j) {
                        if (!have_hash[j]) {
                            prefix_hashes[j] = myds_kvdsk_hash_tokens(tokens, lowkv_segment_end(j));
                            have_hash[j] = true;
                        }
                        end_ok = lowkv_ensure_segment(g, j, prefix_hashes[j], stage, &refs[j]);
                    }
                    if (end_ok) {
                        if (!already || !lowkv_end_refs_match(g, end_slot, bc, refs)) {
                            const double t_end0 = lowkv_now_ms();
                            if (lowkv_write_end_slot(g, end_slot, p_end, end_hash, bc, be, refs, stage)) {
                                MYDS_LOG_I("kvdisk: prompt-end 快照落盘成功 · 槽 %u · 位置 %u (base %u, tail %u) · %.1f MiB · %.1f ms",
                                           end_slot, p_end, be, p_end - be, (double)end_bytes / 1048576.0, lowkv_now_ms() - t_end0);
                            } else {
                                MYDS_LOG_W("kvdisk: prompt-end 快照位置 %u 写盘失败", p_end);
                                ok = false;
                            }
                        }
                    } else {
                        ok = false;
                    }
                    free(stage);
                }
            }
        }
    }

    if (n_async) {
        g->low_job_n = n_async;
        g->low_thr_ok = true;
        if (pthread_create(&g->low_thr, NULL, lowkv_thread_main, g) == 0) {
            g->low_thr_live = true;
        } else {
            MYDS_LOG_W("kvdisk: 后台线程创建失败 ⇒ 改为同步写盘");
            ok = lowkv_run_jobs(g, g->low_jobs, n_async, NULL, NULL) && ok;
        }
    }
    return ok;
}

static uint32_t lowkv_probe_entry(myds_v41_context_t *g, const int32_t *tokens,
    uint32_t min_position, uint32_t max_position, uint32_t *out_entry) {
    if (out_entry) *out_entry = V41_LOWKV_NO_ENTRY;
    if (!g || !g->low_store || !g->snap_enabled || !tokens) return 0;
    lowkv_join(g);   /* Metadata must see the completed pending save. */
    if (max_position < V41_SNAP_MIN) return 0;

    /* 取【最深】的匹配条目：位置越深，免掉的 prefill 越多。
     *
     * 迭代【位置组】而不是【条目】：组的位置可以直接算出来（第 gi 组 = MIN + gi×STRIDE），
     * 而组内所有 way 装的是同一个位置 ⇒ 哈希【每位置只算一次】。
     * ways 增加只增加元信息查询次数，不会对每个 way 重算完整 token 前缀哈希。 */
    uint32_t best_entry = V41_LOWKV_NO_ENTRY, best_pos = 0;
    for (uint32_t remaining = MYDS_SNAP_PINNED; remaining > 0; --remaining) {
        const uint32_t gi = remaining - 1u;
        const uint32_t group_pos = MYDS_SNAP_MIN + gi * MYDS_SNAP_STRIDE;
        if (group_pos > max_position || group_pos <= min_position || group_pos <= best_pos) continue;
        const uint64_t want = myds_kvdsk_hash_tokens(tokens, group_pos);
        for (uint32_t w = 0; w < g->low_ways; ++w) {
            const uint32_t e = lowkv_entry_index(g, gi, w);
            uint32_t p = 0; uint64_t hh = 0, hu = 0;
            if (!myds_kvdsk_slot_meta(g->low_store, e, &p, &hh, &hu)) continue;
            if (p == group_pos && hh == want && hu == lowkv_payload_bytes(p)) {
                best_entry = e; best_pos = group_pos;
                break;
            }
        }
    }

    /* 检查 prompt-end 快照：若比已发现的 pinned 快照更深，直接优选更深的位置 */
    if (g->low_ends) {
        for (uint32_t e = 0; e < V41_LOWKV_ENDS_MAX; ++e) {
            uint32_t p = 0; uint64_t hh = 0, hu = 0;
            if (!myds_kvdsk_slot_meta(g->low_ends, e, &p, &hh, &hu)) continue;
            if (p > max_position || p <= min_position || p <= best_pos) continue;
            const uint32_t bc = lowkv_base_segment_count(p);
            const uint32_t be = lowkv_base_segment_end(bc);
            if (hu != lowkv_end_payload_bytes(p, bc, be)) continue;
            const uint64_t want = myds_kvdsk_hash_tokens(tokens, p);
            if (hh == want) {
                best_entry = e | V41_LOWKV_ENTRY_IS_END;
                best_pos = p;
            }
        }
    }

    if (out_entry) *out_entry = best_entry;
    return best_pos;
}

uint32_t myds_v41_low_kv_probe(myds_v41_context_t *g, const int32_t *tokens,
    uint32_t max_position) {
    return lowkv_probe_entry(g, tokens, 0, max_position, NULL);
}

uint32_t myds_v41_low_kv_probe_deeper(myds_v41_context_t *g, const int32_t *tokens,
    uint32_t memory_position, uint32_t max_position) {
    return lowkv_probe_entry(g, tokens, memory_position, max_position, NULL);
}

uint32_t myds_v41_low_kv_load_deeper(myds_v41_context_t *g, const int32_t *tokens,
    uint32_t memory_position, uint32_t max_position) {
    const double started_ms = lowkv_now_ms();
    uint32_t entry = V41_LOWKV_NO_ENTRY;
    const uint32_t best_pos = lowkv_probe_entry(g, tokens, memory_position, max_position, &entry);
    if (!best_pos || entry == V41_LOWKV_NO_ENTRY || !myds_metal_commit_batch()) return 0;
    lowkv_restore_stage_t stage = {0};
    const bool is_end = (entry & V41_LOWKV_ENTRY_IS_END) != 0;
    const uint32_t slot = is_end ? 0u : (entry / g->low_ways + 1u);
    const bool ok = lowkv_stage_slot(g, entry, best_pos, tokens, &stage) &&
        lowkv_publish_stage(g, slot, best_pos, &stage);
    if (!ok) {
        lowkv_restore_stage_free(&stage);
        MYDS_LOG_W("kvdisk: 位置 %u 的事务恢复失败；保留原内存快照与 live state (load %.3f ms)",
                   best_pos, lowkv_now_ms() - started_ms);
        return 0;
    }
    if (is_end) {
        (void)myds_kvdsk_slot_touch(g->low_ends, entry & ~V41_LOWKV_ENTRY_IS_END);
    } else {
        (void)myds_kvdsk_slot_touch(g->low_store, entry);
    }
    for (uint32_t i = 0; i < stage.count; ++i)
        (void)myds_kvdsk_slot_touch(g->low_segments, stage.ref[i].slot);
    lowkv_restore_stage_free(&stage);
    MYDS_LOG_I("kvdisk: 命中位置 %u (%s) ⇒ 免掉 %u 个 token 的 prefill (load+publish %.3f ms)",
               best_pos, is_end ? "prompt-end" : "pinned", best_pos, lowkv_now_ms() - started_ms);
    return best_pos;
}

uint32_t myds_v41_low_kv_load(myds_v41_context_t *g, const int32_t *tokens,
    uint32_t max_position) {
    return myds_v41_low_kv_load_deeper(g, tokens, 0, max_position);
}

void myds_v41_low_kv_stats(const myds_v41_context_t *g, uint32_t *live_entries, uint32_t *slots) {
    /* 注意：不汇合后台写盘（const，且只读计数）。生产代码目前没有调用方。 */
    if (live_entries) *live_entries = (g && g->low_store) ? myds_kvdsk_live_entries(g->low_store) : 0u;
    if (slots) *slots = (g && g->low_store) ? MYDS_SNAP_PINNED * g->low_ways : 0u;
}

/* ── 低位区落盘的单测夹具（只供 tests/ 使用）── */
bool myds_v41_low_kv_fixture_enable(myds_v41_context_t *g, const char *path, uint64_t identity,
                                    uint32_t ways, bool async_on, uint32_t max_pos) {
    if (!g || !path || g->low_store || !ways || ways > V41_LOWKV_WAYS_MAX || !identity) return false;
    for (uint32_t o = 0; o < 4u; ++o) {
        const uint32_t r = lowkv_owner_ratio(o);
        const uint32_t rows = r ? (max_pos / r) + 1u : 1u;
        if (!g->compressed[o]) g->compressed[o] = myds_metal_buffer_alloc((size_t)rows * V41_LOWKV_LATENT * V41_CACHE_BYTES);
        if (!g->index_cache[o]) g->index_cache[o] = myds_metal_buffer_alloc((size_t)rows * V41_INDEX_DIM * V41_CACHE_BYTES);
        if (!g->compressed[o] || !g->index_cache[o]) return false;
    }
    char segment_path[1400];
    const int n = snprintf(segment_path, sizeof(segment_path), "%s.segments", path);
    if (n <= 0 || (size_t)n >= sizeof(segment_path)) return false;
    char end_path[1400];
    const int en = snprintf(end_path, sizeof(end_path), "%s.ends", path);
    if (en <= 0 || (size_t)en >= sizeof(end_path) ||
        !lowkv_open_stores(g, path, segment_path, end_path, identity, ways, 0)) return false;
    g->low_async = async_on;
    return true;
}

void myds_v41_low_kv_segment_stats(myds_v41_context_t *g, uint32_t *live,
    uint64_t *writes, uint64_t *bytes) {
    lowkv_join(g);
    if (live) *live = g && g->low_segments ? myds_kvdsk_live_entries(g->low_segments) : 0;
    if (writes) *writes = g ? g->low_segment_writes : 0;
    if (bytes) *bytes = g ? g->low_segment_bytes : 0;
}

void myds_v41_low_kv_write_bytes(myds_v41_context_t *g, uint64_t *snapshots, uint64_t *segments) {
    lowkv_join(g);
    if (snapshots) *snapshots = g ? g->low_snapshot_bytes : 0;
    if (segments) *segments = g ? g->low_segment_bytes : 0;
}

myds_kvdsk_t *myds_v41_low_kv_fixture_segments(myds_v41_context_t *g) {
    lowkv_join(g);
    return g ? g->low_segments : NULL;
}

myds_kvdsk_t *myds_v41_low_kv_fixture_ends(myds_v41_context_t *g) {
    lowkv_join(g);
    return g ? g->low_ends : NULL;
}

void myds_v41_low_kv_fixture_fail_segment(myds_v41_context_t *g, uint32_t end) {
    lowkv_join(g);
    if (g) g->low_fail_segment_end = end;
}

void myds_v41_snapshot_fixture_publish(myds_v41_context_t *g, uint32_t slot, uint32_t p) {
    if (!g || slot >= V41_SNAP_SLOTS) return;
    if (g->engram) g->snap_engram[slot] = g->engram->history;
    g->snap_pos[slot] = p;
}

myds_metal_buffer_t *myds_v41_snapshot_get_compressed(myds_v41_context_t *g, uint32_t o) {
    return (g && o < 4) ? g->compressed[o] : NULL;
}

myds_metal_buffer_t *myds_v41_snapshot_get_index_cache(myds_v41_context_t *g, uint32_t o) {
    return (g && o < 4) ? g->index_cache[o] : NULL;
}

/* ── 图片嵌入注入 ──
 * 序列位置落在 [img_emb_start, img_emb_start+img_emb_count) 时，
 * 该位置的基础向量不用词表查表，而取注入数据里的那一份（每维 dim 长）。
 * 其余 (pos, token) 组合照常查表。返回 true 表示已填好 out[0..dim)。
 * ★ 为什么要按【位置】而不是按 token id：图片块所有位置共用同一个 id。 */
static bool v41_image_embedding_at(const myds_context_t *ctx, uint32_t pos,
                                   float *out, uint32_t dim) {
    if (!ctx || !ctx->img_emb_data || ctx->img_emb_n == 0) return false;
    if (dim != V41_DIM) return false;      /* 注入数据按语言模型的嵌入宽度给 */
    /* 区间数是个位数（历史里有几张图就是几个）⇒ 线性扫足够，不必二分 */
    for (uint32_t i = 0; i < ctx->img_emb_n; i++) {
        const uint32_t start = ctx->img_emb_start[i];
        if (pos < start) continue;
        const uint32_t off = pos - start;
        if (off >= ctx->img_emb_count[i]) continue;
        memcpy(out, ctx->img_emb_data[i] + (size_t)off * dim, (size_t)dim * sizeof(float));
        return true;
    }
    return false;
}

static uint32_t ratio_for(uint32_t l) { return l < 2 ? 0 : l < 20 ? 2 : 1; }
static uint32_t owner_for(uint32_t l) { return l < 8 ? 0 : l < 14 ? 1 : l < 20 ? 2 : 3; }
static bool kv_source(uint32_t l) { return l == 2 || l == 8 || l == 14 || l == 20; }
static bool index_source(uint32_t l) { return kv_source(l) || (l >= 24 && l % 4 == 0); }

static bool metadata_u32(const gguf_file_t *gf, const char *suffix, uint32_t expected) {
    char key[160]; snprintf(key, sizeof(key), "deepseek41.%s", suffix);
    const gguf_kv_t *kv = gguf_find_key(gf, key);
    if (!kv || (kv->type != GGUF_TYPE_UINT32 && kv->type != GGUF_TYPE_INT32) ||
        gguf_get_u32(gf, key, UINT32_MAX) != expected) {
        MYDS_LOG_E("V4.1 unsupported/missing metadata %s (expected %u)", key, expected);
        return false;
    }
    return true;
}

static bool metadata_float(const gguf_file_t *gf, const char *suffix, float expected) {
    char key[160]; snprintf(key, sizeof(key), "deepseek41.%s", suffix);
    const gguf_kv_t *kv = gguf_find_key(gf, key);
    if (!kv || kv->type != GGUF_TYPE_FLOAT32 || kv->value.val_f32 != expected) {
        MYDS_LOG_E("V4.1 unsupported/missing metadata %s", key); return false;
    }
    return true;
}

static bool metadata_array(const gguf_file_t *gf, const char *suffix,
                           const uint32_t *expected, uint32_t count) {
    char key[160]; snprintf(key, sizeof(key), "deepseek41.%s", suffix);
    const gguf_kv_t *kv = gguf_find_key(gf, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY || kv->value.val_arr.count != count ||
        (kv->value.val_arr.elem_type != GGUF_TYPE_UINT32 &&
         kv->value.val_arr.elem_type != GGUF_TYPE_INT32) ||
        !kv->value.val_arr.data || memcmp(kv->value.val_arr.data, expected, count * 4u)) {
        MYDS_LOG_E("V4.1 unsupported/missing metadata %s", key); return false;
    }
    return true;
}

static myds_metal_buffer_t *wrap(const gguf_tensor_t *t) {
    return t && t->data ? myds_metal_buffer_from_ptr(t->data, t->size_bytes) : NULL;
}

static myds_metal_buffer_t *weight(const gguf_file_t *gf, uint32_t layer,
                                   const char *suffix, ggml_type type,
                                   uint32_t width, uint32_t rows) {
    char name[128]; snprintf(name, sizeof(name), "blk.%u.%s.weight", layer, suffix);
    const gguf_tensor_t *t = gguf_find_tensor(gf, name);
    if (!t || t->type != type || t->n_dims != (rows ? 2u : 1u) ||
        t->dims[0] != width || (rows && t->dims[1] != rows)) {
        MYDS_LOG_E("V4.1 missing/invalid tensor %s", name); return NULL;
    }
    myds_metal_buffer_t *b = wrap(t);
    if (!b) MYDS_LOG_E("V4.1 cannot map tensor %s", name);
    return b;
}

myds_v41_model_t *myds_v41_model_create(myds_model_t *m, const char *path) {
    if (!m || !path || strcmp(m->arch, "deepseek41") || m->dim != V41_DIM ||
        m->n_layers != V41_LAYERS || m->n_heads != V41_HEADS ||
        m->head_dim != V41_HEAD_DIM || m->q_lora_rank != 1280 ||
        m->n_out_groups != 8 || m->out_group_rank != 1024 ||
        m->n_routed_experts != 384 || m->n_activated_experts != 6 ||
        m->n_shared_experts != 1 || m->moe_expert_dim != 2304 ||
        m->tok_embeddings_type != GGML_TYPE_F16) {
        MYDS_LOG_E("Native V4.1 requires the validated Flash 4-stream text layout");
        return NULL;
    }
    const gguf_file_t *gf = m->gf;
    const struct { const char *key; uint32_t value; } ints[] = {
        {"sliding_window",128}, {"hc_mult",4}, {"hc_sinkhorn_iters",20},
        {"index_n_heads",32}, {"index_head_dim",128}, {"index_topk",512},
        {"candidate_source_layer_id",20}, {"candidate_topk_blocks",2048},
        {"candidate_block_size",8}, {"rope_theta",10000}, {"compress_rope_theta",160000},
    };
    for (size_t i = 0; i < sizeof(ints)/sizeof(*ints); i++)
        if (!metadata_u32(gf, ints[i].key, ints[i].value)) return NULL;
    const struct { const char *key; float value; } floats[] = {
        {"rms_norm_eps",1e-20f}, {"hc_eps",1e-6f}, {"swiglu_limit",10.0f},
        {"routed_scaling_factor",1.5f}, {"rope_scaling.factor",16.0f},
        {"rope_scaling.beta_fast",32.0f}, {"rope_scaling.beta_slow",1.0f},
        {"rope_scaling.original_max_position_embeddings",65536.0f},
    };
    for (size_t i = 0; i < sizeof(floats)/sizeof(*floats); i++)
        if (!metadata_float(gf, floats[i].key, floats[i].value)) return NULL;
    uint32_t ratios[V41_LAYERS];
    for (uint32_t i = 0; i < V41_LAYERS; i++) ratios[i] = ratio_for(i);
    const uint32_t kv[] = {2,8,14,20}, index[] = {2,8,14,20,24,28,32,36};
    if (!metadata_array(gf,"kv_source_layer_ids",kv,4) ||
        !metadata_array(gf,"index_source_layer_ids",index,8) ||
        !metadata_array(gf,"compress_ratios",ratios,V41_LAYERS) ||
        strcmp(gguf_get_str(gf,"deepseek41.scoring_func",""),"sqrtsoftplus") ||
        strcmp(gguf_get_str(gf,"deepseek41.topk_method",""),"noaux_tc") ||
        !gguf_get_bool(gf,"deepseek41.norm_topk_prob",false)) return NULL;
    myds_v41_model_t *v = calloc(1,sizeof(*v));
    if (!v) return NULL;
    v->hc_eps = 1e-6f;
    const gguf_tensor_t *t_legacy_hc = gguf_find_tensor(gf, "output_hc.weight");
    if (!t_legacy_hc) t_legacy_hc = gguf_find_tensor(gf, "output_hc_mix.weight");
    if (t_legacy_hc) {
        MYDS_LOG_W("V4.1 GGUF contains unused legacy output_hc weights; V4.1 folds output via last layer FFN mix.");
    }
#define W(member, name, type, width, rows) \
    do { l->member = weight(gf,i,name,type,width,rows); if (!l->member) goto fail; } while (0)
    for (uint32_t i = 0; i < V41_LAYERS; i++) {
        v41_layer_t *l = &v->layers[i];
        W(attn_fn,"hc_attn_fn",GGML_TYPE_F16,4*V41_DIM,24);
        W(attn_scale,"hc_attn_scale",GGML_TYPE_F32,3,0);
        W(attn_base,"hc_attn_base",GGML_TYPE_F32,24,0);
        W(ffn_fn,"hc_ffn_fn",GGML_TYPE_F16,4*V41_DIM,24);
        W(ffn_scale,"hc_ffn_scale",GGML_TYPE_F32,3,0);
        W(ffn_base,"hc_ffn_base",GGML_TYPE_F32,24,0);
        if (kv_source(i)) {
            W(compressor_kv,"attn_compressor_kv",GGML_TYPE_F16,V41_DIM,512);
            W(compressor_norm,"attn_compressor_norm",GGML_TYPE_F32,512,0);
            if (ratio_for(i)==2) W(compressor_gate,"attn_compressor_gate",GGML_TYPE_F16,V41_DIM,512);
            W(index_k,"indexer.attn_k",GGML_TYPE_F16,512,128);
            W(index_k_norm,"indexer.k_norm",GGML_TYPE_F32,128,0);
        }
        if (index_source(i)) {
            W(index_q,"indexer.attn_q_b",GGML_TYPE_F16,1280,V41_INDEX_HEADS*128);
            W(index_proj,"indexer.proj",GGML_TYPE_F16,V41_DIM,V41_INDEX_HEADS);
        }
        const myds_layer_t *base = &m->layers[i];
        if (!base->router_bias || !base->attn_sinks || !base->shared_gate ||
            base->attn_output_a_type != GGML_TYPE_Q8_0 ||
            base->attn_output_b_type != GGML_TYPE_Q8_0) goto fail;
    }
#undef W
    v->engram = myds_engram_create(gf,path,m->vocab_size,m->dim);
    if (v->engram)
        MYDS_LOG_I("Engram SSD I/O: %u threads%s (MYDS_ENGRAM_IO_THREADS=1 selects serial)",
                   v->engram->io_threads, v->engram->io_threads > 1 ? ", persistent raw-read pool" : ", serial");
    if (!v->engram) goto fail;
    for (uint32_t i=0;i<2;i++) {
        v->engram_proj[i]=wrap(v->engram->projection[i]);
        v->engram_q[i]=wrap(v->engram->q_norm[i]);
        v->engram_k[i]=wrap(v->engram->k_norm[i]);
        if (!v->engram_proj[i] || !v->engram_q[i] || !v->engram_k[i]) goto fail;
    }
    if (!myds_engram_unmap_tables(m->gf,v->engram)) goto fail;
    MYDS_LOG_I("Native V4.1 diagnostic graph: HC4, Engram disk rows, SWA128, 4 KV owners");
    return v;
fail:
    myds_v41_model_free(v); return NULL;
}

void myds_v41_model_free(myds_v41_model_t *v) {
    if (!v) return;
    for (uint32_t i=0;i<V41_LAYERS;i++) {
        v41_layer_t *l=&v->layers[i];
#define F(x) myds_metal_buffer_free(l->x)
        F(attn_fn); F(attn_scale); F(attn_base); F(ffn_fn); F(ffn_scale); F(ffn_base);
        F(compressor_kv); F(compressor_gate); F(compressor_norm);
        F(index_k); F(index_k_norm); F(index_q); F(index_proj);
#undef F
    }
    for (uint32_t i=0;i<2;i++) {
        myds_metal_buffer_free(v->engram_proj[i]); myds_metal_buffer_free(v->engram_q[i]);
        myds_metal_buffer_free(v->engram_k[i]);
    }
    myds_engram_free(v->engram); free(v);
}

myds_v41_context_t *myds_v41_context_create(myds_context_t *ctx) {
    if (!ctx || !ctx->model->v41 || !ctx->cfg.max_seq_len || ctx->cfg.max_seq_len>1048576u)
        return NULL;
    /* Basic buffers are allocated by the shared context constructor. */
    myds_metal_buffer_t *required[] = {ctx->x,ctx->x_norm,ctx->q,ctx->qr,ctx->qr_norm,
        ctx->kv_act,ctx->attn_heads,ctx->attn_low,ctx->attn_out,ctx->ffn_norm_out,
        ctx->ffn_out,ctx->logits,ctx->router_logits,ctx->selected_ids,ctx->selected_weights,
        ctx->moe_gate,ctx->moe_up,ctx->moe_mid,ctx->moe_down,ctx->moe_out,
        ctx->shared_gate_out,ctx->shared_up_out,ctx->shared_swiglu_out,ctx->shared_out};
    for (size_t i=0;i<sizeof(required)/sizeof(*required);i++) if (!required[i]) return NULL;
    if (!ctx->sample_probs || !ctx->sample_indices || !ctx->sample_cands) return NULL;
    myds_v41_context_t *g=calloc(1,sizeof(*g));
    if (!g) return NULL;
    g->model = ctx->model;
    g->capacity=ctx->cfg.max_seq_len;
    g->engram=myds_engram_clone(ctx->model->v41->engram);
    if (!g->engram) goto fail;
    g->gpu=myds_v41_gpu_scratch_create(g->capacity);
    if (!g->gpu) goto fail;
#define A(member, words) do { g->member=myds_metal_buffer_alloc((size_t)(words)*4u); if (!g->member) goto fail; } while (0)
    A(residual,4*V41_DIM); A(after_attn,4*V41_DIM); A(flat_norm,4*V41_DIM);
    A(mix,24); A(attn_split,24); A(ffn_split,24); A(pre,4);
    A(engram_rows[0],MYDS_ENGRAM_INPUT_DIM); A(engram_rows[1],MYDS_ENGRAM_INPUT_DIM);
    A(engram_kv,5*V41_DIM);
    for (uint32_t i=0;i<V41_LAYERS;i++) A(window[i],V41_WINDOW*V41_HEAD_DIM);
    g->snap_enabled=getenv("MYDS_NO_PREFIX_SNAPSHOT")==NULL;
    if (g->snap_enabled) {
        /* 布局写进日志：槽位分配错了不会报错，只会让该命中的请求悄悄退化成全量重算。 */
        MYDS_LOG_I("prefix snapshots: %u slots (%u pinned for positions %u..%u, %u ring), stride %u",
                   (unsigned)MYDS_SNAP_SLOTS, (unsigned)MYDS_SNAP_PINNED,
                   (unsigned)MYDS_SNAP_MIN,
                   (unsigned)(MYDS_SNAP_MIN + (MYDS_SNAP_PINNED - 1u) * MYDS_SNAP_STRIDE),
                   (unsigned)(MYDS_SNAP_SLOTS - 1u - MYDS_SNAP_PINNED),
                   (unsigned)MYDS_SNAP_STRIDE);
    }
    for (uint32_t i=0;i<4;i++) {
        const size_t rows=i==3 ? g->capacity : (g->capacity+1u)/2u;
        g->compressed[i]=myds_metal_buffer_alloc(rows*512u*V41_CACHE_BYTES);
        g->index_cache[i]=myds_metal_buffer_alloc(rows*128u*V41_CACHE_BYTES);
        if (!g->compressed[i] || !g->index_cache[i]) goto fail;
        A(previous_kv[i],512); A(previous_score[i],512);
    }
    A(pool_kv,512); A(pool_score,512); A(latent,512); A(index_k,128);
    A(index_q,V41_INDEX_HEADS*128); A(index_weights,V41_INDEX_HEADS);
    A(index_scores,g->capacity); A(selected,V41_TOPK); A(selected_kv,V41_TOPK*512);
    A(block_scores,(g->capacity+7u)/8u); A(block_mask,(g->capacity+7u)/8u);
    A(block_selected,2048);
#undef A
    g->valid=true;
    return g;
fail:
    myds_v41_context_free(g); return NULL;
}

void myds_v41_context_free(myds_v41_context_t *g) {
    if (!g) return;
    lowkv_join(g);                     /* 先等后台写盘结束：它还在读下面要释放的缓冲 */
    myds_kvdsk_close(g->low_store);
    myds_kvdsk_close(g->low_segments);
    myds_kvdsk_close(g->low_ends);
    g->low_store = NULL;
    g->low_segments = NULL;
    g->low_ends = NULL;
#define F(x) myds_metal_buffer_free(g->x)
    F(residual); F(after_attn); F(flat_norm); F(mix); F(attn_split); F(ffn_split); F(pre);
    F(engram_rows[0]); F(engram_rows[1]); F(engram_kv);
    for (uint32_t i=0;i<V41_LAYERS;i++) F(window[i]);
    for (uint32_t si=0;si<V41_SNAP_SLOTS;si++) {
        F(snap_window[si]);
        for (uint32_t i=0;i<4;i++) {
            F(snap_prev_kv[si][i]);
            F(snap_prev_score[si][i]);
        }
    }
    for (uint32_t i=0;i<4;i++) { F(compressed[i]); F(index_cache[i]); F(previous_kv[i]); F(previous_score[i]); }
    F(pool_kv); F(pool_score); F(latent); F(index_k); F(index_q); F(index_weights);
    F(index_scores); F(selected); F(selected_kv); F(block_scores); F(block_selected); F(block_mask);
#undef F
    if (g->prompt_res) myds_metal_buffer_free(g->prompt_res);
    if (g->prompt_pre) myds_metal_buffer_free(g->prompt_pre);
    if (g->prompt_selected) myds_metal_buffer_free(g->prompt_selected);
    if (g->prompt_block_mask) myds_metal_buffer_free(g->prompt_block_mask);
    if (g->prompt_engram[0]) free(g->prompt_engram[0]);
    if (g->prompt_engram[1]) free(g->prompt_engram[1]);
    v41_chunk_workspace_free(g->chunk_ws);
    myds_v41_gpu_scratch_free(g->gpu); myds_engram_free(g->engram); free(g);
}

myds_v41_context_t *myds_v41_context_create_for_model(const myds_model_t *model) {
    if (!model) return NULL;
    myds_v41_context_t *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->model = model;
    g->snap_enabled = (getenv("MYDS_NO_PREFIX_SNAPSHOT") == NULL);
    return g;
}

static int project(myds_metal_buffer_t *out, const myds_metal_buffer_t *w,
                   const myds_metal_buffer_t *in, uint32_t rows, uint32_t width,
                   ggml_type type, bool round) {
    myds_metal_barrier();
    return ((type == GGML_TYPE_F16 || type == GGML_TYPE_F32)
        ? myds_metal_v41_gemv(out,w,in,rows,width,type)
        : myds_metal_gemv(out,w,in,rows,width,type)) &&
        (!round || myds_metal_v41_bf16(out,rows));
}

static int qkv_fusion_enabled(const char *name,int *cached) {
    if(*cached<0) {const char *v=getenv(name);*cached=!(v&&strcmp(v,"0")==0);}
    return *cached;
}
static int project_decode_qkv(myds_metal_buffer_t *out,const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *in,uint32_t rows,uint32_t width,ggml_type type,bool trace_on) {
    static int enabled=-1;
    if(trace_on||!qkv_fusion_enabled("MYDS_QKV_FUSE_BF16",&enabled))
        return project(out,w,in,rows,width,type,true);
    return myds_metal_v41_gemv_bf16(out,w,in,rows,width,type);
}

/* output-B retains project()'s F16/F32 specialization. The shared-expert
 * caller retains its original generic fallback through the common helper. */
static int project_decode_tail(myds_metal_buffer_t *out,const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *in,uint32_t rows,uint32_t width,ggml_type type,bool trace_on) {
    if(type!=GGML_TYPE_Q8_0) return project(out,w,in,rows,width,type,true);
    return myds_v41_decode_gemv_bf16(out,w,in,rows,width,type,trace_on);
}

static inline bool is_mul_mm_disabled(void) {
    static int cached = -1;
    if (__builtin_expect(cached < 0, 0)) {
        cached = (getenv("MYDS_DISABLE_MUL_MM") != NULL) ? 1 : 0;
    }
    return cached == 1;
}

/* 小 M 时改走 GEMV 路径的阈值。MM 内核的 tile 是 NR0=64 行；当 rows 远小于 64 时
 * （例如 hc_mix 的 rows=24），线程组网格退化成 ceil(rows/64)=1 ⇒ 占用率极低：
 * 实测这一步单独占 Wait Proj 的 ~10%（206 ms/趟），改用 GEMV 后回收。
 * 默认 64 = rows<64 不走 MM。MYDS_MM_MIN_M=<n> 可覆盖（0 = 总是用 MM = 旧行为）。 */
static inline uint32_t mm_min_m(void) {
    static int cached = -1;
    if (__builtin_expect(cached < 0, 0)) {
        const char *v = getenv("MYDS_MM_MIN_M");
        long n = v ? atol(v) : 64;
        cached = (n >= 0 && n < 100000) ? (int)n : 64;
    }
    return (uint32_t)cached;
}

static int project_batch(myds_metal_buffer_t *out, const myds_metal_buffer_t *w,
                         const myds_metal_buffer_t *in, uint32_t rows, uint32_t width,
                         uint32_t n_tok, ggml_type type, bool round) {
    myds_metal_barrier();
    int use_mm = (n_tok >= 32 && !is_mul_mm_disabled());
    if (use_mm) {
        const uint32_t mm_min = mm_min_m();
        if (mm_min && rows < mm_min) use_mm = 0;   /* 小 M：MM 的 64 行 tile 会退化成单块 */
    }
    int ok = 0;
    if (use_mm) {
        ok = myds_metal_mul_mm(out, w, in, rows, width, n_tok, type);
    }
    if (!ok) {
        ok = ((type == GGML_TYPE_F16 || type == GGML_TYPE_F32)
            ? myds_metal_v41_gemv_batch(out, w, in, rows, width, n_tok, type)
            : myds_metal_gemv_batch(out, w, in, rows, width, n_tok, type));
    }
    return ok && (!round || myds_metal_v41_bf16(out, rows * n_tok));
}

static int dispatch_k(const char *kernel, const void *args, size_t size,
                      const myds_metal_buffer_t *const *buffers, size_t count,
                      uint32_t groups, uint32_t threads, size_t scratch) {
    const uint32_t grid[3] = {groups, 1, 1}, block[3] = {threads, 1, 1};
    return myds_metal_dispatch_kernel(kernel, args, size, buffers, count, grid, block, scratch);
}

/* 压缩器（只有 kv_source 层有）。它跨位置有状态：pool2 依赖 previous_kv/previous_score，
 * rope/quant 写入 index_cache/compressed 的第 count-1 行。所以【必须逐 token、按位置
 * 递增顺序】跑 —— prefill 与 decode 共用这一份。 */
static int publish_compressor(myds_context_t *ctx, uint32_t il, uint32_t pos,
                              const myds_metal_buffer_t *x_norm,
                              const myds_metal_buffer_t *pre_pool_kv,
                              const myds_metal_buffer_t *pre_pool_score) {
    myds_v41_context_t *g=ctx->v41;
    const v41_layer_t *l=&ctx->model->v41->layers[il];
    const uint32_t ratio=ratio_for(il), owner=owner_for(il);
    if (!ratio || !kv_source(il)) return 1;
    const uint32_t count=(pos+1u)/ratio;
#define CHECK(x) do { if (!(x)) return 0; } while (0)
    const myds_metal_buffer_t *p_kv = pre_pool_kv;
    if (!p_kv) {
        CHECK(project(g->pool_kv,l->compressor_kv,x_norm,512,V41_DIM,GGML_TYPE_F16,ratio==1));
        p_kv = g->pool_kv;
    }
    if (ratio==2) {
        const myds_metal_buffer_t *p_score = pre_pool_score;
        if (!p_score) {
            CHECK(project(g->pool_score,l->compressor_gate,x_norm,512,V41_DIM,GGML_TYPE_F16,false));
            p_score = g->pool_score;
        }
        CHECK(myds_v41_gpu_pool2(g->latent,p_kv,p_score,
            g->previous_kv[owner],g->previous_score[owner],512,pos));
    } else CHECK(myds_metal_buffer_copy(g->latent,0,p_kv,0,512*4u));
    if ((pos+1u)%ratio==0) {
        const uint32_t start=pos+1u-ratio;
        CHECK(myds_metal_v41_rmsnorm(g->latent,g->latent,l->compressor_norm,512,ctx->model->norm_eps,true));
        CHECK(project(g->index_k,l->index_k,g->latent,128,512,GGML_TYPE_F16,true));
        CHECK(myds_metal_v41_rmsnorm(g->index_k,g->index_k,l->index_k_norm,128,ctx->model->norm_eps,true));
        CHECK(myds_v41_gpu_rope(g->index_k,128,1,start,true,false));
        CHECK(myds_v41_gpu_quant(g->index_k,128,1,2));
        CHECK(myds_v41_gpu_pack_bf16(g->index_cache[owner],(size_t)(count-1)*128u,g->index_k,0,128u));
        CHECK(myds_v41_gpu_rope(g->latent,512,1,start,true,false));
        CHECK(myds_v41_gpu_quant(g->latent,512,1,3));
        CHECK(myds_v41_gpu_pack_bf16(g->compressed[owner],(size_t)(count-1)*512u,g->latent,0,512u));
    }
#undef CHECK
    return 1;
}

/* 【整 chunk 批量】prefill 的压缩器（ratio ∈ {1,2}，只跑在 kv_source 层 = 2/8/14/20）。
 * 逐 token 版把每个 token 拆成 3~11 次 dispatch（4 层合计约 5 万次），其中一半是纯搬运
 * （previous 拷贝、latent 拷贝）—— 每次挪 2 KB 却要付约 9 µs 的 dispatch 开销。
 *
 * 与逐 token 路径逐位相同的依据：
 *   1) ratio==2：提交行是奇数位，它依赖的「前一位」是同 chunk 内的偶数位。chunk 起点为
 *      偶数位、长度为偶数时，配对 (2p, 2p+1) 与逐 token 的 previous_kv/scores 语义完全
 *      对应 ⇒ 同一个 kernel_dsv41_pool2（tail=0）逐元素算出同一结果（无跨线程归约）；
 *   2) ratio==1：latent 就是 pool_kv 自身（逐 token 版是逐行 copy，值相同）；
 *   3) 提交段每个算子都有 batch 版，逐行结果与逐 token 版一致（同一个内核，只是 nef1
 *      的行数不同），bf16 取整参数也保持一致（rmsnorm round_bf16=true）；
 *   4) rope 的位置：逐 token 的 start=pos+1-ratio ⇒ ratio==1 时步长 1、ratio==2 时步长 2，
 *      用内核自带的 stride 参数表达（theta=(start+row*stride)*freq）；
 *   5) 缓存写入的行号：count-1 = (pos+1)/ratio - 1，对提交行而言是【连续】的
 *      ⇒ row0=(base_pos+c_start)/ratio 起、n_commit 行，一次搬运即可。
 *
 * 返回：1 = 已批量完成；0 = 前提不满足（未做任何修改，调用方可安全回退）；
 *      -1 = 已开始 dispatch 后失败（这些算子是原地改写，不能回退重做）⇒ 致命。
 */
static int publish_compressor_chunk(myds_context_t *ctx, uint32_t il,
                                    uint32_t c_start, uint32_t c_len, uint32_t base_pos,
                                    const myds_metal_buffer_t *pool_kv_buf,
                                    const myds_metal_buffer_t *pool_score_buf,
                                    myds_metal_buffer_t *latent_buf,
                                    myds_metal_buffer_t *index_k_buf) {
    myds_v41_context_t *g=ctx->v41;
    const v41_layer_t *l=&ctx->model->v41->layers[il];
    const uint32_t ratio=ratio_for(il), owner=owner_for(il);
    if (!ratio || ratio > 2u || !kv_source(il) || !c_len) return 0;
    if (!pool_kv_buf || !latent_buf || !index_k_buf) return 0;
    /* ratio==2 必须从偶数位开始且长度为偶数，配对才与逐 token 的 previous 语义一致 */
    if (ratio == 2u) {
        if (!pool_score_buf) return 0;
        if (((base_pos + c_start) & 1u) || (c_len & 1u)) return 0;
    }
    const uint32_t n_commit = (ratio == 2u) ? c_len / 2u : c_len;
    if (!n_commit) return 0;
    if (myds_metal_buffer_size(latent_buf) < (size_t)n_commit * 512u * 4u) return 0;
    if (myds_metal_buffer_size(index_k_buf) < (size_t)n_commit * 128u * 4u) return 0;
    const uint32_t row0 = (base_pos + c_start) / ratio;
    if (myds_metal_buffer_size(g->compressed[owner]) < (size_t)(row0 + n_commit) * 512u * V41_CACHE_BYTES) return 0;
    if (myds_metal_buffer_size(g->index_cache[owner]) < (size_t)(row0 + n_commit) * 128u * V41_CACHE_BYTES) return 0;
    /* 1) latent */
    if (ratio == 2u) {
        if (!myds_v41_gpu_pool2_pairs(latent_buf, pool_kv_buf, pool_score_buf, 512u, n_commit)) return -1;
    } else {
        if (!myds_metal_buffer_copy(latent_buf, 0, pool_kv_buf, 0, (size_t)c_len * 512u * 4u)) return -1;
    }
    /* 2) 提交段：整 chunk 一次 */
    /* 逐 token 的 start = pos+1-ratio：ratio==1 ⇒ start=pos=base+c_start+t；ratio==2 ⇒ 提交位是
     * 奇数位 pos=base+c_start+2p+1，start=pos-1=base+c_start+2p。两种情况的【首行】都是
     * base_pos+c_start（ratio==2 时靠 stride=2 逐行错开）。 */
    const uint32_t first_pos = base_pos + c_start;
    if (!myds_metal_v41_rmsnorm_batch(latent_buf, latent_buf, l->compressor_norm, 512u, n_commit,
                                      ctx->model->norm_eps, true)) return -1;
    if (!project_batch(index_k_buf, l->index_k, latent_buf, 128u, 512u, n_commit, GGML_TYPE_F16, true)) return -1;
    if (!myds_metal_v41_rmsnorm_batch(index_k_buf, index_k_buf, l->index_k_norm, 128u, n_commit,
                                      ctx->model->norm_eps, true)) return -1;
    if (!myds_v41_gpu_rope_batch_stride(index_k_buf, 128u, 1u, first_pos, n_commit, ratio, true, false)) return -1;
    if (!myds_v41_gpu_quant(index_k_buf, 128u, n_commit, 2u)) return -1;
    if (!myds_v41_gpu_pack_bf16(g->index_cache[owner], (size_t)row0 * 128u, index_k_buf, 0,
                                n_commit * 128u)) return -1;
    if (!myds_v41_gpu_rope_batch_stride(latent_buf, 512u, 1u, first_pos, n_commit, ratio, true, false)) return -1;
    if (!myds_v41_gpu_quant(latent_buf, 512u, n_commit, 3u)) return -1;
    if (!myds_v41_gpu_pack_bf16(g->compressed[owner], (size_t)row0 * 512u, latent_buf, 0,
                                n_commit * 512u)) return -1;
    /* 3) 跨 chunk 的状态：逐 token 版在【每个偶数位】把 pool_kv/pool_score 拷进 previous_*，
     *    所以 chunk 结束时的状态 = 最后一个偶数位（c_len-2）的值。 */
    if (ratio == 2u && c_len >= 2u) {
        if (!myds_metal_buffer_copy(g->previous_kv[owner], 0, pool_kv_buf,
                                    (size_t)(c_len - 2u) * 512u * 4u, 512u * 4u)) return -1;
        if (!myds_metal_buffer_copy(g->previous_score[owner], 0, pool_score_buf,
                                    (size_t)(c_len - 2u) * 512u * 4u, 512u * 4u)) return -1;
    }
    return 1;
}

/* 块预算：0 = 不限制（默认）。
 *
 * 2026-10-02 经实测，这个「压预算」的做法已被 per-query gather 取代：
 * gather 让内核直接只算选中键，扫描不再浪费，而压预算只是「少选一些键」，
 * 两者叠加会把可见键收缩两次 —— gather+budget=32 在 3k 上下文下实测丢题
 * （budget=0/64 均 8/8，budget=32 挂一题）。
 * ⇒ 默认应当是不限制；预算降到 32 只作为【不用 gather 时】的替代手段。
 * 实测（macstudio M2 Ultra，服务端自报）：
 *   仅 budget=32（无 gather） 5.3k 262.6 / 12.4k 203.8 t/s   8/8
 *   gather + budget=0        5.3k 281.3 / 12.4k 274.9 t/s   8/8
 *   gather + budget=32       5.3k 281.3 / 12.4k 275.2 t/s   3k 挂一题 */
#define V41_BLOCK_BUDGET_DEFAULT 0u

static uint32_t block_budget_cap(uint32_t all) {
    const char *v = getenv("MYDS_BLOCK_BUDGET");
    int n = v ? atoi(v) : (int)V41_BLOCK_BUDGET_DEFAULT;
    if (n > 0 && (uint32_t)n < all) return (uint32_t)n;
    return all;
}

/* indexer 的逐 token 版（decode 路径与 prefill 的回退路径用）。 */
static int publish_indexer_one(myds_context_t *ctx, uint32_t il, uint32_t pos,
                              const myds_metal_buffer_t *x_norm, const myds_metal_buffer_t *qr_norm,
                              myds_metal_buffer_t *pre_index_q,
                              const myds_metal_buffer_t *pre_index_weights,
                              myds_metal_buffer_t *out_selected,
                              const myds_metal_buffer_t *in_block_mask,
                              myds_metal_buffer_t *out_block_mask) {
    myds_v41_context_t *g=ctx->v41;
    const v41_layer_t *l=&ctx->model->v41->layers[il];
    const uint32_t ratio=ratio_for(il), owner=owner_for(il);
    if (!ratio) return 1;
    const uint32_t count=(pos+1u)/ratio;
#define CHECK(x) do { if (!(x)) return 0; } while (0)
    if (count && index_source(il)) {
        myds_metal_buffer_t *q_buf = pre_index_q;
        if (!q_buf) {
            CHECK(project(g->index_q,l->index_q,qr_norm,V41_INDEX_HEADS*128,1280,GGML_TYPE_F16,true));
            q_buf = g->index_q;
        }
        CHECK(myds_v41_gpu_rope(q_buf,128,V41_INDEX_HEADS,pos,true,false));
        CHECK(myds_v41_gpu_quant(q_buf,128,V41_INDEX_HEADS,2));

        const myds_metal_buffer_t *w_buf = pre_index_weights;
        if (!w_buf) {
            CHECK(project(g->index_weights,l->index_proj,x_norm,V41_INDEX_HEADS,V41_DIM,GGML_TYPE_F16,true));
            w_buf = g->index_weights;
        }

        CHECK(myds_v41_gpu_indexscores_bf16(g->index_scores,q_buf,w_buf,g->index_cache[owner],count,V41_INDEX_HEADS));
        if (il==20) {
            uint32_t blocks=(count+7)/8,top=block_budget_cap(blocks<2048?blocks:2048);
            CHECK(myds_v41_gpu_candidates(g->block_scores,g->index_scores,count,pos,ratio));
            CHECK(myds_v41_gpu_topk(g->block_selected,g->block_scores,blocks,top,g->gpu));
            myds_metal_buffer_t *b_mask = out_block_mask ? out_block_mask : g->block_mask;
            CHECK(myds_v41_gpu_topkmask(b_mask,g->block_selected,blocks,top));
        } else if (il>20) {
            const myds_metal_buffer_t *b_mask = in_block_mask ? in_block_mask : g->block_mask;
            CHECK(myds_v41_gpu_filter(g->index_scores,b_mask,count,pos,ratio));
        }
        myds_metal_buffer_t *sel = out_selected ? out_selected : g->selected;
        CHECK(myds_v41_gpu_topk(sel,g->index_scores,count,count<V41_TOPK?count:V41_TOPK,g->gpu));
    }
#undef CHECK
    return 1;
}

/* decode 路径的一份完整发布/选择（= 压缩器 + indexer，逐 token）。 */
static int publish_and_select(myds_context_t *ctx, uint32_t il, uint32_t pos,
                              const myds_metal_buffer_t *x_norm, const myds_metal_buffer_t *qr_norm,
                              const myds_metal_buffer_t *pre_pool_kv,
                              const myds_metal_buffer_t *pre_pool_score,
                              myds_metal_buffer_t *pre_index_q,
                              const myds_metal_buffer_t *pre_index_weights,
                              myds_metal_buffer_t *out_selected,
                              const myds_metal_buffer_t *in_block_mask,
                              myds_metal_buffer_t *out_block_mask) {
    if (!publish_compressor(ctx,il,pos,x_norm,pre_pool_kv,pre_pool_score)) return 0;
    return publish_indexer_one(ctx,il,pos,x_norm,qr_norm,pre_index_q,pre_index_weights,
                               out_selected,in_block_mask,out_block_mask);
}

/* 子组的 token 数：本组最后一个 token 的 count（= 该组分数表的列数）必须让
 * 组内 token 数 × 列数 ≤ 预算。返回 0 表示 1 行都放不下（调用方回退）。 */
static uint32_t indexer_group_len(uint32_t c_len, uint32_t off, uint32_t base_pos,
                                  uint32_t c_start, uint32_t ratio, uint32_t batch_floats) {
    uint32_t glen = c_len - off;
    for (;;) {
        uint32_t nc = (base_pos + c_start + off + glen) / ratio;
        /* 分数表的宽度至少要补到 V41_TOPK：批量 top-k 写 ids 的行步长是 top，
         * 而 prompt_selected 的行步长固定是 V41_TOPK，只有 top == V41_TOPK 时两者一致
         * （见 myds_v41_gpu.c 的 topk_rows_causal）。 */
        uint32_t width = nc < V41_TOPK ? V41_TOPK : nc;
        if ((uint64_t)glen * width <= (uint64_t)batch_floats) return glen;
        uint32_t next = glen > 1u ? (uint32_t)((uint64_t)batch_floats / width) : 0u;
        if (!next || next >= glen) return 0;
        glen = next;
    }
}

/* 【整 chunk 批量】prefill Step 4 的 indexer。要求调用方已用 project_batch 把整 chunk 的
 * index_q / index_weights 投影好（pre_index_q = cw->index_q_buf 等）。
 *
 * 与逐 token 路径逐位相同的依据：
 *   1) 压缩器仍在外面逐 token 跑（index_cache 的写入单调递增）⇒ 批量 indexer 读到的
 *      压缩键与逐 token 时完全一样；
 *   2) 每个内核都自带【逐行】因果宽度，行号就是本组内的 token 序号：
 *        - kernel_glm_indexer_score_chunk_direct: row >= (causal_start+tok+1)/ratio 写 -INF；
 *        - kernel_argsort_*_causal:    width = min(ne00, (causal_start+i01+1)/ratio)，
 *                                      i01 就是行号（grid.x % ne01）；
 *        - kernel_dsv41_candidate_blocks/_filter: visible = min(width, (start+row+1)/ratio)；
 *      ⇒ 每行看到的列数恰好等于逐 token 的 count；多算出来的列全被宽度挡在排序之外。
 *   3) 投影/rope/quant 本来就是整 chunk 做（project_batch），逐 token 循环里的那两个
 *      rope/quant 只是把它们拆碎 ⇒ 合并回整 chunk 是同一批内核、同一组参数。
 *
 * 分数表是「组内 token 数 × 本组最大 count」的矩形，缓冲预算自适应动态伸缩
 * （batch_floats），所以把 chunk 再切成若干 token 子组，
 * 子组宽度取该组【最后一个 token】的 count。
 *
 * 返回值：1 = 已批量完成；0 = 前提不满足（调用方回退到逐 token，此时【未做任何修改】）；
 *        -1 = 已经开始 dispatch 后失败（rope/quant 是原地改写，不能回退重复做）——致命。 */
static int publish_indexer_chunk(myds_context_t *ctx, uint32_t il, uint32_t c_start, uint32_t c_len,
                                 uint32_t base_pos, myds_metal_buffer_t *index_q_buf,
                                 const myds_metal_buffer_t *index_weights_buf,
                                 myds_metal_buffer_t *prompt_selected, size_t sel_bytes,
                                 myds_metal_buffer_t *prompt_block_mask, size_t mask_bytes,
                                 myds_metal_buffer_t *q_view, myds_metal_buffer_t *w_view,
                                 myds_metal_buffer_t *sel_view, myds_metal_buffer_t *mask_view) {
    myds_v41_context_t *g=ctx->v41;
    const uint32_t ratio=ratio_for(il), owner=owner_for(il);
    const uint32_t n_comp_all=(base_pos + c_start + c_len)/ratio;
    const uint32_t mask_stride=(uint32_t)(mask_bytes/4u);
    const uint32_t batch_floats=myds_v41_gpu_indexer_batch_floats(g->gpu);
    if (!ratio || !index_source(il) || !c_len) return 0;
    /* 逐 token 路径里 count==0 的 token 整段跳过（连 rope/quant 都不做）；只有
     * 「序列第一个 token + ratio==2 的层」会出现。批量路径没有对应「整行不写」的语义
     * （因果宽度 0 会让 merge 的无符号减法下溢），所以直接回退。 */
    if (!n_comp_all || (base_pos + c_start + 1u)/ratio == 0u) return 0;
    if (V41_INDEX_HEADS != 32u) return 0;
    if (n_comp_all > batch_floats) return 0;
    if (mask_bytes%4u || !mask_stride) return 0;
    /* 压缩键的【表宽】：不足 V41_TOPK 时补到 V41_TOPK（多出来的列由因果宽度挡成
     * -INF，既多算了什么也不选），这样 top == V41_TOPK == prompt_selected 的行步长。
     * 缓存里必须真的有 V41_TOPK 行，否则按逐 token 回退（缓存行数 = 位置容量/ratio）。 */
    const uint32_t cache_rows=(uint32_t)(myds_metal_buffer_size(g->index_cache[owner])/(128u*V41_CACHE_BYTES));
    if (n_comp_all<V41_TOPK && cache_rows<V41_TOPK) return 0;
    myds_metal_buffer_t *scores_buf=myds_v41_gpu_indexer_scores(g->gpu);
    myds_metal_buffer_t *blocks_buf=myds_v41_gpu_indexer_blocks(g->gpu);
    myds_metal_buffer_t *block_sel_buf=myds_v41_gpu_indexer_block_selected(g->gpu);
    if (!scores_buf || !blocks_buf || !block_sel_buf) return 0;
    if (!q_view || !w_view || !sel_view || !mask_view) return 0;
    /* 预检：先把子组划分算完（不做任何 dispatch）。rope/quant 是原地改写，一旦开始
     * 就不能再回退重做，所以放不下必须在这里发现。 */
    for (uint32_t off=0; off<c_len; ) {
        uint32_t glen=indexer_group_len(c_len,off,base_pos,c_start,ratio,batch_floats);
        if (!glen) return 0;
        off+=glen;
    }
    /* 1) index_q 的 rope + quant：逐 token 版本在 publish_and_select 里逐行做，参数
     *    完全相同（start_pos = 该位置、heads = 32、width = 128、compressed = true）。 */
    if (!myds_v41_gpu_rope_batch(index_q_buf,128,V41_INDEX_HEADS,base_pos + c_start,c_len,true,false)) return -1;
    if (!myds_v41_gpu_quant(index_q_buf,128,V41_INDEX_HEADS*c_len,2)) return -1;
    for (uint32_t off=0; off<c_len; ) {
        const uint32_t glen=indexer_group_len(c_len,off,base_pos,c_start,ratio,batch_floats);
        const uint32_t nc_raw=(base_pos + c_start + off + glen)/ratio;
        const uint32_t nc=nc_raw<V41_TOPK?V41_TOPK:nc_raw;   /* 表宽（见上） */
        const uint32_t g_start=base_pos + c_start + off;
        if (!glen || !nc_raw) return -1;
        if (!myds_metal_buffer_update_sub_view(q_view,index_q_buf,
                (size_t)off*V41_INDEX_HEADS*128u*sizeof(float),
                (size_t)glen*V41_INDEX_HEADS*128u*sizeof(float)) ||
            !myds_metal_buffer_update_sub_view(w_view,index_weights_buf,
                (size_t)off*V41_INDEX_HEADS*sizeof(float),
                (size_t)glen*V41_INDEX_HEADS*sizeof(float)) ||
            !myds_metal_buffer_update_sub_view(sel_view,prompt_selected,
                (size_t)(c_start+off)*sel_bytes,(size_t)glen*sel_bytes) ||
            !myds_metal_buffer_update_sub_view(mask_view,prompt_block_mask,
                (size_t)(c_start+off)*mask_bytes,(size_t)glen*mask_bytes)) return -1;
        /* 分数表：glen 行 × nc 列，行主序。 */
        if (!myds_v41_gpu_indexscores_chunk_bf16(scores_buf,q_view,w_view,g->index_cache[owner],
                nc,V41_INDEX_HEADS,glen,g_start,ratio)) return -1;
        if (il==20) {
            /* 块掩码：只要本行的块【全部被选中】，掩码前缀就与逐 token 一致
             * （前 blocks_t 列先清成不可见、再被 scatter 打开）。这里 nc ≤ 16384
             * ⇒ blocks ≤ 2048 ⇒ top = blocks = 全选，行内顺序不影响结果。 */
            const uint32_t blocks=(nc+7u)/8u;
            const uint32_t top_blocks=block_budget_cap(blocks<2048u?blocks:2048u);
            if (!myds_v41_gpu_candidates_rows(blocks_buf,scores_buf,nc,glen,g_start,ratio,mask_stride)) return -1;
            if (!myds_v41_gpu_topk_causal(block_sel_buf,blocks_buf,blocks,top_blocks,glen,
                    g_start,ratio,g->gpu)) return -1;
            if (!myds_v41_gpu_topkmask_rows(mask_view,block_sel_buf,blocks,top_blocks,glen,mask_stride)) return -1;
        } else if (il>20) {
            if (!myds_v41_gpu_filter_rows(scores_buf,mask_view,nc,glen,g_start,ratio,mask_stride)) return -1;
        }
        /* 最终选择：top 取全局 V41_TOPK（≤ nc），逐行宽度由因果参数限制到 count。 */
        if (!myds_v41_gpu_topk_causal(sel_view,scores_buf,nc,nc<V41_TOPK?nc:V41_TOPK,
                glen,g_start,ratio,g->gpu)) return -1;
        off+=glen;
    }
    return 1;
}

static inline bool check_logits_finite(const float *logits, uint32_t n) {
#if defined(__ARM_NEON)
    uint32_t i = 0;
    const uint32x4_t mask = vdupq_n_u32(0x7F800000u);
    for (; i + 16 <= n; i += 16) {
        uint32x4_t v0 = vld1q_u32((const uint32_t *)(logits + i));
        uint32x4_t v1 = vld1q_u32((const uint32_t *)(logits + i + 4));
        uint32x4_t v2 = vld1q_u32((const uint32_t *)(logits + i + 8));
        uint32x4_t v3 = vld1q_u32((const uint32_t *)(logits + i + 12));
        uint32x4_t c0 = vceqq_u32(vandq_u32(v0, mask), mask);
        uint32x4_t c1 = vceqq_u32(vandq_u32(v1, mask), mask);
        uint32x4_t c2 = vceqq_u32(vandq_u32(v2, mask), mask);
        uint32x4_t c3 = vceqq_u32(vandq_u32(v3, mask), mask);
        uint32x4_t any = vorrq_u32(vorrq_u32(c0, c1), vorrq_u32(c2, c3));
        if (__builtin_expect(vmaxvq_u32(any) != 0, 0)) return false;
    }
    for (; i < n; i++) {
        if (!isfinite(logits[i])) return false;
    }
    return true;
#else
    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(logits[i])) return false;
    }
    return true;
#endif
}

int32_t myds_v41_forward(myds_context_t *ctx, int32_t token, uint32_t pos, bool need_logits) {
    myds_model_t *m=ctx->model;
    myds_v41_context_t *g=ctx->v41;
    myds_v41_model_t *v=m->v41;
    uint32_t il=0;
    const char *phase="input";
    if (!g || token<0 || (uint32_t)token>=m->vocab_size || pos>=g->capacity ||
        pos>=ctx->cfg.max_seq_len) return -1;
    if (!myds_metal_commit_batch()) return -1;
    if (pos==0) { g->next_position=0; g->valid=true; myds_engram_reset(g->engram); }
    if (!g->valid || pos!=g->next_position) {
        MYDS_LOG_E("V4.1 expected consecutive token position %u, received %u",g->next_position,pos);
        return -1;
    }
    const bool trace_on = myds_trace_active();
    /* A：decode 分段提交（MYDS_DECODE_SPLIT=N，N 段；未设/0/1 = 关 = 与以前完全一致）。
     * 在层边界把已编码的部分 commit（不等），GPU 先跑前面的层，CPU 同时编码后面的层。
     * 内核与顺序不变；缓冲是默认 tracked ⇒ 同一 queue 上跨 command buffer 的读写顺序由 Metal 保证。
     * 上限约 7%（2026-10-04 powermetrics：GPU 已 93% 活跃）。trace 时不分段。 */
    static int decode_split = -1;
    if (decode_split < 0) {
        const char *e = getenv("MYDS_DECODE_SPLIT");
        const int n = e ? atoi(e) : 0;
        decode_split = (n > 1 && n <= (int)V41_LAYERS) ? n : 0;
    }
#define CHECK(x) do { if (!(x)) goto fail; } while (0)
#define TRACE(stage,buf,n) do { if (trace_on) CHECK(myds_trace_f32(stage,il,pos,buf,n)); } while (0)
    CHECK(myds_engram_prepare_token(g->engram,token,pos));
    for (uint32_t i=0;i<2;i++) CHECK(myds_metal_buffer_write(g->engram_rows[i],0,
        g->engram->row_values[i],MYDS_ENGRAM_INPUT_DIM*4u));
    float *res=myds_metal_buffer_data(g->residual);
    if (v41_image_embedding_at(ctx,pos,res,V41_DIM)) {
        /* 图片块：该位置用注入向量（已填进 res） */
    } else {
        const uint16_t *row=(const uint16_t *)myds_metal_buffer_data(m->tok_embeddings)+(size_t)token*V41_DIM;
        for (uint32_t j=0;j<V41_DIM;j++) res[j]=myds_fp16_to_fp32(row[j]);
    }
    for (uint32_t j=1;j<4;j++) memcpy(res+j*V41_DIM,res,V41_DIM*4u);
    const float initial_pre[4]={1,0,0,0};
    CHECK(myds_metal_buffer_write(g->pre,0,initial_pre,sizeof(initial_pre)));
    TRACE("embedding",g->residual,4*V41_DIM);
    CHECK(myds_metal_begin_batch());
    for (il=0;il<V41_LAYERS;il++) {
        const myds_layer_t *base=&m->layers[il];
        const v41_layer_t *l=&v->layers[il];
        const bool compressed=ratio_for(il)!=0;
        phase="HC attention input";
        if (il==1 || il==14) {
            uint32_t e=il==1?0:1;
            CHECK(project(g->engram_kv,v->engram_proj[e],g->engram_rows[e],5*V41_DIM,MYDS_ENGRAM_INPUT_DIM,GGML_TYPE_F16,true));
            CHECK(myds_metal_v41_engram_add(g->residual,g->engram_kv,v->engram_q[e],v->engram_k[e],V41_DIM,m->norm_eps));
            TRACE("engram_out",g->residual,4*V41_DIM);
        }
        CHECK(myds_metal_v41_hc_mix(g->attn_split,g->residual,l->attn_fn,GGML_TYPE_F16,l->attn_scale,l->attn_base,g->flat_norm,g->mix,V41_DIM,m->norm_eps,v->hc_eps));
        TRACE("hc_attn_mix",g->mix,24); TRACE("hc_attn_split",g->attn_split,24);
        const myds_metal_buffer_t *hc_in_pre = (il == 0 ? g->pre : g->ffn_split);
        TRACE("hc_attn_pre",hc_in_pre,4);
        CHECK(myds_metal_v41_hc_weighted_sum(ctx->x,g->residual,hc_in_pre,V41_DIM));
        TRACE("hc_attn_input",ctx->x,V41_DIM);
        CHECK(myds_metal_v41_rmsnorm(ctx->x_norm,ctx->x,base->attn_norm,V41_DIM,m->norm_eps,true));
        TRACE("attn_norm",ctx->x_norm,V41_DIM);
        phase="attention projections";
        CHECK(project_decode_qkv(ctx->qr,base->attn_q_a,ctx->x_norm,1280,V41_DIM,base->attn_q_a_type,trace_on));
        TRACE("q_a_raw",ctx->qr,1280);
        CHECK(myds_metal_v41_rmsnorm(ctx->qr_norm,ctx->qr,base->attn_q_a_norm,1280,m->norm_eps,true));
        TRACE("q_a_norm",ctx->qr_norm,1280);
        CHECK(project_decode_qkv(ctx->q,base->attn_q_b,ctx->qr_norm,V41_HEADS*512,1280,base->attn_q_b_type,trace_on));
        TRACE("q_b",ctx->q,V41_HEADS*512);
        CHECK(project_decode_qkv(ctx->kv_act,base->attn_kv,ctx->x_norm,512,V41_DIM,base->attn_kv_type,trace_on));
        TRACE("kv_raw",ctx->kv_act,512);
        CHECK(myds_metal_v41_rmsnorm(ctx->kv_act,ctx->kv_act,base->attn_kv_a_norm,512,m->norm_eps,true));
        TRACE("kv_norm",ctx->kv_act,512);
        static int qkv_post_enabled=-1;
        if(!trace_on&&qkv_fusion_enabled("MYDS_QKV_FUSE_POST",&qkv_post_enabled)) {
            CHECK(myds_v41_gpu_qkv_finish(ctx->q,ctx->kv_act,g->window[il],pos,compressed));
        } else {
            CHECK(myds_v41_gpu_rope(ctx->q,512,V41_HEADS,pos,compressed,false));
            CHECK(myds_v41_gpu_rope(ctx->kv_act,512,1,pos,compressed,false));
            TRACE("kv_rope",ctx->kv_act,512); TRACE("q_rope",ctx->q,V41_HEADS*512);
            CHECK(myds_v41_gpu_quant(ctx->kv_act,512,1,1));
            TRACE("kv_quant",ctx->kv_act,512);
            CHECK(myds_metal_buffer_copy(g->window[il],(size_t)(pos%128)*512*4u,ctx->kv_act,0,512*4u));
        }
        phase="compressed KV / indexer";
        CHECK(publish_and_select(ctx,il,pos,ctx->x_norm,ctx->qr_norm,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
        uint32_t count=compressed?(pos+1u)/ratio_for(il):0;
        uint32_t top=count<V41_TOPK?count:V41_TOPK;
        uint32_t nraw=pos+1u<128?pos+1u:128;
        phase="attention reduction";
        static int attn_direct_enabled = -1;
        if (attn_direct_enabled < 0) {
            const char *env = getenv("MYDS_ATTN_DIRECT_PACK");
            attn_direct_enabled = (env == NULL || atoi(env) != 0);
        }
        if (attn_direct_enabled) {
            CHECK(myds_v41_gpu_attention_direct(ctx->attn_heads,base->attn_sinks,ctx->q,
                                                g->window[il],nraw,(pos+1u-nraw)%128,
                                                count ? g->compressed[owner_for(il)] : NULL,
                                                count ? g->selected : NULL,
                                                count,top,g->gpu));
        } else {
            if (count) CHECK(myds_v41_gpu_gather_bf16(g->selected_kv,g->compressed[owner_for(il)],g->selected,count,top));
            CHECK(myds_v41_gpu_attention(ctx->attn_heads,base->attn_sinks,ctx->q,g->window[il],nraw,(pos+1u-nraw)%128,g->selected_kv,top,g->gpu));
        }
        CHECK(myds_v41_gpu_rope_round(ctx->attn_heads,512,V41_HEADS,pos,compressed,true));
        CHECK(myds_metal_attn_output_a_bf16(ctx->attn_low,base->attn_output_a,ctx->attn_heads,8,4096,1024,base->attn_output_a_type));
        CHECK(project_decode_tail(ctx->attn_out,base->attn_output_b,ctx->attn_low,V41_DIM,8192,base->attn_output_b_type,trace_on));
        TRACE("attn_out",ctx->attn_out,V41_DIM);
        phase="HC FFN input";
        CHECK(myds_metal_v41_hc_expand(g->after_attn,ctx->attn_out,g->residual,g->attn_split,V41_DIM));
        TRACE("after_attn",g->after_attn,4*V41_DIM);
        CHECK(myds_metal_v41_hc_mix(g->ffn_split,g->after_attn,l->ffn_fn,GGML_TYPE_F16,l->ffn_scale,l->ffn_base,g->flat_norm,g->mix,V41_DIM,m->norm_eps,v->hc_eps));
        TRACE("hc_ffn_mix",g->mix,24); TRACE("hc_ffn_split",g->ffn_split,24);
        CHECK(myds_metal_v41_hc_weighted_sum(ctx->x,g->after_attn,g->attn_split,V41_DIM));
        TRACE("hc_ffn_input",ctx->x,V41_DIM);
        CHECK(myds_metal_v41_rmsnorm(ctx->ffn_norm_out,ctx->x,base->ffn_norm,V41_DIM,m->norm_eps,true));
        TRACE("ffn_norm",ctx->ffn_norm_out,V41_DIM);
        phase="MoE";
        CHECK(myds_v41_moe(ctx,il,(uint32_t)token,pos));
        CHECK(myds_metal_v41_hc_expand(g->residual,ctx->ffn_out,g->after_attn,g->ffn_split,V41_DIM));
        TRACE("layer_out",g->residual,4*V41_DIM);
        if (decode_split && !trace_on && il + 1u < V41_LAYERS &&
            ((il + 1u) * (uint32_t)decode_split) / V41_LAYERS != (il * (uint32_t)decode_split) / V41_LAYERS) {
            CHECK(myds_metal_flush_batch());
            CHECK(myds_metal_begin_batch());
        }
    }
    phase="output head";
    if (need_logits) {
        CHECK(myds_metal_v41_hc_weighted_sum(ctx->x,g->residual,g->ffn_split,V41_DIM));
        CHECK(myds_metal_v41_rmsnorm(ctx->x_norm,ctx->x,m->output_norm,V41_DIM,m->norm_eps,true));
        CHECK(project(ctx->logits,m->output,ctx->x_norm,m->vocab_size,V41_DIM,m->output_type,false));
        CHECK(myds_metal_commit_batch());
        if (trace_on) CHECK(myds_trace_f32("logits",0,pos,ctx->logits,m->vocab_size));
        const float *logits=myds_metal_buffer_data(ctx->logits);
        if (!check_logits_finite(logits, m->vocab_size)) goto fail;
    } else {
        CHECK(myds_metal_commit_batch());
    }
    g->next_position=pos+1u;
    return need_logits ? myds_sample_next_token(myds_metal_buffer_data(ctx->logits),m->vocab_size,&ctx->cfg,ctx,NULL,0) : 0;
fail:
    MYDS_LOG_E("V4.1 native graph failed at pos %u layer %u: %s",pos,il,phase);
    (void)myds_metal_commit_batch(); g->valid=false; return -1;
#undef CHECK
#undef TRACE
}

struct v41_chunk_workspace_s {
    uint32_t capacity;
    myds_metal_buffer_t *x_buf;
    myds_metal_buffer_t *x_norm_buf;
    myds_metal_buffer_t *qr_buf;
    myds_metal_buffer_t *qr_norm_buf;
    myds_metal_buffer_t *q_buf;
    myds_metal_buffer_t *kv_act_buf;
    myds_metal_buffer_t *attn_heads_buf;
    /* B 方案：每 query 640 个可见键的紧凑段（f16）。只在 MYDS_ATTN_GATHER=1 时分配，
     * 否则为 NULL（旧路径）。尺寸 = C * 640 * 512 * 2 字节（chunk 512 ⇒ 335 MB）。 */
    myds_metal_buffer_t *kv_gathered;
    myds_metal_buffer_t *attn_low_buf;
    myds_metal_buffer_t *attn_out_buf;
    myds_metal_buffer_t *after_attn_buf;
    myds_metal_buffer_t *attn_split_buf;
    myds_metal_buffer_t *ffn_split_buf;
    myds_metal_buffer_t *ffn_norm_buf;
    myds_metal_buffer_t *router_logits_buf;
    myds_metal_buffer_t *shared_gate_buf;
    myds_metal_buffer_t *shared_up_buf;
    myds_metal_buffer_t *shared_swiglu_buf;
    myds_metal_buffer_t *shared_out_buf;
    myds_metal_buffer_t *engram_rows_buf[2];
    myds_metal_buffer_t *engram_proj_buf[2];
    myds_metal_buffer_t *selected_ids_buf;
    myds_metal_buffer_t *selected_weights_buf;
    myds_metal_buffer_t *route_probs_buf;
    myds_metal_buffer_t *route_scores_buf;
    myds_metal_buffer_t *route_ids_f32_buf;
    myds_metal_buffer_t *pool_kv_buf;
    myds_metal_buffer_t *pool_score_buf;
    /* 压缩器批量化（publish_compressor_chunk）的临时缓冲：latent 的 512 维行与
     * index_k 的 128 维行，各按 chunk 内【提交行数】计（ratio==1 是 c_len，ratio==2 是 c_len/2）。 */
    myds_metal_buffer_t *comp_latent_buf;
    myds_metal_buffer_t *comp_index_k_buf;
    myds_metal_buffer_t *index_q_buf;
    myds_metal_buffer_t *index_weights_buf;
    myds_metal_buffer_t *moe_gate_buf;
    myds_metal_buffer_t *moe_up_buf;
    myds_metal_buffer_t *moe_mid_buf;
    myds_metal_buffer_t *moe_down_buf;
    myds_metal_buffer_t *moe_out_buf;
    myds_metal_buffer_t *ffn_out_buf;
    myds_metal_buffer_t *layer_out_buf;

    /* MoE 的批大小（token 数）可以与 attention 的 chunk 解耦：attention 仍按 chunk 跑
     * （数值路径完全不变），只把各 chunk 的 MoE 输入累积起来、攒够再跑一次 MoE ——
     * 因为 MoE 每层都要读【全部 n_routed_experts 个 expert】的权重（~3.8 GB），
     * 这份开销与 token 数无关、只与 MoE 被调用的【次数】有关。
     * moe_batch == capacity 时，下面这些指针就是原缓冲本身（零拷贝，与旧行为一致）。 */
    uint32_t moe_batch;
    myds_metal_buffer_t *moe_norm_acc, *moe_ids_acc, *moe_w_acc;
    myds_metal_buffer_t *moe_shared_acc, *moe_after_acc, *moe_split_acc, *moe_out_acc;

    myds_metal_buffer_t *x[V41_MAX_CHUNK];
    myds_metal_buffer_t *x_norm[V41_MAX_CHUNK];
    myds_metal_buffer_t *qr[V41_MAX_CHUNK];
    myds_metal_buffer_t *qr_norm[V41_MAX_CHUNK];
    myds_metal_buffer_t *q[V41_MAX_CHUNK];
    myds_metal_buffer_t *kv_act[V41_MAX_CHUNK];
    myds_metal_buffer_t *attn_out[V41_MAX_CHUNK];
    myds_metal_buffer_t *after_attn[V41_MAX_CHUNK];
    myds_metal_buffer_t *attn_split[V41_MAX_CHUNK];
    myds_metal_buffer_t *ffn_split[V41_MAX_CHUNK];
    myds_metal_buffer_t *ffn_norm[V41_MAX_CHUNK];
    myds_metal_buffer_t *router_logits[V41_MAX_CHUNK];
    myds_metal_buffer_t *shared_out[V41_MAX_CHUNK];
    myds_metal_buffer_t *engram_rows[2][V41_MAX_CHUNK];
    /* engram 投影的整 chunk 输出（每个 engram 层一个缓冲），以及每 token 的视图。
     * 以前这里只有单个 token 的 g->engram_kv，逐 token 调用把 [25600 x 6144] 的
     * F16 权重（314 MB）整个读一遍 —— 每个 token 一遍。 */
    myds_metal_buffer_t *engram_proj_view[2][V41_MAX_CHUNK];
    myds_metal_buffer_t *selected_ids[V41_MAX_CHUNK];
    myds_metal_buffer_t *selected_weights[V41_MAX_CHUNK];
    myds_metal_buffer_t *route_probs[V41_MAX_CHUNK];
    myds_metal_buffer_t *route_scores[V41_MAX_CHUNK];
    myds_metal_buffer_t *route_ids_f32[V41_MAX_CHUNK];
    myds_metal_buffer_t *pool_kv[V41_MAX_CHUNK];
    myds_metal_buffer_t *pool_score[V41_MAX_CHUNK];
    myds_metal_buffer_t *index_q[V41_MAX_CHUNK];
    myds_metal_buffer_t *index_weights[V41_MAX_CHUNK];
    myds_metal_buffer_t *moe_gate[V41_MAX_CHUNK];
    myds_metal_buffer_t *moe_up[V41_MAX_CHUNK];
    myds_metal_buffer_t *moe_mid[V41_MAX_CHUNK];
    myds_metal_buffer_t *moe_down[V41_MAX_CHUNK];
    myds_metal_buffer_t *moe_out[V41_MAX_CHUNK];
    myds_metal_buffer_t *ffn_out[V41_MAX_CHUNK];
    myds_metal_buffer_t *layer_out[V41_MAX_CHUNK];

    myds_metal_buffer_t *res_view[V41_MAX_CHUNK];
    myds_metal_buffer_t *pre_view[V41_MAX_CHUNK];
    myds_metal_buffer_t *cur_res_view;
    myds_metal_buffer_t *cur_pre_view;
    myds_metal_buffer_t *cur_selected_view;
    myds_metal_buffer_t *cur_mask_view;
    /* 批量 indexer 用的分组视图（整 chunk 缓冲 / 提示缓冲按子组切片） */
    myds_metal_buffer_t *cur_index_q_view;
    myds_metal_buffer_t *cur_index_weights_view;
    myds_metal_buffer_t *moe_view;
    myds_metal_buffer_t *shared_view;
    myds_metal_buffer_t *after_view;
    myds_metal_buffer_t *split_view;
    myds_metal_buffer_t *hc_flat_norm_buf;
    myds_metal_buffer_t *hc_mix_buf;
};

static void v41_chunk_workspace_free(v41_chunk_workspace_t *cw) {
    if (!cw) return;
    for (uint32_t t = 0; t < cw->capacity; t++) {
        if (cw->x[t]) myds_metal_buffer_free(cw->x[t]);
        if (cw->x_norm[t]) myds_metal_buffer_free(cw->x_norm[t]);
        if (cw->qr[t]) myds_metal_buffer_free(cw->qr[t]);
        if (cw->qr_norm[t]) myds_metal_buffer_free(cw->qr_norm[t]);
        if (cw->q[t]) myds_metal_buffer_free(cw->q[t]);
        if (cw->kv_act[t]) myds_metal_buffer_free(cw->kv_act[t]);
        if (cw->attn_out[t]) myds_metal_buffer_free(cw->attn_out[t]);
        if (cw->after_attn[t]) myds_metal_buffer_free(cw->after_attn[t]);
        if (cw->attn_split[t]) myds_metal_buffer_free(cw->attn_split[t]);
        if (cw->ffn_split[t]) myds_metal_buffer_free(cw->ffn_split[t]);
        if (cw->ffn_norm[t]) myds_metal_buffer_free(cw->ffn_norm[t]);
        if (cw->router_logits[t]) myds_metal_buffer_free(cw->router_logits[t]);
        if (cw->shared_out[t]) myds_metal_buffer_free(cw->shared_out[t]);
        if (cw->engram_rows[0][t]) myds_metal_buffer_free(cw->engram_rows[0][t]);
        if (cw->engram_rows[1][t]) myds_metal_buffer_free(cw->engram_rows[1][t]);
        if (cw->engram_proj_view[0][t]) myds_metal_buffer_free(cw->engram_proj_view[0][t]);
        if (cw->engram_proj_view[1][t]) myds_metal_buffer_free(cw->engram_proj_view[1][t]);
        if (cw->selected_ids[t]) myds_metal_buffer_free(cw->selected_ids[t]);
        if (cw->selected_weights[t]) myds_metal_buffer_free(cw->selected_weights[t]);
        if (cw->route_probs[t]) myds_metal_buffer_free(cw->route_probs[t]);
        if (cw->route_scores[t]) myds_metal_buffer_free(cw->route_scores[t]);
        if (cw->route_ids_f32[t]) myds_metal_buffer_free(cw->route_ids_f32[t]);
        if (cw->pool_kv[t]) myds_metal_buffer_free(cw->pool_kv[t]);
        if (cw->pool_score[t]) myds_metal_buffer_free(cw->pool_score[t]);
        if (cw->index_q[t]) myds_metal_buffer_free(cw->index_q[t]);
        if (cw->index_weights[t]) myds_metal_buffer_free(cw->index_weights[t]);
        if (cw->moe_gate[t]) myds_metal_buffer_free(cw->moe_gate[t]);
        if (cw->moe_up[t]) myds_metal_buffer_free(cw->moe_up[t]);
        if (cw->moe_mid[t]) myds_metal_buffer_free(cw->moe_mid[t]);
        if (cw->moe_down[t]) myds_metal_buffer_free(cw->moe_down[t]);
        if (cw->moe_out[t]) myds_metal_buffer_free(cw->moe_out[t]);
        if (cw->ffn_out[t]) myds_metal_buffer_free(cw->ffn_out[t]);
        if (cw->layer_out[t]) myds_metal_buffer_free(cw->layer_out[t]);
        if (cw->res_view[t]) myds_metal_buffer_free(cw->res_view[t]);
        if (cw->pre_view[t]) myds_metal_buffer_free(cw->pre_view[t]);
    }
    if (cw->x_buf) myds_metal_buffer_free(cw->x_buf);
    if (cw->x_norm_buf) myds_metal_buffer_free(cw->x_norm_buf);
    if (cw->qr_buf) myds_metal_buffer_free(cw->qr_buf);
    if (cw->qr_norm_buf) myds_metal_buffer_free(cw->qr_norm_buf);
    if (cw->q_buf) myds_metal_buffer_free(cw->q_buf);
    if (cw->kv_act_buf) myds_metal_buffer_free(cw->kv_act_buf);
    if (cw->attn_heads_buf) myds_metal_buffer_free(cw->attn_heads_buf);
    if (cw->attn_low_buf) myds_metal_buffer_free(cw->attn_low_buf);
    if (cw->attn_out_buf) myds_metal_buffer_free(cw->attn_out_buf);
    if (cw->after_attn_buf) myds_metal_buffer_free(cw->after_attn_buf);
    if (cw->attn_split_buf) myds_metal_buffer_free(cw->attn_split_buf);
    if (cw->ffn_split_buf) myds_metal_buffer_free(cw->ffn_split_buf);
    if (cw->ffn_norm_buf) myds_metal_buffer_free(cw->ffn_norm_buf);
    if (cw->router_logits_buf) myds_metal_buffer_free(cw->router_logits_buf);
    if (cw->shared_gate_buf) myds_metal_buffer_free(cw->shared_gate_buf);
    if (cw->shared_up_buf) myds_metal_buffer_free(cw->shared_up_buf);
    if (cw->shared_swiglu_buf) myds_metal_buffer_free(cw->shared_swiglu_buf);
    if (cw->shared_out_buf) myds_metal_buffer_free(cw->shared_out_buf);
    if (cw->engram_rows_buf[0]) myds_metal_buffer_free(cw->engram_rows_buf[0]);
    if (cw->engram_rows_buf[1]) myds_metal_buffer_free(cw->engram_rows_buf[1]);
    if (cw->engram_proj_buf[0]) myds_metal_buffer_free(cw->engram_proj_buf[0]);
    if (cw->engram_proj_buf[1]) myds_metal_buffer_free(cw->engram_proj_buf[1]);
    if (cw->selected_ids_buf) myds_metal_buffer_free(cw->selected_ids_buf);
    if (cw->selected_weights_buf) myds_metal_buffer_free(cw->selected_weights_buf);
    if (cw->route_probs_buf) myds_metal_buffer_free(cw->route_probs_buf);
    if (cw->route_scores_buf) myds_metal_buffer_free(cw->route_scores_buf);
    if (cw->route_ids_f32_buf) myds_metal_buffer_free(cw->route_ids_f32_buf);
    if (cw->pool_kv_buf) myds_metal_buffer_free(cw->pool_kv_buf);
    if (cw->pool_score_buf) myds_metal_buffer_free(cw->pool_score_buf);
    if (cw->comp_latent_buf) myds_metal_buffer_free(cw->comp_latent_buf);
    if (cw->comp_index_k_buf) myds_metal_buffer_free(cw->comp_index_k_buf);
    if (cw->index_q_buf) myds_metal_buffer_free(cw->index_q_buf);
    if (cw->index_weights_buf) myds_metal_buffer_free(cw->index_weights_buf);
    if (cw->moe_gate_buf) myds_metal_buffer_free(cw->moe_gate_buf);
    if (cw->moe_up_buf) myds_metal_buffer_free(cw->moe_up_buf);
    if (cw->moe_mid_buf) myds_metal_buffer_free(cw->moe_mid_buf);
    if (cw->moe_down_buf) myds_metal_buffer_free(cw->moe_down_buf);
    if (cw->moe_out_buf) myds_metal_buffer_free(cw->moe_out_buf);
    if (cw->ffn_out_buf) myds_metal_buffer_free(cw->ffn_out_buf);
    if (cw->layer_out_buf) myds_metal_buffer_free(cw->layer_out_buf);
    if (cw->cur_res_view) myds_metal_buffer_free(cw->cur_res_view);
    if (cw->cur_pre_view) myds_metal_buffer_free(cw->cur_pre_view);
    if (cw->cur_selected_view) myds_metal_buffer_free(cw->cur_selected_view);
    if (cw->cur_mask_view) myds_metal_buffer_free(cw->cur_mask_view);
    if (cw->cur_index_q_view) myds_metal_buffer_free(cw->cur_index_q_view);
    if (cw->cur_index_weights_view) myds_metal_buffer_free(cw->cur_index_weights_view);
    if (cw->moe_view) myds_metal_buffer_free(cw->moe_view);
    if (cw->shared_view) myds_metal_buffer_free(cw->shared_view);
    if (cw->after_view) myds_metal_buffer_free(cw->after_view);
    if (cw->split_view) myds_metal_buffer_free(cw->split_view);
    if (cw->hc_flat_norm_buf) myds_metal_buffer_free(cw->hc_flat_norm_buf);
    if (cw->hc_mix_buf) myds_metal_buffer_free(cw->hc_mix_buf);
    if (cw->moe_batch > cw->capacity) {   /* 别名时不能重复释放 */
        if (cw->moe_norm_acc) myds_metal_buffer_free(cw->moe_norm_acc);
        if (cw->moe_ids_acc) myds_metal_buffer_free(cw->moe_ids_acc);
        if (cw->moe_w_acc) myds_metal_buffer_free(cw->moe_w_acc);
        if (cw->moe_shared_acc) myds_metal_buffer_free(cw->moe_shared_acc);
        if (cw->moe_after_acc) myds_metal_buffer_free(cw->moe_after_acc);
        if (cw->moe_split_acc) myds_metal_buffer_free(cw->moe_split_acc);
        if (cw->moe_out_acc) myds_metal_buffer_free(cw->moe_out_acc);
    }
    free(cw);
}

static v41_chunk_workspace_t *v41_chunk_workspace_create(uint32_t capacity, uint32_t exp_dim,
                                                         uint32_t moe_batch_tokens) {
    if (!capacity || capacity > V41_MAX_CHUNK) return NULL;
    v41_chunk_workspace_t *cw = calloc(1, sizeof(*cw));
    if (!cw) return NULL;
    cw->capacity = capacity;
    size_t C = capacity;
    /* MoE 的批容量（token 数）—— 至少一个 chunk。调用方按 MYDS_MOE_CHUNKS 与 prompt
     * 长度给出；只影响 MoE 的调用次数，不影响 attention 的数值。 */
    size_t B = (moe_batch_tokens > capacity) ? (size_t)moe_batch_tokens : C;
    cw->moe_batch = (uint32_t)B;

#define ALLOC_BUF(name, bytes) do { cw->name = myds_metal_buffer_alloc(bytes); if (!cw->name) goto fail; } while(0)
    ALLOC_BUF(x_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(x_norm_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(qr_buf, C * 1280 * sizeof(float));
    ALLOC_BUF(qr_norm_buf, C * 1280 * sizeof(float));
    ALLOC_BUF(q_buf, C * V41_HEADS * 512 * sizeof(float));
    ALLOC_BUF(kv_act_buf, C * 512 * sizeof(float));
    ALLOC_BUF(attn_heads_buf, C * V41_HEADS * 512 * sizeof(float));
    /* B 方案（gather）默认开：仅当显式 MYDS_ATTN_GATHER=0 时跳过分配。 */
    {
        const char *gv = getenv("MYDS_ATTN_GATHER");
        if (!(gv && atoi(gv) == 0))
            ALLOC_BUF(kv_gathered, C * 640u * 512u * sizeof(uint16_t));
    }
    ALLOC_BUF(attn_low_buf, C * 8192 * sizeof(float));
    ALLOC_BUF(attn_out_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(after_attn_buf, C * 4 * V41_DIM * sizeof(float));
    ALLOC_BUF(attn_split_buf, C * 24 * sizeof(float));
    ALLOC_BUF(ffn_split_buf, C * 24 * sizeof(float));
    ALLOC_BUF(ffn_norm_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(router_logits_buf, C * 384 * sizeof(float));
    ALLOC_BUF(shared_gate_buf, C * exp_dim * sizeof(float));
    ALLOC_BUF(shared_up_buf, C * exp_dim * sizeof(float));
    ALLOC_BUF(shared_swiglu_buf, C * exp_dim * sizeof(float));
    ALLOC_BUF(shared_out_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(engram_rows_buf[0], C * MYDS_ENGRAM_INPUT_DIM * sizeof(float));
    ALLOC_BUF(engram_rows_buf[1], C * MYDS_ENGRAM_INPUT_DIM * sizeof(float));
    ALLOC_BUF(engram_proj_buf[0], (size_t)C * 5 * V41_DIM * sizeof(float));
    ALLOC_BUF(engram_proj_buf[1], (size_t)C * 5 * V41_DIM * sizeof(float));
    ALLOC_BUF(selected_ids_buf, C * 6 * sizeof(int32_t));
    ALLOC_BUF(selected_weights_buf, C * 6 * sizeof(float));
    ALLOC_BUF(route_probs_buf, C * 384 * sizeof(float));
    ALLOC_BUF(route_scores_buf, C * 384 * sizeof(float));
    ALLOC_BUF(route_ids_f32_buf, C * 6 * sizeof(float));
    ALLOC_BUF(pool_kv_buf, C * 512 * sizeof(float));
    ALLOC_BUF(pool_score_buf, C * 512 * sizeof(float));
    ALLOC_BUF(comp_latent_buf, C * 512 * sizeof(float));
    ALLOC_BUF(comp_index_k_buf, C * 128 * sizeof(float));
    /* 复用注意力输出缓冲：attn_heads_buf (C*32768*4) 在 indexer 完成后才由 attention_chunk 写入，
     * 生命周期与 index_q_buf (C*4096*4) 完全正交，直接作为 sub_view 复用，省去 32 MiB 独占显存。 */
    cw->index_q_buf = myds_metal_buffer_sub_view(cw->attn_heads_buf, 0, C * V41_INDEX_HEADS * 128 * sizeof(float));
    if (!cw->index_q_buf) goto fail;
    ALLOC_BUF(index_weights_buf, C * V41_INDEX_HEADS * sizeof(float));
    ALLOC_BUF(moe_gate_buf, B * 6 * exp_dim * sizeof(float));
    ALLOC_BUF(moe_up_buf, B * 6 * exp_dim * sizeof(float));
    ALLOC_BUF(moe_mid_buf, B * 6 * exp_dim * sizeof(float));
    ALLOC_BUF(moe_down_buf, B * 6 * V41_DIM * sizeof(float));
    ALLOC_BUF(moe_out_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(ffn_out_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(layer_out_buf, C * 4 * V41_DIM * sizeof(float));
    ALLOC_BUF(hc_flat_norm_buf, C * 4 * V41_DIM * sizeof(float));
    ALLOC_BUF(hc_mix_buf, C * 24 * sizeof(float));
    if (B > C) {
        /* 解耦模式：MoE 的输入/输出各自独立成批缓冲，attention 侧原缓冲继续按 chunk 用。 */
        ALLOC_BUF(moe_norm_acc, B * V41_DIM * sizeof(float));
        ALLOC_BUF(moe_ids_acc, B * 6 * sizeof(int32_t));
        ALLOC_BUF(moe_w_acc, B * 6 * sizeof(float));
        ALLOC_BUF(moe_shared_acc, B * V41_DIM * sizeof(float));
        ALLOC_BUF(moe_after_acc, B * 4 * V41_DIM * sizeof(float));
        ALLOC_BUF(moe_split_acc, B * 24 * sizeof(float));
        ALLOC_BUF(moe_out_acc, B * V41_DIM * sizeof(float));
    } else {
        /* 单 chunk 批：直接指向原缓冲，零拷贝、与旧行为一致。 */
        cw->moe_norm_acc = cw->ffn_norm_buf;
        cw->moe_ids_acc = cw->selected_ids_buf;
        cw->moe_w_acc = cw->selected_weights_buf;
        cw->moe_shared_acc = cw->shared_out_buf;
        cw->moe_after_acc = cw->after_attn_buf;
        cw->moe_split_acc = cw->ffn_split_buf;
        cw->moe_out_acc = cw->moe_out_buf;
    }
#undef ALLOC_BUF

    for (size_t t = 0; t < C; t++) {
#define SUB_V(arr, buf, offset_elems, count_elems) do { \
    cw->arr[t] = myds_metal_buffer_sub_view(cw->buf, (t * (offset_elems)) * sizeof(float), (count_elems) * sizeof(float)); \
    if (!cw->arr[t]) goto fail; \
} while(0)
        SUB_V(x, x_buf, V41_DIM, V41_DIM);
        SUB_V(x_norm, x_norm_buf, V41_DIM, V41_DIM);
        SUB_V(qr, qr_buf, 1280, 1280);
        SUB_V(qr_norm, qr_norm_buf, 1280, 1280);
        SUB_V(q, q_buf, V41_HEADS * 512, V41_HEADS * 512);
        SUB_V(kv_act, kv_act_buf, 512, 512);
        SUB_V(attn_out, attn_out_buf, V41_DIM, V41_DIM);
        SUB_V(after_attn, after_attn_buf, 4 * V41_DIM, 4 * V41_DIM);
        SUB_V(attn_split, attn_split_buf, 24, 24);
        SUB_V(ffn_split, ffn_split_buf, 24, 24);
        SUB_V(ffn_norm, ffn_norm_buf, V41_DIM, V41_DIM);
        SUB_V(router_logits, router_logits_buf, 384, 384);
        SUB_V(shared_out, shared_out_buf, V41_DIM, V41_DIM);
#undef SUB_V
        cw->engram_rows[0][t] = myds_metal_buffer_sub_view(cw->engram_rows_buf[0], (t * MYDS_ENGRAM_INPUT_DIM) * sizeof(float), MYDS_ENGRAM_INPUT_DIM * sizeof(float));
        cw->engram_rows[1][t] = myds_metal_buffer_sub_view(cw->engram_rows_buf[1], (t * MYDS_ENGRAM_INPUT_DIM) * sizeof(float), MYDS_ENGRAM_INPUT_DIM * sizeof(float));
        if (!cw->engram_rows[0][t] || !cw->engram_rows[1][t]) goto fail;
        cw->engram_proj_view[0][t] = myds_metal_buffer_sub_view(cw->engram_proj_buf[0], (size_t)t * 5 * V41_DIM * sizeof(float), 5 * V41_DIM * sizeof(float));
        cw->engram_proj_view[1][t] = myds_metal_buffer_sub_view(cw->engram_proj_buf[1], (size_t)t * 5 * V41_DIM * sizeof(float), 5 * V41_DIM * sizeof(float));
        if (!cw->engram_proj_view[0][t] || !cw->engram_proj_view[1][t]) goto fail;

        cw->selected_ids[t] = myds_metal_buffer_sub_view(cw->selected_ids_buf, t * 6 * sizeof(int32_t), 6 * sizeof(int32_t));
        if (!cw->selected_ids[t]) goto fail;
        cw->selected_weights[t] = myds_metal_buffer_sub_view(cw->selected_weights_buf, t * 6 * sizeof(float), 6 * sizeof(float));
        if (!cw->selected_weights[t]) goto fail;
        cw->route_probs[t] = myds_metal_buffer_sub_view(cw->route_probs_buf, t * 384 * sizeof(float), 384 * sizeof(float));
        if (!cw->route_probs[t]) goto fail;
        cw->route_scores[t] = myds_metal_buffer_sub_view(cw->route_scores_buf, t * 384 * sizeof(float), 384 * sizeof(float));
        if (!cw->route_scores[t]) goto fail;
        cw->route_ids_f32[t] = myds_metal_buffer_sub_view(cw->route_ids_f32_buf, t * 6 * sizeof(float), 6 * sizeof(float));
        if (!cw->route_ids_f32[t]) goto fail;

        cw->pool_kv[t] = myds_metal_buffer_sub_view(cw->pool_kv_buf, t * 512 * sizeof(float), 512 * sizeof(float));
        if (!cw->pool_kv[t]) goto fail;
        cw->pool_score[t] = myds_metal_buffer_sub_view(cw->pool_score_buf, t * 512 * sizeof(float), 512 * sizeof(float));
        if (!cw->pool_score[t]) goto fail;
        cw->index_q[t] = myds_metal_buffer_sub_view(cw->index_q_buf, t * V41_INDEX_HEADS * 128 * sizeof(float), V41_INDEX_HEADS * 128 * sizeof(float));
        if (!cw->index_q[t]) goto fail;
        cw->index_weights[t] = myds_metal_buffer_sub_view(cw->index_weights_buf, t * V41_INDEX_HEADS * sizeof(float), V41_INDEX_HEADS * sizeof(float));
        if (!cw->index_weights[t]) goto fail;

        cw->moe_gate[t] = myds_metal_buffer_sub_view(cw->moe_gate_buf, t * 6 * exp_dim * sizeof(float), 6 * exp_dim * sizeof(float));
        if (!cw->moe_gate[t]) goto fail;
        cw->moe_up[t] = myds_metal_buffer_sub_view(cw->moe_up_buf, t * 6 * exp_dim * sizeof(float), 6 * exp_dim * sizeof(float));
        if (!cw->moe_up[t]) goto fail;
        cw->moe_mid[t] = myds_metal_buffer_sub_view(cw->moe_mid_buf, t * 6 * exp_dim * sizeof(float), 6 * exp_dim * sizeof(float));
        if (!cw->moe_mid[t]) goto fail;
        cw->moe_down[t] = myds_metal_buffer_sub_view(cw->moe_down_buf, t * 6 * V41_DIM * sizeof(float), 6 * V41_DIM * sizeof(float));
        if (!cw->moe_down[t]) goto fail;

        cw->moe_out[t] = myds_metal_buffer_sub_view(cw->moe_out_buf, (t * V41_DIM) * sizeof(float), V41_DIM * sizeof(float));
        cw->ffn_out[t] = myds_metal_buffer_sub_view(cw->ffn_out_buf, (t * V41_DIM) * sizeof(float), V41_DIM * sizeof(float));
        cw->layer_out[t] = myds_metal_buffer_sub_view(cw->layer_out_buf, (t * 4 * V41_DIM) * sizeof(float), 4 * V41_DIM * sizeof(float));
        cw->res_view[t] = myds_metal_buffer_create_view();
        cw->pre_view[t] = myds_metal_buffer_create_view();
        if (!cw->moe_out[t] || !cw->ffn_out[t] || !cw->layer_out[t] || !cw->res_view[t] || !cw->pre_view[t]) goto fail;
    }
    cw->cur_res_view = myds_metal_buffer_create_view();
    cw->cur_pre_view = myds_metal_buffer_create_view();
    cw->cur_selected_view = myds_metal_buffer_create_view();
    cw->cur_mask_view = myds_metal_buffer_create_view();
    cw->cur_index_q_view = myds_metal_buffer_create_view();
    cw->cur_index_weights_view = myds_metal_buffer_create_view();
    cw->moe_view = myds_metal_buffer_create_view();
    cw->shared_view = myds_metal_buffer_create_view();
    cw->after_view = myds_metal_buffer_create_view();
    cw->split_view = myds_metal_buffer_create_view();
    if (!cw->cur_res_view || !cw->cur_pre_view || !cw->cur_selected_view || !cw->cur_mask_view ||
        !cw->cur_index_q_view || !cw->cur_index_weights_view ||
        !cw->moe_view || !cw->shared_view || !cw->after_view || !cw->split_view) goto fail;
    return cw;
fail:
    v41_chunk_workspace_free(cw);
    return NULL;
}

uint32_t myds_v41_resolve_macroblock_size(void) {
    const char *mb_env = getenv("MYDS_PREFILL_MACROBLOCK_SIZE");
    long val = 4096L;
    if (mb_env && *mb_env) {
        char *endptr = NULL;
        errno = 0;
        long parsed = strtol(mb_env, &endptr, 10);
        if (errno == 0 && endptr != mb_env && parsed > 0) {
            val = parsed;
        } else {
            val = 4096L;
        }
    }
    if (val < 512L) val = 512L;
    if (val > 16384L) val = 16384L;
    uint32_t macroblock_size = (uint32_t)val;
    if (macroblock_size % V41_SNAP_STRIDE != 0) {
        macroblock_size = ((macroblock_size + V41_SNAP_STRIDE - 1u) / V41_SNAP_STRIDE) * V41_SNAP_STRIDE;
    }
    if (macroblock_size > 16384u) macroblock_size = 16384u;
    return macroblock_size;
}

int myds_v41_prefill_tokens(myds_context_t *ctx, const int32_t *tokens,
                            uint32_t n_tokens, volatile int *cancel_flag) {
    if (!ctx || !tokens || n_tokens == 0) return 1;
    myds_model_t *m = ctx->model;
    myds_v41_context_t *g = ctx->v41;
    myds_v41_model_t *v = m ? m->v41 : NULL;
    if (!m || !g || !v) return -1;
    lowkv_join(g);   /* prefill 会写压缩 KV 与快照槽 */
    if (!myds_metal_commit_batch()) return -1;

    uint32_t base_pos = ctx->pos;
    const bool trace_on = myds_trace_active();
    if (base_pos + n_tokens > g->capacity || base_pos + n_tokens > ctx->cfg.max_seq_len) {
        MYDS_LOG_E("V4.1 prefill prompt length exceeds context capacity (%u + %u > %u)",
                   base_pos, n_tokens, g->capacity);
        return -1;
    }

    /* ratio==1 的层（l>=20）一个 chunk 的 n_comp 就是【绝对位置】，所以键范围
     * CARRY + c_len + n_comp 最坏到 CARRY + chunk + (base_pos + n_tokens)。
     * 超过 chunk attention 能覆盖的键数时，旧版会在跑到【第 20 层】才返回 0，
     * 于是一个超长 prompt 会白跑几秒再报一句「chunked attention failed」。
     * 这里在分配任何缓冲之前就判定，并直接说清上限与开关。 */
    {
        const uint32_t key_cap = myds_v41_gpu_chunk_key_max(g->gpu);
        if (key_cap && base_pos + n_tokens > key_cap) {
            MYDS_LOG_E("V4.1 prefill needs %u keys but chunked attention covers %u "
                       "(position %u + %u tokens). Raise MYDS_CHUNK_KEY_MAX or shorten the prompt.",
                       base_pos + n_tokens, key_cap, base_pos, n_tokens);
            return -1;
        }
    }

    if (base_pos == 0) {
        g->next_position = 0;
        g->valid = true;
        myds_engram_reset(g->engram);
    }
    if (!g->valid || base_pos != g->next_position) {
        MYDS_LOG_E("V4.1 expected consecutive token position %u, received %u",
                   g->next_position, base_pos);
        return -1;
    }

    const char *chunk_env = getenv("MYDS_PREFILL_CHUNK_SIZE");
    uint32_t chunk_size = chunk_env ? (uint32_t)atoi(chunk_env) : 256u;
    if (chunk_size < 2u) chunk_size = 2u;
    if (chunk_size % 2u != 0) chunk_size = (chunk_size / 2u) * 2u;
    if (chunk_size > V41_MAX_CHUNK) {
        static bool clamped_warned = false;
        if (!clamped_warned) {
            clamped_warned = true;
            MYDS_LOG_W("V4.1 prefill chunk size %u clamped to the attention limit %u",
                       chunk_size, V41_MAX_CHUNK);
        }
        chunk_size = V41_MAX_CHUNK;
    }
    if (n_tokens < chunk_size) {
        chunk_size = n_tokens;
        if (chunk_size % 2u != 0) chunk_size++;
        if (chunk_size < 2u) chunk_size = 2u;
        if (chunk_size > V41_MAX_CHUNK) chunk_size = V41_MAX_CHUNK;
    }

    size_t res_tok_bytes = 4u * V41_DIM * sizeof(float);
    size_t pre_tok_bytes = 24u * sizeof(float);
    size_t sel_bytes = V41_TOPK * sizeof(int32_t);
    const uint32_t max_comp = (base_pos + n_tokens) / 1u;
    const uint32_t max_blocks = (max_comp + 7u) / 8u;
    uint32_t mask_stride = (max_blocks + 31u) / 32u * 32u;
    if (mask_stride < 256u) mask_stride = 256u;
    const uint32_t cap_blocks = (g->capacity + 7u) / 8u;
    if (mask_stride > cap_blocks) mask_stride = cap_blocks;
    size_t mask_bytes = (size_t)mask_stride * sizeof(float);

    /* Macroblock: 把超长 prompt 切成固定上限（默认 4096 tokens）的 macroblock。
     * 每个 macroblock 跑完整 40 层，从而 prompt 级工作区（prompt_res, prompt_block_mask 等）
     * 只需按 macroblock 大小分配，彻底消除 262K 上下文时 64.52 GiB 的内存峰值，降至 ~1 GiB。 */
    const uint32_t macroblock_size = myds_v41_resolve_macroblock_size();
    const uint32_t mb_cap_needed = n_tokens < macroblock_size ? n_tokens : macroblock_size;

    /* MoE 的批跨几个 attention chunk。默认 1（= 旧行为：每个 chunk 跑一次 MoE）。
     * 加大它不改变 attention 的数值，只是把 MoE 的调用次数降下来 —— MoE 每层要读
     * 全部 n_routed_experts 个 expert 的权重，与 token 数无关。 */
    uint32_t moe_chunks = 1;
    {
        const char *mc = getenv("MYDS_MOE_CHUNKS");
        if (mc) { int v = atoi(mc); if (v > 1) moe_chunks = (uint32_t)v; }
        if (moe_chunks > 16u) moe_chunks = 16u;
        /* 内存护栏：批缓冲约 0.44 MB/token（MoE 的 gate/up/mid/down + 累积缓冲） */
        if (moe_chunks > 1u && (size_t)chunk_size * moe_chunks > MYDS_MOE_BATCH_MAX_TOKENS)
            moe_chunks = (uint32_t)(MYDS_MOE_BATCH_MAX_TOKENS / chunk_size);
        if (moe_chunks < 1u) moe_chunks = 1u;
        if (moe_chunks > 1u) {
            uint32_t need = (uint32_t)(((size_t)mb_cap_needed + chunk_size - 1u) / chunk_size);  /* 不需要超过一个 macroblock 的 chunk 数 */
            if (moe_chunks > need) moe_chunks = need;
            if (moe_chunks < 1u) moe_chunks = 1u;
        }
    }
    const uint32_t moe_batch_tokens = chunk_size * moe_chunks;
    if (!g->chunk_ws || g->chunk_ws->capacity < chunk_size || g->chunk_ws->moe_batch < moe_batch_tokens) {
        v41_chunk_workspace_free(g->chunk_ws);
        g->chunk_ws = v41_chunk_workspace_create(chunk_size, m->moe_expert_dim, moe_batch_tokens);
    }
    v41_chunk_workspace_t *cw = g->chunk_ws;

    /* 复用 prompt 级工作区（按 macroblock 容量分配，避免长请求每次 prefill 申请释放上吉字节引发抖动） */
    const bool prompt_ws_complete = g->prompt_res && g->prompt_pre && g->prompt_selected &&
                                    g->prompt_block_mask && g->prompt_engram[0] && g->prompt_engram[1];
    uint32_t cap = (mb_cap_needed + 511u) / 512u * 512u;
    if (!prompt_ws_complete || g->prompt_tokens_cap < cap || g->prompt_mask_bytes_cap < mask_bytes) {
        size_t cap_res_size = (size_t)cap * 4u * V41_DIM;
        myds_metal_buffer_t *n_res = myds_metal_buffer_alloc(cap_res_size * sizeof(float));
        myds_metal_buffer_t *n_pre = myds_metal_buffer_alloc((size_t)cap * pre_tok_bytes);
        myds_metal_buffer_t *n_selected = myds_metal_buffer_alloc((size_t)cap * sel_bytes);
        myds_metal_buffer_t *n_block_mask = myds_metal_buffer_alloc((size_t)cap * mask_bytes);
        float *n_engram0 = (float *)malloc((size_t)cap * MYDS_ENGRAM_INPUT_DIM * sizeof(float));
        float *n_engram1 = (float *)malloc((size_t)cap * MYDS_ENGRAM_INPUT_DIM * sizeof(float));

        if (!n_res || !n_pre || !n_selected || !n_block_mask || !n_engram0 || !n_engram1) {
            if (n_res) myds_metal_buffer_free(n_res);
            if (n_pre) myds_metal_buffer_free(n_pre);
            if (n_selected) myds_metal_buffer_free(n_selected);
            if (n_block_mask) myds_metal_buffer_free(n_block_mask);
            if (n_engram0) free(n_engram0);
            if (n_engram1) free(n_engram1);
            MYDS_LOG_E("V4.1 prefill failed to allocate workspace buffers (cap=%u)", cap);
            return -1;
        }

        /* 成功申请完整一组后再释放旧组并原子替换，确保容量发布与指针完整性原子对齐 */
        if (g->prompt_res) myds_metal_buffer_free(g->prompt_res);
        if (g->prompt_pre) myds_metal_buffer_free(g->prompt_pre);
        if (g->prompt_selected) myds_metal_buffer_free(g->prompt_selected);
        if (g->prompt_block_mask) myds_metal_buffer_free(g->prompt_block_mask);
        if (g->prompt_engram[0]) free(g->prompt_engram[0]);
        if (g->prompt_engram[1]) free(g->prompt_engram[1]);

        g->prompt_res = n_res;
        g->prompt_pre = n_pre;
        g->prompt_selected = n_selected;
        g->prompt_block_mask = n_block_mask;
        g->prompt_engram[0] = n_engram0;
        g->prompt_engram[1] = n_engram1;
        g->prompt_tokens_cap = cap;
        g->prompt_mask_bytes_cap = mask_bytes;
    }

    myds_metal_buffer_t *prompt_res = g->prompt_res;
    myds_metal_buffer_t *prompt_pre = g->prompt_pre;
    myds_metal_buffer_t *prompt_selected = g->prompt_selected;
    myds_metal_buffer_t *prompt_block_mask = g->prompt_block_mask;
    float *engram_all[2] = { g->prompt_engram[0], g->prompt_engram[1] };

    if (!prompt_res || !prompt_pre || !prompt_selected || !prompt_block_mask ||
        !engram_all[0] || !engram_all[1] || !cw) {
        MYDS_LOG_E("V4.1 prefill failed to allocate workspace buffers");
        return -1;
    }

    int do_prof = getenv("MYDS_PROFILE_PREFILL") ? 1 : 0;
    const bool batch_trace_prefix = (getenv("MYDS_BATCH_TRACE_PREFIX") != NULL);
    const bool opt_disable_mm_engram = (getenv("MYDS_DISABLE_MM_ENGRAM") != NULL);
    const bool opt_skip_q_a = (getenv("MYDS_SKIP_Q_A") != NULL);
    const bool opt_skip_kv = (getenv("MYDS_SKIP_KV") != NULL);
    const bool opt_skip_norm = (getenv("MYDS_SKIP_NORM") != NULL);
    const bool opt_skip_kv_rope = (getenv("MYDS_SKIP_KV_ROPE") != NULL);
    const bool opt_skip_kv_quant = (getenv("MYDS_SKIP_KV_QUANT") != NULL);
    const bool opt_skip_q_b = (getenv("MYDS_SKIP_Q_B") != NULL);
    const bool opt_skip_q_rope = (getenv("MYDS_SKIP_Q_ROPE") != NULL);
    const bool opt_comp_per_token = (getenv("MYDS_COMPRESSOR_PER_TOKEN") != NULL);
    const bool opt_idx_per_token = (getenv("MYDS_INDEXER_PER_TOKEN") != NULL);
    const bool opt_prof_split = (getenv("MYDS_PROFILE_SPLIT") != NULL);
    const bool opt_prof_split2 = (getenv("MYDS_PROFILE_SPLIT2") != NULL);
    const bool opt_prof_split3 = (getenv("MYDS_PROFILE_SPLIT3") != NULL);
    const bool opt_prof_moe = (getenv("MYDS_PROFILE_MOE") != NULL);
    const bool opt_prof_step1 = (getenv("MYDS_PROFILE_STEP1") != NULL);
    /* 后缀收缩（suffix shrink，见 docs/SUFFIX_SHRINK_SPEC.md）：只处理因果依赖锥内的
     * 尾部行。阈值对齐 ds4：total_count >= 8192。
     *
     * 默认【开】（2026-10-03 定）：真实负载是【纯顺序增长】—— 实测 62/62 次复用走
     * 活序列（`resume from cached prefix`）、0 次走中途快照，而收缩不碰活序列
     * ⇒ 零代价拿满收益（12.4k: 283→425 t/s）。要回旧行为用 `MYDS_SUFFIX_SHRINK=0`。
     * （跨会话「兄弟请求」复用中途快照时会退化，那类负载需配 MYDS_SUFFIX_SHRINK_SHIFT。） */
    const char *suffix_env = getenv("MYDS_SUFFIX_SHRINK");
    const bool opt_suffix_shrink = !(suffix_env && atoi(suffix_env) == 0);
    /* 只在【per-query gather 路径】（默认）下开启：那条路径把每个 query 可见的
     * 640 个键（128 滑窗 + 512 picks）gather 成紧凑段，内核看到的输入与 chunk
     * 边界无关 ⇒ 收缩后各行结果与不收缩【逐位相同】。
     * 关掉 gather（MYDS_ATTN_GATHER=0）时注意力走 slab 掩码路径，键在 slab 里的
     * 块对齐随 chunk 长度变化 ⇒ 极点值可能差最后几位，本版未验证那种组合，直接不收缩。 */
    const char *gather_env = getenv("MYDS_ATTN_GATHER");
    const bool shrink_gather_ok = (!gather_env) || atoi(gather_env) != 0;
    uint32_t shrink_min = 512u;
    const char *smin = getenv("MYDS_SUFFIX_SHRINK_MIN_TOKENS");
    if (smin && *smin) {
        long v = atol(smin);
        if (v >= 0) shrink_min = (uint32_t)v;
    }
    const bool suffix_shrink = opt_suffix_shrink && shrink_gather_ok &&
                               (base_pos + n_tokens) >= shrink_min;
    /* 收缩前置段的「有效快照窗口」（见 docs/verification/2026-10-03-suffix-shrink-snapshot.md）：
     * 层 il 的窗口环在 p 处【内容正确】的前提是 p ≥ B - shift（推导见层循环里的注释）。
     *   shift = 0（默认）⇒ 与 6a 完全一致：只有 macroblock 末尾能发布快照，
     *                    前置段仍从 S(il)-127 起跳（零额外开销）。
     *   shift > 0       ⇒ 前置段从 A 起跑（chunk 末端落在 512 网格上），并在
     *                    [B-shift, B] 上发布快照 ⇒ 每个 macroblock 多出 shift/512
     *                    个可复用点。代价：层 il 多算 shift 行 KV（不做 attention/FFN/MoE）。 */
    uint32_t shrink_shift = 0;
    {
        const char *se = getenv("MYDS_SUFFIX_SHRINK_SHIFT");
        if (se && *se) {
            const long v = atol(se);
            if (v > 0) {
                shrink_shift = (uint32_t)v;
                /* 只有 512 网格点才可能成为快照点 ⇒ 向上取整到 STRIDE。 */
                if (shrink_shift % V41_SNAP_STRIDE)
                    shrink_shift = (shrink_shift / V41_SNAP_STRIDE + 1u) * V41_SNAP_STRIDE;
                if (shrink_shift > macroblock_size) shrink_shift = macroblock_size;
            } else if (v < 0) {
                MYDS_LOG_W("MYDS_SUFFIX_SHRINK_SHIFT=%s 非法（需 >= 0），按 0 处理", se);
            }
        }
    }
    double prof_s1_norm = 0, prof_s1_mix = 0, prof_s1_split = 0, prof_s1_wsum = 0, prof_s1_norm2 = 0;
    const bool opt_disable_mm_attn_a = (getenv("MYDS_DISABLE_MM_ATTN_A") != NULL);
    double prof_engram = 0, prof_embedding = 0, prof_total_layers = 0;
    double prof_step1 = 0, prof_step2 = 0, prof_step3 = 0, prof_step4 = 0, prof_step5 = 0;
    double prof_step6 = 0, prof_step7 = 0, prof_step8 = 0;
    double prof_wait_qkv = 0, prof_wait_swa = 0, prof_wait_ffn_router = 0, prof_wait_moe = 0;
    /* 诊断：把「Wait MoE」这个窗口拆开 —— 它实际包含 Step5..8 全部（attn_out 投影、
     * FFN In、shared+router、MoE GEMM、以及 epilogue 的 add_bf16/hc_expand），
     * 所以不能直接当成「MoE 的耗时」。三段在 MYDS_PROFILE_MOE=1 时才测。 */
    double prof_wait_attnB = 0, prof_wait_ffnpre = 0, prof_wait_moe_gemm = 0;
    uint64_t disp_ringattn = 0, disp_compressor = 0;   /* 诊断：分段 dispatch 数 */
    double prof_wait_indexer = 0;   /* 诊断：Step 4 里 indexer 部分的 GPU 等待 */
    double prof_wait_attn2 = 0;     /* 诊断：ring+attention 的 GPU 等待 */
    double prof_wait_bf16 = 0;      /* 诊断：attn_heads 的 bf16 转换 */
    uint64_t disp_step14 = 0, disp_step58 = 0, disp_layer_all = 0;  /* 诊断 */
    uint64_t disp_step4loop = 0;  /* 诊断：仅 Step4 逐 token 循环 */
    struct timeval ptv0, ptv1, ptv_all0, ptv_all1;
#define TIME_START() do { if (do_prof) gettimeofday(&ptv0, NULL); } while(0)
#define TIME_ACC(acc) do { if (do_prof) { gettimeofday(&ptv1, NULL); acc += (ptv1.tv_sec - ptv0.tv_sec) * 1000.0 + (ptv1.tv_usec - ptv0.tv_usec) / 1000.0; } } while(0)

    uint32_t il = 0;
    int idx_fallback_warned = 0;   /* 批量 indexer 回退只提示一次 */
    const char *phase = "engram prepare";

#define CHECK(x) do { if (!(x)) goto fail; } while (0)
#define TRACE(stage,buf,n) do { if (trace_on) CHECK(myds_trace_f32(stage,il,pos,buf,n)); } while (0)

    if (do_prof) gettimeofday(&ptv_all0, NULL);

    for (uint32_t mb_start = 0; mb_start < n_tokens; mb_start += macroblock_size) {
        uint32_t mb_len = n_tokens - mb_start;
        if (mb_len > macroblock_size) mb_len = macroblock_size;
        const uint32_t mb_base_pos = base_pos + mb_start;

        phase = "engram prepare";
        TIME_START();
        for (uint32_t t = 0; t < mb_len; t++) {
            uint32_t pos = mb_base_pos + t;
            if (!myds_engram_prepare_token(g->engram, tokens[mb_start + t], (uint64_t)pos)) {
                MYDS_LOG_E("V4.1 engram prepare failed at pos %u", pos);
                goto fail;
            }
            snapshot_store_engram(g, pos + 1u, (mb_start + t + 1u) == n_tokens);   /* 处理完 pos ⇒ 已推进到 pos+1 */
            memcpy(engram_all[0] + (size_t)t * MYDS_ENGRAM_INPUT_DIM,
                   g->engram->row_values[0], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
            memcpy(engram_all[1] + (size_t)t * MYDS_ENGRAM_INPUT_DIM,
                   g->engram->row_values[1], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
        }
        TIME_ACC(prof_engram);

        phase = "embedding lookup";
        TIME_START();
        const uint16_t *embd_table = (const uint16_t *)myds_metal_buffer_data(m->tok_embeddings);
        for (uint32_t t = 0; t < mb_len; t++) {
            int32_t tok = tokens[mb_start + t];
            if (tok < 0 || (uint32_t)tok >= m->vocab_size) {
                MYDS_LOG_E("V4.1 invalid token %d at pos %u", tok, mb_base_pos + t);
                goto fail;
            }
            float r[4 * V41_DIM];
            if (v41_image_embedding_at(ctx, mb_base_pos + t, r, V41_DIM)) {
                /* 图片块：该位置的向量来自注入（上面已填 r[0..V41_DIM)）*/
                const uint16_t *unused_row = embd_table;   /* 仅为保持下方代码结构清晰 */
                (void)unused_row;
                for (uint32_t j = 0; j < V41_DIM; j++) {
                    r[j + V41_DIM] = r[j];
                    r[j + 2 * V41_DIM] = r[j];
                    r[j + 3 * V41_DIM] = r[j];
                }
            } else {
                const uint16_t *row = embd_table + (size_t)tok * V41_DIM;
                for (uint32_t j = 0; j < V41_DIM; j++) {
                    float v_emb = myds_fp16_to_fp32(row[j]);
                    r[j] = v_emb;
                    r[j + V41_DIM] = v_emb;
                    r[j + 2 * V41_DIM] = v_emb;
                    r[j + 3 * V41_DIM] = v_emb;
                }
            }
            const float initial_pre[24] = {1.0f, 0.0f, 0.0f, 0.0f};
            if (!myds_metal_buffer_write(prompt_res, (size_t)t * res_tok_bytes, r, res_tok_bytes)) goto fail;
            if (!myds_metal_buffer_write(prompt_pre, (size_t)t * pre_tok_bytes, initial_pre, pre_tok_bytes)) goto fail;

            if (batch_trace_prefix) {
                memcpy(myds_metal_buffer_data(g->residual), r, res_tok_bytes);
                if (!myds_trace_f32("embedding", 0, mb_base_pos + t, g->residual, 4 * V41_DIM)) goto fail;
            }
        }
        TIME_ACC(prof_embedding);

        phase = "layer loop";
        uint32_t moe_off = 0;   /* 当前 MoE 批已经攒了多少 token */
        for (il = 0; il < V41_LAYERS; il++) {
            if (cancel_flag && *cancel_flag) goto cancelled;
            const myds_layer_t *base = &m->layers[il];
            const v41_layer_t *l = &v->layers[il];
            const bool compressed = ratio_for(il) != 0;
            moe_off = 0;

            /* ── 后缀收缩：本层只处理因果锥内的尾部行 ────────────────────────────
             * 参考点取【本 macroblock 的末尾】B（不是整条 prompt 的末尾）：
             *   层 il 的输出只覆盖 [S(il), B)，S(il) = B - (1 + (39-il)*127) - shift；
             *   [A, S(il)) 这段只算 KV（填窗口环）—— 它是「进入 S 时的 carry 引导」。
             *   shift = 0 时引导段从 S(il)-127 起跳（6a 原样）；shift > 0 时从 A 起跳。
             * 取 B 的理由：每个 macroblock 都要把最后一层的窗口环推进到自己的末尾，
             * 供下一个 macroblock 的 carry 使用；「相对 B 的锥」恰好同时满足
             * 「KV 环维护」与「正确性」—— 因为 S(il-1) = S(il) - 127，上一层的输出
             * 范围天生比本层宽 127 行，正好覆盖本层进入 S 所需的输入。
             * 短 macroblock（B-A 小于锥宽）⇒ s 被夹到 A ⇒ 整块全量，等价于关掉收缩。 */
            const uint32_t total_eff = mb_base_pos + mb_len;
            const uint32_t sh_need = 1u + (V41_LAYERS - 1u - il) * 127u;
            const bool shrink_layer = suffix_shrink && il >= 20u;
            /* 本层窗口环在 p 处【内容有效】的边界：p ≥ s(il)+1（s(il) = A + sh_full_rel）。
             * 推导：环在 p 处装着 [p-128,p-1] 的 KV，要它们全对，这些行的输入残差必须都由
             * 层 il-1 的【全量段】写出；层 il-1 的全量段起点 = s(il)-127
             * ⇒ p-128 ≥ s(il)-127 ⇒ p ≥ s(il)+1。
             * 发布要求【全部 40 层】都存过同一个 p ⇒ 全局紧界 = s(39)+1 = B - shift
             * （s(39) = max(mb_base, B-1-shift)）；shift=0 时就是 B ⇒ 只有
             * macroblock 末尾/序列末尾能发布（即 6a 行为；锥被夹到 A 时所有层都
             * 满足 p ≥ A+1，与 6a 一样在各网格点发布 —— 那时整块都是全量，环本来就对）。 */
            const uint32_t sh_valid_from = (total_eff > shrink_shift) ? total_eff - shrink_shift : 0u;
            uint32_t sh_loop_rel = 0, sh_full_rel = 0;
            if (shrink_layer) {
                const uint32_t sub = sh_need + shrink_shift;
                uint32_t s = (total_eff > sub) ? total_eff - sub : 0u;
                if (s < mb_base_pos) s = mb_base_pos;
                sh_full_rel = s - mb_base_pos;
                if (shrink_shift == 0u) {
                    /* 默认（6a）路径：前置段只从 S(il)-127 起跳（层 20 从 A）。
                     * 与已验证的 6a 逐位一致，且不多算任何 KV。 */
                    uint32_t skv = (s > 127u) ? s - 127u : 0u;
                    if (skv < mb_base_pos) skv = mb_base_pos;
                    sh_loop_rel = (il == 20u) ? 0u : (skv - mb_base_pos);
                }
                /* shift>0：sh_loop_rel = 0 ⇒ 前置段从 A 起跑；A 是 STRIDE 的倍数，
                 * 所以各层 chunk 末端都落在 512 网格上（否则中途快照点根本无法发布）。 */
            }
            if (shrink_layer && il == 20u) {
                static bool shrink_logged = false;
                if (!shrink_logged) {
                    shrink_logged = true;
                    MYDS_LOG_I("V4.1 suffix shrink[snapshot-guard v2]: front=%s shift=%u valid_from=%u "
                               "(total=%u, layer20 rows=%u..%u of %u)",
                               shrink_shift ? "all" : "skv", shrink_shift, sh_valid_from,
                               base_pos + n_tokens, sh_full_rel, mb_len, mb_len);
                }
            }

            for (uint32_t c_start = sh_loop_rel; c_start < mb_len; ) {
                if (cancel_flag && *cancel_flag) goto cancelled;
                uint32_t c_len = mb_len - c_start;
                if (c_len > chunk_size) c_len = chunk_size;
                /* shift>0：chunk 末端还要落在【绝对位置】的 512 网格上 —— 快照点只可能存在于
                 * 网格点，而上面那句「模式边界必须落在 chunk 边界」会把 chunk 切到非网格的
                 * s(il) 上，之后的 chunk 就全部错位、再也发布不了快照（实测：shift=2048 时
                 * 续跑点仍然退到 8192 —— 就是这个错位造成的）。默认（shift=0）不动，
                 * 保持 6a 行为（那时的错位恰好让中途快照发布不了，是我们想要的）。 */
                if (shrink_shift) {
                    const uint32_t abs0 = mb_base_pos + c_start;
                    const uint32_t grid = ((abs0 + c_len) / V41_SNAP_STRIDE) * V41_SNAP_STRIDE;
                    if (grid > abs0) c_len = grid - abs0;   /* 只在能缩短时对齐（不会变成 0） */
                }
                /* 全量/前端的边界必须落在 chunk 边界上（一个 chunk 不能走两条路径） */
                if (shrink_layer && c_start < sh_full_rel && c_start + c_len > sh_full_rel)
                    c_len = sh_full_rel - c_start;
                /* kv_only：只做「窗口环需要的 KV 投影」（层 20 还要做压缩器），
                 * 不做 Q_A/Q_B 投影、attention / 输出 / FFN / MoE，也不累加 moe_off。
                 * 注意窗口环拷贝对两条路径都要做 —— 它正是前置段存在的目的。 */
                const bool kv_only = shrink_layer && (c_start + c_len <= sh_full_rel);

                if (il == 1 || il == 14) {
                    uint32_t e = (il == 1 ? 0 : 1);
                    CHECK(myds_metal_buffer_write(cw->engram_rows_buf[e], 0,
                        engram_all[e] + (size_t)c_start * MYDS_ENGRAM_INPUT_DIM,
                        (size_t)c_len * MYDS_ENGRAM_INPUT_DIM * sizeof(float)));
                }

                CHECK(myds_metal_begin_batch());

                myds_metal_buffer_update_sub_view(cw->cur_res_view, prompt_res, (size_t)c_start * res_tok_bytes, (size_t)c_len * res_tok_bytes);
                myds_metal_buffer_update_sub_view(cw->cur_pre_view, prompt_pre, (size_t)c_start * pre_tok_bytes, (size_t)c_len * pre_tok_bytes);

                if (trace_on) {
                    for (uint32_t t = 0; t < c_len; t++) {
                        uint32_t idx = c_start + t;
                        myds_metal_buffer_update_sub_view(cw->res_view[t], prompt_res, (size_t)idx * res_tok_bytes, res_tok_bytes);
                        myds_metal_buffer_update_sub_view(cw->pre_view[t], prompt_pre, (size_t)idx * pre_tok_bytes, 4 * sizeof(float));
                    }
                }

                uint64_t disp_layer0 = myds_metal_dispatch_count();  /* 诊断 */
                /* 本次迭代 Step1-4 结束时的计数（诊断）。默认 = 起点：若本迭代没走到 Step4 末尾，
                 * Step5-8 就按「整段都算 5-8」计，而不会出现负数回绕。 */
                uint64_t disp_mark4 = disp_layer0;

                /* Step 1: HC Attention Inputs for all tokens in chunk */
                TIME_START();
                phase = "HC attention input";

                if (il == 1 || il == 14) {
                    uint32_t e = (il == 1 ? 0 : 1);
                    myds_metal_barrier();
                    int e_batched = 0;
                    if (!opt_disable_mm_engram)
                        e_batched = myds_metal_mul_mm(cw->engram_proj_buf[e], v->engram_proj[e],
                            cw->engram_rows_buf[e], 5 * V41_DIM, MYDS_ENGRAM_INPUT_DIM, c_len, GGML_TYPE_F16);
                    if (!e_batched) {
                        for (uint32_t t = 0; t < c_len; t++)
                            CHECK(project(cw->engram_proj_view[e][t], v->engram_proj[e], cw->engram_rows[e][t],
                                5 * V41_DIM, MYDS_ENGRAM_INPUT_DIM, GGML_TYPE_F16, true));
                    }
                    if (trace_on) {
                        for (uint32_t t = 0; t < c_len; t++) {
                            uint32_t idx = c_start + t;
                            myds_metal_buffer_update_sub_view(cw->res_view[t], prompt_res, (size_t)idx * res_tok_bytes, res_tok_bytes);
                        }
                        for (uint32_t t = 0; t < c_len; t++) {
                            uint32_t pos = mb_base_pos + c_start + t;   /* TRACE 宏要用 */
                            CHECK(myds_metal_v41_engram_add(cw->res_view[t], cw->engram_proj_view[e][t], v->engram_q[e], v->engram_k[e], V41_DIM, m->norm_eps));
                            TRACE("engram_out", cw->res_view[t], 4 * V41_DIM);
                        }
                    } else {
                        CHECK(myds_metal_v41_engram_add_batch(cw->cur_res_view, cw->engram_proj_buf[e], v->engram_q[e], v->engram_k[e], V41_DIM, c_len, m->norm_eps));
                    }
                }

                /* Batched HC Mix: RMSNorm + GEMM + Split across the entire chunk */
                CHECK(myds_metal_v41_rmsnorm_batch(cw->hc_flat_norm_buf, cw->cur_res_view, NULL, 4 * V41_DIM, c_len, m->norm_eps, false));
                if (do_prof && opt_prof_step1) { TIME_START(); CHECK(myds_metal_commit_batch()); TIME_ACC(prof_s1_norm); CHECK(myds_metal_begin_batch()); }
                CHECK(project_batch(cw->hc_mix_buf, l->attn_fn, cw->hc_flat_norm_buf, 24, 4 * V41_DIM, c_len, GGML_TYPE_F16, false));
                if (do_prof && opt_prof_step1) { TIME_START(); CHECK(myds_metal_commit_batch()); TIME_ACC(prof_s1_mix); CHECK(myds_metal_begin_batch()); }
                CHECK(myds_metal_v41_hc_split_batch(cw->attn_split_buf, cw->hc_mix_buf, l->attn_scale, l->attn_base, c_len, v->hc_eps));
                if (do_prof && opt_prof_step1) { TIME_START(); CHECK(myds_metal_commit_batch()); TIME_ACC(prof_s1_split); CHECK(myds_metal_begin_batch()); }

                if (trace_on) {
                    for (uint32_t t = 0; t < c_len; t++) {
                        uint32_t pos = mb_base_pos + c_start + t;
                        TRACE("hc_attn_split", cw->attn_split[t], 24);
                        TRACE("hc_attn_pre", cw->pre_view[t], 4);
                    }
                }

                CHECK(myds_metal_v41_hc_weighted_sum_batch(cw->x_buf, cw->cur_res_view, cw->cur_pre_view, V41_DIM, c_len, pre_tok_bytes));
                if (do_prof && opt_prof_step1) { TIME_START(); CHECK(myds_metal_commit_batch()); TIME_ACC(prof_s1_wsum); CHECK(myds_metal_begin_batch()); }
                CHECK(myds_metal_v41_rmsnorm_batch(cw->x_norm_buf, cw->x_buf, base->attn_norm, V41_DIM, c_len, m->norm_eps, true));
                if (do_prof && opt_prof_step1) { TIME_START(); CHECK(myds_metal_commit_batch()); TIME_ACC(prof_s1_norm2); CHECK(myds_metal_begin_batch()); }

                if (trace_on) {
                    for (uint32_t t = 0; t < c_len; t++) {
                        uint32_t pos = mb_base_pos + c_start + t;
                        TRACE("hc_attn_input", cw->x[t], V41_DIM);
                        TRACE("attn_norm", cw->x_norm[t], V41_DIM);
                    }
                }
                TIME_ACC(prof_step1);

                /* Step 2: Batched Q_A and KV projections (reads weights ONCE for all c_len tokens)
                 * ★ 前缀段（kv_only）只需要 KV：Q_A/Q_B 都不必算（窗口环只用 KV）。
                 *   层 20 保持原样（它要建 owner 3 的全局状态）。 */
                const bool kv_only_skip_q = kv_only && il >= 21u;
                TIME_START();
                phase = "batched attention projections";
                if (!opt_skip_q_a && !kv_only_skip_q)
                    CHECK(project_batch(cw->qr_buf, base->attn_q_a, cw->x_norm_buf, 1280, V41_DIM, c_len, base->attn_q_a_type, true));
                if (!opt_skip_kv)
                    CHECK(project_batch(cw->kv_act_buf, base->attn_kv, cw->x_norm_buf, 512, V41_DIM, c_len, base->attn_kv_type, true));

                if (!opt_skip_norm) {
                    if (!kv_only_skip_q)
                        CHECK(myds_metal_v41_rmsnorm_batch(cw->qr_norm_buf, cw->qr_buf, base->attn_q_a_norm, 1280, c_len, m->norm_eps, true));
                    CHECK(myds_metal_v41_rmsnorm_batch(cw->kv_act_buf, cw->kv_act_buf, base->attn_kv_a_norm, 512, c_len, m->norm_eps, true));
                }

                /* Batched RoPE & Quant for KV */
                if (!opt_skip_kv_rope)
                    CHECK(myds_v41_gpu_rope_batch(cw->kv_act_buf, 512, 1, mb_base_pos + c_start, c_len, compressed, false));
                if (!opt_skip_kv_quant)
                    CHECK(myds_v41_gpu_quant(cw->kv_act_buf, 512, c_len, 1));

                if (trace_on && !kv_only) {   /* 前缀段不算 Q（il>=21），Q 的 trace 无意义 */
                    for (uint32_t t = 0; t < c_len; t++) {
                        uint32_t idx = c_start + t;
                        uint32_t pos = mb_base_pos + idx;
                        TRACE("q_a_raw", cw->qr[t], 1280);
                        TRACE("q_a_norm", cw->qr_norm[t], 1280);
                        TRACE("kv_raw", cw->kv_act[t], 512);
                        TRACE("kv_norm", cw->kv_act[t], 512);
                        TRACE("kv_rope", cw->kv_act[t], 512);
                        TRACE("kv_quant", cw->kv_act[t], 512);
                    }
                }
                TIME_ACC(prof_step2);

                /* Step 3: Batched Q_B projection (reads 21 MB weight ONCE for all c_len tokens) */
                TIME_START();
                if (kv_only) { /* 前端只喂窗口环，Q 完全不用算 */ }
                else {
                if (!opt_skip_q_b)
                    CHECK(project_batch(cw->q_buf, base->attn_q_b, cw->qr_norm_buf, V41_HEADS * 512, 1280, c_len, base->attn_q_b_type, false));

                /* Batched RoPE + BF16 rounding for Q */
                if (!opt_skip_q_rope)
                    CHECK(myds_v41_gpu_rope_batch_round(cw->q_buf, 512, V41_HEADS, mb_base_pos + c_start, c_len, compressed, false));

                if (trace_on) {
                    for (uint32_t t = 0; t < c_len; t++) {
                        uint32_t pos = mb_base_pos + c_start + t;
                        TRACE("q_b", cw->q[t], V41_HEADS * 512);
                        TRACE("q_rope", cw->q[t], V41_HEADS * 512);
                    }
                }
                }
                TIME_ACC(prof_step3);

                if (do_prof) {
                    TIME_START();
                    CHECK(myds_metal_commit_batch());
                    TIME_ACC(prof_wait_qkv);
                    CHECK(myds_metal_begin_batch());
                }

                /* Step 4: SWA Attention reduction & indexer */
                TIME_START();
                uint32_t ratio = ratio_for(il);
                if (ratio) {
                    if (kv_source(il)) {
                        CHECK(project_batch(cw->pool_kv_buf, l->compressor_kv, cw->x_norm_buf, 512, V41_DIM, c_len, GGML_TYPE_F16, ratio == 1));
                        if (ratio == 2) {
                            CHECK(project_batch(cw->pool_score_buf, l->compressor_gate, cw->x_norm_buf, 512, V41_DIM, c_len, GGML_TYPE_F16, false));
                        }
                    }
                    if (index_source(il) && !kv_only) {
                        CHECK(project_batch(cw->index_q_buf, l->index_q, cw->qr_norm_buf, V41_INDEX_HEADS * 128, 1280, c_len, GGML_TYPE_F16, true));
                        CHECK(project_batch(cw->index_weights_buf, l->index_proj, cw->x_norm_buf, V41_INDEX_HEADS, V41_DIM, c_len, GGML_TYPE_F16, true));
                    }
                }

                uint64_t disp_loop0 = myds_metal_dispatch_count();  /* 诊断 */
                if (kv_source(il)) {
                    phase = "compressed KV";
                    int comp_batched = 0;
                    if (!opt_comp_per_token)
                        comp_batched = publish_compressor_chunk(ctx, il, c_start, c_len, mb_base_pos,
                            cw->pool_kv_buf, ratio == 2 ? cw->pool_score_buf : NULL,
                            cw->comp_latent_buf, cw->comp_index_k_buf);
                    if (comp_batched < 0) {
                        MYDS_LOG_E("V4.1 chunk compressor failed mid-flight (layer %u chunk %u)", il, c_start);
                        goto fail;
                    }
                    if (!comp_batched) {
                        uint64_t dc0 = myds_metal_dispatch_count();
                        for (uint32_t t = 0; t < c_len; t++) {
                            uint32_t pos = mb_base_pos + c_start + t;
                            CHECK(publish_compressor(ctx, il, pos, cw->x_norm[t], cw->pool_kv[t],
                                                     ratio == 2 ? cw->pool_score[t] : NULL));
                        }
                        disp_compressor += myds_metal_dispatch_count() - dc0;
                    }
                }
                if (index_source(il) && !kv_only) {
                    int batched = 0;
                    if (!opt_idx_per_token) {
                        batched = publish_indexer_chunk(ctx, il, c_start, c_len, mb_base_pos,
                            cw->index_q_buf, cw->index_weights_buf, prompt_selected, sel_bytes,
                            prompt_block_mask, mask_bytes, cw->cur_index_q_view, cw->cur_index_weights_view,
                            cw->cur_selected_view, cw->cur_mask_view);
                        if (batched < 0) {
                            MYDS_LOG_E("V4.1 chunk indexer failed mid-flight (layer %u chunk %u)", il, c_start);
                            goto fail;
                        }
                        if (!batched && !idx_fallback_warned) {
                            idx_fallback_warned = 1;
                            MYDS_LOG_W("V4.1 chunk indexer skipped at layer %u, per-token fallback (first token / small cache)", il);
                        }
                    }
                    if (!batched) {
                        phase = "compressed KV / indexer";
                        for (uint32_t t = 0; t < c_len; t++) {
                            uint32_t idx = c_start + t;
                            uint32_t pos = mb_base_pos + idx;
                            uint32_t count = compressed ? (pos + 1u) / ratio : 0;
                            if (!myds_metal_buffer_update_sub_view(cw->cur_selected_view, prompt_selected, (size_t)idx * sel_bytes, sel_bytes) ||
                                !myds_metal_buffer_update_sub_view(cw->cur_mask_view, prompt_block_mask, (size_t)idx * mask_bytes, mask_bytes)) {
                                goto fail;
                            }
                            CHECK(publish_indexer_one(ctx, il, pos, cw->x_norm[t], cw->qr_norm[t],
                                                      index_source(il) ? cw->index_q[t] : NULL,
                                                      index_source(il) ? cw->index_weights[t] : NULL,
                                                      (count && index_source(il)) ? cw->cur_selected_view : NULL,
                                                      (il > 20 && index_source(il)) ? cw->cur_mask_view : NULL,
                                                      (il == 20) ? cw->cur_mask_view : NULL));
                        }
                    }
                }

                /* 诊断：把 Step 4 切成 indexer / attention 两段，各自测 GPU 等待。
                 * 不开这个开关时行为与原来完全一致（一次 commit 都不多）。 */
                if (do_prof && opt_prof_split) {
                    TIME_START();
                    CHECK(myds_metal_commit_batch());
                    TIME_ACC(prof_wait_indexer);
                    CHECK(myds_metal_begin_batch());
                }

                uint64_t dr0 = myds_metal_dispatch_count();   /* 诊断 */
                /* carry：进入第一个【全量】chunk 之前，把前 127 行的 KV 从窗口环搬进
                 * 注意力 slab。收缩后这个位置不再是 c_start==0（前置段可能已经占掉
                 * 若干 chunk），但窗口环此刻恰好是 [S-128, S)（前置段刚写进去）
                 * ⇒ 用 sh_full_abs 当 base_pos 调用与原来完全同一支内核。
                 * ★ 位置 0 没有「前面的 127 行」，carry_stage 对 base_pos==0 返回失败；
                 *   锥被夹到 A=0（首块 + 大 shift，或小 macroblock）时 sh_full_rel==0、
                 *   c_start==0 ⇒ 必须跳过（否则整次 prefill 直接失败，实测 HTTP 500）。 */
                const uint32_t carry_pos = mb_base_pos + c_start;
                if (shrink_layer) {
                    if (c_start == sh_full_rel && carry_pos != 0u) {
                        phase = "carry stage";
                        if (!myds_v41_gpu_carry_stage(g->gpu, g->window[il], carry_pos)) {
                            MYDS_LOG_E("carry stage failed at layer %u (pos=%u)", il, carry_pos);
                            goto fail;
                        }
                    }
                } else if (c_start == 0 && carry_pos != 0u) {
                    phase = "carry stage";
                    if (!myds_v41_gpu_carry_stage(g->gpu, g->window[il], mb_base_pos)) {
                        MYDS_LOG_E("carry stage failed at layer %u (mb_base_pos=%u)", il, mb_base_pos);
                        goto fail;
                    }
                }

                {
                    phase = "window ring";
                    uint32_t t0 = c_len > V41_WINDOW ? c_len - V41_WINDOW : 0;
                    uint32_t nw = c_len - t0;
                    uint32_t slot0 = (mb_base_pos + c_start + t0) % V41_WINDOW;
                    uint32_t run1 = V41_WINDOW - slot0; if (run1 > nw) run1 = nw;
                    if (run1 && !myds_metal_buffer_copy(g->window[il], (size_t)slot0 * 512 * 4u, cw->kv_act_buf, (size_t)t0 * 512 * 4u, (size_t)run1 * 512 * 4u)) goto fail;
                    uint32_t run2 = nw - run1;
                    if (run2 && !myds_metal_buffer_copy(g->window[il], 0, cw->kv_act_buf, (size_t)(t0 + run1) * 512 * 4u, (size_t)run2 * 512 * 4u)) goto fail;
                }

                if (kv_only) { /* 前端：attention 整体跳过（下面 !kv_only 段） */ }
                else {
                {
                    uint32_t n_comp = compressed ? (mb_base_pos + c_start + c_len) / ratio : 0;
                    phase = "chunk attention";
                    if (n_comp) {
                        if (!myds_metal_buffer_update_sub_view(cw->cur_selected_view, prompt_selected,
                                (size_t)c_start * sel_bytes, (size_t)c_len * sel_bytes)) goto fail;
                    }
                    if (!myds_v41_gpu_attention_chunk_bf16(cw->attn_heads_buf, base->attn_sinks, cw->q_buf,
                            cw->kv_act_buf,
                            n_comp ? g->compressed[owner_for(il)] : NULL,
                            n_comp ? cw->cur_selected_view : NULL,
                            cw->kv_gathered,
                            c_len, n_comp, V41_TOPK, mb_base_pos, c_start, ratio, g->gpu)) {
                        MYDS_LOG_E("chunked attention failed (c_len=%u n_comp=%u)", c_len, n_comp);
                        goto fail;
                    }
                }

                }
                if (!kv_only) {
                if (do_prof && opt_prof_split2) {
                    TIME_START();
                    CHECK(myds_metal_commit_batch());
                    TIME_ACC(prof_wait_attn2);
                    CHECK(myds_metal_begin_batch());
                }

                /* Batched inverse RoPE + BF16 rounding */
                CHECK(myds_v41_gpu_rope_batch_round(cw->attn_heads_buf, 512, V41_HEADS, mb_base_pos + c_start, c_len, compressed, true));
                disp_ringattn += myds_metal_dispatch_count() - dr0;   /* 诊断 */

                if (do_prof) { disp_step4loop += myds_metal_dispatch_count() - disp_loop0; }  /* 诊断 */

                myds_metal_barrier();
                int a_ok = 0;
                if (!opt_disable_mm_attn_a)
                    a_ok = myds_metal_mul_mm_grouped(cw->attn_low_buf, base->attn_output_a,
                            cw->attn_heads_buf, 1024, 4096, 8, c_len, base->attn_output_a_type);
                if (!a_ok)
                    a_ok = myds_metal_attn_output_a_batch(cw->attn_low_buf, base->attn_output_a,
                            cw->attn_heads_buf, 8, 4096, 1024, c_len, base->attn_output_a_type);
                CHECK(a_ok);
                CHECK(myds_metal_v41_bf16(cw->attn_low_buf, 8192 * c_len));
                TIME_ACC(prof_step4);
                if (do_prof) { disp_mark4 = myds_metal_dispatch_count(); disp_step14 += disp_mark4 - disp_layer0; }  /* 诊断 */

                if (do_prof) {
                    TIME_START();
                    CHECK(myds_metal_commit_batch());
                    TIME_ACC(prof_wait_swa);
                    CHECK(myds_metal_begin_batch());
                }

                /* Step 5: Batched Attention Output B (reads 17.8 MB weight ONCE for all c_len tokens) */
                TIME_START();
                phase = "batched attention output b";
                CHECK(project_batch(cw->attn_out_buf, base->attn_output_b, cw->attn_low_buf, V41_DIM, 8192, c_len, base->attn_output_b_type, true));
                TIME_ACC(prof_step5);
                if (do_prof && opt_prof_moe) {
                    TIME_START();
                    CHECK(myds_metal_commit_batch());
                    TIME_ACC(prof_wait_attnB);
                    CHECK(myds_metal_begin_batch());
                }

                /* Step 6: HC FFN Input for all tokens in chunk */
                TIME_START();
                phase = "HC FFN input";
                CHECK(myds_metal_v41_hc_expand_batch(cw->after_attn_buf, cw->attn_out_buf, cw->cur_res_view, cw->attn_split_buf, V41_DIM, c_len));

                /* Batched HC Mix: RMSNorm + GEMM + Split across the entire chunk */
                CHECK(myds_metal_v41_rmsnorm_batch(cw->hc_flat_norm_buf, cw->after_attn_buf, NULL, 4 * V41_DIM, c_len, m->norm_eps, false));
                CHECK(project_batch(cw->hc_mix_buf, l->ffn_fn, cw->hc_flat_norm_buf, 24, 4 * V41_DIM, c_len, GGML_TYPE_F16, false));
                CHECK(myds_metal_v41_hc_split_batch(cw->ffn_split_buf, cw->hc_mix_buf, l->ffn_scale, l->ffn_base, c_len, v->hc_eps));

                if (trace_on) {
                    for (uint32_t t = 0; t < c_len; t++) {
                        uint32_t pos = mb_base_pos + c_start + t;
                        TRACE("attn_out", cw->attn_out[t], V41_DIM);
                        TRACE("after_attn", cw->after_attn[t], 4 * V41_DIM);
                        TRACE("hc_ffn_split", cw->ffn_split[t], 24);
                    }
                }

                CHECK(myds_metal_v41_hc_weighted_sum_batch(cw->x_buf, cw->after_attn_buf, cw->attn_split_buf, V41_DIM, c_len, 96));
                CHECK(myds_metal_v41_rmsnorm_batch(cw->ffn_norm_buf, cw->x_buf, base->ffn_norm, V41_DIM, c_len, m->norm_eps, true));

                if (trace_on) {
                    for (uint32_t t = 0; t < c_len; t++) {
                        uint32_t pos = mb_base_pos + c_start + t;
                        TRACE("hc_ffn_input", cw->x[t], V41_DIM);
                        TRACE("ffn_norm", cw->ffn_norm[t], V41_DIM);
                    }
                }
                TIME_ACC(prof_step6);

                /* Step 7: Batched Router Gate & Shared Experts (reads 10 MB shared weights + 1.5 MB router ONCE) */
                TIME_START();
                phase = "batched MoE shared and router";
                uint32_t width = m->moe_expert_dim;
                CHECK(project_batch(cw->router_logits_buf, base->router_gate, cw->ffn_norm_buf, m->n_routed_experts, V41_DIM, c_len, base->router_gate_type, false));
                CHECK(project_batch(cw->shared_gate_buf, base->shared_gate, cw->ffn_norm_buf, width, V41_DIM, c_len, base->shared_gate_type, true));
                CHECK(project_batch(cw->shared_up_buf, base->shared_up, cw->ffn_norm_buf, width, V41_DIM, c_len, base->shared_up_type, true));
                CHECK(myds_v41_shared_swiglu(cw->shared_swiglu_buf, cw->shared_gate_buf, cw->shared_up_buf, width * c_len, m->swiglu_clamp_exp));
                CHECK(project_batch(cw->shared_out_buf, base->shared_down, cw->shared_swiglu_buf, V41_DIM, width, c_len, base->shared_down_type, true));

                /* Step 7b: Batched MoE routing for all tokens in chunk on GPU */
                phase = "batched MoE routing";
                uint32_t k = m->n_activated_experts;
                if (trace_on) {
                    for (uint32_t t = 0; t < c_len; t++) {
                        uint32_t pos = mb_base_pos + c_start + t;   /* TRACE 需要 pos */
                        (void)pos;
                        TRACE("route_logits", cw->router_logits[t], m->n_routed_experts);
                    }
                }
                CHECK(myds_v41_route_batch(cw->selected_ids_buf, cw->selected_weights_buf,
                                           cw->route_probs_buf, cw->route_scores_buf,
                                           cw->router_logits_buf, base->router_bias,
                                           m->n_routed_experts, k, m->routed_scaling_factor, c_len));
                if (trace_on) {
                    for (uint32_t t = 0; t < c_len; t++) {
                        uint32_t pos = mb_base_pos + c_start + t;   /* TRACE 需要 pos */
                        (void)pos;
                        TRACE("route_probs", cw->route_probs[t], m->n_routed_experts);
                        TRACE("route_weights", cw->selected_weights[t], k);
                        const myds_metal_buffer_t *id_buffers[] = {cw->selected_ids[t], cw->route_ids_f32[t]};
                        CHECK(dispatch_k("kernel_myds_v41_ids_f32", &k, sizeof(k), id_buffers, 2, 1, 32, 0));
                        TRACE("route_ids", cw->route_ids_f32[t], k);
                    }
                }
                TIME_ACC(prof_step7);
                if (do_prof && opt_prof_moe) {
                    TIME_START();
                    CHECK(myds_metal_commit_batch());
                    TIME_ACC(prof_wait_ffnpre);
                    CHECK(myds_metal_begin_batch());
                }

                /* Single commit for all Attention + FFN + Shared + Routing in the chunk */
                if (do_prof) {
                    TIME_START();
                    CHECK(myds_metal_commit_batch());
                    TIME_ACC(prof_wait_ffn_router);
                    CHECK(myds_metal_begin_batch());
                } else {
                    myds_metal_barrier();
                }

                /* Step 8: Routed MoE & Layer output.
                 * 本 chunk 的 MoE 输入先攒进批缓冲（moe_batch > capacity 时才需要搬），
                 * 攒满 —— 或到了 prompt 末尾 —— 再跑一次 MoE。attention 的数值路径与
                 * chunk 大小完全不变，所以结果与「每 chunk 一次 MoE」逐位相同。 */
                TIME_START();
                if (cw->moe_batch > cw->capacity) {
                    CHECK(myds_metal_buffer_copy(cw->moe_norm_acc, (size_t)moe_off * V41_DIM * sizeof(float),
                                                 cw->ffn_norm_buf, 0, (size_t)c_len * V41_DIM * sizeof(float)));
                    CHECK(myds_metal_buffer_copy(cw->moe_ids_acc, (size_t)moe_off * 6 * sizeof(int32_t),
                                                 cw->selected_ids_buf, 0, (size_t)c_len * 6 * sizeof(int32_t)));
                    CHECK(myds_metal_buffer_copy(cw->moe_w_acc, (size_t)moe_off * 6 * sizeof(float),
                                                 cw->selected_weights_buf, 0, (size_t)c_len * 6 * sizeof(float)));
                    CHECK(myds_metal_buffer_copy(cw->moe_shared_acc, (size_t)moe_off * V41_DIM * sizeof(float),
                                                 cw->shared_out_buf, 0, (size_t)c_len * V41_DIM * sizeof(float)));
                    CHECK(myds_metal_buffer_copy(cw->moe_after_acc, (size_t)moe_off * 4 * V41_DIM * sizeof(float),
                                                 cw->after_attn_buf, 0, (size_t)c_len * 4 * V41_DIM * sizeof(float)));
                    CHECK(myds_metal_buffer_copy(cw->moe_split_acc, (size_t)moe_off * 24 * sizeof(float),
                                                 cw->ffn_split_buf, 0, (size_t)c_len * 24 * sizeof(float)));
                }
                moe_off += c_len;

                if (moe_off == cw->moe_batch || c_start + c_len == mb_len) {
                    /* 注：本分支只在全量 chunk 里可达（moe_off 只在全量 chunk 里累加） */
                    const uint32_t mb_moe_len = moe_off;
                    const uint32_t mb_moe_start = c_start + c_len - mb_moe_len;   /* 本批在 prompt 里的起点 */
                    myds_metal_barrier();
                    CHECK(myds_metal_routed_moe_batch(cw->moe_out_acc, cw->moe_norm_acc,
                        base->ffn_gate_exps, base->ffn_up_exps, base->ffn_down_exps,
                        cw->moe_ids_acc, cw->moe_w_acc,
                        cw->moe_gate_buf, cw->moe_up_buf, cw->moe_mid_buf, cw->moe_down_buf,
                        V41_DIM, width, m->n_routed_experts, k, mb_moe_len,
                        base->ffn_gate_exps_type, base->ffn_up_exps_type, base->ffn_down_exps_type,
                        m->swiglu_clamp_exp));
                    myds_metal_barrier();
                    if (do_prof && opt_prof_moe) {
                        TIME_START();
                        CHECK(myds_metal_commit_batch());
                        TIME_ACC(prof_wait_moe_gemm);
                        CHECK(myds_metal_begin_batch());
                    }

                    /* 收尾仍按 attention chunk 逐个做：hc_expand 要用该 chunk 自己的
                     * after_attn / ffn_split（它们都被保存在批缓冲里）。 */
                    for (uint32_t e0 = 0; e0 < mb_moe_len; e0 += chunk_size) {
                        uint32_t e_len = mb_moe_len - e0;
                        if (e_len > chunk_size) e_len = chunk_size;
                        const uint32_t e_start = mb_moe_start + e0;
                        const bool aliased = (cw->moe_batch == cw->capacity);
                        if (!aliased) {
                            if (!myds_metal_buffer_update_sub_view(cw->moe_view, cw->moe_out_acc,
                                    (size_t)e0 * V41_DIM * sizeof(float), (size_t)e_len * V41_DIM * sizeof(float)) ||
                                !myds_metal_buffer_update_sub_view(cw->shared_view, cw->moe_shared_acc,
                                    (size_t)e0 * V41_DIM * sizeof(float), (size_t)e_len * V41_DIM * sizeof(float)) ||
                                !myds_metal_buffer_update_sub_view(cw->after_view, cw->moe_after_acc,
                                    (size_t)e0 * 4 * V41_DIM * sizeof(float), (size_t)e_len * 4 * V41_DIM * sizeof(float)) ||
                                !myds_metal_buffer_update_sub_view(cw->split_view, cw->moe_split_acc,
                                    (size_t)e0 * 24 * sizeof(float), (size_t)e_len * 24 * sizeof(float))) {
                                myds_metal_barrier();
                                MYDS_LOG_E("V4.1 MoE batch sub-view failed (e0=%u mb_moe_len=%u)", e0, mb_moe_len);
                                goto fail;
                            }
                        }
                        const myds_metal_buffer_t *moe_b = aliased ? cw->moe_out_buf : cw->moe_view;
                        const myds_metal_buffer_t *sh_b = aliased ? cw->shared_out_buf : cw->shared_view;
                        const myds_metal_buffer_t *af_b = aliased ? cw->after_attn_buf : cw->after_view;
                        const myds_metal_buffer_t *sp_b = aliased ? cw->ffn_split_buf : cw->split_view;

                        CHECK(myds_metal_v41_add_bf16(cw->ffn_out_buf, moe_b, sh_b, V41_DIM * e_len));
                        myds_metal_barrier();
                        CHECK(myds_metal_v41_hc_expand_batch(cw->layer_out_buf, cw->ffn_out_buf, af_b, sp_b, V41_DIM, e_len));

                        if (trace_on) {
                            for (uint32_t t = 0; t < e_len; t++) {
                                uint32_t pos = mb_base_pos + e_start + t;
                                const myds_metal_buffer_t *sv = aliased ? cw->shared_out[t] :
                                    myds_metal_buffer_sub_view(cw->moe_shared_acc,
                                        (size_t)(e0 + t) * V41_DIM * sizeof(float), V41_DIM * sizeof(float));
                                const myds_metal_buffer_t *rv = aliased ? cw->moe_out[t] :
                                    myds_metal_buffer_sub_view(cw->moe_out_acc,
                                        (size_t)(e0 + t) * V41_DIM * sizeof(float), V41_DIM * sizeof(float));
                                int traced = sv && rv &&
                                    myds_trace_f32("shared_out", il, pos, sv, V41_DIM) &&
                                    myds_trace_f32("routed_out", il, pos, rv, V41_DIM) &&
                                    myds_trace_f32("ffnblock", il, pos, cw->ffn_out[t], V41_DIM) &&
                                    myds_trace_f32("layer_out", il, pos, cw->layer_out[t], 4 * V41_DIM);
                                if (!aliased) {
                                    if (sv) myds_metal_buffer_free((myds_metal_buffer_t *)sv);
                                    if (rv) myds_metal_buffer_free((myds_metal_buffer_t *)rv);
                                }
                                if (!traced) goto fail;
                            }
                        }

                        CHECK(myds_metal_buffer_copy(prompt_res, (size_t)e_start * res_tok_bytes,
                                                     cw->layer_out_buf, 0, (size_t)e_len * res_tok_bytes));
                        CHECK(myds_metal_buffer_copy(prompt_pre, (size_t)e_start * pre_tok_bytes,
                                                     sp_b, 0, (size_t)e_len * pre_tok_bytes));
                    }
                    moe_off = 0;
                }
                }  /* !kv_only */
                /* ★ 前缀快照：本层在本 chunk 结束处的窗口环。时刻它正好含 [p-128, p-1]，
                 * 而这 128 槽随后会被后面的 token 覆盖 ⇒ 只能在这里存下来（事后无法回溯）。
                 * 与 engram 的保存用同一个确定性槽位映射，故两者必然属于同一个 p。
                 *
                 * ★★ 有效性门（2026-10-03，实测教训）：本层的环只有在 p ≥ s(il)+1 时
                 * 内容才对；不满足就不写（也绝不让最深的那一层发布）—— 否则快照会带着
                 * 「位置对得上、内容却来自旧残差」的环被兄弟请求选中续跑，输出静默错掉
                 * （上机实证：front=all 但无此门时两臂在恢复点逐字节不同，max|Δ|=57.6）。 */
                {
                    const uint32_t sp = mb_base_pos + c_start + c_len;
                    if (!shrink_layer || sp > mb_base_pos + sh_full_rel) {
                        if (!snapshot_store_window(g, il, sp, sp == base_pos + n_tokens)) goto fail;
                    }
                    /* ★ 注意：拒绝时【不能】顺手清槽位 —— 槽是按位置模数映射的，同一个槽可能
                     * 正存着另一个（有效的）已发布位置。2026-10-03 実测踩过：删掉这一句之前，
                     * 末块里被拒的 sp 会把「本块末尾 B」的已发布快照一并清掉（实测 x/l 两臂
                     * 的续跑点于是从 8192 退到 4096）。staged 标记留着无害：它只对同一个 p 生效，
                     * 而位置单调递增 ⇒ 同一个 p 不会再被访问。 */
                }
                TIME_ACC(prof_step8);

                TIME_START();
                CHECK(myds_metal_commit_batch());
                TIME_ACC(prof_wait_moe);
                if (do_prof) {  /* 诊断 */
                    disp_layer_all += myds_metal_dispatch_count() - disp_layer0;
                    /* ★ 减【本次迭代】的 Step4 末尾标记，不是跨迭代累加的 disp_step14
                     *   （旧写法从第 2 次迭代起为负，按 uint64 回绕成 ~1.8e19）。 */
                    disp_step58 += myds_metal_dispatch_count() - disp_mark4;
                }
                c_start += c_len;
            } /* chunk loop */
        } /* layer loop */

        /* 每个 macroblock 跑完 40 层后，把该 macroblock 最后一个 token 的 residual / pre 保存进 g->residual / g->pre */
        {
            size_t last_mb_idx = mb_len - 1u;
            CHECK(myds_metal_buffer_copy(g->residual, 0, prompt_res, last_mb_idx * res_tok_bytes, res_tok_bytes));
            CHECK(myds_metal_buffer_copy(g->pre, 0, prompt_pre, last_mb_idx * pre_tok_bytes, 4 * sizeof(float)));
            CHECK(myds_metal_commit_batch());
        }
    } /* macroblock loop */

    if (do_prof) {
        gettimeofday(&ptv_all1, NULL);
        prof_total_layers = (ptv_all1.tv_sec - ptv_all0.tv_sec) * 1000.0 + (ptv_all1.tv_usec - ptv_all0.tv_usec) / 1000.0;
        printf("--- PREFILL PROFILE (%u tokens, %u layers) ---\n", n_tokens, V41_LAYERS);
        printf("Engram Prepare        : %8.2f ms\n", prof_engram);
        printf("Embedding Lookup      : %8.2f ms\n", prof_embedding);
        printf("Step 1 (HC Attn In)   : %8.2f ms\n", prof_step1);
        printf("Step 2 (Q_A & KV GEMV): %8.2f ms\n", prof_step2);
        printf("Step 3 (Q_B GEMV)     : %8.2f ms\n", prof_step3);
        printf("Wait Proj (GPU)       : %8.2f ms\n", prof_wait_qkv);
        printf("Step 4 (SWA Attention): %8.2f ms\n", prof_step4);
        printf("Wait SWA (GPU)        : %8.2f ms\n", prof_wait_swa);
        if (opt_prof_split)
            printf("  ...of which indexer  : %8.2f ms\n", prof_wait_indexer);
        if (opt_prof_split2)
            printf("  ...ring+attention    : %8.2f ms\n", prof_wait_attn2);
        if (opt_prof_split3)
            printf("  ...bf16(attn_heads)  : %8.2f ms  (rope = remainder)\n", prof_wait_bf16);
        printf("Step 5 (Attn Out B)   : %8.2f ms\n", prof_step5);
        printf("Step 6 (HC FFN In)    : %8.2f ms\n", prof_step6);
        printf("Step 7 (Shared+Route) : %8.2f ms\n", prof_step7);
        printf("Wait FFN+R (GPU)      : %8.2f ms\n", prof_wait_ffn_router);
        printf("Step 8 (MoE dispatch) : %8.2f ms\n", prof_step8);
        if (opt_prof_step1) {
            printf("  [Step1] norm=%.1f mix=%.1f split=%.1f wsum=%.1f norm2=%.1f ms\n",
                   prof_s1_norm, prof_s1_mix, prof_s1_split, prof_s1_wsum, prof_s1_norm2);
        }
        printf("Wait MoE (GPU)        : %8.2f ms\n", prof_wait_moe);
        if (opt_prof_moe) {
            printf("  ...attn_out B       : %8.2f ms\n", prof_wait_attnB);
            printf("  ...FFN-in+sh+router : %8.2f ms\n", prof_wait_ffnpre);
            printf("  ===> MoE GEMMs      : %8.2f ms\n", prof_wait_moe_gemm);
            printf("  ...epilogue(add/hc) : %8.2f ms\n",
                   prof_wait_moe - prof_wait_attnB - prof_wait_ffnpre - prof_wait_moe_gemm);
        }
        printf("Total Layers Wall     : %8.2f ms\n", prof_total_layers);
        {   /* 诊断：dispatch 次数 */
            uint64_t layer_iters = (uint64_t)V41_LAYERS * (uint64_t)((n_tokens + chunk_size - 1u) / (chunk_size ? chunk_size : 1u));
            double kvpl = layer_iters ? (double)disp_layer_all / (double)layer_iters : 0.0;
            printf("--- DISPATCH COUNT (诊断) ---\n");
            printf("Step1-4 dispatches    : %10llu\n", (unsigned long long)disp_step14);
            printf("Step5-8 dispatches    : %10llu\n", (unsigned long long)disp_step58);
            printf("Layer dispatches total: %10llu\n", (unsigned long long)disp_layer_all);
            printf("Layer iterations      : %10llu\n", (unsigned long long)layer_iters);
            printf("Step4 loop dispatches : %10llu\n", (unsigned long long)disp_step4loop);
            printf("Seg: compressor       : %10llu  (%.1f / layer-iter)\n",
                   (unsigned long long)disp_compressor, (double)disp_compressor / (double)layer_iters);
            printf("Seg: ring+attn+bf16+rope: %9llu  (%.1f / layer-iter, %.1f / chunk-attn-call)\n",
                   (unsigned long long)disp_ringattn, (double)disp_ringattn / (double)layer_iters,
                   (double)disp_ringattn / (double)layer_iters);
            printf("dispatches / layer-iter: %8.1f  (Step1-4: %.1f, Step4loop: %.1f)\n",
                   kvpl, layer_iters ? (double)disp_step14 / (double)layer_iters : 0.0,
                   layer_iters ? (double)disp_step4loop / (double)layer_iters : 0.0);
        }
        printf("Total Steps Sum       : %8.2f ms\n",
               prof_step1 + prof_step2 + prof_step3 + prof_wait_qkv + prof_step4 + prof_wait_indexer + prof_wait_attn2 + prof_wait_bf16 + prof_wait_swa +
               prof_step5 + prof_step6 + prof_step7 + prof_wait_ffn_router + prof_step8 + prof_wait_moe);
        fflush(stdout);
    }

    ctx->pos = base_pos + n_tokens;
    g->next_position = ctx->pos;
    g->valid = true;
#undef CHECK
#undef TRACE

    if (ctx->cfg.temperature > 0.0f) {
        for (uint32_t i = 0; i < n_tokens; i++) {
            ctx->rng_state = ctx->rng_state * 6364136223846793005ULL + 1ULL;
        }
    }

    return 1;

cancelled:
    (void)myds_metal_commit_batch();
    return -2;

fail:
    MYDS_LOG_E("V4.1 prefill failed at layer %u: %s", il, phase);
    (void)myds_metal_commit_batch();
    g->valid = false;
    v41_chunk_workspace_free(g->chunk_ws);
    g->chunk_ws = NULL;
    if (g->prompt_res) { myds_metal_buffer_free(g->prompt_res); g->prompt_res = NULL; }
    if (g->prompt_pre) { myds_metal_buffer_free(g->prompt_pre); g->prompt_pre = NULL; }
    if (g->prompt_selected) { myds_metal_buffer_free(g->prompt_selected); g->prompt_selected = NULL; }
    if (g->prompt_block_mask) { myds_metal_buffer_free(g->prompt_block_mask); g->prompt_block_mask = NULL; }
    if (g->prompt_engram[0]) { free(g->prompt_engram[0]); g->prompt_engram[0] = NULL; }
    if (g->prompt_engram[1]) { free(g->prompt_engram[1]); g->prompt_engram[1] = NULL; }
    g->prompt_tokens_cap = 0;
    g->prompt_mask_bytes_cap = 0;
    return -1;
}
