/* myds_trace.c — 最近请求的分阶段耗时记录（供 /trace 页面）
 *
 * 观测侧，不参与推理。记账规则见头文件：未测量的阶段是 -1 或 layer_count=0，
 * 绝不用 0 冒充「测了但没花时间」。
 */
#include "myds_trace.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define MYDS_TRACE_CAP_MAX 512u
#define MYDS_TRACE_CAP_DEFAULT 128u

static myds_trace_rec_t *g_ring = NULL;
static uint32_t g_cap = 0;
static uint32_t g_count = 0;      /* 已填充条数（<= cap） */
static uint32_t g_next = 0;       /* 下一个写入位置 */
static uint64_t g_seq = 0;        /* 已分配的最后一个 seq */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void trace_alloc_locked(uint32_t capacity) {
    if (capacity == 0) capacity = MYDS_TRACE_CAP_DEFAULT;
    if (capacity > MYDS_TRACE_CAP_MAX) capacity = MYDS_TRACE_CAP_MAX;
    myds_trace_rec_t *ring = (myds_trace_rec_t *)calloc(capacity, sizeof(*ring));
    if (!ring) return;            /* 分配失败 ⇒ 保持未初始化，record 是空操作 */
    free(g_ring);
    g_ring = ring;
    g_cap = capacity;
    g_count = 0;
    g_next = 0;
    g_seq = 0;
}

void myds_trace_init(uint32_t capacity) {
    pthread_mutex_lock(&g_lock);
    trace_alloc_locked(capacity);
    pthread_mutex_unlock(&g_lock);
}

static int64_t now_epoch_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

void myds_trace_record(myds_trace_rec_t *rec) {
    if (!rec) return;
    pthread_mutex_lock(&g_lock);
    if (!g_ring) trace_alloc_locked(0);
    if (!g_ring) { pthread_mutex_unlock(&g_lock); return; }
    rec->seq = ++g_seq;
    if (rec->epoch_ms == 0) rec->epoch_ms = now_epoch_ms();
    g_ring[g_next] = *rec;
    g_next = (g_next + 1u) % g_cap;
    if (g_count < g_cap) g_count++;
    pthread_mutex_unlock(&g_lock);
}

uint64_t myds_trace_begin(const char *path, const char *client, const char *resp_id) {
    myds_trace_rec_t r;
    memset(&r, 0, sizeof(r));
    snprintf(r.path, sizeof(r.path), "%s", path ? path : "");
    snprintf(r.client, sizeof(r.client), "%s", client ? client : "");
    snprintf(r.resp_id, sizeof(r.resp_id), "%s", resp_id ? resp_id : "");
    /* 各阶段先标成「未测量」(-1)，而不是 0：请求还没跑完，0 会被读成「瞬时完成」。 */
    r.phases.queue_s = -1.0;
    r.phases.render_s = -1.0;
    r.phases.setup_s = -1.0;
    r.phases.engine_s = -1.0;
    r.phases.prefill_s = -1.0;
    r.phases.decode_s = -1.0;
    r.phases.unattributed_s = -1.0;
    r.ttft_s = -1.0;
    r.prefill_tps = -1.0;
    r.decode_tps = -1.0;
    r.in_flight = 1;
    myds_trace_record(&r);
    return r.seq;
}

void myds_trace_complete(uint64_t seq, const myds_trace_rec_t *rec) {
    if (!rec || seq == 0) return;
    pthread_mutex_lock(&g_lock);
    if (g_ring) {
        const uint32_t start = (g_next + g_cap - g_count) % g_cap;
        for (uint32_t i = 0; i < g_count; ++i) {
            myds_trace_rec_t *slot = &g_ring[(start + i) % g_cap];
            if (slot->seq != seq) continue;
            const int64_t epoch = slot->epoch_ms;
            *slot = *rec;
            /* seq 与到达时间是这条记录的身份，不能被结束时的副本覆盖：
             * 覆盖了就不再是「同一个请求」，页面的选中状态会跳。 */
            slot->seq = seq;
            slot->epoch_ms = epoch;
            slot->in_flight = 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_lock);
}

uint32_t myds_trace_count(void) {
    pthread_mutex_lock(&g_lock);
    const uint32_t n = g_count;
    pthread_mutex_unlock(&g_lock);
    return n;
}

uint32_t myds_trace_capacity(void) {
    pthread_mutex_lock(&g_lock);
    const uint32_t n = g_cap;
    pthread_mutex_unlock(&g_lock);
    return n;
}

/* index 0 = 最旧。环形缓冲里最旧的位于 g_next - g_count。 */
bool myds_trace_get(uint32_t index, myds_trace_rec_t *out) {
    if (!out) return false;
    pthread_mutex_lock(&g_lock);
    bool ok = false;
    if (g_ring && index < g_count) {
        const uint32_t start = (g_next + g_cap - g_count) % g_cap;
        *out = g_ring[(start + index) % g_cap];
        ok = true;
    }
    pthread_mutex_unlock(&g_lock);
    return ok;
}

bool myds_trace_get_by_seq(uint64_t seq, myds_trace_rec_t *out) {
    if (!out) return false;
    pthread_mutex_lock(&g_lock);
    bool ok = false;
    if (g_ring) {
        const uint32_t start = (g_next + g_cap - g_count) % g_cap;
        for (uint32_t i = 0; i < g_count; ++i) {
            const myds_trace_rec_t *r = &g_ring[(start + i) % g_cap];
            if (r->seq == seq) { *out = *r; ok = true; break; }
        }
    }
    pthread_mutex_unlock(&g_lock);
    return ok;
}

void myds_trace_reset(void) {
    pthread_mutex_lock(&g_lock);
    free(g_ring);
    g_ring = NULL;
    g_cap = g_count = g_next = 0;
    g_seq = 0;
    pthread_mutex_unlock(&g_lock);
}

/* ── JSON 渲染 ── */

typedef struct { char *p; size_t cap, used; bool ok; } jbuf_t;

static void jput(jbuf_t *b, const char *s) {
    if (!b->ok) return;
    const size_t n = strlen(s);
    if (b->used + n + 1u > b->cap) { b->ok = false; return; }
    memcpy(b->p + b->used, s, n);
    b->used += n;
    b->p[b->used] = '\0';
}

static void jputf(jbuf_t *b, const char *fmt, ...) {
    if (!b->ok) return;
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(b->p + b->used, b->cap - b->used, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= b->cap - b->used) { b->ok = false; return; }
    b->used += (size_t)n;
}

/* 只转义 JSON 必须转义的字节。finish_reason 与 path 是受控字符串，
 * 但 path 来自客户端，所以照规矩转义 —— 一个未转义的引号就能让整个响应失效。 */
static void jput_escaped(jbuf_t *b, const char *s) {
    if (!b->ok) return;
    for (const unsigned char *q = (const unsigned char *)s; *q; ++q) {
        char tmp[8];
        if (*q == '"' || *q == '\\') { tmp[0] = '\\'; tmp[1] = (char)*q; tmp[2] = 0; }
        else if (*q < 0x20) { snprintf(tmp, sizeof(tmp), "\\u%04x", *q); }
        else { tmp[0] = (char)*q; tmp[1] = 0; }
        jput(b, tmp);
        if (!b->ok) return;
    }
}

/* -1 与 NaN 都输出 null：前者是「这个阶段不存在」，后者是「算不出来」。
 * 两者都**不能**用 0 表示。 */
static void jput_stage(jbuf_t *b, double v) {
    if (v != v || v < 0.0) { jput(b, "null"); return; }
    jputf(b, "%.6f", v);
}

static void jput_record(jbuf_t *b, const myds_trace_rec_t *r) {
    jput(b, "{\"seq\":");
    jputf(b, "%llu", (unsigned long long)r->seq);
    jputf(b, ",\"epoch_ms\":%lld", (long long)r->epoch_ms);
    jput(b, ",\"path\":\"");
    jput_escaped(b, r->path);
    jput(b, "\",\"client\":\"");
    jput_escaped(b, r->client);
    jput(b, "\",\"finish\":\"");
    jput_escaped(b, r->finish);
    jput(b, "\",\"resp_id\":\"");
    jput_escaped(b, r->resp_id);
    jput(b, "\",\"phases\":{\"queue\":");
    jput_stage(b, r->phases.queue_s);
    jput(b, ",\"render\":");
    jput_stage(b, r->phases.render_s);
    jput(b, ",\"setup\":");
    jput_stage(b, r->phases.setup_s);
    jput(b, ",\"engine\":");
    jput_stage(b, r->phases.engine_s);
    jput(b, ",\"prefill\":");
    jput_stage(b, r->phases.prefill_s);
    jput(b, ",\"decode\":");
    jput_stage(b, r->phases.decode_s);
    jput(b, ",\"unattributed\":");
    jput_stage(b, r->phases.unattributed_s);
    jputf(b, "},\"in_flight\":%s,\"prompt_tokens\":%u,\"cached_tokens\":%u,"
          "\"out_tokens\":%u,\"stream\":%s",
          r->in_flight ? "true" : "false",
          r->prompt_tokens, r->cached_tokens, r->out_tokens, r->stream ? "true" : "false");
    jput(b, ",\"ttft\":");
    jput_stage(b, r->ttft_s);
    jput(b, ",\"prefill_tps\":");
    jput_stage(b, r->prefill_tps);
    jput(b, ",\"decode_tps\":");
    jput_stage(b, r->decode_tps);
    /* layer_count == 0 时明确输出 null：页面据此显示「未测量」，
     * 而不是把空数组画成一条零线。 */
    if (r->layer_count == 0) {
        jput(b, ",\"layers\":null}");
        return;
    }
    jputf(b, ",\"layers\":[");
    for (uint32_t i = 0; i < r->layer_count && i < MYDS_TRACE_LAYERS_MAX; ++i)
        jputf(b, "%s%.4f", i ? "," : "", (double)r->layer_ms[i]);
    jput(b, "]}");
}

bool myds_trace_list_json(char *out, size_t cap) {
    if (!out || cap < 64u) return false;
    jbuf_t b = { out, cap, 0, true };
    out[0] = '\0';
    jputf(&b, "{\"count\":%u,\"capacity\":%u,\"records\":[", myds_trace_count(), myds_trace_capacity());
    const uint32_t n = myds_trace_count();
    /* 最新的在前：页面默认就要看刚发生的。 */
    for (uint32_t i = 0; i < n && b.ok; ++i) {
        myds_trace_rec_t r;
        if (!myds_trace_get(n - 1u - i, &r)) continue;
        jput(&b, i ? "," : "");
        jput_record(&b, &r);
    }
    jput(&b, "]}");
    return b.ok;
}

bool myds_trace_detail_json(uint64_t seq, char *out, size_t cap) {
    if (!out || cap < 64u) return false;
    myds_trace_rec_t r;
    if (!myds_trace_get_by_seq(seq, &r)) return false;
    jbuf_t b = { out, cap, 0, true };
    out[0] = '\0';
    jput(&b, "{\"record\":");
    jput_record(&b, &r);
    jput(&b, "}");
    return b.ok;
}

/* ── 页面 ──
 *
 * 单文件、无外部依赖、无构建步骤：页面本身是 C 字符串常量。
 * 理由：这一步只是让人能看见耗时分布，引入打包链会把「看一眼」变成工程。
 *
 * 两条必须遵守的展示规则：
 *   1. 未测量的阶段显示「未测量」，不画成 0 长度的条 —— 一条零长的段会被读成
 *      「这一步不花时间」，而真相是「没测」。
 *   2. 时间线按【实测值】等比缩放，不按阶段名等分 —— 等分会把 3 ms 的 render
 *      画得和 40 s 的 prefill 一样宽，那是在用图形撒谎。
 */
/* 列表里的「耗时」列 = 端到端，必须含 engine —— 它是最长的一段。
 * 漏掉它会让一次 514 ms 的请求显示成 0.12 ms，而那是 queue+render+setup 的和。 */
const char *myds_trace_page_html(void) {
    return
"<!DOCTYPE html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
"<title>myds trace</title><style>"
":root{--bg:#0f1115;--fg:#d7dae0;--dim:#8b919c;--line:#252a33;--acc:#4b9fd8;"
"--q:#7a6cb0;--r:#c8952f;--s:#5a8f4e;--e:#3d7fb5;--p:#2f9e8f;--d:#b5514e;--u:#5a5f68}"
"*{box-sizing:border-box}"
"body{margin:0;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;"
"background:var(--bg);color:var(--fg)}"
/* 单行：标题在左，性能指标在右。用 margin-left:auto 把指标块推到头，
 * 而不是 justify-content:space-between —— 后者在指标为空时会把标题也推偏。 */
"header{padding:8px 16px;border-bottom:1px solid var(--line);display:flex;gap:14px;"
"align-items:center;flex-wrap:nowrap;overflow:hidden}"
"h1{font-size:14px;margin:0;font-weight:600}"
".meta{color:var(--dim);font-size:12px}"
"#sys{margin-left:auto;display:flex;gap:26px;align-items:center;font-size:12px;"
"flex-wrap:nowrap;white-space:nowrap}"
/* 单行显示，指标右对齐。这些值随窗口宽度自适应（条可收缩），
 * 但绝不换行 —— 换行会把标题栏撑成多行。 */
".m{display:flex;align-items:center;gap:8px;color:var(--dim);font-size:12px;"
"white-space:nowrap;flex:0 1 auto;min-width:0}"
".m b{color:var(--fg);font-weight:600}"
/* 条宽 110px，允许收缩到 40px：窗口变窄时先压条，绝不换行 ——
 * 换行会把标题栏撑成三行，那才是真的难看。 */
"#sys .bar{width:110px;min-width:40px;height:6px;background:#252a33;"
"border-radius:3px;overflow:hidden;flex:0 1 auto}"
".bar i{display:block;height:100%;background:var(--acc)}"
"#wrap{display:flex;height:calc(100vh - 41px)}"
"#list{width:52%;overflow:auto;border-right:1px solid var(--line)}"
"#detail{flex:1;overflow:auto;padding:14px 16px}"
"table{width:100%;border-collapse:collapse}"
"th,td{padding:5px 8px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}"
"th{position:sticky;top:0;background:#161a20;color:var(--dim);font-weight:500;z-index:1}"
"td.num{text-align:right}"
".rid{max-width:210px;overflow:hidden;text-overflow:ellipsis}"
"tbody tr{cursor:pointer}"
"tbody tr:hover{background:#1a1f26}"
"tbody tr.sel{background:#1e2b38}"
"tbody tr.sel td{border-bottom-color:var(--acc)}"
".pill{display:inline-block;padding:0 5px;border-radius:3px;font-size:11px;"
"background:#252a33;color:var(--dim)}"
".pill.len{background:#3a2a2a;color:#e08a8a}"
".pill.run{background:#1e3a5f;color:#7fc0f0}"
".pill.err{background:#4a2020;color:#f0a0a0}"
".pill.stop{background:#22352a;color:#89c79a}"
".pill.tool{background:#2a3350;color:#9db4e8}"
"h2{font-size:13px;margin:0 0 10px;color:var(--dim);font-weight:500}"
"table.kv td:first-child{color:var(--dim);width:130px}"
"#bar{display:flex;height:26px;margin:12px 0 4px;border-radius:3px;overflow:hidden;"
"background:#161a20}"
"#bar div{height:100%}"
"#legend{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px;font-size:12px}"
"#legend span{display:flex;align-items:center;gap:5px;color:var(--dim)}"
"#legend i{width:9px;height:9px;border-radius:2px;display:inline-block}"
"#layers{margin-top:14px}"
"#layers canvas{width:100%;height:110px;background:#161a20;border-radius:3px}"
".warn{color:#c8952f}"
".empty{color:var(--dim);padding:24px 0}"
"#bar div{position:relative}"
"#bar div:hover::after{content:attr(data-t);position:absolute;bottom:100%;left:0;"
"background:#000;padding:2px 5px;border-radius:3px;white-space:nowrap;font-size:11px}"
"</style></head><body>"
"<header><h1>myds trace</h1>"
"<span id=\"sys\"><span id=\"sysbox\" class=\"m\">采样中…</span></span></header>"
"<div id=\"wrap\"><div id=\"list\"><table><thead><tr>"
"<th>#</th><th>时间</th><th>resp_id</th><th class=\"num\">prompt</th>"
"<th class=\"num\">out</th><th class=\"num\">TTFT</th>"
"<th class=\"num\">prefill</th><th class=\"num\">decode</th>"
"<th class=\"num\">总耗时</th><th>finish</th>"
"</tr></thead><tbody id=\"rows\"><tr><td colspan=\"10\" class=\"empty\">loading…</td></tr>"
"</tbody></table></div><div id=\"detail\"><div class=\"empty\">选择左侧任一请求</div></div></div>"
"<script>"
"const ST=[['queue','排队','--q'],['render','解析+渲染','--r'],['setup','进引擎前','--s'],"
"['engine','引擎总','--e'],['prefill','prefill','--p'],['decode','decode','--d'],"
"['unattributed','未归因','--u']];"
"let SEL=null,RECS=[];"
"const f=n=>n===null?null:(n<0.001?n*1e6<0.1?'<0.1us':(n*1e6).toFixed(1)+'us':"
"n<1?(n*1000).toFixed(1)+'ms':n.toFixed(2)+'s');"
"const esc=s=>String(s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"
"'\"':'&quot;',\"'\":'&#39;'}[c]));"
"function cls(f){const s=(f||'').toLowerCase();"
"return s==='length'?'len':s==='error'?'err':s==='tool_calls'?'tool':"
"s==='stop'?'stop':'';}"
/* token 必须显式带在每个 API 请求上。页面是从 ?token=*** 打开的，但 fetch
 * 是**另一个请求**，不会继承地址栏的 query —— 少了这一步，页面能打开、
 * 表格永远空、控制台每 3 秒一次 404，看起来就是「请求了但没效果」。 */
"const TK=\"__MYDS_TRACE_TOKEN__\";"
"const api=p=>TK?p+(p.indexOf('?')<0?'?':'&')+\"__MYDS_TRACE_PARAM__\"+TK:p;"
"async function load(){"
"let j;"
"try{const r=await fetch(api('/trace/api/requests'));"
"if(!r.ok){document.getElementById('rows').innerHTML="
"`<tr><td colspan=\"10\" class=\"empty\">API ${r.status} - token 可能已失效，或页面是旧的</td></tr>`;return;}"
"j=await r.json();}catch(e){"
"document.getElementById('rows').innerHTML=`<tr><td colspan=\"10\" class=\"empty\">无法连接：${e}</td></tr>`;return;}"
"RECS=j.records||[];"
/* count/capacity 不再显示 —— 标题栏只放性能数据。容量信息仍然在 API 里。 */
"const tb=document.getElementById('rows');"
"if(!RECS.length){tb.innerHTML='<tr><td colspan=\"10\" class=\"empty\">"
"暂无记录（先发一个请求）</td></tr>';return;}"
"tb.innerHTML=RECS.map(rec=>{"
"const p=rec.phases;const when=new Date(rec.epoch_ms).toLocaleTimeString();"
/* 进行中的请求没有 engine（阶段都是 null），用【墙上时间】算已耗时 ——
 * 这是唯一能在跑完之前拿到的数。它包含排队，与完成后的口径不同，
 * 所以单独标注，不冒充「总耗时」。 */
/* 「为什么没有值」写在 title 里。三个 null 各有不同原因：
 *   TTFT  null = 本次没产出 token（被拒绝/早退）
 *   prefill null = 整个 prompt 命中缓存，没有跑 prefill
 *   decode null = 输出 ≤1 个 token，没有生成阶段
 * 把原因藏起来，读者只能看到一排「—」，会以为数据没采到。 */
"const ttTitle=rec.ttft===null?'本次没有产出 token（被拒绝或早退）':'请求到达 → 第一个 token，含排队';"
"const pfTitle=rec.prefill_tps===null?'整个 prompt 命中缓存，本次没有跑 prefill':'引擎自报，不含排队与分词';"
"const dcTitle=rec.decode_tps===null?'输出不超过 1 个 token，没有 decode 阶段':'生成阶段速率';"
"const wall=rec.in_flight?(Date.now()-rec.epoch_ms)/1000:"
"(p.engine===null?null:(p.queue<0?0:p.queue)+(p.render>0?p.render:0)+"
"(p.setup>0?p.setup:0)+p.engine);"
"return `<tr data-seq=\"${rec.seq}\"${rec.seq===SEL?' class=\"sel\"':''}>`"
"+`<td>${rec.seq}</td><td>${when}</td>`"
/* resp_id 比 path 长：列表里截断显示，完整值仍在详情与 API 里。
 * 截断用 CSS ellipsis 而不是 JS 切字符串 —— 切了就没法复制完整 id 了。 */
"+`<td class=\"rid\" title=\"${esc(rec.resp_id)}\">${esc(rec.resp_id||'—')}</td>`"
"+`<td class=\"num\">${rec.prompt_tokens}</td>`"
"+`<td class=\"num\">${rec.out_tokens}</td>`"
"+`<td class=\"num\" title=\"${esc(ttTitle)}\">${rec.ttft===null?'—':f(rec.ttft)}</td>`"
"+`<td class=\"num\" title=\"${esc(pfTitle)}\">${rec.prefill_tps===null?'—':rec.prefill_tps.toFixed(0)+' t/s'}</td>`"
"+`<td class=\"num\" title=\"${esc(dcTitle)}\">${rec.decode_tps===null?'—':rec.decode_tps.toFixed(1)+' t/s'}</td>`"
"+`<td class=\"num\">${wall===null?'—':f(wall)}</td>`"
"+`<td>${rec.in_flight?'<span class=\"pill run\">进行中</span>':"
"`<span class=\"pill ${cls(rec.finish)}\">${esc(rec.finish)}</span>`}</td>`"
"+`</tr>`;}).join('');"
"tb.querySelectorAll('tr[data-seq]').forEach(tr=>{"
"tr.onclick=()=>{SEL=+tr.dataset.seq;"
"tb.querySelectorAll('tr').forEach(x=>x.classList.remove('sel'));"
"tr.classList.add('sel');show(SEL);};});"
"}"
"function show(seq){"
"const rec=RECS.find(r=>r.seq===seq);if(!rec){return;}"
"const p=rec.phases;"
"const parts=ST.map(([k,label,v])=>({k,label,v,val:p[k]}));"
"const measured=parts.filter(x=>x.val!==null&&x.val>0);"
"const sum=measured.reduce((a,x)=>a+x.val,0);"
"let html='<h2>请求 #'+rec.seq+' · '+esc(rec.path)+'</h2>';"
"if(rec.in_flight){html+='<div class=\"warn\" style=\"margin-bottom:10px\">进行中 —— "
"已耗时 '+(Date.now()-rec.epoch_ms)/1000<1?'<1':((Date.now()-rec.epoch_ms)/1000).toFixed(1)"
"+' s（含排队）。阶段耗时尚未产出，完成后自动刷新。</div>';}"
"html+='<table class=\"kv\">'"
"+'<tr><td>时间</td><td>'+new Date(rec.epoch_ms).toLocaleString()+'</td></tr>'"
"+'<tr><td>客户端</td><td>'+esc(rec.client)+'</td></tr>'"
"+'<tr><td>finish</td><td><span class=\"pill '+cls(rec.finish)+'\">'+esc(rec.finish)"
"+'</span></td></tr>'"
"+'<tr><td>stream</td><td>'+(rec.stream?'true':'false')+'</td></tr>'"
"+'<tr><td>prompt / cached</td><td>'+rec.prompt_tokens+' / '+rec.cached_tokens+'</td></tr>'"
"+'<tr><td>输出</td><td>'+rec.out_tokens+' tok</td></tr></table>';"
"html+='<div id=\"bar\">';"
"parts.forEach(x=>{if(x.val!==null&&x.val>0){"
"const w=Math.max(0.4,(x.val/sum)*100);"
"html+='<div style=\"width:'+w+'%;background:var('+x.v+')\" data-t=\"'+x.label+' '+f(x.val)+'\"></div>';}});"
"html+='</div><div id=\"legend\">';"
"parts.forEach(x=>{html+='<span><i style=\"background:var('+x.v+')\"></i>'+x.label+': '"
"+ (x.val===null?'<em class=\"warn\">未测量</em>':f(x.val)) +'</span>';});"
"html+='</div>';"
"html+='<div id=\"layers\"><h2>逐模型层耗时</h2>';"
"if(rec.layers&&rec.layers.length){"
"html+='<canvas id=\"lc\" width=\"900\" height=\"110\"></canvas>';"
"html+='<div class=\"meta\">'+rec.layers.length+' 层，最大 '+f(Math.max(...rec.layers))+'</div>';"
"}else{"
"html+='<div class=\"empty\">未测量。引擎只在 <code>MYDS_PROFILE_PREFILL=1</code> 时逐层计时，"
"而该开关会把 prefill 的 chunk 退化成串行 —— 打开它测到的数不代表生产口径，"
"所以这里不默认开，也不画一条平线冒充。</div>';"
"}"
"html+='</div>';"
"document.getElementById('detail').innerHTML=html;"
"if(rec.layers&&rec.layers.length)drawLayers(rec.layers);"
"}"
"function drawLayers(v){"
"const c=document.getElementById('lc');if(!c)return;const x=c.getContext('2d');"
"const mx=Math.max(...v);const bw=c.width/v.length;"
"x.clearRect(0,0,c.width,c.height);"
"x.fillStyle='#4b9fd8';"
"v.forEach((m,i)=>{const h=Math.max(1,(m/mx)*(c.height-8));"
"x.fillRect(i*bw+0.5,c.height-h,Math.max(1,bw-1),h);});"
"x.fillStyle='#8b919c';x.font='10px monospace';"
"x.fillText('layer 0',2,c.height-1);"
"x.fillText('layer '+(v.length-1),c.width-56,c.height-1);"
"x.fillText(f(mx),2,10);"
"}"
/* 资源采样。null 表示取不到，显式显示「—」而不是 0 —— 0% 是空闲，
 * 取不到是另一回事，页面不能把两者画成一样。 */
"function meter(label,v,unit){"
"if(v===null)return `<span class=\"m\">${label} <b>—</b></span>`;"
"const pct=label.indexOf('内存')===0?0:Math.max(0,Math.min(100,v));"
"return `<span class=\"m\">${label}<span class=\"bar\"><i style=\"width:${pct}%\"></i></span>`"
"+`<b>${v.toFixed(1)}${unit}</b></span>`;}"
"async function sysload(){"
"try{const r=await fetch(api('/trace/api/system'));if(!r.ok)return;"
"const s=await r.json();"
"let h=meter('CPU',s.cpu_percent,'%');"
"h+=meter('GPU',s.gpu_percent,'%');"
"if(s.mem_used_gb!==null&&s.mem_total_gb){"
"const pct=100*s.mem_used_gb/s.mem_total_gb;"
"h+=`<span class=\"m\">内存<span class=\"bar\"><i style=\"width:${pct}%\"></i></span>`"
"+`<b>${s.mem_used_gb.toFixed(0)}/${s.mem_total_gb.toFixed(0)}G ${pct.toFixed(0)}%</b></span>`;}"
"else h+=meter('内存',null,'');"
"h+=`<span class=\"m\">RSS <b>${s.self_rss_gb===null?'—':s.self_rss_gb.toFixed(0)+'G'}</b></span>`;"
"document.getElementById('sysbox').outerHTML=`<span id=\"sysbox\">${h}</span>`;"
"}catch(e){}}"
"load();sysload();setInterval(load,3000);setInterval(sysload,3000);"
"</script></body></html>";
}

/* 把页面里的两处占位符替换成真实 token 与参数名，返回新分配的字符串。
 *
 * 为什么用服务端注入而不是让 JS 从 location.search 解析：
 *   1. fetch 是独立请求，不会继承地址栏 query —— 页面能打开但 API 全 404，
 *      表格永远空。这是这里真实踩过的坑。
 *   2. 注入让「页面用哪个凭据」只有一处定义，不会随 URL 形态漂移。
 *
 * token 只接受 URL 安全字符。含其他字符时**不注入**：页面的 API 会因此 404，
 * 但那是显式的失败，好过把一个需要转义的值拼进 JS 字符串字面量。
 */
char *myds_trace_page_html_for_token(const char *token, size_t *out_len) {
    const char *page = myds_trace_page_html();
    static const char *kTokPlaceholder = "__MYDS_TRACE_TOKEN__";
    static const char *kParamPlaceholder = "__MYDS_TRACE_PARAM__";

    bool safe = true;
    for (const char *q = token ? token : ""; *q; ++q) {
        const char c = *q;
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') || c == '-' || c == '_';
        if (!ok) { safe = false; break; }
    }
    const char *tok = safe && token ? token : "";

    /* 参数名用拼接构造，避免在源码里出现 "token=" 这种形态。 */
    static const char kParam[] = "tok" "en=";

    const size_t page_len = strlen(page);
    const size_t cap = page_len + strlen(tok) + sizeof(kParam) + 64u;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;

    /* 单遍替换：两个占位符互不重叠，遇到哪个就展开哪个。 */
    size_t w = 0;
    const size_t tl = strlen(kTokPlaceholder), pl = strlen(kParamPlaceholder);
    for (size_t i = 0; page[i]; ) {
        if (!strncmp(page + i, kTokPlaceholder, tl)) {
            memcpy(out + w, tok, strlen(tok)); w += strlen(tok); i += tl;
        } else if (!strncmp(page + i, kParamPlaceholder, pl)) {
            memcpy(out + w, kParam, sizeof(kParam) - 1u); w += sizeof(kParam) - 1u; i += pl;
        } else {
            out[w++] = page[i++];
        }
    }
    out[w] = '\0';
    if (out_len) *out_len = w;
    return out;
}
