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
    jputf(b, "},\"prompt_tokens\":%u,\"cached_tokens\":%u,\"out_tokens\":%u,\"stream\":%s",
          r->prompt_tokens, r->cached_tokens, r->out_tokens, r->stream ? "true" : "false");
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
const char *myds_trace_page_html(void) {
    return
"<!DOCTYPE html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
"<title>myds trace</title><style>"
":root{--bg:#0f1115;--fg:#d7dae0;--dim:#8b919c;--line:#252a33;--acc:#4b9fd8;"
"--q:#7a6cb0;--r:#c8952f;--s:#5a8f4e;--e:#3d7fb5;--p:#2f9e8f;--d:#b5514e;--u:#5a5f68}"
"*{box-sizing:border-box}"
"body{margin:0;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;"
"background:var(--bg);color:var(--fg)}"
"header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;"
"align-items:baseline;flex-wrap:wrap}"
"h1{font-size:14px;margin:0;font-weight:600}"
".meta{color:var(--dim);font-size:12px}"
"#wrap{display:flex;height:calc(100vh - 45px)}"
"#list{width:52%;overflow:auto;border-right:1px solid var(--line)}"
"#detail{flex:1;overflow:auto;padding:14px 16px}"
"table{width:100%;border-collapse:collapse}"
"th,td{padding:5px 8px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}"
"th{position:sticky;top:0;background:#161a20;color:var(--dim);font-weight:500;z-index:1}"
"td.num{text-align:right}"
"tbody tr{cursor:pointer}"
"tbody tr:hover{background:#1a1f26}"
"tbody tr.sel{background:#1e2b38}"
"tbody tr.sel td{border-bottom-color:var(--acc)}"
".pill{display:inline-block;padding:0 5px;border-radius:3px;font-size:11px;"
"background:#252a33;color:var(--dim)}"
".pill.len{background:#3a2a2a;color:#e08a8a}"
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
"<header><h1>myds trace</h1><span class=\"meta\" id=\"hdr\">loading…</span>"
"<span class=\"meta\">点击一行查看该请求的分层耗时</span></header>"
"<div id=\"wrap\"><div id=\"list\"><table><thead><tr>"
"<th>#</th><th>时间</th><th>path</th><th class=\"num\">prompt</th>"
"<th class=\"num\">out</th><th class=\"num\">耗时</th><th>finish</th>"
"</tr></thead><tbody id=\"rows\"><tr><td colspan=\"7\" class=\"empty\">loading…</td></tr>"
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
"function total(p){return p.engine!==null?p.queue+(p.render||0)+(p.setup||0)+"
"(p.engine-(p.prefill||0)-(p.decode||0)>0?0:0)+p.engine:null;}"
"function cls(f){const s=(f||'').toLowerCase();"
"return s==='length'?'len':s==='error'?'err':s==='tool_calls'?'tool':"
"s==='stop'?'stop':'';}"
"async function load(){"
"const r=await fetch('/trace/api/requests');const j=await r.json();"
"RECS=j.records||[];"
"document.getElementById('hdr').textContent='count '+j.count+' / capacity '+j.capacity;"
"const tb=document.getElementById('rows');"
"if(!RECS.length){tb.innerHTML='<tr><td colspan=\"7\" class=\"empty\">"
"暂无记录（先发一个请求）</td></tr>';return;}"
"tb.innerHTML=RECS.map(rec=>{"
"const p=rec.phases;const wall=p.engine===null?null:"
"p.queue+(p.render||0)+(p.setup||0);"
"const when=new Date(rec.epoch_ms).toLocaleTimeString();"
"return '<tr data-seq=\"'+rec.seq+'\"'+(rec.seq===SEL?' class=\"sel\"':'')+'>'"
"+'<td>'+rec.seq+'</td><td>'+when+'</td>"
"+'<td>'+esc(rec.path)+'</td>"
"+'<td class=\"num\">'+rec.prompt_tokens+'</td>"
"+'<td class=\"num\">'+rec.out_tokens+'</td>"
"+'<td class=\"num\">'+(wall===null?'—':f(wall))+'</td>"
"+'<td><span class=\"pill '+cls(rec.finish)+'\">'+esc(rec.finish)+'</span></td>"
"+'</tr>';}).join('');"
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
"load();setInterval(load,3000);"
"</script></body></html>";
}
