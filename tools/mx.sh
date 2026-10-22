#!/bin/bash
# =================================================================
# mx.sh — 部署与 A/B 测试的唯一入口
#
# 这个脚本存在的理由：今天一场实验踩了 9 个装置坑，每个都伪装成"模型有问题"。
# 逐条固化在断言里（不是提示，是硬失败）：
#   ① 编译机 minos > 运行机 ⇒ dyld 直接 SIGKILL，0 字节输出，无 crash 报告
#   ② 只传 bin 不传 metallib ⇒ 服务起不来（它按 exe 同目录找）
#   ③ env 手写/复用旧快照 ⇒ 几轮实验全跑在默认值上，白跑
#   ④ env 覆盖没有真正替换 ⇒ 两臂配置相同，0-ULP 假通过
#   ⑤ 看门狗挂在 SSH 会话上 ⇒ 断线先死，进程裸奔到 panic
#   ⑥ 看门狗盯 RSS 而不是 wired ⇒ 盯错了对象（wired 才是不可回收的）
#   ⑦ pgrep -f 匹配到自己的 shell ⇒ "残留检查"永远假阳性
#   ⑧ kill -9 后只 sleep 0.2 ⇒ 旧进程还没归还内存，新进程已开始 mmap
#   ⑨ macOS 没有 timeout(1) / setsid(1)
#
# 用法（主机默认 macstudio，可 MX_HOST 覆盖）：
#   ./tools/mx.sh doctor                          # 跑之前先看环境对不对
#   ./tools/mx.sh ship                            # 传 bin+metallib，双端 md5 + minos 校验
#   ./tools/mx.sh env                             # 从活着的生产进程抓 env（唯一可信来源）
#   ./tools/mx.sh run A=MYDS_PROMPT_MASK_BITS=0 B=MYDS_PROMPT_MASK_BITS=1 --tokens 200000
#   ./tools/mx.sh run A= --tokens 200000,500000   # 无覆盖 = 生产原样；多长度做剂量-响应
#   ./tools/mx.sh restore                         # 收工：清场 + 恢复生产 + 校验
#   ./tools/mx.sh status                          # 一眼看状态
#
# run 的语义：停生产 → 逐臂起新进程 → 探针 → 关 → 下一臂 → 恢复生产。
# 每臂都有独立看门狗（盯 wired）与独立日志目录。
# =================================================================

set -u

HOST=${MX_HOST:-macstudio}
PROD_PORT=${MX_PROD_PORT:-8187}
TEST_PORT=${MX_TEST_PORT:-8199}
RUN=${MX_RUN:-/tmp/mx}
WIRED_LIMIT=${MX_WIRED_LIMIT:-190}
MIN_FREE_GB=${MX_MIN_FREE_GB:-40}
REPO=~/Projects/myds
FLEET=~/Projects/deploy
MODEL=${MX_MODEL:-$HOME/hf-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf}
CTX=${MX_CTX:-524288}
ALIAS=${MX_ALIAS:-mx}

die()  { echo "❌ $*" >&2; exit 1; }
ok()   { echo "   ✅ $*"; }
info() { echo "   ·  $*"; }
step() { echo "── $* ──"; }

R() { ssh -o BatchMode=yes -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=20 "$HOST" "$@"; }

# base64 打包：把 env 文件内容安全地穿过 ssh 的引号层
pack()   { base64 < /dev/stdin | tr -d '\n'; }
unpack() { base64 -d; }
# 目录名里的 @ 在某些 shell 里要转义，统一换成 _（arm@tokens → arm_at_tokens）
safetag() { echo "${1//@/_}"; }

# 远端小工具（避免依赖 macOS 没有的 timeout/setsid）
remote_lib() {
    cat <<'LIB'
port_pid() { lsof -nP -t -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | head -1; }
nds_count() { pgrep -x myds 2>/dev/null | wc -l | tr -d ' '; }
wired_gb() { vm_stat | awk '/Pages wired down/ {printf "%d", $4*16384/1e9}'; }
avail_gb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END {printf "%.0f", (f+i+s)*16384/1e9}'; }
# kill 之后必须等到进程真的消失：mmap 数百 GB 的进程拆映射远不止 0.2 秒
kill_wait() {
    local p="$1" n=0
    [ -z "$p" ] && return 0
    kill -TERM "$p" 2>/dev/null
    for n in $(seq 1 300); do kill -0 "$p" 2>/dev/null || return 0; sleep 0.2; done
    kill -9 "$p" 2>/dev/null
    for n in $(seq 1 300); do kill -0 "$p" 2>/dev/null || return 0; sleep 0.2; done
    return 1
}
pack() { echo "$1" | base64 | tr -d '\n'; }
unpack() { echo "$1" | base64 -d; }
LIB
}

# ---------------------------------------------------------------------------
cmd_doctor() {
    echo "═══════════ mx doctor ═══════════"
    step "1/6 本机工具链"
    command -v scp >/dev/null || die "缺 scp"
    command -v otool >/dev/null || die "缺 otool（Command Line Tools）"
    ok "scp / otool 就绪"

    step "2/6 远端可达性"
    R 'echo ok' >/dev/null 2>&1 || die "$HOST 不可达（网络/SSH）"
    ok "$HOST 可达"

    step "3/6 二进制 minos vs 远端系统版本"
    local BIN="$REPO/bin/myds"
    [ -f "$BIN" ] || die "缺 $BIN —— 先 make bin/myds 与 make bin/myds.metallib"
    [ -f "$REPO/bin/myds.metallib" ] || die "缺 bin/myds.metallib —— make bin/myds 不会重建它，要 make bin/myds.metallib"
    local minos hostver
    minos=$(otool -l "$BIN" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')
    hostver=$(R 'sw_vers -productVersion' | tr -d '\r')
    info "本机二进制 minos=$minos   远端 macOS=$hostver"
    awk -v a="$minos" -v b="$hostver" 'BEGIN{split(a,x,".");split(b,y,".");exit ((x[1]+0)>(y[1]+0))?1:0}' \
        || die "minos($minos) > 远端系统($hostver)：dyld 会直接 SIGKILL，先改 Makefile 的 DEPLOY_TARGET 并 rm -rf build 重编"
    ok "minos 兼容"   # awk exit 1 表示 minos>运行机 ⇒ 拒绝

    step "4/6 远端端口与进程"
    R "$(remote_lib)
        echo \"  prod:$PROD_PORT=\$(port_pid $PROD_PORT)\"
        echo \"  test:$TEST_PORT=\$(port_pid $TEST_PORT)\"
        echo \"  myds 进程数=\$(nds_count)\"
        echo \"  可用=\$(avail_gb) GB  wired=\$(wired_gb) GB\""
    local t
    t=$(R "lsof -nP -t -iTCP:$TEST_PORT -sTCP:LISTEN 2>/dev/null | head -1" | tr -d '\r')
    [ -z "$t" ] || die "测试端口 $TEST_PORT 被占（pid $t），先 ./tools/mx.sh restore"
    ok "测试端口空闲"

    step "5/6 可用内存"
    local avail prod
    avail=$(R "$(remote_lib); avail_gb" | tr -d '\r')
    prod=$(R "$(remote_lib); port_pid $PROD_PORT" | tr -d '\r')
    if [ -n "$prod" ]; then
        # 生产在跑时可用内存必然很低（模型常驻 wired ~162 GB）。
        # 这里只报告，不判定：真正需要独占内存的是 run，它在停生产之后才检查。
        info "可用 ${avail} GB（生产 pid=$prod 占用中，门槛留到停生产后再判）"
    else
        [ "${avail:-0}" -ge "$MIN_FREE_GB" ] || die "可用 ${avail} GB < ${MIN_FREE_GB} GB"
        ok "可用 ${avail} GB"
    fi

    step "6/6 远端仓库与 metallib"
    R "ls -la $REPO/bin/myds.metallib 2>/dev/null | awk '{print \"  remote metallib:\", \$5}'"
    ok "doctor 通过"
}

# ---------------------------------------------------------------------------
cmd_ship() {
    echo "═══════════ mx ship ═══════════"
    local BIN="$REPO/bin/myds" LIBF="$REPO/bin/myds.metallib"
    [ -f "$BIN" ] && [ -f "$LIBF" ] || die "缺 bin 或 metallib"

    local minos hostver
    minos=$(otool -l "$BIN" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')
    hostver=$(R 'sw_vers -productVersion' | tr -d '\r')
    awk -v a="$minos" -v b="$hostver" 'BEGIN{split(a,x,".");split(b,y,".");exit ((x[1]+0)>(y[1]+0))?1:0}' \
        || die "minos($minos) > 远端($hostver)，拒绝上传（传了也起不来）"
    ok "minos=$minos ≤ 远端 $hostver"

    local mb lb
    mb=$(md5 -q "$BIN"); lb=$(md5 -q "$LIBF")
    scp -q "$BIN" "$HOST:$REPO/bin/myds" || die "上传 bin 失败"
    scp -q "$LIBF" "$HOST:$REPO/bin/myds.metallib" || die "上传 metallib 失败"
    ok "已上传 bin + metallib（必须成对，服务按 exe 同目录找 metallib）"

    local rb rl
    rb=$(R "md5 -q $REPO/bin/myds" | tr -d '\r')
    rl=$(R "md5 -q $REPO/bin/myds.metallib" | tr -d '\r')
    [ "$mb" = "$rb" ] || die "bin md5 不一致：本地 $mb 远端 $rb"
    [ "$lb" = "$rl" ] || die "metallib md5 不一致：本地 $lb 远端 $rl"
    ok "双端 md5 一致  bin=${mb:0:12}  lib=${lb:0:12}"
    echo "$mb $lb" > "$RUN/bin.md5" 2>/dev/null || { mkdir -p "$RUN"; echo "$mb $lb" > "$RUN/bin.md5"; }
}

# ---------------------------------------------------------------------------
cmd_env() {
    mkdir -p "$RUN"
    echo "═══════════ mx env ═══════════"
    local p
    p=$(R "lsof -nP -t -iTCP:$PROD_PORT -sTCP:LISTEN 2>/dev/null | head -1" | tr -d '\r')
    [ -n "$p" ] || die "生产端口 $PROD_PORT 没有进程 —— env 只能从活着的生产进程抓（这是唯一可信来源）"
    info "生产 pid=$p"
    R "ps eww -p $p | tr ' ' '\n' | grep '^MYDS_' | sort -u" > "$RUN/env.prod" || die "抓 env 失败"
    local n
    n=$(wc -l < "$RUN/env.prod" | tr -d ' ')
    [ "$n" -ge 20 ] || die "只抓到 $n 个 MYDS_ 变量，不可信（阈值 20）"
    ok "$RUN/env.prod 共 $n 个变量"
    # 关键项必须存在，否则说明生产启动方式变了
    for k in MYDS_PREFILL_CHUNK_SIZE MYDS_MOE_CHUNKS MYDS_CHUNK_KEY_MAX; do
        grep -q "^$k=" "$RUN/env.prod" || info "⚠️ 缺 $k（该实验臂将吃默认值）"
    done
}

# 生成某臂的完整 env：生产 env 去掉被覆盖的键，再追加覆盖项
# ④ 的关键：必须"删除+追加"，不能只追加 —— 否则重复键里哪个生效取决于实现
build_env_file() {
    local out="$1" overrides="$2"
    mkdir -p "$(dirname "$out")"
    : > "$out"
    local keys=""
    if [ -n "$overrides" ]; then
        IFS=',' read -ra pairs <<< "$overrides"
        for kv in "${pairs[@]}"; do
            [ -z "$kv" ] && continue
            case "$kv" in
                -*) keys="$keys ${kv#-}" ;;   # -KEY = 只删除，不追加
                *)  keys="$keys ${kv%%=*}" ;;
            esac
        done
    fi
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local k=${line%%=*}
        local skip=0
        for kk in $keys; do [ "$k" = "$kk" ] && skip=1; done
        [ "$skip" = "1" ] || echo "$line" >> "$out"
    done < "$RUN/env.prod"
    if [ -n "$overrides" ]; then
        IFS=',' read -ra pairs <<< "$overrides"
        for kv in "${pairs[@]}"; do
            [ -z "$kv" ] && continue
            case "$kv" in
                -*) : ;;                      # -KEY：已在上一步删除，不追加
                *)  echo "$kv" >> "$out" ;;
            esac
        done
    fi
}

# ---------------------------------------------------------------------------
# 按 token 数推导所需的 ctx 与 prompt 上限：ctx 必须 ≥ tokens，否则 server 直接拒绝；
# MYDS_PREFILL_MAX_TOKENS 是源码里的保护阈值（默认 327680），实验必须显式放宽
# 否则 400K/500K 会被它以「exceeds the tested-safe limit」拒绝。
arm_ctx_for() {
    local n="$1"
    local c=$(( (n + 8191) / 8192 * 8192 ))
    [ "$c" -lt "$CTX" ] && c="$CTX"
    echo "$c"
}

arm_serve() {   # tag overrides tokens
    local tag="$1" ovr="${2:-}" ntok="${3:-0}"
    local dir="$RUN/$(safetag "$tag")"
    mkdir -p "$dir"; rm -f "$dir"/sampler.* "$dir"/server.log
    local actx actxmax
    actx=$(arm_ctx_for "$ntok")
    # 覆盖项里若显式给了 ctx / 上限，则以显式为准；否则注入按 token 推导的值
    case ",$ovr," in
        *MYDS_PREFILL_MAX_TOKENS=*) : ;;
        *) ovr="${ovr:+$ovr,}MYDS_PREFILL_MAX_TOKENS=$actx" ;;
    esac
    case ",$ovr," in
        *MYDS_CHUNK_KEY_MAX=*) : ;;
        *) ovr="${ovr:+$ovr,}MYDS_CHUNK_KEY_MAX=$actx" ;;
    esac
    build_env_file "$dir/env" "$ovr"
    
    # 停测试端口残留（并等它真的消失）
    R "$(remote_lib)
        p=\$(port_pid $TEST_PORT); [ -n \"\$p\" ] && kill_wait \$p
        exit 0"

    # 起被测进程（env 用 base64 文件注入，避免 shell 引号地狱）
    base64 < "$dir/env" > "$dir/env.b64"
    R "mkdir -p $dir" || die "[$tag] 远端目录创建失败"
    scp -q "$dir/env.b64" "$HOST:$dir/env.b64" || die "[$tag] env 上传失败"
    R "$(remote_lib)
        cd $dir || exit 1
        cat env.b64 | base64 -d > env.use
        set -a; . ./env.use; set +a
        \"$REPO/bin/myds\" -m $MODEL --port $TEST_PORT --host 127.0.0.1 \
            --ctx-size $actx --temp 0 --top-p 1.0 --max-tokens 262144 \
            --thinking off --alias $ALIAS > server.log 2>&1 &
        for i in \$(seq 1 500); do
            curl -s -m 2 http://127.0.0.1:$TEST_PORT/health 2>/dev/null | grep -q ok && break
            sleep 1
        done
        curl -s -m 3 http://127.0.0.1:$TEST_PORT/health" > "$dir/health.txt" 2>&1
    grep -q ok "$dir/health.txt" || { cat "$dir/health.txt" | tail -5; die "[$tag] 服务没起来"; }
    ok "[$tag] 服务就绪"

    # ⑤⑥ 看门狗：nohup+disown 脱离会话，盯 wired
    local limit_b64
    R "$(remote_lib)
        cd $dir || exit 1
        WIRED_LIMIT=$WIRED_LIMIT nohup $REPO/tools/rss_sampler.sh \
            --port $TEST_PORT --limit 200 --out $dir/sampler.log --interval 1 \
            > /dev/null 2>&1 &
        disown
        sleep 3
        cat sampler.log.hb 2>/dev/null || echo 0" > "$dir/hb.txt" 2>&1
    local hb; hb=$(tr -d '\r\n' < "$dir/hb.txt")
    [ -n "$hb" ] && [ "$hb" != "0" ] || die "[$tag] 看门狗没起来 —— 拒绝在无护栏状态下跑长上下文"
    ok "[$tag] 看门狗心跳 $hb"
}

arm_probe() {   # tag tokens [gen]
    local tag="$1" ntok="$2" ngen="${3:-1}"
    local dir="$RUN/$(safetag "$tag")"
    R "python3 - $TEST_PORT $ALIAS $ntok $dir $ngen <<'PY'
import json,sys,time,urllib.request,random
port,alias,tgt,out,ngen = sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],int(sys.argv[5])
FILL=('The compressor reads rows from the table and folds them into the residual stream '
      'before the next sublayer begins its work. ')
chars=int(tgt*5.347)
p='[n %d] '%random.getrandbits(60)+(FILL*200000)[:chars]+'\n\nSummarize.'
body=json.dumps({'model':alias,'messages':[{'role':'user','content':p}],'max_tokens':ngen,
                 'temperature':0,'stream':True,'stream_options':{'include_usage':True}}).encode()
req=urllib.request.Request('http://127.0.0.1:%s/v1/chat/completions'%port,data=body,
                           headers={'Content-Type':'application/json'})
t0=time.monotonic(); ttft=None; pt=None; chunks=[]; tl=None; ntok=0
try:
    with urllib.request.urlopen(req,timeout=5400) as s:
        for raw in s:
            line=raw.decode('utf-8','replace').strip()
            if not line.startswith('data:'): continue
            d=line[5:].strip()
            if d=='[DONE]': break
            try: j=json.loads(d)
            except Exception: continue
            if j.get('usage'): pt=j['usage']['prompt_tokens']
            c=j.get('choices') or []
            if c:
                dl=c[0].get('delta') or {}
                if dl.get('content'):
                    now=time.monotonic()-t0
                    if ttft is None: ttft=now
                    tl=now
                    ntok+=1
                    chunks.append(dl['content'])
    open(out+'/out.txt','w').write(''.join(chunks))
    # decode 速率：(生成数-1)/（最后一个 token 时刻 - 第一个 token 时刻）
    dec=0.0
    if ntok>1 and tl is not None and ttft is not None and tl>ttft:
        dec=(ntok-1)/(tl-ttft)
    if pt and ttft:
        print('OK %d %.1f %.1f dec=%.2f n=%d'%(pt,ttft,pt/ttft,dec,ntok))
    else:
        print('EMPTY')
except Exception as e:
    print('FAIL %s'%type(e).__name__)
PY" > "$dir/probe_$ntok.txt" 2>&1

    # 峰值与断言（prefill batching / mask 只在首个请求时打印，所以必须探测后再查）
    R "$(remote_lib)
        cd $dir
        pw=\$(cat sampler.log.peakwired 2>/dev/null || echo '?')
        trip=\$(cat sampler.log.trip 2>/dev/null || echo no)
        mb=\$(grep -oE 'macroblock=[0-9]+' server.log | head -1)
        mask=\$(grep -oE 'prompt block mask: [a-z-]+' server.log | head -1)
        echo \"PEAK \$pw \$trip \$mb \$mask\"" > "$dir/assert.txt" 2>&1

    # 服务端自己记的 prefill/decode 才权威（探针是客户端视角，会漏算）
    R "grep -a 'Completed' $dir/server.log 2>/dev/null | tail -1" > "$dir/done.txt" 2>&1

    local res peak trip mb mask
    res=$(tr -d '\r' < "$dir/probe_$ntok.txt" | tail -1)
    read -r _ peak trip mb mask <<< "$(tr -d '\r' < "$dir/assert.txt" | tail -1)"
    local srvdec
    srvdec=$(tr -d '\r' < "$dir/done.txt" 2>/dev/null | grep -oE 'decode: [0-9]+ tok in [0-9.]+s = [0-9.]+ t/s' | sed 's/decode: //')
    printf "   %-10s %-8s %-40s peak=%-4sGB %s  srv[%s]\n" \
        "$tag" "$ntok" "$res" "${peak:-?}" "${trip:-?}" "${srvdec:-n/a}"
    echo "$tag $ntok $res ${peak:-?} ${trip:-?} ${mb:-} ${mask:-} ${srvdec:-n/a}" >> "$RUN/results.txt"

    if [ "${trip:-no}" != "no" ]; then
        die "[$tag] 看门狗因 wired 超限杀进程（$trip）—— 立即停止，不要再往上加长度"
    fi
}

arm_down() {
    local tag="$1" dir="$RUN/$(safetag "$1")"
    R "$(remote_lib)
        pkill -f 'rss_sampler.sh.*--port $TEST_PORT' 2>/dev/null
        p=\$(port_pid $TEST_PORT); [ -n \"\$p\" ] && kill_wait \$p
        sleep 2
        exit 0"
    ok "[$tag] 已收"
}

# ---------------------------------------------------------------------------
cmd_run() {
    echo "═══════════ mx run ═══════════"
    mkdir -p "$RUN"
    local arms=() tokens="" gen=1
    local a
    for a in "$@"; do
        case "$a" in
            --tokens) : ;;
            --tokens=*) tokens="${a#*=}" ;;
            *=*) arms+=("$a") ;;
            *) [ "$tokens" = "" ] && tokens="$a" ;;
        esac
    done
    # 解析 --tokens N 形式
    local prev=""
    for a in "$@"; do
        [ "$prev" = "--tokens" ] && tokens="$a"
        [ "$prev" = "--gen" ] && gen="$a"
        prev="$a"
    done
    [ ${#arms[@]} -gt 0 ] || die "至少给一臂，例如 A=MYDS_PROMPT_MASK_BITS=1（无覆盖写 A=）"
    [ -n "$tokens" ] || tokens="200000"

    cmd_doctor   # 失败直接退出（die 用 exit，不能用 || 兜底重试）
    cmd_env
    L=$(md5 -q "$RUN/env.prod")

    step "停生产（独占）"
    R "$(remote_lib)
        p=\$(port_pid $PROD_PORT)
        [ -n \"\$p\" ] && kill_wait \$p
        for i in \$(seq 1 400); do [ \"\$(nds_count)\" = '0' ] && break; sleep 0.5; done
        for p in \$(pgrep -x myds); do kill -9 \$p; done
        sleep 8
        echo \"  myds=\$(nds_count)  wired=\$(wired_gb)GB\""
    local left avail2
    left=$(R "$(remote_lib); nds_count" | tr -d '\r')
    [ "$left" = "0" ] || die "还有 $left 个 myds 残留，拒绝开始（会假 OOM）"
    avail2=$(R "$(remote_lib); avail_gb" | tr -d '\r')
    [ "${avail2:-0}" -ge "$MIN_FREE_GB" ] \
        || die "停生产后可用仅 ${avail2} GB < ${MIN_FREE_GB} GB —— 拒绝实验"
    ok "生产已停，无残留；可用 ${avail2} GB"

    : > "$RUN/results.txt"
    local tok armtag
    IFS=',' read -ra TOKS <<< "$tokens"
    for armtag in "${arms[@]}"; do
        local tag="${armtag%%=*}" ovr="${armtag#*=}"
        [ "$tag" = "$armtag" ] && ovr=""
        for tok in "${TOKS[@]}"; do
            local t2="$tag@$tok"
            step "臂 $t2  覆盖[${ovr:-无}]"
            arm_serve "$t2" "$ovr" "$tok"
            arm_probe "$t2" "$tok" "$gen"
            arm_down "$t2"
        done
    done

    echo
    step "结果"
    printf "   %-14s %-9s %-30s %-8s %-6s %s\n" arm tokens result peak trip cfg
    awk '{printf "   %-14s %-9s %-30s %-8s %-6s %s %s\n",$1,$2,$3" "$4" "$5,$6,$7,$8}' "$RUN/results.txt" 2>/dev/null | head -30

    step "恢复生产"
    cmd_restore
}

# ---------------------------------------------------------------------------
cmd_restore() {
    echo "═══════════ mx restore ═══════════"
    R "$(remote_lib)
        pkill -f rss_sampler.sh 2>/dev/null
        p=\$(port_pid $TEST_PORT); [ -n \"\$p\" ] && kill_wait \$p
        for p in \$(pgrep -x myds); do kill -TERM \$p; done
        for i in \$(seq 1 400); do [ \"\$(nds_count)\" = '0' ] && break; sleep 0.5; done
        for p in \$(pgrep -x myds); do kill -9 \$p; done
        sleep 8
        echo \"  myds=\$(nds_count)  wired=\$(wired_gb)GB\"" \
        || die "清场失败"
    local left; left=$(R "$(remote_lib); nds_count" | tr -d '\r')
    [ "$left" = "0" ] || die "仍有 $left 个 myds"

    R "cd $FLEET && nohup ./start_fleet.sh > /tmp/r_mx.log 2>&1 & sleep 55
       for i in \$(seq 1 400); do curl -s -m 2 http://127.0.0.1:$PROD_PORT/health 2>/dev/null | grep -q ok && break; sleep 2; done
       echo \"  生产 pid=\$(lsof -nP -t -iTCP:$PROD_PORT -sTCP:LISTEN | head -1) health=\$(curl -s -m 5 http://127.0.0.1:$PROD_PORT/health)\"" \
        || die "启动生产失败，看 /tmp/r_mx.log"
    local n; n=$(R "$(remote_lib); nds_count" | tr -d '\r')
    [ "$n" = "1" ] || die "生产 myds 进程数=$n（应为 1）"
    ok "生产已恢复，进程数 1"
}

# ---------------------------------------------------------------------------
cmd_status() {
    echo "═══════════ mx status ═══════════"
    R "$(remote_lib)
        echo \"  系统: \$(sw_vers -productVersion)\"
        echo \"  uptime: \$(uptime | sed 's/.*up //;s/,.*//')\"
        echo \"  生产 $PROD_PORT: pid=\$(port_pid $PROD_PORT) health=\$(curl -s -m 3 http://127.0.0.1:$PROD_PORT/health)\"
        echo \"  测试 $TEST_PORT: pid=\$(port_pid $TEST_PORT)\"
        echo \"  myds 进程数=\$(nds_count)\"
        echo \"  可用=\$(avail_gb) GB  wired=\$(wired_gb) GB\"
        p=\$(port_pid $PROD_PORT)
        [ -n \"\$p\" ] && echo \"  RSS=\$(ps -o rss= -p \$p | awk '{printf \"%.1f GB\", \$1/1048576}')  \$(ps -o command= -p \$p | grep -oE '\-\-ctx-size [0-9]+')\"" \
        || die "远端不可达"
}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 纯逻辑自检：不碰远端，验证两处最容易错的推导
cmd_selftest() {
    echo "═══════════ mx selftest ═══════════"
    local fail=0

    step "1/3 minos 比较（编译机 vs 运行机）"
    chk_minos() {  # $1=minos $2=hostver $3=期望(ok/bad)
        local r
        if awk -v a="$1" -v b="$2" 'BEGIN{split(a,x,".");split(b,y,".");exit ((x[1]+0)>(y[1]+0))?1:0}'; then r=ok; else r=bad; fi
        if [ "$r" = "$3" ]; then ok "minos=$1 vs host=$2 → $r"; else echo "   ❌ minos=$1 vs host=$2 → $r（期望 $3）"; fail=1; fi
    }
    chk_minos 27.0 26.7.1 bad      # 今天真实踩过：dyld 直接 SIGKILL
    chk_minos 26.0 26.7.1 ok
    chk_minos 26.0 27.0.1 ok       # 升级后：低 minos 在高系统上仍可跑
    chk_minos 27.0 27.0.1 ok

    step "2/3 env 覆盖：必须真替换，不能出现重复键"
    local tmp
    tmp=$(mktemp -d)
    printf 'MYDS_A=1\nMYDS_MOE_CHUNKS=9\nMYDS_B=2\nMYDS_CHUNK_KEY_MAX=524288\n' > "$tmp/env.prod"
    RUN="$tmp" build_env_file "$tmp/out" "MYDS_MOE_CHUNKS=16,MYDS_NEW=z"
    if [ "$(grep -c '^MYDS_MOE_CHUNKS=' "$tmp/out")" = "1" ]; then ok "被覆盖的键只出现一次"; else echo "   ❌ MYDS_MOE_CHUNKS 出现 $(grep -c '^MYDS_MOE_CHUNKS=' "$tmp/out") 次"; fail=1; fi
    if grep -qx 'MYDS_MOE_CHUNKS=16' "$tmp/out"; then ok "覆盖值生效 (16)"; else echo "   ❌ 覆盖值未生效"; fail=1; fi
    if grep -qx 'MYDS_NEW=z' "$tmp/out"; then ok "新增键追加成功"; else echo "   ❌ 新增键缺失"; fail=1; fi
    for keep in MYDS_A=1 MYDS_B=2 MYDS_CHUNK_KEY_MAX=524288; do
        grep -qx "$keep" "$tmp/out" || { echo "   ❌ 未覆盖的键被丢弃: $keep"; fail=1; }
    done
    ok "未覆盖的键全部保留"
    rm -rf "$tmp"

    step "3/3 必需构件成对存在"
    for f in "$REPO/bin/myds" "$REPO/bin/myds.metallib" "$REPO/tools/rss_sampler.sh"; do
        if [ -x "$f" ] || [ -f "$f" ]; then ok "存在 $f"; else echo "   ❌ 缺 $f"; fail=1; fi
    done

    echo
    [ "$fail" = "0" ] && { echo "✅ selftest 全部通过"; return 0; } || die "selftest 有失败项"
}

case "${1:-}" in
    doctor|env|ship|run|restore|status|selftest) cmd="$1"; shift; "cmd_$cmd" "$@" ;;
    *) sed -n '2,40p' "$0"; exit 1 ;;
esac
