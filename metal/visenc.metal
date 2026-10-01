/* visenc.metal — DeepSeek V4.1 视觉编码器专用的 Metal 内核
 *
 * 忠实移植自 ds4 的 metal/deepseek4_vision.metal 与 metal/glm53_vision.metal。
 * ★ 移植原则：算术次序、舍入位置、常量一字不改 —— 逐层对拍才有意义。
 *
 * 与 ds4 的唯一差异：ds4 的偏置在显存里是 BF16（ushort），每用一次现转 F32；
 * 本引擎在加载期已把偏置无损展宽成 F32（BF16 是 F32 的高 16 位）。
 * glm53_bf16_to_f32(bias) 本身是精确的，所以两者结果【完全一致】，
 * 只是省掉了内层转换。
 *
 * 「每步都舍回 BF16」是这套编码器的性格：它是 BF16 训练的，
 * 每个中间量都要落回 BF16。少一处舍入，误差会一路累积。
 */

struct visenc_rows_args  { uint width;  uint rows; };
struct visenc_rms_args   { uint width;  uint rows;  float eps; };
struct visenc_qkv_args   { uint rows;   uint grid_width; };
struct visenc_attn_args  { uint rows;   float scale; };
struct visenc_align_args { uint grid_height; uint grid_width; uint output_rows; };

/* 就近偶数舍入到 BF16。NaN / Inf 原样保留（否则会把非有限值变成有限值）。 */
static inline float visenc_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    bits += 0x00007fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

/* erf 的多项式近似（Abramowitz–Stegun 7.1.26）。与 ds4 逐字一致 ——
 * ★ 不能用 Metal 自带的 erf()：那是另一套实现，结果会有末位差异。 */
static inline float visenc_erf(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = abs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = (((((1.061405429f * t - 1.453152027f) * t) +
                       1.421413741f) * t - 0.284496736f) * t +
                       0.254829592f) * t;
    return sign * (1.0f - p * exp(-a * a));
}

/* 就地舍回 BF16（每步之间的那个 round） */
kernel void kernel_visenc_round(
        constant visenc_rows_args &args,
        device float              *x,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    x[off] = visenc_round_bf16(x[off]);
}

/* 加偏置（【不】舍入；调用方随后单独 dispatch_round，与 ds4 的分工一致） */
kernel void kernel_visenc_bias(
        constant visenc_rows_args &args,
        device float              *x,
        device const float        *bias,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    x[(ulong)gid.y * args.width + gid.x] += bias[gid.x];
}

/* V4.1 的偏置+残差：★ 两次舍入（先舍 bias 和，再舍加残差）
 * 少一次舍入就会与 ds4 分叉 —— 这是 V4.1 相对 V4.0 的特有行为。 */
kernel void kernel_visenc_bias_residual41(
        constant visenc_rows_args &args,
        device float              *x,
        device const float        *bias,
        device const float        *residual,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    const float projected = visenc_round_bf16(x[off] + bias[gid.x]);
    x[off] = visenc_round_bf16(projected + residual[off]);
}

/* 加残差并舍回 */
kernel void kernel_visenc_add_residual(
        constant visenc_rows_args &args,
        device float              *x,
        device const float        *residual,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    x[off] = visenc_round_bf16(x[off] + residual[off]);
}

/* QKV 拆分 + 偏置 + 2D RoPE。
 *
 * 布局要点（照抄 ds4）：head_dim = 64，一个 simdgroup(32 lane) 覆盖半个头：
 *   q0 = qkv[base + lane]，q1 = qkv[base + lane + 32]
 *   q = (q0*cs - q1*sn, q1*cs + q0*sn)
 * 位置是【二维】的：lane < 16 用行号 y，否则用列号 x；
 * 频率下标 = lane & 15，inv_freq = 10000^(-idx/16)。
 * 偏置在加完后、【立刻】舍回 BF16，RoPE 的输出也舍回 —— 共 3 处舍入。 */
kernel void kernel_visenc_qkv_rope(
        constant visenc_qkv_args &args,
        device const float       *qkv,
        device const float       *bias,
        device float             *q,
        device float             *k,
        device float             *v,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint head = group.y;
    if (row >= args.rows || head >= 16u) return;
    const ulong qkv_base = (ulong)row * 3072u + (ulong)head * 64u;
    const ulong out_base = (ulong)row * 1024u + (ulong)head * 64u;

    const float q0 = visenc_round_bf16(qkv[qkv_base + lane] +
                                       bias[(ulong)head * 64u + lane]);
    const float q1 = visenc_round_bf16(qkv[qkv_base + lane + 32u] +
                                       bias[(ulong)head * 64u + lane + 32u]);
    const float k0 = visenc_round_bf16(qkv[qkv_base + 1024u + lane] +
                                       bias[1024u + (ulong)head * 64u + lane]);
    const float k1 = visenc_round_bf16(qkv[qkv_base + 1024u + lane + 32u] +
                                       bias[1024u + (ulong)head * 64u + lane + 32u]);

    const uint y = row / args.grid_width;
    const uint x = row - y * args.grid_width;
    const uint pos = lane < 16u ? y : x;
    const uint freq_index = lane & 15u;
    const float inv_freq = powr(10000.0f, -(float)freq_index / 16.0f);
    const float angle = (float)pos * inv_freq;
    const float cs = cos(angle);
    const float sn = sin(angle);

    q[out_base + lane] = visenc_round_bf16(q0 * cs - q1 * sn);
    q[out_base + lane + 32u] = visenc_round_bf16(q1 * cs + q0 * sn);
    k[out_base + lane] = visenc_round_bf16(k0 * cs - k1 * sn);
    k[out_base + lane + 32u] = visenc_round_bf16(k1 * cs + k0 * sn);

    v[out_base + lane] = visenc_round_bf16(qkv[qkv_base + 2048u + lane] +
                                           bias[2048u + (ulong)head * 64u + lane]);
    v[out_base + lane + 32u] =
        visenc_round_bf16(qkv[qkv_base + 2048u + lane + 32u] +
                          bias[2048u + (ulong)head * 64u + lane + 32u]);
}

/* 全序列注意力（无因果掩码 —— 视觉是一整张图，不是自回归序列）。
 * 每行一个 simdgroup，key 维在 32 个 lane 上做 simd_sum 归约。
 * 在线 softmax（保留 max/denom 滚动），与 ds4 同样的稳定化写法。
 * ★ 输出【不】舍入；调用方随后单独 dispatch_round。 */
kernel void kernel_visenc_attention(
        constant visenc_attn_args &args,
        device const float        *q,
        device const float        *k,
        device const float        *v,
        device float              *out,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint head = group.y;
    if (row >= args.rows || head >= 16u) return;
    const ulong base = (ulong)row * 1024u + (ulong)head * 64u;
    const float q0 = q[base + lane];
    const float q1 = q[base + lane + 32u];
    float acc0 = 0.0f, acc1 = 0.0f;
    float max_score = -INFINITY;
    float denom = 0.0f;
    for (uint key_row = 0; key_row < args.rows; key_row++) {
        const ulong kb = (ulong)key_row * 1024u + (ulong)head * 64u;
        float score = simd_sum(q0 * k[kb + lane] + q1 * k[kb + lane + 32u]);
        score *= args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ? 0.0f : exp(max_score - next_max);
        const float new_scale = exp(score - next_max);
        denom = denom * old_scale + new_scale;
        acc0 = acc0 * old_scale + new_scale * v[kb + lane];
        acc1 = acc1 * old_scale + new_scale * v[kb + lane + 32u];
        max_score = next_max;
    }
    out[base + lane] = acc0 / denom;
    out[base + lane + 32u] = acc1 / denom;
}

/* SwiGLU 拆分（V4.1 版：silu 之后也舍回 BF16，输出再舍一次）。
 * 输入是拼接的 [gate(2816) | up(2816)]，按行交错：source = row*width*2 + d */
kernel void kernel_visenc_swiglu_split41(
        constant visenc_rows_args &args,
        device const float        *gate_up,
        device float              *out,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong source = (ulong)gid.y * args.width * 2u + gid.x;
    const float gate = gate_up[source];
    const float up = gate_up[source + args.width];
    float activated = gate / (1.0f + exp(-gate));
    activated = visenc_round_bf16(activated);          /* V4.1：silu 后舍一次 */
    out[(ulong)gid.y * args.width + gid.x] = visenc_round_bf16(activated * up);
}

/* 下采样重排：等价 F.unfold(x, 3, stride=3)，通道在最外层，随后是 3x3 邻域行主序。
 * 越界补 0。输入 [gh*gw, 1024] → 输出 [ceil(gh/3)*ceil(gw/3), 9216] */
kernel void kernel_visenc_aligner_reorder(
        constant visenc_align_args &args,
        device const float         *x,
        device float               *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= 9216u || row >= args.output_rows) return;
    const uint output_width = (args.grid_width + 2u) / 3u;
    const uint block_y = row / output_width;
    const uint block_x = row - block_y * output_width;
    const uint channel = d / 9u;
    const uint within = d - channel * 9u;
    const uint source_y = block_y * 3u + within / 3u;
    const uint source_x = block_x * 3u + within % 3u;
    float value = 0.0f;
    if (source_y < args.grid_height && source_x < args.grid_width) {
        const ulong source_row = (ulong)source_y * args.grid_width + source_x;
        value = x[source_row * 1024u + channel];
    }
    out[(ulong)row * 9216u + d] = value;
}

/* aligner 第一层之后：加偏置 → 舍 BF16 → GELU → 再舍 BF16。
 * ★ GELU 用精确公式 0.5x(1+erf(x/√2))，erf 用上面那个多项式（与 ds4 一致） */
kernel void kernel_visenc_gelu_bias(
        constant visenc_rows_args &args,
        device const float        *x,
        device const float        *bias,
        device float              *out,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    const float value = visenc_round_bf16(x[off] + bias[gid.x]);
    out[off] = visenc_round_bf16(
        0.5f * value * (1.0f + visenc_erf(value * 0.7071067811865475f)));
}
