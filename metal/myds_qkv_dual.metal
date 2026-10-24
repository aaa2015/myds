// Opt-in ordinary one-token Qa/KV partitioned Q4_K projection.
// Assemble after dense.metal and moe.metal, which define the production ABI,
// Q4_K helper and BF16 rounding function.  The helper itself is unchanged.
//
// Host contract: both weights are Q4_K; Qa=1280x5120, KV=512x5120;
// NR2/NSG4, FC600=short(4), groups=(224,1,1), threads=(32,4,1).
// Each argument buffer is the original 112-byte ds4_metal_args_mul_mv, set
// exactly as two independent myds_metal_v41_gemv_bf16 calls would set it.
// Bind 256 bytes of scratch at threadgroup(0), matching the native provider
// (the Q4 helper currently does not access it).  Weights, the shared input
// and both outputs use their original independent buffers and offsets.
// Host must reject wrong shapes/ranges/alignment/aliasing before encoding.
// No concatenation, staging, output copy, changed reduction, or fallback.

kernel void kernel_myds_qkv_dual_q4_k_bf16_nr2(
        constant ds4_metal_args_mul_mv &args_qa [[buffer(0)]],
        constant ds4_metal_args_mul_mv &args_kv [[buffer(1)]],
        device const char *weight_qa [[buffer(2)]],
        device const char *weight_kv [[buffer(3)]],
        device const char *input [[buffer(4)]],
        device char *output_qa [[buffer(5)]],
        device char *output_kv [[buffer(6)]],
        threadgroup char *scratch [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (FC_mul_mv_nsg != 4 || group.x >= 224u || group.y != 0u || group.z != 0u)
        return;

    // All threads in a group take the same partition.  Remapping only the
    // group index reproduces each native launch's row/lane/SG assignment.
    const bool second = group.x >= 160u;
    constant ds4_metal_args_mul_mv &args = second ? args_kv : args_qa;
    device const char *weight = second ? weight_kv : weight_qa;
    device char *output = second ? output_kv : output_qa;
    uint3 local_group = group;
    if (second) local_group.x -= 160u;

    kernel_mul_mv_q4_K_f32_impl<2, constant ds4_metal_args_mul_mv &, true>(
        args, weight, input, output, scratch, local_group, lane, sg);
}
