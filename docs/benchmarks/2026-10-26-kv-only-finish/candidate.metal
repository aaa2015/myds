kernel void kernel_myds_probe_kv_only_finish(
        constant dsv41_qkv_finish_args &args,
        device float *q, device float *kv, device float *window,
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if(group<64u) return;
    const uint column=(group-64u)*32u+lane;
    float value=kv[column];
    if(column>=448u) {
        const uint pair=(column-448u)/2u;
        const uint i=448u+pair*2u;
        const float theta=float(args.pos)*args.frequencies[pair];
        const float c=precise::cos(theta),s=precise::sin(theta);
        const float re=kv[i],im=kv[i+1u];
        value=dsv41_bf16((column&1u)?re*s+im*c:re*c-im*s);
    }
    /* Match rope's BF16 store, quant's BF16 input, E4M3 with power-of-two
     * scale, then quant's final BF16 store. Non-tail values only enter quant. */
    value=dsv41_bf16(value);
    const float amax=simd_max(abs(value));
    const float scale=dsv41_pow2_ceil(max(amax,1.0e-4f)*(1.0f/448.0f));
    const float result=dsv41_bf16(copysign(dsv4_e4m3fn_dequant(abs(value)/scale),value)*scale);
    kv[column]=result;
    window[(args.pos%128u)*512u+column]=result;
}
