// Standalone component probe. Append after unchanged dense.metal/moe.metal;
// this file is deliberately absent from the production shader source list.
// Exact contract: ordinary one-token Output_A Q4_K, 8x(1024 by 4096), NR2,
// NSG4, groups=(128,1,8), threads=(32,4,1), dynamic TG=256B, BF16 store.
// Buffers retain the original 136-byte id ABI and slots 0=args,1=w,2=x,3=y.
// Only address/shape metadata becomes compile-time constant. The original
// Q4_K helper, source reduction order and BF16 epilogue are called unchanged.
// Constant propagation can still change fast-math codegen; real full-output
// GPU bit comparison is mandatory. No parity or speed is claimed by source.
kernel void kernel_myds_outa_q4_v41_fixed_bf16(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void)args;
    (void)tiitg;
    const uint group=tgpig.z;
    tgpig.z=0;
    device const char * src0_cur=src0s+group*2359296u;
    device const char * src1_cur=src1+group*16384u;
    device       char * dst_cur=dst+group*4096u;
    ds4_metal_args_mul_mv args0={
        4096,1024,1,144,2304,2359296,2359296,
        4096,1,1,4,16384,131072,131072,
        1024,1,2,1,1,
    };
    kernel_mul_mv_q4_K_f32_impl<2,ds4_metal_args_mul_mv,true>(
        args0,src0_cur,src1_cur,dst_cur,shmem,tgpig,tiisg,sgitg);
}
