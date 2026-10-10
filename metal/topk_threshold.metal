#include <metal_stdlib>
using namespace metal;

// Scalar F32 Top512 candidate. All decision/state transitions remain on GPU.
// Raw positive-normal IEEE keys have exactly the reference comparison order.
struct myds_topk_threshold_args { uint cols, parts; };
static inline bool threshold_positive(uint bits) {
    return bits >= 0x00800000u && bits < 0x7f800000u;
}
static inline void threshold_finish(device uint *control, device atomic_uint *stats,
                                    uint reason) {
    control[0] = reason == 0u ? 1u : 0u;
    control[1] = reason;
    atomic_fetch_add_explicit(stats, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(stats + (reason ? 2u : 1u), 1u, memory_order_relaxed);
    if (reason) atomic_fetch_add_explicit(stats + reason + 2u, 1u, memory_order_relaxed);
}

kernel void kernel_myds_topk_threshold_range_parts(
    constant myds_topk_threshold_args &a [[buffer(0)]],
    device const uint *scores [[buffer(1)]], device uint4 *parts [[buffer(2)]],
    uint part [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    uint lo=0xffffffffu, hi=0u, count=0u, flags=0u;
    for (uint i=0u;i<4u;i++) {
        const uint col=part*1024u+uint(tid)+i*256u;
        if (col<a.cols) {
            const uint bits=scores[col];
            flags|=((bits&0x7fffffffu)>0x7f800000u)?1u:0u;
            flags|=bits==0x7f800000u?2u:0u;
            if (threshold_positive(bits)) { lo=min(lo,bits); hi=max(hi,bits); count++; }
        }
    }
    threadgroup uint4 partial[8];
    const uint4 value=uint4(simd_min(lo),simd_max(hi),simd_sum(count),simd_or(flags));
    if (lane==0) partial[sg]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) {
        uint4 result=uint4(0xffffffffu,0u,0u,0u);
        for(uint i=0;i<8u;i++) {
            result.x=min(result.x,partial[i].x);result.y=max(result.y,partial[i].y);
            result.z+=partial[i].z;result.w|=partial[i].w;
        }
        parts[part]=result;
    }
}

kernel void kernel_myds_topk_threshold_range_reduce(
    constant myds_topk_threshold_args &a [[buffer(0)]],
    device const uint4 *parts [[buffer(1)]], device uint *control [[buffer(2)]],
    ushort tid [[thread_index_in_threadgroup]]) {
    // At most 256 partitions. One thread publishes immutable range metadata.
    if(tid!=0) return;
    uint lo=0xffffffffu,hi=0u,count=0u,flags=0u;
    for(uint p=0;p<a.parts;p++) {
        const uint4 v=parts[p];lo=min(lo,v.x);hi=max(hi,v.y);count+=v.z;flags|=v.w;
    }
    const uint reason=(flags&1u)?1u:(flags&2u)?2u:count<512u?3u:0u;
    uint shift=0u;
    if(!reason) {
        const uint delta=hi-lo;
        const uint length=delta?32u-clz(delta):0u;
        shift=length>8u?length-8u:0u;
    }
    for(uint i=0;i<12u;i++)control[i]=0u;
    control[1]=reason;control[2]=lo;control[3]=hi;control[4]=count;
    control[5]=flags;control[6]=shift;control[9]=a.parts;
}

kernel void kernel_myds_topk_threshold_hist_parts(
    constant myds_topk_threshold_args &a [[buffer(0)]],
    device const uint *scores [[buffer(1)]],device const uint *control [[buffer(2)]],
    device uint *hist [[buffer(3)]],uint part [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    if(control[1]!=0u)return; // uniform, before all threadgroup barriers
    threadgroup atomic_uint bins[256];
    atomic_store_explicit(bins+tid,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=0;i<4u;i++) {
        const uint col=part*1024u+uint(tid)+i*256u;
        if(col<a.cols) {
            const uint bits=scores[col];
            if(threshold_positive(bits)) {
                const uint bin=(bits-control[2])>>control[6];
                atomic_fetch_add_explicit(bins+bin,1u,memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hist[part*256u+uint(tid)]=atomic_load_explicit(bins+tid,memory_order_relaxed);
}

kernel void kernel_myds_topk_threshold_hist_reduce(
    constant myds_topk_threshold_args &a [[buffer(0)]],
    device const uint *hist [[buffer(1)]], device uint *control [[buffer(2)]],
    device uint2 *offsets [[buffer(3)]],ushort tid [[thread_index_in_threadgroup]]) {
    if(control[1]!=0u) { if(uint(tid)<a.parts)offsets[tid]=uint2(0u);return; }
    threadgroup uint totals[256], counts[256];
    threadgroup uint pivot;
    uint total=0u;
    for(uint p=0;p<a.parts;p++)total+=hist[p*256u+uint(tid)];
    totals[tid]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        uint retained=0u,bucket=255u;
        for(int b=255;b>=0;b--) { retained+=totals[b]; if(retained>=512u){bucket=uint(b);break;} }
        pivot=bucket;
        control[7]=control[2]+(bucket<<control[6]);control[8]=retained;
        if(retained<512u)control[1]=5u;
        else if(retained>1024u)control[1]=4u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    uint count=0u;
    if(uint(tid)<a.parts)for(uint b=pivot;b<256u;b++)count+=hist[uint(tid)*256u+b];
    counts[tid]=count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(uint(tid)<a.parts) {
        uint offset=0u;for(uint p=0;p<uint(tid);p++)offset+=counts[p];
        offsets[tid]=uint2(offset,count);
    }
}

kernel void kernel_myds_topk_threshold_compact(
    constant myds_topk_threshold_args &a [[buffer(0)]],
    device const uint *scores [[buffer(1)]],device uint *control [[buffer(2)]],
    device const uint2 *offsets [[buffer(3)]],device uint2 *pairs [[buffer(4)]],
    uint part [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if(control[1]!=0u)return;
    uint bits[4];uint mask=0u;
    for(uint i=0;i<4u;i++) {
        const uint col=part*1024u+uint(tid)+i*256u;
        bits[i]=col<a.cols?scores[col]:0u;
        if(col<a.cols&&threshold_positive(bits[i])&&bits[i]>=control[7])mask|=1u<<i;
    }
    const uint count=popcount(mask), prefix=simd_prefix_exclusive_sum(count), total=simd_sum(count);
    threadgroup atomic_uint allocated;
    if(tid==0)atomic_store_explicit(&allocated,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base=0u;
    if(lane==0)base=atomic_fetch_add_explicit(&allocated,total,memory_order_relaxed);
    base=simd_broadcast_first(base)+prefix+offsets[part].x;
    for(uint i=0;i<4u;i++)if(mask&(1u<<i)) {
        if(base<1024u)pairs[base]=uint2(bits[i],part*1024u+uint(tid)+i*256u);
        base++;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        const uint written=atomic_load_explicit(&allocated,memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_uint *)(control+10),written,memory_order_relaxed);
    }
}

kernel void kernel_myds_topk_threshold_final(
    constant myds_topk_threshold_args &a [[buffer(0)]],
    device uint *control [[buffer(1)]],device uint2 *pairs [[buffer(2)]],
    device int *ids [[buffer(3)]],device atomic_uint *stats [[buffer(4)]],
    threadgroup uint2 *tile [[threadgroup(0)]],ushort tid [[thread_index_in_threadgroup]]) {
    if(control[1]!=0u) { if(tid==0)threshold_finish(control,stats,control[1]);return; }
    for(uint i=uint(tid);i<1024u;i+=512u)
        tile[i]=i<control[8]?pairs[i]:uint2(0u,0xffffffffu);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint k=2u;k<=1024u;k*=2u)for(uint j=k/2u;j;j/=2u) {
        for(uint i=uint(tid);i<1024u;i+=512u) {
            const uint other=i^j;
            if(other>i) {
                const bool exchange=(i&k)==0u?tile[i].x<tile[other].x:tile[i].x>tile[other].x;
                if(exchange){const uint2 tmp=tile[i];tile[i]=tile[other];tile[other]=tmp;}
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0) {
        uint reason=0u;
        if(control[8]<512u||control[8]>1024u||control[10]!=control[8])reason=5u;
        for(uint i=0u;i<512u;i++)
            if(!threshold_positive(tile[i].x)||tile[i].y>=a.cols||tile[i].x<control[7])reason=5u;
        if(!reason) {
            for(uint i=1u;i<512u;i++)if(tile[i-1u].x<=tile[i].x)reason=6u;
            if(control[8]>512u&&tile[511].x<=tile[512].x)reason=6u;
        }
        threshold_finish(control,stats,reason);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    for(uint i=uint(tid);i<1024u;i+=512u)pairs[i]=tile[i];
    if(control[0])ids[tid]=int(tile[tid].y);
}
