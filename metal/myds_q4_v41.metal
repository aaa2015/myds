// V4.1 ordinary scalar Dense Q4_K candidates. No existing symbol changes.
// Host must accept only the named full shapes, NR2/NSG4, contiguous one-token
// ranges, disjoint writers/readers and relative ranges below UINT32_MAX.
// addr: fixed one-token addresses/row strides but runtime nb from args.ne00.
// nb: the same candidate with fixed nb. No explicit K unroll/reassociation.
#include <metal_stdlib>
using namespace metal;
#ifdef FC_MUL_MV
#define MYDS_Q4_V41_NSG FC_mul_mv_nsg
#else
constant short MYDS_Q4_V41_NSG [[function_constant(600)]];
#endif
struct myds_q4_v41_args {
    int ne00, ne01, ne02;
    ulong nb00, nb01, nb02, nb03;
    int ne10, ne11, ne12;
    ulong nb10, nb11, nb12, nb13;
    int ne0, ne1, nr0;
    short r2, r3;
};
struct myds_q4_v41_block { half d, dmin; uchar scales[12]; uchar qs[128]; };
static_assert(sizeof(myds_q4_v41_args)==112, "production argument ABI");
static_assert(sizeof(myds_q4_v41_block)==144, "raw Q4_K block ABI");
static inline float myds_q4_v41_bf16(float x) {
    uint u=as_type<uint>(x);
    if ((u & 0x7f800000u)!=0x7f800000u) u += 0x7fffu + ((u >> 16u) & 1u);
    return as_type<float>(u & 0xffff0000u);
}
template<uint ROWS, uint WIDTH, bool FIXED_NB, bool BF16>
static inline void myds_q4_v41_body(
        constant myds_q4_v41_args &args,
        device const char *src0, device const char *src1, device char *dst,
        uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG=MYDS_Q4_V41_NSG;
    constexpr uint16_t kmask1=0x3f3f,kmask2=0x0f0f,kmask3=0xc0c0;
    const short ix=tiisg/8;
    const short it=tiisg%8;
    const short iq=it/4;
    const short ir=it%4;
    const int nb=FIXED_NB ? int(WIDTH/256u) : args.ne00/256;
    const uint first_row=(tgpig.x*uint(NSG)+uint(sgitg))*2u;
    constexpr uint ROW_BYTES=(WIDTH/256u)*144u;
    const uint offset0=first_row*ROW_BYTES;
    device const float *y=(device const float *)src1;
    float yl[16],yh[16];
    float sumf[2]={0.f};
    device const float *y4=y+ix*256+64*iq+8*ir;
    uint16_t sc16[4];
    thread const uint8_t *sc8=(thread const uint8_t *)sc16;
    for(int ib=ix;ib<nb;ib+=4) {
        float4 sumy={0.f,0.f,0.f,0.f};
        for(short i=0;i<8;++i) {
            yl[i+0]=y4[i+0]; sumy[0]+=yl[i+0];
            yl[i+8]=y4[i+32]; sumy[1]+=yl[i+8];
            yh[i+0]=y4[i+128]; sumy[2]+=yh[i+0];
            yh[i+8]=y4[i+160]; sumy[3]+=yh[i+8];
        }
        const uint block_offset=offset0+uint(ib)*144u;
        device const myds_q4_v41_block *block=(device const myds_q4_v41_block *)(src0+block_offset);
        device const uint16_t *sc=(device const uint16_t *)block->scales+iq;
        device const uint16_t *q1=(device const uint16_t *)block->qs+16*iq+4*ir;
        device const half *dh=&block->d;
        for(short row=0;row<2;row++) {
            sc16[0]=sc[0]&kmask1;
            sc16[1]=sc[2]&kmask1;
            sc16[2]=((sc[4]>>0)&kmask2)|((sc[0]&kmask3)>>2);
            sc16[3]=((sc[4]>>4)&kmask2)|((sc[2]&kmask3)>>2);
            device const uint16_t *q2=q1+32;
            float4 acc1={0.f,0.f,0.f,0.f};
            float4 acc2={0.f,0.f,0.f,0.f};
            _Pragma("clang loop unroll(full)") for(short i=0;i<4;++i) {
                acc1[0]+=yl[2*i+0]*(q1[i]&0x000F);
                acc1[1]+=yl[2*i+1]*(q1[i]&0x0F00);
                acc1[2]+=yl[2*i+8]*(q1[i]&0x00F0);
                acc1[3]+=yl[2*i+9]*(q1[i]&0xF000);
                acc2[0]+=yh[2*i+0]*(q2[i]&0x000F);
                acc2[1]+=yh[2*i+1]*(q2[i]&0x0F00);
                acc2[2]+=yh[2*i+8]*(q2[i]&0x00F0);
                acc2[3]+=yh[2*i+9]*(q2[i]&0xF000);
            }
            sumf[row]+=dh[0]*((acc1[0]+1.f/256.f*acc1[1])*sc8[0]+
                              (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f+
                              (acc2[0]+1.f/256.f*acc2[1])*sc8[4]+
                              (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f)-
                       dh[1]*(sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
            q1+=ROW_BYTES/2u;
            sc+=ROW_BYTES/2u;
            dh+=ROW_BYTES/2u;
        }
        y4+=4*256;
    }
    device float *dst_f32=(device float *)dst;
    for(int row=0;row<2;++row) {
        float sum_all=simd_sum(sumf[row]);
        if(tiisg==0) dst_f32[first_row+row]=BF16 ? myds_q4_v41_bf16(sum_all) : sum_all;
    }
}
#define MYDS_Q4_V41_KERNEL(NAME, ROWS, WIDTH, FIXED, BF16) \
kernel void NAME(constant myds_q4_v41_args &args [[buffer(0)]], \
 device const char *w [[buffer(1)]], device const char *x [[buffer(2)]], \
 device char *y [[buffer(3)]], threadgroup char *scratch [[threadgroup(0)]], \
 uint3 group [[threadgroup_position_in_grid]], \
 ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) { \
 myds_q4_v41_body<ROWS,WIDTH,FIXED,BF16>(args,w,x,y,group,lane,sg); (void)scratch; }
MYDS_Q4_V41_KERNEL(kernel_myds_q4_v41_head_addr_f32,129280,5120,false,false)
MYDS_Q4_V41_KERNEL(kernel_myds_q4_v41_head_nb_f32,129280,5120,true,false)
MYDS_Q4_V41_KERNEL(kernel_myds_q4_v41_qb_addr_bf16,32768,1280,false,true)
MYDS_Q4_V41_KERNEL(kernel_myds_q4_v41_qb_nb_bf16,32768,1280,true,true)
MYDS_Q4_V41_KERNEL(kernel_myds_q4_v41_ob_addr_bf16,5120,8192,false,true)
MYDS_Q4_V41_KERNEL(kernel_myds_q4_v41_ob_nb_bf16,5120,8192,true,true)
#undef MYDS_Q4_V41_KERNEL

