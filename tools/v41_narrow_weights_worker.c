/* CPU-only diagnostic worker. No Metal, model writes, or inference linkage.
 * The exact E4M3 encoder and scale search below follow kernelpool/ds4
 * 7d9903b2ed6bd74c14aed06a89a591a14d613e9e (reused by 55b1e75).
 * Diagnostics scan every block/value instead of returning after the first
 * rejected block. Non-finite F16 blocks are rejected before ceilf->int.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint64_t elements, blocks, failed_blocks, values_without_accepted_block;
    uint64_t unencodable_at_all_scales, roundtrip_mismatches, shader_f32_mismatches;
    uint64_t positive_zero, negative_zero, subnormal, normal, infinity, nan;
    uint64_t failed_scale_trials, invalid_scales;
    uint64_t witness_scale_values_tested, under_witness_scale_mismatch_count, blocks_without_finite_witness_scale;
    uint64_t first_kind, first_index, first_source_bits, first_reconstructed_bits;
    uint64_t first_e0, first_scale_bits;
} v41_audit_stats;
_Static_assert(sizeof(float)==4 && sizeof(_Float16)==2, "requires IEEE F32/F16 storage");

static uint16_t read16(const uint8_t *p) { return (uint16_t)(p[0] | ((uint16_t)p[1] << 8)); }
static uint32_t read32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void write16(uint8_t *p, uint16_t x) { p[0]=(uint8_t)x; p[1]=(uint8_t)(x>>8); }
static void write32(uint8_t *p, uint32_t x) {
    p[0]=(uint8_t)x; p[1]=(uint8_t)(x>>8); p[2]=(uint8_t)(x>>16); p[3]=(uint8_t)(x>>24);
}
static uint32_t fbits(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static float asfloat(uint32_t u) { float x; memcpy(&x,&u,4); return x; }

/* Integer widening preserves signed zero and every NaN payload for reporting. */
static uint32_t half_bits_to_f32(uint16_t h) {
    uint32_t sign=(uint32_t)(h&0x8000u)<<16, m=h&1023u, e=(h>>10)&31u;
    if (e==31) return sign | 0x7f800000u | (m<<13);
    if (e) return sign | ((e+112u)<<23) | (m<<13);
    if (!m) return sign;
    int exp=-14;
    while (!(m&1024u)) { m<<=1; --exp; }
    return sign | ((uint32_t)(exp+127)<<23) | ((m&1023u)<<13);
}
static void classify(v41_audit_stats *s, uint32_t bits, bool half) {
    uint32_t mant=bits&(half?1023u:0x7fffffu), exp=bits&(half?0x7c00u:0x7f800000u);
    if (!exp && !mant) { if (bits&(half?0x8000u:0x80000000u)) s->negative_zero++; else s->positive_zero++; }
    else if (!exp) s->subnormal++;
    else if (exp==(half?0x7c00u:0x7f800000u)) { if (mant) s->nan++; else s->infinity++; }
    else s->normal++;
}
static void failure(v41_audit_stats *s, uint64_t kind, uint64_t i, uint32_t a,
                    uint32_t b, int e0, uint32_t scale) {
    if (!s->first_kind || i<s->first_index) {
        s->first_kind=kind; s->first_index=i; s->first_source_bits=a;
        s->first_reconstructed_bits=b; s->first_e0=(uint64_t)(int64_t)e0; s->first_scale_bits=scale;
    }
}

/* Verbatim encoder body from 7d9903b; only the unused annotation is omitted. */
static bool ds41_e4m3_code(float q, uint8_t *code) {
    const uint8_t sign = signbit(q) ? 0x80u : 0u;
    const float a = fabsf(q);
    if (a == 0.0f) { *code = sign; return true; }
    if (!(a <= 448.0f)) return false;
    int e;
    const float f = frexpf(a, &e);   /* a = f * 2^e, f in [0.5, 1) */
    if (e - 1 >= -6) {
        const float m = f * 16.0f - 8.0f;   /* 1.mmm: m in [0, 8) */
        if (m != floorf(m)) return false;
        *code = (uint8_t)(sign | (uint8_t)((e - 1 + 7) << 3) | (uint8_t)m);
        return true;
    }
    const float m = a * 512.0f;
    if (m != floorf(m) || m < 1.0f || m > 7.0f) return false;
    *code = (uint8_t)(sign | (uint8_t)m);
    return true;
}

uint32_t v41_audit_abi(void) { return (uint32_t)sizeof(v41_audit_stats); }

int v41_audit_router(const uint8_t *src, uint64_t n, uint8_t *codes,
                     uint8_t *restored, v41_audit_stats *s) {
    if (!src || !codes || !restored || !s || n>SIZE_MAX/4u) return -1;
    memset(s,0,sizeof(*s)); s->elements=n;
    for (uint64_t i=0;i<n;i++) {
        uint32_t a=read32(src+i*4), b=a&0xffff0000u;
        classify(s,a,false); write16(codes+i*2,(uint16_t)(a>>16)); write32(restored+i*4,b);
        if (a!=b) { s->roundtrip_mismatches++; failure(s,1,i,a,b,0,0); }
    }
    return 0;
}

/* One contiguous 32-row stripe, codes in source row-major order, scales and
 * valid mask in block-column order. Invalid blocks have code=0, scale=qNaN,
 * restored=F16 qNaN, mask=0. Their SHA is diagnostic, never a representation.
 */
int v41_audit_e4m3(const uint8_t *src, uint64_t cols, uint8_t *codes,
                    uint8_t *scales, uint8_t *restored, uint8_t *valid,
                    v41_audit_stats *s) {
    if (!src || !codes || !scales || !restored || !valid || !s || !cols || cols%32u || cols>SIZE_MAX/64u) return -1;
    memset(s,0,sizeof(*s)); s->elements=32u*cols; s->blocks=cols/32u;
    memset(codes,0,(size_t)s->elements); memset(valid,0,(size_t)s->blocks);
    for (uint64_t i=0;i<s->elements;i++) { uint16_t h=read16(src+2*i); classify(s,h,true); write16(restored+2*i,0x7e00u); }
    for (uint64_t bj=0;bj<s->blocks;bj++) {
        float vals[1024], amax=0.0f;
        uint16_t original[1024]; uint8_t selected[1024], any[1024]={0};
        bool nonfinite=false, fit=false; int selected_e=0; uint32_t first_witness_bad=1024u;
        write32(scales+4*bj,0x7fc00000u);
        for (uint32_t r=0;r<32;r++) for (uint32_t j=0;j<32;j++) {
            uint64_t i=(uint64_t)r*cols+bj*32u+j; uint32_t k=r*32u+j;
            uint16_t h=read16(src+2*i); original[k]=h;
            if ((h&0x7c00u)==0x7c00u) { nonfinite=true; failure(s,2,i,h,0x7e00u,0,0); vals[k]=0; }
            else { vals[k]=asfloat(half_bits_to_f32(h)); float v=fabsf(vals[k]); if (v>amax) amax=v; }
        }
        if (nonfinite) {
            s->failed_blocks++; s->values_without_accepted_block+=1024u;
            s->blocks_without_finite_witness_scale++;
            /* No reference scale exists: avoid undefined Inf->int conversion. */
            continue;
        }
        const int e0 = amax > 0.0f ? (int)ceilf(log2f(amax / 448.0f)) : 0;
        for (int d=0;!fit && d<3;d++) {
            const int e = e0 + (d == 1 ? 1 : d == 2 ? -1 : 0);
            const float inv = ldexpf(1.0f, -e);
            const float scale = ldexpf(1.0f, e);
            bool trial=true; uint8_t trial_codes[1024];
            if (!isfinite(inv) || !isfinite(scale) || !(scale>0)) { s->invalid_scales++; s->failed_scale_trials++; continue; }
            for (uint32_t k=0;k<1024;k++) {
                bool ok=ds41_e4m3_code(vals[k]*inv,&trial_codes[k]);
                if (ok) any[k]=1; else trial=false;
                if (d==0) {
                    s->witness_scale_values_tested++;
                    bool exact=ok;
                    if (ok) {
                        uint8_t c=trial_codes[k];
                        uint16_t hdiv=(uint16_t)(((uint16_t)(c&0x80u)<<8) | ((uint16_t)(c&0x7fu)<<7));
                        float back=asfloat(half_bits_to_f32(hdiv))*(scale*256.0f);
                        exact=fbits(back)==half_bits_to_f32(original[k]);
                    }
                    if (!exact) {
                        s->under_witness_scale_mismatch_count++;
                        if (first_witness_bad==1024u) first_witness_bad=k;
                    }
                }
            }
            if (trial) { fit=true; selected_e=e; memcpy(selected,trial_codes,sizeof(selected)); }
            else s->failed_scale_trials++;
        }
        if (!fit) {
            s->failed_blocks++; s->values_without_accepted_block+=1024u;
            bool individual_failure=false;
            for (uint32_t r=0;r<32;r++) for (uint32_t j=0;j<32;j++) {
                uint32_t k=r*32u+j;
                if (!any[k]) { individual_failure=true; s->unencodable_at_all_scales++; failure(s,3,(uint64_t)r*cols+bj*32u+j,original[k],0x7e00u,e0,0); }
            }
            /* Different values may require incompatible scales even if each
             * fits some trial. Still expose a first rejected block/value. */
            if (!individual_failure) {
                uint32_t k=first_witness_bad<1024u ? first_witness_bad : 0u;
                failure(s,7,(uint64_t)(k/32u)*cols+bj*32u+k%32u,original[k],0x7e00u,e0,0);
            }
            continue;
        }
        const float scale=ldexpf(1.0f,selected_e), s256=scale*256.0f;
        write32(scales+4*bj,fbits(scale)); valid[bj]=1;
        for (uint32_t r=0;r<32;r++) for (uint32_t j=0;j<32;j++) {
            uint32_t k=r*32u+j; uint64_t i=(uint64_t)r*cols+bj*32u+j;
            uint8_t c=selected[k]; codes[i]=c;
            /* Exact scalar equivalent of ds4_e4m3x4_div256() * s256. */
            uint16_t div256=(uint16_t)(((uint16_t)(c&0x80u)<<8) | ((uint16_t)(c&0x7fu)<<7));
            float decoded=asfloat(half_bits_to_f32(div256))*s256;
            _Float16 back=(_Float16)decoded; uint16_t h; memcpy(&h,&back,2); write16(restored+2*i,h);
            if (fbits(decoded)!=half_bits_to_f32(original[k])) { s->shader_f32_mismatches++; failure(s,5,i,original[k],h,e0,fbits(scale)); }
            if (h!=original[k]) { s->roundtrip_mismatches++; failure(s,4,i,original[k],h,e0,fbits(scale)); }
        }
    }
    return 0;
}
