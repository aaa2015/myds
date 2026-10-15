#define _POSIX_C_SOURCE 200809L
#include "myds_v41_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if(!(c)){fprintf(stderr,"[FAIL] %s:%d: %s\n",__FILE__,__LINE__,#c);return 0;} } while(0)
static void fill(myds_metal_buffer_t *b,unsigned seed) {
    float *p=myds_metal_buffer_data(b);
    for(size_t i=0;i<myds_metal_buffer_size(b)/4;i++)p[i]=(float)((int)((i+seed)%53)-26)/64.0f;
}
static int exact(myds_metal_buffer_t *a,myds_metal_buffer_t *b,size_t count) {
    const float *x=myds_metal_buffer_data(a),*y=myds_metal_buffer_data(b);
    for(size_t i=0;i<count;i++)if(!isfinite(x[i])||!isfinite(y[i]))return 0;
    return !memcmp(x,y,count*4);
}
struct fixture {
    uint32_t rows,comp_rows;
    myds_metal_buffer_t *q,*kv,*window,*comp,*ids,*gathered,*sinks,*out,*ref,*tmp;
};
static int fixture_create(struct fixture *f,uint32_t c,uint32_t n) {
    memset(f,0,sizeof(*f));f->rows=c;f->comp_rows=n;
#define ALLOC(member,bytes) do { f->member=myds_metal_buffer_alloc(bytes);CHECK(f->member); } while(0)
    ALLOC(q,(size_t)c*64*512*4);ALLOC(kv,(size_t)c*512*4);ALLOC(window,128*512*4);
    ALLOC(comp,(size_t)n*512*4);ALLOC(ids,(size_t)c*512*4);ALLOC(gathered,(size_t)c*640*512*2);
    ALLOC(sinks,64*4);ALLOC(out,(size_t)c*64*512*4);ALLOC(ref,(size_t)c*64*512*4);ALLOC(tmp,(size_t)c*64*512*4);
#undef ALLOC
    fill(f->q,0);fill(f->kv,7);fill(f->window,11);fill(f->comp,13);
    memset(myds_metal_buffer_data(f->sinks),0,256);
    int32_t *ids=myds_metal_buffer_data(f->ids);
    for(uint32_t t=0;t<c;t++)for(uint32_t j=0;j<512;j++)ids[(size_t)t*512+j]=(int32_t)((j*127+t*17)%n);
    return 1;
}
static void fixture_free(struct fixture *f) {
    myds_metal_buffer_t *b[]={f->q,f->kv,f->window,f->comp,f->ids,f->gathered,f->sinks,f->out,f->ref,f->tmp};
    for(size_t i=0;i<sizeof(b)/sizeof(*b);i++)myds_metal_buffer_free(b[i]);
}
static int attention(struct fixture *f,myds_v41_gpu_scratch_t *s,myds_metal_buffer_t *out,
                     uint32_t c,uint32_t n,uint32_t pos,myds_metal_buffer_t *gathered) {
    return myds_v41_gpu_attention_chunk(out,f->sinks,f->q,f->kv,n?f->comp:NULL,n?f->ids:NULL,
                                       gathered,c,n,512,pos,0,n?1:0,s);
}
static int test_compact_resource(void) {
    const uint32_t caps[]={8192,65536,262144};
    const size_t compact=2176u*512u*2u+(size_t)2048u*2176u*2u;
    for(size_t i=0;i<3;i++){
        myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(caps[i]);CHECK(s);
        CHECK(myds_v41_gpu_chunk_key_max(s)==caps[i]);
        CHECK(myds_v41_gpu_chunk_workspace_bytes(s)==compact);
        myds_v41_gpu_scratch_free(s);
    }
    printf("  [PASS] compact workspace = %zu bytes across 8K/64K/256K contexts\n",compact);
    return 1;
}
static int test_mixed_paths(int disabled) {
    if(disabled)return 1;
    struct fixture f;CHECK(fixture_create(&f,256,4096));
    const uint32_t gathered_rows[]={1,256};
    for(int same_batch=0;same_batch<2;same_batch++)for(size_t k=0;k<2;k++){
        myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(8192),*r=myds_v41_gpu_scratch_create(8192);CHECK(s&&r);
        size_t bytes=myds_v41_gpu_chunk_workspace_bytes(s);
        CHECK(myds_metal_begin_batch());
        CHECK(attention(&f,s,f.tmp,256,0,3840,f.gathered));
        if(!same_batch){CHECK(myds_metal_commit_batch());CHECK(myds_metal_begin_batch());}
        CHECK(attention(&f,s,f.tmp,gathered_rows[k],4096,4096-gathered_rows[k],f.gathered));
        if(!same_batch){CHECK(myds_metal_commit_batch());CHECK(myds_metal_begin_batch());}
        CHECK(myds_v41_gpu_carry_stage(s,f.window,4096));
        CHECK(attention(&f,s,f.out,256,0,4096,f.gathered));
        CHECK(myds_v41_gpu_carry_stage(r,f.window,4096));
        CHECK(attention(&f,r,f.ref,256,0,4096,f.gathered));
        CHECK(myds_metal_commit_batch());
        CHECK(exact(f.out,f.ref,(size_t)256*64*512));
        CHECK(myds_v41_gpu_chunk_workspace_bytes(s)==bytes);
        myds_v41_gpu_scratch_free(s);myds_v41_gpu_scratch_free(r);
    }
    fixture_free(&f);printf("  [PASS] window256 -> gather1/256 -> window256, separate/same batch, byte exact\n");return 1;
}
static int test_slab_grow_carry(void) {
    struct fixture f;CHECK(fixture_create(&f,32,4096));
    myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(8192),*r=myds_v41_gpu_scratch_create(8192);CHECK(s&&r);
    size_t compact=myds_v41_gpu_chunk_workspace_bytes(s);
    /* Pre-grow the reference, then give both workspaces the same GPU-staged
     * carry. The tested workspace grows while that carry is still in flight. */
    CHECK(myds_metal_begin_batch());CHECK(attention(&f,r,f.tmp,32,4096,4096,NULL));CHECK(myds_metal_commit_batch());
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_carry_stage(s,f.window,4096));CHECK(myds_v41_gpu_carry_stage(r,f.window,4096));
    myds_metal_inject_copy_failure(1);
    CHECK(!attention(&f,s,f.out,32,4096,4096,NULL));
    CHECK(myds_v41_gpu_chunk_workspace_bytes(s)==compact);
    myds_metal_inject_copy_failure(0);
    CHECK(attention(&f,s,f.out,32,4096,4096,NULL));CHECK(attention(&f,r,f.ref,32,4096,4096,NULL));
    CHECK(myds_metal_commit_batch());CHECK(exact(f.out,f.ref,(size_t)32*64*512));
    const uint32_t full_rows=(127u+2048u+8192u+31u)/32u*32u;
    CHECK(myds_v41_gpu_chunk_workspace_bytes(s)==(size_t)full_rows*(1024u+2048u*2u));
    CHECK(myds_v41_gpu_chunk_workspace_bytes(s)>compact);
    myds_v41_gpu_scratch_free(s);myds_v41_gpu_scratch_free(r);fixture_free(&f);
    printf("  [PASS] lazy slab growth preserves in-flight carry, failure state and fallback coverage\n");return 1;
}
static int test_strategy_threshold(int threshold) {
    if(!threshold)return 1;
    struct fixture f;CHECK(fixture_create(&f,32,4096));
    myds_metal_buffer_t *low_ids=myds_metal_buffer_alloc((size_t)32*512*4);CHECK(low_ids);
    int32_t *p=myds_metal_buffer_data(low_ids);
    for(uint32_t t=0;t<32;t++)for(uint32_t j=0;j<512;j++)p[t*512+j]=(j*127+t*17)%1024;
    const uint32_t ns[]={1024,4096,0};
    for(int same_batch=0;same_batch<2;same_batch++){
        CHECK(!setenv("MYDS_ATTN_GATHER_MIN_KEYS","2048",1));
        myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(8192);CHECK(s);
        CHECK(!setenv("MYDS_ATTN_GATHER_MIN_KEYS","0",1));
        myds_v41_gpu_scratch_t *r[3];myds_metal_buffer_t *out[3],*ref[3];
        for(size_t i=0;i<3;i++){
            r[i]=myds_v41_gpu_scratch_create(8192);out[i]=myds_metal_buffer_alloc((size_t)32*64*512*4);
            ref[i]=myds_metal_buffer_alloc((size_t)32*64*512*4);CHECK(r[i]&&out[i]&&ref[i]);
        }
        CHECK(!setenv("MYDS_ATTN_GATHER_MIN_KEYS","2048",1));
        size_t bytes=myds_v41_gpu_chunk_workspace_bytes(s);
        CHECK(myds_metal_begin_batch());
        for(size_t i=0;i<3;i++){
            uint32_t n=ns[i];const myds_metal_buffer_t *ids=n==1024?low_ids:f.ids;
            CHECK(myds_v41_gpu_carry_stage(s,f.window,4096));CHECK(myds_v41_gpu_carry_stage(r[i],f.window,4096));
            CHECK(myds_v41_gpu_attention_chunk(out[i],f.sinks,f.q,f.kv,n?f.comp:NULL,n?ids:NULL,
                                               f.gathered,32,n,512,4096,0,n?1:0,s));
            CHECK(myds_v41_gpu_attention_chunk(ref[i],f.sinks,f.q,f.kv,n?f.comp:NULL,n?ids:NULL,
                                               n==4096?f.gathered:NULL,32,n,512,4096,0,n?1:0,r[i]));
            if(!same_batch){CHECK(myds_metal_commit_batch());if(i<2)CHECK(myds_metal_begin_batch());}
        }
        if(same_batch)CHECK(myds_metal_commit_batch());
        for(size_t i=0;i<3;i++){CHECK(exact(out[i],ref[i],(size_t)32*64*512));
            myds_v41_gpu_scratch_free(r[i]);myds_metal_buffer_free(out[i]);myds_metal_buffer_free(ref[i]);}
        CHECK(myds_v41_gpu_chunk_workspace_bytes(s)==bytes);myds_v41_gpu_scratch_free(s);
    }
    const char *bad[]={"-1","8193","garbage",""};
    for(size_t i=0;i<4;i++){CHECK(!setenv("MYDS_ATTN_GATHER_MIN_KEYS",bad[i],1));CHECK(!myds_v41_gpu_scratch_create(8192));}
    CHECK(!setenv("MYDS_ATTN_GATHER_MIN_KEYS","2048",1));
    myds_metal_buffer_free(low_ids);fixture_free(&f);
    printf("  [PASS] opt-in threshold low/high/window sequence, separate/same batch, exact forced-path references\n");return 1;
}
static int test_window_boundaries(void) {
    const uint32_t max=MYDS_V41_CHUNK_MAX,cs[]={1,2,31,32,127,128,129,MYDS_V41_CHUNK_MAX};
    myds_metal_buffer_t *q=myds_metal_buffer_alloc((size_t)max*64*512*4),*kv=myds_metal_buffer_alloc((size_t)max*512*4),
                        *h=myds_metal_buffer_alloc((size_t)max*64*512*4),*sink=myds_metal_buffer_alloc(256);
    CHECK(q&&kv&&h&&sink);memset(myds_metal_buffer_data(q),0,myds_metal_buffer_size(q));
    memset(myds_metal_buffer_data(sink),0,256);float *v=myds_metal_buffer_data(kv);
    for(size_t i=0;i<(size_t)max*512;i++)v[i]=1.0f;
    for(size_t ci=0;ci<sizeof(cs)/sizeof(*cs);ci++){
        myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(8192);CHECK(s);
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_attention_chunk(h,sink,q,kv,NULL,NULL,NULL,cs[ci],0,512,0,0,0,s));
        CHECK(myds_metal_commit_batch());const float *out=myds_metal_buffer_data(h);
        for(uint32_t t=0;t<cs[ci];t++){
            uint32_t visible=t+1u<128u?t+1u:128u;float ref=(float)visible/(visible+1.0f);
            for(uint32_t head=0;head<64;head++)for(uint32_t d=0;d<512;d++){
                float got=out[((size_t)t*64+head)*512+d];CHECK(isfinite(got)&&fabsf(got-ref)<4e-6f);
            }
        }
        myds_v41_gpu_scratch_free(s);
    }
    myds_metal_buffer_free(q);myds_metal_buffer_free(kv);myds_metal_buffer_free(h);myds_metal_buffer_free(sink);
    printf("  [PASS] window causal/sink oracle through max chunk, including padded final block\n");return 1;
}
static int test_pack_and_gather(void) {
    const uint32_t rows=65,count=rows*512;
    myds_metal_buffer_t *src=myds_metal_buffer_alloc((size_t)(count+5)*4),*packed=myds_metal_buffer_alloc((size_t)(count+7)*2),
                        *ids=myds_metal_buffer_alloc(rows*4),*a=myds_metal_buffer_alloc((size_t)count*4),*b=myds_metal_buffer_alloc((size_t)count*4);
    CHECK(src&&packed&&ids&&a&&b);int32_t *ip=myds_metal_buffer_data(ids);for(uint32_t i=0;i<rows;i++)ip[i]=(i*17)%rows;
    for(uint32_t mode=2;mode<=3;mode++){
        fill(src,mode);CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_quant(src,512,rows,mode));CHECK(myds_metal_commit_batch());
        memset(myds_metal_buffer_data(packed),0xa5,myds_metal_buffer_size(packed));
        CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_pack_bf16(packed,3,src,0,count));CHECK(myds_metal_commit_batch());
        const uint32_t *sp=myds_metal_buffer_data(src);const uint16_t *pp=myds_metal_buffer_data(packed);
        for(uint32_t i=0;i<count;i++){CHECK((sp[i]&65535u)==0);CHECK(((uint32_t)pp[3+i]<<16)==sp[i]);}
        for(uint32_t i=0;i<3;i++)CHECK(pp[i]==0xa5a5);for(uint32_t i=count+3;i<count+7;i++)CHECK(pp[i]==0xa5a5);
        myds_metal_buffer_t *view=myds_metal_buffer_sub_view(packed,6,(size_t)count*2);CHECK(view);
        CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_gather(a,src,ids,rows,rows));
        CHECK(myds_v41_gpu_gather_bf16(b,view,ids,rows,rows));CHECK(myds_metal_commit_batch());CHECK(exact(a,b,count));myds_metal_buffer_free(view);
    }
    /* Exact bit transport includes signed zero, tiny BF16 and large values
     * outside IEEE F16's range; reading the codes as half would fail this. */
    uint32_t bits[]={0x00000000,0x80000000,0x00010000,0x7f7f0000,0xff7f0000,0x7f800000,0xff800000,0x7fc10000};
    memcpy(myds_metal_buffer_data(src),bits,sizeof(bits));CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_pack_bf16(packed,0,src,0,8));CHECK(myds_metal_commit_batch());
    for(size_t i=0;i<8;i++)CHECK(((uint32_t)((uint16_t*)myds_metal_buffer_data(packed))[i]<<16)==bits[i]);
    memcpy((uint32_t*)myds_metal_buffer_data(src)+5,bits,sizeof(bits));
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_pack_bf16(packed,2,src,5,8));CHECK(myds_metal_commit_batch());
    for(size_t i=0;i<8;i++)CHECK(((uint32_t)((uint16_t*)myds_metal_buffer_data(packed))[2+i]<<16)==bits[i]);
    ((int32_t*)myds_metal_buffer_data(ids))[0]=0;
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_pack_bf16(packed,0,src,0,512));
    CHECK(myds_v41_gpu_gather_bf16(b,packed,ids,1,1));CHECK(myds_metal_commit_batch());
    CHECK(!memcmp(myds_metal_buffer_data(src),myds_metal_buffer_data(b),512*4));
    CHECK(!myds_v41_gpu_pack_bf16(packed,SIZE_MAX,src,0,8));CHECK(!myds_v41_gpu_pack_bf16(packed,0,src,SIZE_MAX,8));
    myds_metal_buffer_free(src);myds_metal_buffer_free(packed);myds_metal_buffer_free(ids);myds_metal_buffer_free(a);myds_metal_buffer_free(b);
    printf("  [PASS] mode2/3 BF16 bits, offset guards, special bit patterns and exact gather\n");return 1;
}
static int test_indexscores_bf16(void) {
    const uint32_t rows=65,tokens=9,blocks=(rows+7)/8;
    myds_metal_buffer_t *keys=myds_metal_buffer_alloc(rows*128*4),*packed=myds_metal_buffer_alloc(rows*128*2),
                        *q=myds_metal_buffer_alloc(tokens*32*128*4),*w=myds_metal_buffer_alloc(tokens*32*4),
                        *a=myds_metal_buffer_alloc(tokens*rows*4),*b=myds_metal_buffer_alloc(tokens*rows*4),
                        *mask=myds_metal_buffer_alloc(blocks*sizeof(float));
    CHECK(keys&&packed&&q&&w&&a&&b&&mask);fill(keys,3);fill(q,7);fill(w,11);
    float *mp=(float*)myds_metal_buffer_data(mask);
    for(uint32_t bk=0;bk<blocks;bk++) mp[bk] = (bk % 2 == 0) ? 0.0f : -INFINITY;
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_quant(keys,128,rows,2));
    CHECK(myds_v41_gpu_pack_bf16(packed,0,keys,0,rows*128));CHECK(myds_metal_commit_batch());
    const uint32_t heads[]={1,7,32};for(size_t i=0;i<3;i++){
        CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_indexscores(a,q,w,keys,rows,heads[i]));
        CHECK(myds_v41_gpu_indexscores_bf16(b,q,w,packed,rows,heads[i]));CHECK(myds_metal_commit_batch());CHECK(exact(a,b,rows));

        /* Masked indexscores 测试：对比 全量+filter 与 直接masked 内核输出，要求 100% 逐位全同 */
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexscores(a,q,w,keys,rows,heads[i]));
        CHECK(myds_v41_gpu_filter(a,mask,rows,rows,1));
        CHECK(myds_v41_gpu_indexscores_masked_bf16(b,q,w,packed,mask,rows,heads[i]));
        CHECK(myds_metal_commit_batch());
        CHECK(!memcmp(myds_metal_buffer_data(a),myds_metal_buffer_data(b),(size_t)rows*4));
        const float *bp=(const float*)myds_metal_buffer_data(b);
        for(uint32_t r=0;r<rows;r++) {
            if ((r / 8) % 2 != 0) CHECK(bp[r] == -INFINITY);
            else CHECK(isfinite(bp[r]));
        }
    }
    for(int tiled=0;tiled<2;tiled++){
        CHECK(!setenv("MYDS_INDEXER_TILED",tiled?"1":"0",1));
        CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_indexscores_chunk(a,q,w,keys,rows,32,tokens,60,2));
        CHECK(myds_v41_gpu_indexscores_chunk_bf16(b,q,w,packed,rows,32,tokens,60,2));CHECK(myds_metal_commit_batch());
        CHECK(!memcmp(myds_metal_buffer_data(a),myds_metal_buffer_data(b),(size_t)tokens*rows*4));
        const float *p=myds_metal_buffer_data(b);for(uint32_t t=0;t<tokens;t++)for(uint32_t r=0;r<rows;r++)
            CHECK(r<(60+t+1)/2?isfinite(p[t*rows+r]):p[t*rows+r]==-INFINITY);
    }
    unsetenv("MYDS_INDEXER_TILED");
    CHECK(!setenv("MYDS_INDEXER_CHUNK_EXACT","0",1));
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_indexscores_chunk_bf16(a,q,w,packed,rows,32,tokens,60,2));CHECK(myds_metal_commit_batch());
    CHECK(!setenv("MYDS_INDEXER_CHUNK_EXACT","1",1));
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_indexscores_chunk_bf16(b,q,w,packed,rows,32,tokens,60,2));CHECK(myds_metal_commit_batch());
    CHECK(!memcmp(myds_metal_buffer_data(a),myds_metal_buffer_data(b),(size_t)tokens*rows*4));
    unsetenv("MYDS_INDEXER_CHUNK_EXACT");
    myds_metal_buffer_free(keys);myds_metal_buffer_free(packed);myds_metal_buffer_free(q);
    myds_metal_buffer_free(w);myds_metal_buffer_free(a);myds_metal_buffer_free(b);myds_metal_buffer_free(mask);
    printf("  [PASS] BF16 index scores byte exact: scalar/direct/chunk/tiled, masked early exit and causal tails\n");return 1;
}
static int test_indexscores_selected_blocks(void) {
    const uint32_t shapes[]={1,7,8,9,33,16385,32769,131077};
    myds_v41_gpu_indexer_compact_counters(NULL,true);
    for(size_t si=0;si<sizeof(shapes)/sizeof(*shapes);si++) {
        const uint32_t rows=shapes[si],blocks=(rows+7u)/8u,top=rows<512u?rows:512u;
        const size_t score_bytes=(size_t)rows*4u,cache_bytes=(size_t)rows*128u*2u;
        myds_metal_buffer_t *cache_owner=myds_metal_buffer_alloc(cache_bytes+32u),
            *a_owner=myds_metal_buffer_alloc(score_bytes+32u),*b_owner=myds_metal_buffer_alloc(score_bytes+32u);
        CHECK(cache_owner&&a_owner&&b_owner);
        memset(myds_metal_buffer_data(cache_owner),0xa5,cache_bytes+32u);
        memset(myds_metal_buffer_data(a_owner),0xa5,score_bytes+32u);
        memset(myds_metal_buffer_data(b_owner),0xa5,score_bytes+32u);
        /* BF16 cache deliberately has only ushort alignment; score views have
         * guard words before/after and do not start at owner offset zero. */
        myds_metal_buffer_t *cache=myds_metal_buffer_sub_view(cache_owner,6u,cache_bytes),
            *a=myds_metal_buffer_sub_view(a_owner,16u,score_bytes),*b=myds_metal_buffer_sub_view(b_owner,16u,score_bytes),
            *q=myds_metal_buffer_alloc(32u*128u*4u),*w=myds_metal_buffer_alloc(32u*4u),
            *mask=myds_metal_buffer_alloc((size_t)blocks*4u),*ids=myds_metal_buffer_alloc((size_t)blocks*4u),
            *sel_a=myds_metal_buffer_alloc((size_t)top*4u),*sel_b=myds_metal_buffer_alloc((size_t)top*4u);
        CHECK(cache&&a&&b&&q&&w&&mask&&ids&&sel_a&&sel_b);fill(q,7);
        for(uint32_t h=0;h<32u;h++)((float*)myds_metal_buffer_data(w))[h]=0.25f+(float)h/64.0f;
        uint16_t *cp=myds_metal_buffer_data(cache);
        for(uint32_t r=0;r<rows;r++)for(uint32_t d=0;d<128u;d++)
            cp[(size_t)r*128u+d]=(uint16_t)(0x3c00u+(si%2u?d%127u:(r*17u+d)%511u));
        /* This fixture uses scratch only for dense top-k. The surrounding
         * --threshold suite intentionally sets a larger attention threshold;
         * isolate its construction policy and restore the suite immediately. */
        const char *threshold=getenv("MYDS_ATTN_GATHER_MIN_KEYS");
        char *saved_threshold=threshold?strdup(threshold):NULL;
        CHECK(!threshold||saved_threshold);
        CHECK(!setenv("MYDS_ATTN_GATHER_MIN_KEYS","0",1));
        myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(rows<2u?2u:rows);
        int restore_error=saved_threshold?setenv("MYDS_ATTN_GATHER_MIN_KEYS",saved_threshold,1):unsetenv("MYDS_ATTN_GATHER_MIN_KEYS");
        free(saved_threshold);CHECK(!restore_error&&scratch);
        for(unsigned pattern=0;pattern<3;pattern++) {
            float *mp=myds_metal_buffer_data(mask);int32_t *ip=myds_metal_buffer_data(ids);uint32_t selected=0;
            for(uint32_t bk=0;bk<blocks;bk++) {
                const bool allowed=pattern==0||(pattern==2&&(bk%3u==0u||bk+1u==blocks));
                mp[bk]=allowed?0.0f:-INFINITY;
            }
            /* Reversed block order shows scoring order does not define tie IDs.
             * Only the reference dense top-k network decides final ordering. */
            for(uint32_t bk=blocks;bk>0;bk--)if(mp[bk-1u]==0.0f)ip[selected++]=(int32_t)(bk-1u);
            myds_metal_set_concurrent(0);myds_metal_set_auto_barrier(1);
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_masked_bf16(a,q,w,cache,mask,rows,32));
            CHECK(myds_v41_gpu_topk(sel_a,a,rows,top,scratch));CHECK(myds_metal_commit_batch());
            myds_metal_set_concurrent(1);myds_metal_set_auto_barrier(0);
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_selected_blocks_bf16(b,q,w,cache,mask,ids,rows,selected)==1);
            CHECK(myds_v41_gpu_topk(sel_b,b,rows,top,scratch));CHECK(myds_metal_commit_batch());
            CHECK(!memcmp(myds_metal_buffer_data(a),myds_metal_buffer_data(b),score_bytes));
            CHECK(!memcmp(myds_metal_buffer_data(sel_a),myds_metal_buffer_data(sel_b),(size_t)top*4u));
            const float *bp=myds_metal_buffer_data(b);
            for(uint32_t r=0;r<rows;r++)CHECK(mp[r/8u]==0.0f?isfinite(bp[r]):bp[r]==-INFINITY);
        }
        /* An absent optional PSO returns before even the -inf fill, so the
         * original scorer can safely run. Invalid shapes fail explicitly. */
        myds_v41_gpu_inject_indexer_compact_unavailable(true);
        CHECK(myds_v41_gpu_indexscores_selected_blocks_bf16(b,q,w,cache,mask,ids,rows,0)==0);
        CHECK(!memcmp(myds_metal_buffer_data(a),myds_metal_buffer_data(b),score_bytes));
        myds_v41_gpu_inject_indexer_compact_unavailable(false);
        CHECK(myds_v41_gpu_indexscores_selected_blocks_bf16(b,q,w,cache,mask,ids,rows,blocks+1u)==-1);
        CHECK(myds_v41_gpu_indexscores_selected_blocks_bf16(b,q,w,cache,NULL,ids,rows,0)==-1);
        CHECK(myds_v41_gpu_indexscores_selected_blocks_bf16(q,q,w,cache,mask,ids,rows,0)==-1);
        const uint8_t *ap=myds_metal_buffer_data(a_owner),*bp=myds_metal_buffer_data(b_owner),*guard=myds_metal_buffer_data(cache_owner);
        for(size_t i=0;i<16u;i++)CHECK(ap[i]==0xa5&&bp[i]==0xa5&&ap[16u+score_bytes+i]==0xa5&&bp[16u+score_bytes+i]==0xa5);
        for(size_t i=0;i<6u;i++)CHECK(guard[i]==0xa5);
        for(size_t i=6u+cache_bytes;i<cache_bytes+32u;i++)CHECK(guard[i]==0xa5);
        myds_v41_gpu_scratch_free(scratch);
        myds_metal_buffer_t *bufs[]={cache,a,b,q,w,mask,ids,sel_a,sel_b,cache_owner,a_owner,b_owner};
        for(size_t i=0;i<sizeof(bufs)/sizeof(*bufs);i++)myds_metal_buffer_free(bufs[i]);
    }
    myds_v41_gpu_indexer_compact_counters_t counters;myds_v41_gpu_indexer_compact_counters(&counters,false);
    CHECK(counters.encoded==24u&&counters.unavailable==8u&&counters.requests==32u);
    CHECK(counters.scored_row_slots<=counters.dense_rows+8u*24u);
    myds_metal_set_concurrent(0);myds_metal_set_auto_barrier(1);
    printf("  [PASS] selected-block scorer original dense scores/top-k IDs exact, finite/-inf ties, 1..128k tails, canaries, concurrent and unavailable PSO\n");return 1;
}

/* Compare actual GPU IDs, including untouched short-row tails, in three
 * paths: original dense, separate NaN classification, fused classification.
 * A CPU sort would not preserve the partition/network-dependent tie order.
 * unavailable: 1=all prefix optional PSOs, 2=fused sort, 3=fused merge. */
static int test_topk_prefix_case(uint32_t cols,uint32_t top,uint32_t rows,
    uint32_t start,uint32_t ratio,unsigned pattern,int concurrent,int unavailable,
    myds_v41_gpu_scratch_t *scratch) {
    const size_t score_bytes=(size_t)cols*rows*4u,id_bytes=(size_t)top*rows*4u;
    myds_metal_buffer_t *so=myds_metal_buffer_alloc(score_bytes+32u),*owners[3]={0},*out[3]={0};
    CHECK(so);memset(myds_metal_buffer_data(so),0xa5,score_bytes+32u);
    myds_metal_buffer_t *scores=myds_metal_buffer_sub_view(so,16u,score_bytes);CHECK(scores);
    for(unsigned path=0;path<3u;path++) {
        owners[path]=myds_metal_buffer_alloc(id_bytes+32u);CHECK(owners[path]);
        memset(myds_metal_buffer_data(owners[path]),0xa5,id_bytes+32u);
        out[path]=myds_metal_buffer_sub_view(owners[path],16u,id_bytes);CHECK(out[path]);
    }
    uint32_t nth=1u,max_threads=myds_metal_kernel_max_threads(ratio?
        "kernel_argsort_f32_i32_desc_causal":"kernel_argsort_f32_i32_desc");
    CHECK(max_threads);
    while(nth<cols&&2u*nth<=max_threads)nth*=2u;
    const uint32_t parts=(cols+nth-1u)/nth,blocktop=top<nth?top:nth;
    const uint32_t last=cols-(parts-1u)*nth;
    const uint32_t work=parts==1u?top:(parts-1u)*blocktop+(last<blocktop?last:blocktop);
    const uint32_t layout=ratio?(parts==1u?top:cols):work;
    const bool eligible=parts>=16u&&layout>2u*blocktop&&top<=512u;
    float *sp=myds_metal_buffer_data(scores);
    for(uint32_t row=0;row<rows;row++)for(uint32_t col=0;col<cols;col++) {
        const uint32_t h=col*2654435761u+row*739u+17u;
        uint32_t width=ratio?(start+row+1u)/ratio:cols;if(width>cols)width=cols;
        float v=(float)((int)(h%4001u)-2000)/256.0f+1.0f/1024.0f;
        if(pattern==1)v=3.25f;
        if(pattern==2&&col%19u)v=-INFINITY;
        if(pattern==3)v=-INFINITY;
        if(pattern==4&&col%17u==0u)v=col%2u?INFINITY:-INFINITY;
        if(pattern==5)v=col%2u?-0.0f:0.0f;
        if(pattern==6&&(col%1023u==0u||col+1u==cols)) {
            uint32_t bits=col%2u?0x7fc12345u:0xff800001u;
            memcpy(&v,&bits,4u); /* Quiet/signaling and signed NaNs. */
        }
        if(pattern==7)v=NAN;
        if(pattern==8&&ratio&&col>=width)v=NAN;
        if(pattern==9&&(col+1u)%nth==0u) {
            uint32_t bits=(col/nth+row)%2u?0xffc12345u:0x7f800001u;
            memcpy(&v,&bits,4u); /* The last value of each original partition. */
        }
        if(pattern==10&&width&&col+1u==width) {
            uint32_t bits=row%2u?0xff800001u:0x7fc12345u;
            memcpy(&v,&bits,4u); /* A NaN in the last visible run guards ALL merges. */
        }
        if(pattern==11&&((row%3u==1u&&col==0u)||(row%3u==2u&&col>=width))) {
            uint32_t bits=row%3u==1u?0x7f800001u:0xffc12345u;
            memcpy(&v,&bits,4u); /* Mixed finite, visible and future-only NaN rows. */
        }
        sp[(size_t)row*cols+col]=v;
    }
    void *copy=malloc(score_bytes);CHECK(copy);memcpy(copy,sp,score_bytes);
    uint64_t dense_dispatches=0u;
    for(unsigned path=0;path<3u;path++) {
        CHECK(!setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE",path?"1":"0",1));
        int env_error=path!=2u&&pattern%2u?unsetenv("MYDS_INDEXER_TOPK_NAN_FUSE"):
            setenv("MYDS_INDEXER_TOPK_NAN_FUSE",path==2u?"1":"0",1);
        CHECK(!env_error); /* Both absent and explicit zero use old PSOs. */
        myds_v41_gpu_topk_prefix_counters(NULL,true);myds_v41_gpu_topk_nan_fuse_counters(NULL,true);
        const bool missing=path&&(unavailable==1||(path==2u&&unavailable>=2));
        myds_v41_gpu_inject_topk_prefix_unavailable(path&&unavailable==1);
        myds_v41_gpu_inject_topk_nan_fuse_unavailable(path==2u?
            unavailable==2?MYDS_TOPK_NAN_FUSE_UNAVAILABLE_SORT:
            unavailable==3?MYDS_TOPK_NAN_FUSE_UNAVAILABLE_MERGE:0u:0u);
        myds_metal_set_concurrent(path?concurrent:0);myds_metal_set_auto_barrier(path?!concurrent:1);
        myds_metal_dispatch_count_reset();CHECK(myds_metal_begin_batch());
        CHECK(ratio?myds_v41_gpu_topk_causal(out[path],scores,cols,top,rows,start,ratio,scratch):
            myds_v41_gpu_topk(out[path],scores,cols,top,scratch));
        CHECK(myds_metal_commit_batch());
        myds_v41_gpu_inject_topk_prefix_unavailable(false);
        myds_v41_gpu_inject_topk_nan_fuse_unavailable(0u);
        const uint64_t dispatches=myds_metal_dispatch_count();
        if(!path)dense_dispatches=dispatches;
        else CHECK(!memcmp(myds_metal_buffer_data(out[0]),myds_metal_buffer_data(out[path]),id_bytes));
        CHECK(!memcmp(copy,sp,score_bytes));
        const int32_t *ids=myds_metal_buffer_data(out[path]);
        for(uint32_t row=0;row<rows;row++) {
            uint32_t width=ratio?(start+row+1u)/ratio:cols;if(width>cols)width=cols;
            const uint32_t written=width<top?width:top;
            for(uint32_t j=0;j<written;j++)CHECK(ids[(size_t)row*top+j]>=0&&(uint32_t)ids[(size_t)row*top+j]<width);
            for(uint32_t j=written;j<top;j++)CHECK((uint32_t)ids[(size_t)row*top+j]==0xa5a5a5a5u);
        }
        myds_v41_gpu_topk_prefix_counters_t counters;myds_v41_gpu_topk_prefix_counters(&counters,false);
        myds_v41_gpu_topk_nan_fuse_counters_t nf;myds_v41_gpu_topk_nan_fuse_counters(&nf,false);
        const bool requested=path&&eligible,encoded=requested&&!missing;
        CHECK(counters.requests==(requested?1u:0u));
        CHECK(counters.unavailable==(requested&&missing?1u:0u));
        CHECK(counters.encoded==(encoded?1u:0u));
        CHECK(counters.requests==counters.encoded+counters.unavailable);
        /* Separate scans add one dispatch. Fused scans add no dispatch versus
         * dense sorting, including PSO preflight fallback and ineligible calls. */
        CHECK(dispatches==dense_dispatches+(path==1u&&encoded?1u:0u));
        CHECK(nf.requests==(path==2u&&requested?1u:0u));
        CHECK(nf.encoded==(path==2u&&encoded?1u:0u));
        CHECK(nf.unavailable==(path==2u&&requested&&missing?1u:0u));
        CHECK(nf.classified_partitions==(path==2u&&encoded?(uint64_t)parts*rows:0u));
        CHECK(nf.requests==nf.encoded+nf.unavailable);
        if(encoded) {
            CHECK(counters.classified_rows==rows&&counters.merge_dispatches>0u);
            CHECK(counters.prefix_write_slots<=counters.dense_write_slots);
            /* A large score table may have only empty/tiny visible causal
             * prefixes. Strict savings require a visible sorted run longer
             * than top; table width alone does not imply removable writes. */
            bool removable=false;
            for(uint32_t row=0;row<rows;row++) {
                uint32_t width=ratio?(start+row+1u)/ratio:cols;if(width>cols)width=cols;
                uint32_t visible_work=0u;
                if(width)visible_work=ratio?((width-1u)/nth)*blocktop+
                    (((width-1u)%nth+1u)<blocktop?((width-1u)%nth+1u):blocktop):layout;
                if(visible_work>top)removable=true;
            }
            if(removable)CHECK(counters.prefix_write_slots<counters.dense_write_slots);
            else CHECK(counters.prefix_write_slots==counters.dense_write_slots);
        } else CHECK(counters.classified_rows==0u&&counters.merge_dispatches==0u&&counters.dense_write_slots==0u&&counters.prefix_write_slots==0u);
        const uint8_t *g=myds_metal_buffer_data(owners[path]);
        for(size_t j=0;j<16u;j++)CHECK(g[j]==0xa5&&g[16u+id_bytes+j]==0xa5);
    }
    const uint8_t *g=myds_metal_buffer_data(so);
    for(size_t j=0;j<16u;j++)CHECK(g[j]==0xa5&&g[16u+score_bytes+j]==0xa5);
    free(copy);myds_metal_buffer_free(scores);myds_metal_buffer_free(so);
    for(unsigned path=0;path<3u;path++){myds_metal_buffer_free(out[path]);myds_metal_buffer_free(owners[path]);}
    return 1;
}

static int test_topk_prefix_reuse(bool fused,uint32_t unavailable) {
    const uint32_t cols=16407u,starts[]={16364u,16406u,16364u},top=512u,rows[]={1u,3u,3u};
    myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(cols);CHECK(scratch);
    myds_metal_buffer_t *scores[3]={0},*ref[3]={0},*owners[3]={0},*out[3]={0};
    for(unsigned c=0;c<3u;c++) {
        const size_t bytes=(size_t)rows[c]*top*4u;
        scores[c]=myds_metal_buffer_alloc((size_t)rows[c]*cols*4u);
        ref[c]=myds_metal_buffer_alloc(bytes);owners[c]=myds_metal_buffer_alloc(bytes+32u);
        CHECK(scores[c]&&ref[c]&&owners[c]);
        memset(myds_metal_buffer_data(owners[c]),0xa5,bytes+32u);
        out[c]=myds_metal_buffer_sub_view(owners[c],16u,bytes);CHECK(out[c]);
        float *p=myds_metal_buffer_data(scores[c]);
        for(uint32_t row=0;row<rows[c];row++)for(uint32_t col=0;col<cols;col++) {
            float v=(float)((int)((col*127u+row*31u+c*19u)%751u)-375)/64.0f;
            /* Growth retains flags consumed by the first call. Reuse then
             * switches the same rows from visible NaN back to finite values. */
            if(c==1u&&(col%1023u==0u||col+1u==cols))v=NAN;
            if(c==2u&&col>=starts[c]+1u+row)v=NAN; /* future only, ignore */
            p[(size_t)row*cols+col]=v;
        }
        CHECK(!setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE","0",1));
        CHECK(!setenv("MYDS_INDEXER_TOPK_NAN_FUSE","0",1));
        myds_metal_set_concurrent(0);myds_metal_set_auto_barrier(1);
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_topk_causal(ref[c],scores[c],cols,top,rows[c],starts[c],1u,scratch));
        CHECK(myds_metal_commit_batch());
    }
    CHECK(!setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE","1",1));
    CHECK(!setenv("MYDS_INDEXER_TOPK_NAN_FUSE",fused?"1":"0",1));
    myds_v41_gpu_inject_topk_nan_fuse_unavailable(unavailable);
    myds_v41_gpu_topk_prefix_counters(NULL,true);
    myds_v41_gpu_topk_nan_fuse_counters(NULL,true);
    myds_metal_set_concurrent(1);myds_metal_set_auto_barrier(0);
    CHECK(myds_metal_begin_batch());
    for(unsigned c=0;c<3u;c++)
        CHECK(myds_v41_gpu_topk_causal(out[c],scores[c],cols,top,rows[c],starts[c],1u,scratch));
    CHECK(myds_metal_commit_batch());
    myds_v41_gpu_inject_topk_nan_fuse_unavailable(0u);
    myds_v41_gpu_topk_prefix_counters_t counters;myds_v41_gpu_topk_prefix_counters(&counters,false);
    CHECK(counters.requests==3u&&counters.encoded==(unavailable?0u:3u)&&counters.unavailable==(unavailable?3u:0u));
    CHECK(counters.classified_rows==(unavailable?0u:7u));
    if(!unavailable)CHECK(counters.prefix_write_slots<counters.dense_write_slots);
    else CHECK(counters.merge_dispatches==0u&&counters.prefix_write_slots==0u&&counters.dense_write_slots==0u);
    myds_v41_gpu_topk_nan_fuse_counters_t nf;myds_v41_gpu_topk_nan_fuse_counters(&nf,false);
    uint32_t nth=1u,max_threads=myds_metal_kernel_max_threads("kernel_argsort_f32_i32_desc_causal");
    while(nth<cols&&2u*nth<=max_threads)nth*=2u;
    CHECK(nf.requests==(fused?3u:0u)&&nf.encoded==(fused&&!unavailable?3u:0u)&&nf.unavailable==(unavailable?3u:0u));
    CHECK(nf.classified_partitions==(fused&&!unavailable?7ull*((cols+nth-1u)/nth):0u));
    for(unsigned c=0;c<3u;c++) {
        const size_t bytes=(size_t)rows[c]*top*4u;
        CHECK(!memcmp(myds_metal_buffer_data(ref[c]),myds_metal_buffer_data(out[c]),bytes));
        const uint8_t *g=myds_metal_buffer_data(owners[c]);
        for(size_t j=0;j<16u;j++)CHECK(g[j]==0xa5&&g[16u+bytes+j]==0xa5);
        myds_metal_buffer_free(out[c]);myds_metal_buffer_free(owners[c]);
        myds_metal_buffer_free(ref[c]);myds_metal_buffer_free(scores[c]);
    }
    myds_v41_gpu_scratch_free(scratch);
    printf("  [PASS] top-k %s (missing=%u) same-batch scratch/flag growth and reuse: 1->3->3 rows, finite->visible NaN->future NaN, including NaN partition becoming empty, original IDs and canaries, concurrent explicit barriers\n",fused?"fused NaN":"separate NaN",unavailable);
    return 1;
}

static int test_topk_nan_fuse_config(myds_v41_gpu_scratch_t *scratch) {
    myds_metal_buffer_t *scores=myds_metal_buffer_alloc(32u*4u),*ids=myds_metal_buffer_alloc(6u*4u);
    CHECK(scores&&ids);fill(scores,19u);
    float copy[32];memcpy(copy,myds_metal_buffer_data(scores),sizeof(copy));
    /* Strict validation applies before encoding, even to ineligible calls. */
    const char *invalid[]={"","2","-1","01","true"," 1","1 "};
    for(unsigned i=0;i<sizeof(invalid)/sizeof(*invalid);i++) {
        CHECK(!setenv("MYDS_INDEXER_TOPK_NAN_FUSE",invalid[i],1));
        memset(myds_metal_buffer_data(ids),0xa5,24u);
        myds_v41_gpu_topk_prefix_counters(NULL,true);myds_v41_gpu_topk_nan_fuse_counters(NULL,true);
        myds_metal_dispatch_count_reset();CHECK(myds_metal_begin_batch());
        CHECK(!myds_v41_gpu_topk(ids,scores,32u,6u,scratch));
        CHECK(myds_metal_commit_batch());CHECK(myds_metal_dispatch_count()==0u);
        for(unsigned j=0;j<6u;j++)CHECK(((uint32_t *)myds_metal_buffer_data(ids))[j]==0xa5a5a5a5u);
        CHECK(!memcmp(copy,myds_metal_buffer_data(scores),sizeof(copy)));
        myds_v41_gpu_topk_prefix_counters_t tp;myds_v41_gpu_topk_prefix_counters(&tp,false);
        myds_v41_gpu_topk_nan_fuse_counters_t nf;myds_v41_gpu_topk_nan_fuse_counters(&nf,false);
        CHECK(tp.requests==0u&&tp.encoded==0u&&tp.unavailable==0u&&tp.classified_rows==0u&&tp.merge_dispatches==0u);
        CHECK(nf.requests==0u&&nf.encoded==0u&&nf.unavailable==0u&&nf.classified_partitions==0u);
    }
    CHECK(!unsetenv("MYDS_INDEXER_TOPK_NAN_FUSE"));
    myds_metal_buffer_free(scores);myds_metal_buffer_free(ids);
    printf("  [PASS] fused NaN flag strict validation: 7 invalid values reject before encoding or writes\n");
    return 1;
}

static int test_topk_prefix(void) {
    const char *threshold=getenv("MYDS_ATTN_GATHER_MIN_KEYS");
    char *saved=threshold?strdup(threshold):NULL;CHECK(!threshold||saved);
    const char *nan_env=getenv("MYDS_INDEXER_TOPK_NAN_FUSE");
    char *saved_nan=nan_env?strdup(nan_env):NULL;CHECK(!nan_env||saved_nan);
    CHECK(!setenv("MYDS_INDEXER_TOPK_NAN_FUSE","0",1));
    CHECK(!setenv("MYDS_ATTN_GATHER_MIN_KEYS","0",1));
    myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(131077u);CHECK(scratch);
    const uint32_t cols[]={1u,31u,511u,512u,513u,1024u,1025u,1537u,2049u,5304u,32768u,131072u,131077u};
    unsigned cases=0u;
    for(unsigned i=0;i<sizeof(cols)/sizeof(*cols);i++)for(unsigned pattern=0;pattern<8u;pattern++) {
        uint32_t top=cols[i]<512u?cols[i]:512u;
        CHECK(test_topk_prefix_case(cols[i],top,1u,0u,0u,pattern,pattern%2u,0,scratch));cases++;
    }
    /* Derive the partition size from this device's actual sort PSO. At
     * nth=1024 these widths are 15360/15361: exactly 15 then 16 partitions. */
    uint32_t nth=1u,max_threads=myds_metal_kernel_max_threads("kernel_argsort_f32_i32_desc");
    CHECK(max_threads>=512u);
    while(2u*nth<=max_threads)nth*=2u;
    const uint32_t boundary[]={15u*nth,15u*nth+1u};
    CHECK(boundary[1]<=131077u);
    for(unsigned i=0;i<2u;i++)for(unsigned pattern=0;pattern<8u;pattern++) {
        CHECK(test_topk_prefix_case(boundary[i],512u,1u,0u,0u,pattern,pattern%2u,0,scratch));cases++;
    }
    printf("  [PASS] prefix partition boundary: nth=%u, widths=%u/%u (15/16 parts)\n",nth,boundary[0],boundary[1]);
    const uint32_t tops[]={1u,6u,31u,511u,512u,513u,2048u};
    for(unsigned i=0;i<sizeof(tops)/sizeof(*tops);i++)for(unsigned pattern=0;pattern<8u;pattern++) {
        CHECK(test_topk_prefix_case(16407u,tops[i],1u,0u,0u,pattern,pattern%2u,0,scratch));cases++;
    }
    for(unsigned pattern=0;pattern<9u;pattern++) {
        CHECK(test_topk_prefix_case(16407u,512u,7u,16364u,1u,pattern,pattern%2u,0,scratch));cases++;
        CHECK(test_topk_prefix_case(15361u,512u,7u,0u,2u,pattern,pattern%2u,0,scratch));cases++;
        CHECK(test_topk_prefix_case(15361u,512u,7u,1017u,1u,pattern,pattern%2u,0,scratch));cases++;
    }
    CHECK(test_topk_prefix_case(32768u,512u,1u,0u,0u,2u,1,1,scratch));cases++;
    CHECK(test_topk_prefix_case(16407u,512u,7u,16364u,1u,6u,1,1,scratch));cases++;
    for(unsigned pattern=9u;pattern<12u;pattern++) {
        CHECK(test_topk_prefix_case(32768u,512u,1u,0u,0u,pattern,pattern%2u,0,scratch));cases++;
        CHECK(test_topk_prefix_case(16407u,512u,7u,16364u,1u,pattern,pattern%2u,0,scratch));cases++;
        CHECK(test_topk_prefix_case(32768u,512u,7u,65529u,2u,pattern,pattern%2u,0,scratch));cases++;
        CHECK(test_topk_prefix_case(32768u,512u,7u,0u,2u,pattern,pattern%2u,0,scratch));cases++;
    }
    const uint32_t tiny_tops[]={1u,6u,31u,511u,512u};
    for(unsigned i=0;i<sizeof(tiny_tops)/sizeof(*tiny_tops);i++) {
        CHECK(test_topk_prefix_case(32768u,tiny_tops[i],1u,0u,0u,10u,1,0,scratch));cases++;
    }
    for(int missing=2;missing<=3;missing++) {
        CHECK(test_topk_prefix_case(32768u,512u,1u,0u,0u,10u,1,missing,scratch));cases++;
        CHECK(test_topk_prefix_case(16407u,512u,7u,16364u,1u,11u,1,missing,scratch));cases++;
    }
    CHECK(test_topk_prefix_reuse(false,0u));CHECK(test_topk_prefix_reuse(true,0u));
    CHECK(test_topk_prefix_reuse(true,MYDS_TOPK_NAN_FUSE_UNAVAILABLE_SORT));
    CHECK(test_topk_prefix_reuse(true,MYDS_TOPK_NAN_FUSE_UNAVAILABLE_MERGE));
    CHECK(test_topk_nan_fuse_config(scratch));
    myds_v41_gpu_topk_prefix_counters_t counters;
    myds_v41_gpu_topk_prefix_counters(&counters,true);myds_v41_gpu_topk_prefix_counters(&counters,false);
    CHECK(counters.requests==0u&&counters.encoded==0u&&counters.unavailable==0u&&counters.classified_rows==0u&&counters.merge_dispatches==0u&&counters.dense_write_slots==0u&&counters.prefix_write_slots==0u);
    myds_v41_gpu_topk_nan_fuse_counters_t nf;
    myds_v41_gpu_topk_nan_fuse_counters(&nf,true);myds_v41_gpu_topk_nan_fuse_counters(&nf,false);
    CHECK(nf.requests==0u&&nf.encoded==0u&&nf.unavailable==0u&&nf.classified_partitions==0u);
    myds_v41_gpu_scratch_free(scratch);CHECK(!unsetenv("MYDS_INDEXER_TOPK_PREFIX_MERGE"));
    int e=saved?setenv("MYDS_ATTN_GATHER_MIN_KEYS",saved,1):unsetenv("MYDS_ATTN_GATHER_MIN_KEYS");free(saved);CHECK(!e);
    e=saved_nan?setenv("MYDS_INDEXER_TOPK_NAN_FUSE",saved_nan,1):unsetenv("MYDS_INDEXER_TOPK_NAN_FUSE");free(saved_nan);CHECK(!e);
    myds_metal_set_concurrent(0);myds_metal_set_auto_barrier(1);
    printf("  [PASS] top-k prefix %u three-path GPU cases (dense/separate/fused): full IDs, 1..131077 columns, finite ties, signed zero, +/-INF, masked/all -INF, quiet/signaling/signed/partition-end/last-visible NaNs, mixed causal short/empty/future rows, canaries, explicit/auto barriers, separate and independent fused sort/merge PSO fallback, counters\n",cases);
    return 1;
}

static int test_attention_bf16(void) {
    struct fixture f;CHECK(fixture_create(&f,129,4096));
    myds_metal_buffer_t *storage=myds_metal_buffer_alloc((size_t)4096*512*2+6);CHECK(storage);
    myds_metal_buffer_t *packed=myds_metal_buffer_sub_view(storage,6,(size_t)4096*512*2);CHECK(packed);
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_quant(f.comp,512,4096,3));
    CHECK(myds_v41_gpu_pack_bf16(packed,0,f.comp,0,4096*512));CHECK(myds_metal_commit_batch());
    const uint32_t cs[]={1,2,31,32,127,128,129};
    for(int gather=0;gather<2;gather++)for(size_t i=0;i<7;i++){
        myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(8192),*r=myds_v41_gpu_scratch_create(8192);CHECK(s&&r);
        CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_carry_stage(s,f.window,4096));CHECK(myds_v41_gpu_carry_stage(r,f.window,4096));
        CHECK(attention(&f,s,f.out,cs[i],4096,4096,gather?f.gathered:NULL));
        CHECK(myds_v41_gpu_attention_chunk_bf16(f.ref,f.sinks,f.q,f.kv,packed,f.ids,gather?f.gathered:NULL,
                                               cs[i],4096,512,4096,0,1,r));
        CHECK(myds_metal_commit_batch());CHECK(exact(f.out,f.ref,(size_t)cs[i]*64*512));
        myds_v41_gpu_scratch_free(s);myds_v41_gpu_scratch_free(r);
    }
    myds_metal_buffer_free(packed);myds_metal_buffer_free(storage);fixture_free(&f);
    printf("  [PASS] F32/packed BF16 chunk attention byte exact, gather/slab, ushort-aligned views and short tails\n");return 1;
}
static int test_partial_picks(void) {
    struct fixture f;CHECK(fixture_create(&f,2,2));
    memset(myds_metal_buffer_data(f.q),0,myds_metal_buffer_size(f.q));
    float *kv=myds_metal_buffer_data(f.kv),*comp=myds_metal_buffer_data(f.comp);
    for(size_t i=0;i<2*512;i++){kv[i]=1;comp[i]=1;}
    int32_t *ids=myds_metal_buffer_data(f.ids);ids[0]=ids[1]=0;
    myds_metal_buffer_t *packed=myds_metal_buffer_alloc(2*512*2);CHECK(packed);
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_pack_bf16(packed,0,f.comp,0,2*512));CHECK(myds_metal_commit_batch());
    for(int side=0;side<2;side++){
        myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(8192);CHECK(s);
        memset(myds_metal_buffer_data(f.gathered),0xff,myds_metal_buffer_size(f.gathered));
        CHECK(myds_metal_begin_batch());
        CHECK(side?myds_v41_gpu_attention_chunk_bf16(f.out,f.sinks,f.q,f.kv,packed,f.ids,f.gathered,2,2,1,0,0,2,s):
                   myds_v41_gpu_attention_chunk(f.out,f.sinks,f.q,f.kv,f.comp,f.ids,f.gathered,2,2,1,0,0,2,s));
        CHECK(myds_metal_commit_batch());const float *out=myds_metal_buffer_data(f.out);
        for(size_t i=0;i<2*64*512;i++){float ref=i<64*512?0.5f:0.75f;CHECK(isfinite(out[i])&&fabsf(out[i]-ref)<4e-6f);}
        myds_v41_gpu_scratch_free(s);
    }
    myds_metal_buffer_free(packed);fixture_free(&f);
    printf("  [PASS] causal partial picks overwrite unused gather slots, F32/BF16 CPU oracle\n");return 1;
}
int main(int argc,char **argv) {
    int disabled=argc>1&&!strcmp(argv[1],"--slab");
    int threshold=argc>1&&!strcmp(argv[1],"--threshold");
    if(disabled&&setenv("MYDS_ATTN_GATHER","0",1))return 1;
    if(!disabled&&setenv("MYDS_ATTN_GATHER","1",1))return 1;
    if(setenv("MYDS_ATTN_GATHER_MIN_KEYS",threshold?"2048":"0",1))return 1;
    if(!myds_metal_init("metal"))return 1;
    if(argc>1&&!strcmp(argv[1],"--topk-prefix-only")) {
        int ok=test_topk_prefix();myds_metal_cleanup();return ok?0:1;
    }
    int ok=test_compact_resource()&&test_mixed_paths(disabled)&&test_slab_grow_carry()&&test_strategy_threshold(threshold)&&test_window_boundaries()&&
           test_pack_and_gather()&&test_indexscores_bf16()&&test_indexscores_selected_blocks()&&test_topk_prefix()&&test_attention_bf16()&&test_partial_picks();
    myds_metal_cleanup();if(ok)printf("ALL V4.1 GPU REGRESSIONS PASSED (%s)\n",disabled?"gather disabled":threshold?"explicit threshold":"gather enabled");return ok?0:1;
}
