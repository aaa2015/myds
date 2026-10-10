/* Real F16 GEMV -> BF16 -> output-copy graph acceptance. No GGUF is needed.
 * This translation unit includes the unchanged production runtime to retain
 * Completed-CB timestamps and inject a missing PSO. --host-only does not create
 * a Metal device. Other modes must run inside an exclusive GPU window.
 * Mode is fixed at 1 throughout the component: A1/A2 use explicit legacy APIs,
 * B uses the explicit candidate. This is API A-B-A, not environment A-B-A.
 */
#include "myds_metal.m"
#include <errno.h>
#include <inttypes.h>
#include <sys/stat.h>

static const char *shader_dir="bin", *output_dir;
static unsigned samples=5, repetitions=32, native_files;
static int host_only, expect_missing, no_timing;
static void fail(const char *file,int line,const char *expr) {
    fprintf(stderr,"FAIL %s:%d %s\n",file,line,expr); fflush(stdout); fflush(stderr); exit(1);
}
#define REQUIRE(x) do {if(!(x))fail(__FILE__,__LINE__,#x);} while(0)
#define SCHEMA "\"schema_version\":1"
static const uint32_t shapes[][2]={{4096,1280},{32,5120},{512,5120},{128,512},
    {1,4},{3,132},{5,508},{31,516},{33,1284},{127,5116},{129,5124},
    {511,1280},{513,5120},{4095,1280},{4097,1280}};
static const uint32_t tie_bits[]={0x3f807fffu,0x3f808000u,0x3f808001u,
    0x3f817fffu,0x3f818000u,0x3f818001u,0xbf807fffu,0xbf808000u,
    0xbf808001u,0xbf817fffu,0xbf818000u,0xbf818001u};
struct guarded {myds_metal_buffer_t *owner,*view;size_t bytes,offset;};
struct fixture {uint32_t rows,width,storage_rows;unsigned seed,pattern;
    struct guarded w,x,raw,ref,out,ref_copy,out_copy;};
static uint64_t fnv(const void *data,size_t n) {
    uint64_t h=UINT64_C(14695981039346656037);const uint8_t *p=data;
    for(size_t i=0;i<n;++i){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
static uint32_t random_bits(uint32_t i,unsigned seed) {
    uint32_t z=i*747796405u+seed*2891336453u;
    z=((z>>((z>>28u)+4u))^z)*277803737u;return (z>>22u)^z;
}
static uint32_t bf16(uint32_t u) {
    if((u&0x7f800000u)!=0x7f800000u)u+=0x7fffu+((u>>16u)&1u);
    return u&0xffff0000u;
}
static struct guarded alloc(size_t bytes,size_t offset) {
    struct guarded b={.bytes=bytes,.offset=offset};
    REQUIRE(bytes&&offset>=64u);b.owner=myds_metal_buffer_alloc(offset+bytes+64u);REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,offset+bytes+64u);
    b.view=myds_metal_buffer_sub_view(b.owner,offset,bytes);REQUIRE(b.view);return b;
}
static void guards(struct guarded b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(size_t i=0;i<b.offset;++i)REQUIRE(p[i]==0xa5);
    for(size_t i=0;i<64u;++i)REQUIRE(p[b.offset+b.bytes+i]==0xa5);
}
static void release(struct guarded b) {guards(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);}
static void fill_x(struct guarded b,unsigned seed,unsigned pattern) {
    float *x=myds_metal_buffer_data(b.view);
    for(uint32_t i=0;i<b.bytes/4u;++i){
        if(pattern){uint32_t u=tie_bits[i%12u];memcpy(x+i,&u,4u);}
        else x[i]=((int)(random_bits(i,seed)%1023u)-511)/257.0f;
    }
}
static struct fixture create(uint32_t rows,uint32_t width,unsigned seed,unsigned pattern) {
    struct fixture f={.rows=rows,.width=width,.storage_rows=(rows+1u)&~1u,.seed=seed,.pattern=pattern};
    f.w=alloc((size_t)f.storage_rows*width*2u,64u);f.x=alloc((size_t)width*4u,80u);
    f.raw=alloc((size_t)rows*4u,96u);f.ref=alloc(f.raw.bytes,112u);f.out=alloc(f.raw.bytes,128u);
    f.ref_copy=alloc(f.raw.bytes,144u);f.out_copy=alloc(f.raw.bytes,160u);
    uint16_t *w=myds_metal_buffer_data(f.w.view);memset(w,0,f.w.bytes);
    if(pattern)for(uint32_t row=0;row<rows;++row)w[(size_t)row*width+row%width]=0x3c00u;
    else for(size_t i=0;i<(size_t)rows*width;++i){uint32_t z=random_bits((uint32_t)i,17u);w[i]=(uint16_t)((0x2400u+z%0x800u)|((z>>16u)&0x8000u));}
    /* Odd NR2 output rows have an explicitly allocated zero-filled read-only
     * weight row. This row is exported; a guard starts after it. */
    fill_x(f.x,seed,pattern);return f;
}
static void destroy(struct fixture *f) {release(f->w);release(f->x);release(f->raw);release(f->ref);release(f->out);release(f->ref_copy);release(f->out_copy);}
static void poison(struct guarded b){memset(myds_metal_buffer_data(b.view),0xa5,b.bytes);}
static void policy(unsigned c,unsigned a){myds_metal_set_concurrent((int)c);myds_metal_set_auto_barrier((int)a);}
static void verify_encoder(void){REQUIRE(g_batch_enc&&g_batch_enc.dispatchType==(myds_metal_get_concurrent()?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial));}
static void input_join(const struct fixture *f,struct guarded out) {
    if(myds_metal_get_auto_barrier())myds_metal_barrier();
    else {const myds_metal_buffer_t *deps[]={f->w.view,f->x.view,out.view};myds_metal_barrier_buffers(deps,3u);}
}
static void output_join(struct guarded out){if(!myds_metal_get_auto_barrier())myds_metal_barrier_buffer(out.view);}
static int projected(const struct fixture *f,struct guarded out,bool candidate) {
    input_join(f,out);
    if(candidate){if(!myds_metal_v41_gemv_f16_bf16(out.view,f->w.view,f->x.view,f->rows,f->width))return 0;}
    else {
        if(!myds_metal_v41_gemv(out.view,f->w.view,f->x.view,f->rows,f->width,GGML_TYPE_F16))return 0;
        if(!myds_metal_get_auto_barrier())myds_metal_barrier_buffer(out.view);
        if(!myds_metal_v41_bf16(out.view,f->rows))return 0;
    }
    output_join(out);return 1;
}
static void consume(struct guarded out,struct guarded copied) {
    const myds_metal_buffer_t *deps[]={out.view,copied.view};myds_metal_barrier_buffers(deps,2u);
    REQUIRE(myds_metal_buffer_copy(copied.view,0,out.view,0,out.bytes));
    /* Complete this reader before a later iteration writes OUT. The copy is
     * a fixture consumer, not a claim about actual full-model downstream work. */
    myds_metal_barrier_buffers(deps,2u);
}
static id<MTLCommandBuffer> graph(const struct fixture *f,struct guarded out,struct guarded copied,bool candidate,unsigned reps) {
    REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
    for(unsigned i=0;i<reps;++i){REQUIRE(projected(f,out,candidate));consume(out,copied);}
    verify_encoder();REQUIRE(myds_metal_commit_batch());REQUIRE(cb.status==MTLCommandBufferStatusCompleted);return cb;
}
static void raw_reference(struct fixture *f) {
    REQUIRE(myds_metal_begin_batch());input_join(f,f->raw);
    REQUIRE(myds_metal_v41_gemv(f->raw.view,f->w.view,f->x.view,f->rows,f->width,GGML_TYPE_F16));
    output_join(f->raw);verify_encoder();id<MTLCommandBuffer> cb=g_batch_cb;
    REQUIRE(myds_metal_commit_batch());REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
    graph(f,f->ref,f->ref_copy,false,1u);
}
static unsigned low16(struct guarded b) {const uint32_t *p=myds_metal_buffer_data(b.view);unsigned n=0;for(size_t i=0;i<b.bytes/4u;++i)n+=(p[i]&0xffffu)!=0;return n;}
static void finite(struct guarded b){const float *p=myds_metal_buffer_data(b.view);for(size_t i=0;i<b.bytes/4u;++i){uint32_t u;memcpy(&u,p+i,4u);REQUIRE(isfinite(p[i])&&u!=0xa5a5a5a5u);}guards(b);}
static bool full_equal(struct fixture *f,struct guarded measured,struct guarded copy) {
    const uint32_t *raw=myds_metal_buffer_data(f->raw.view),*ref=myds_metal_buffer_data(f->ref.view);
    bool equal=true;for(uint32_t i=0;i<f->rows;++i)equal&=bf16(raw[i])==ref[i];
    equal&=!memcmp(ref,myds_metal_buffer_data(measured.view),f->ref.bytes);
    equal&=!memcmp(ref,myds_metal_buffer_data(copy.view),f->ref.bytes);
    equal&=!memcmp(ref,myds_metal_buffer_data(f->ref_copy.view),f->ref.bytes);return equal;
}
static void write_blob(const char *name,const void *data,size_t bytes) {
    char path[4096];REQUIRE(output_dir);int n=snprintf(path,sizeof(path),"%s/%s",output_dir,name);REQUIRE(n>0&&(size_t)n<sizeof(path));
    FILE *file=fopen(path,"rb");
    if(file){uint8_t *old=malloc(bytes?bytes:1u);REQUIRE(old);REQUIRE(fread(old,1,bytes,file)==bytes&&fgetc(file)==EOF&&!memcmp(old,data,bytes));REQUIRE(!fclose(file));free(old);return;}
    REQUIRE(errno==ENOENT);file=fopen(path,"wx");REQUIRE(file);REQUIRE(fwrite(data,1,bytes,file)==bytes&&!fclose(file));++native_files;
}
static void file_item(const char *prefix,const char *role,const char *dtype,struct guarded b) {
    char name[160];int n=snprintf(name,sizeof(name),"%s-%s.%s",prefix,role,!strcmp(dtype,"f32")?"f32":"f16");REQUIRE(n>0&&(size_t)n<sizeof(name));
    const void *data=myds_metal_buffer_data(b.view);write_blob(name,data,b.bytes);
    printf("{\"path\":\"%s\",\"role\":\"%s\",\"dtype\":\"%s\",\"bytes\":%zu,\"fnv1a64\":\"%016" PRIx64 "\",\"view_offset\":%zu}",name,role,dtype,b.bytes,fnv(data,b.bytes),b.offset);
    uint8_t *guard=malloc(b.offset+64u);REQUIRE(guard);const uint8_t *owner=myds_metal_buffer_data(b.owner);
    memcpy(guard,owner,b.offset);memcpy(guard+b.offset,owner+b.offset+b.bytes,64u);
    n=snprintf(name,sizeof(name),"%s-%s-guards.bin",prefix,role);REQUIRE(n>0&&(size_t)n<sizeof(name));write_blob(name,guard,b.offset+64u);
    printf(",{\"path\":\"%s\",\"role\":\"%s-guards\",\"dtype\":\"bytes\",\"bytes\":%zu,\"fnv1a64\":\"%016" PRIx64 "\"}",name,role,b.offset+64u,fnv(guard,b.offset+64u));free(guard);
}
static void files(const char *asset,const char *measured,struct fixture *f,struct guarded out,struct guarded copy) {
    /* Immutable seeded weights are shared across policies/arms/samples. Each
     * descriptor still binds the complete payload; avoid duplicating hundreds
     * of identical matrices in the evidence archive. */
    char weight_asset[64];snprintf(weight_asset,sizeof(weight_asset),"weights-r%u-w%u-p%u",f->rows,f->width,f->pattern);
    printf(",\"files\":[");file_item(weight_asset,"weight","f16",f->w);printf(",");file_item(asset,"input","f32",f->x);printf(",");
    file_item(asset,"reference-raw","f32",f->raw);printf(",");file_item(asset,"reference-output","f32",f->ref);printf(",");file_item(asset,"reference-consumer","f32",f->ref_copy);printf(",");
    file_item(measured,"measured-output","f32",out);printf(",");file_item(measured,"measured-consumer","f32",copy);printf("]");
}
static void shape_fields(const struct fixture *f) {
    printf(",\"rows\":%u,\"width\":%u,\"weight_storage_rows\":%u,\"weight_seed\":17,\"input_seed\":%u,\"pattern\":%u,\"actual_mode\":%d,\"nsg\":%u,\"nr0\":2,\"threads\":%u,\"groups\":%u,\"dynamic_tg_bytes\":256,\"concurrent\":%d,\"auto_barrier\":%d",f->rows,f->width,f->storage_rows,f->seed,f->pattern,myds_metal_f16_gemv_bf16_fuse_enabled(),MIN(8u,(f->width+127u)/128u),32u*MIN(8u,(f->width+127u)/128u),(f->rows+1u)/2u,myds_metal_get_concurrent(),myds_metal_get_auto_barrier());
    id<MTLComputePipelineState> pso=g_pso_f16_gemv_bf16[MIN(8u,(f->width+127u)/128u)-1u];
    printf(",\"candidate_max_threads\":%lu,\"candidate_simd_width\":%lu,\"candidate_static_tg_bytes\":%lu",(unsigned long)pso.maxTotalThreadsPerThreadgroup,(unsigned long)pso.threadExecutionWidth,(unsigned long)pso.staticThreadgroupMemoryLength);
}
static void numeric(unsigned n,unsigned shape,unsigned pattern,unsigned c,unsigned a) {
    policy(c,a);struct fixture f=create(shapes[shape][0],shapes[shape][1],31u+n,pattern);raw_reference(&f);
    myds_metal_f16_gemv_bf16_fuse_counters(NULL,true);uint64_t before=myds_metal_dispatch_count();
    graph(&f,f.out,f.out_copy,true,1u);uint64_t dispatches=myds_metal_dispatch_count()-before;
    myds_metal_f16_gemv_bf16_fuse_counters_t count;myds_metal_f16_gemv_bf16_fuse_counters(&count,false);
    bool equal=full_equal(&f,f.out,f.out_copy);char stem[64];snprintf(stem,sizeof(stem),"case%u",n);
    printf("{" SCHEMA ",\"kind\":\"parity\",\"case\":%u,\"shape\":%u,\"production_shape\":%s,\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"reference_raw_gpu_oracle\":true,\"full_f32_bit_exact\":%s,\"finite\":true,\"canaries\":true,\"caller_output_join\":true,\"output_copy_consumer\":true,\"consumer_war_join\":true,\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"failures\":%" PRIu64 ",\"candidate_graph_dispatches\":%" PRIu64 ",\"raw_low16_nonzero\":%u",n,shape,shape<4u?"true":"false",equal?"true":"false",count.requests,count.encoded,count.failures,dispatches,low16(f.raw));
    shape_fields(&f);files(stem,stem,&f,f.out,f.out_copy);puts("}");fflush(stdout);
    REQUIRE(equal&&dispatches==2u&&count.requests==1u&&count.encoded==1u&&!count.failures);
    finite(f.raw);finite(f.ref);finite(f.out);finite(f.ref_copy);finite(f.out_copy);destroy(&f);
}
static void reuse(unsigned case_id) {
    unsigned c=case_id/2u,a=case_id%2u;policy(c,a);struct fixture f=create(4096,1280,400u,0u);
    struct guarded sources[7],raws[7],refs[7],outs[7],rcopies[7],ocopies[7];
    for(unsigned i=0;i<7u;++i){sources[i]=alloc(f.x.bytes,80u);fill_x(sources[i],401u+i,0u);raws[i]=alloc(f.raw.bytes,96u);refs[i]=alloc(f.raw.bytes,112u);outs[i]=alloc(f.raw.bytes,128u);rcopies[i]=alloc(f.raw.bytes,144u);ocopies[i]=alloc(f.raw.bytes,160u);}
    myds_metal_f16_gemv_bf16_fuse_counters(NULL,true);REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
    for(unsigned i=0;i<7u;++i){
        const myds_metal_buffer_t *xdeps[]={f.x.view,sources[i].view};myds_metal_barrier_buffers(xdeps,2u);
        REQUIRE(myds_metal_buffer_copy(f.x.view,0,sources[i].view,0,f.x.bytes));myds_metal_barrier_buffers(xdeps,2u);
        input_join(&f,f.ref);REQUIRE(myds_metal_v41_gemv(f.ref.view,f.w.view,f.x.view,f.rows,f.width,GGML_TYPE_F16));
        myds_metal_barrier_buffer(f.ref.view);REQUIRE(myds_metal_buffer_copy(raws[i].view,0,f.ref.view,0,f.ref.bytes));
        myds_metal_barrier_buffer(f.ref.view);REQUIRE(myds_metal_v41_bf16(f.ref.view,f.rows));output_join(f.ref);
        consume(f.ref,f.ref_copy);REQUIRE(projected(&f,f.out,true));consume(f.out,f.out_copy);
        const myds_metal_buffer_t *deps[]={f.ref.view,f.out.view,f.ref_copy.view,f.out_copy.view};myds_metal_barrier_buffers(deps,4u);
        REQUIRE(myds_metal_buffer_copy(refs[i].view,0,f.ref.view,0,f.ref.bytes));REQUIRE(myds_metal_buffer_copy(outs[i].view,0,f.out.view,0,f.out.bytes));
        REQUIRE(myds_metal_buffer_copy(rcopies[i].view,0,f.ref_copy.view,0,f.ref_copy.bytes));REQUIRE(myds_metal_buffer_copy(ocopies[i].view,0,f.out_copy.view,0,f.out_copy.bytes));
        myds_metal_barrier_buffers(deps,4u);
    }
    verify_encoder();REQUIRE(myds_metal_commit_batch());REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
    myds_metal_f16_gemv_bf16_fuse_counters_t count;myds_metal_f16_gemv_bf16_fuse_counters(&count,false);
    printf("{" SCHEMA ",\"kind\":\"reuse\",\"case\":%u,\"concurrent\":%u,\"auto_barrier\":%u,\"same_command_buffer\":true,\"cb_count\":1,\"sequence\":[1,3,3],\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"snapshot_copies_add_dependencies\":true,\"input_raw_and_snapshot_war\":true,\"caller_output_join\":true,\"consumer_war_join\":true,\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"failures\":%" PRIu64 ",\"items\":[",case_id,c,a,count.requests,count.encoded,count.failures);
    bool equal=true;for(unsigned i=0;i<7u;++i){if(i)printf(",");struct fixture snap=f;snap.x=sources[i];snap.raw=raws[i];snap.ref=refs[i];snap.ref_copy=rcopies[i];snap.seed=401u+i;bool same=full_equal(&snap,outs[i],ocopies[i]);equal&=same;
        printf("{\"item\":%u,\"full_f32_bit_exact\":%s",i,same?"true":"false");shape_fields(&snap);char stem[64];snprintf(stem,sizeof(stem),"reuse%u-item%u",case_id,i);files(stem,stem,&snap,outs[i],ocopies[i]);printf("}");}
    puts("]}");fflush(stdout);REQUIRE(equal&&count.requests==7u&&count.encoded==7u&&!count.failures);
    for(unsigned i=0;i<7u;++i){finite(raws[i]);finite(refs[i]);finite(outs[i]);finite(rcopies[i]);finite(ocopies[i]);release(sources[i]);release(raws[i]);release(refs[i]);release(outs[i]);release(rcopies[i]);release(ocopies[i]);}destroy(&f);
}
static void rejected(unsigned case_id) {
    struct fixture f=create(32,5120,71u,0u);policy(1,0);poison(f.out);
    myds_metal_buffer_t *out=f.out.view;const myds_metal_buffer_t *w=f.w.view,*x=f.x.view;
    myds_metal_buffer_t ob=*out,wb=*w,xb=*x;uint32_t rows=f.rows,width=f.width;
    id<MTLComputePipelineState> saved=g_pso_f16_gemv_bf16[7];
    switch(case_id){case 0:out=NULL;break;case 1:w=NULL;break;case 2:x=NULL;break;case 3:rows=0;break;case 4:width=0;break;case 5:width=5119;break;case 6:rows=UINT32_MAX;break;
        case 7:ob.size-=4u;out=&ob;break;case 8:xb.size-=4u;x=&xb;break;case 9:wb.size-=2u;w=&wb;break;
        case 10:ob.offset+=4u;out=&ob;break;case 11:xb.offset+=4u;x=&xb;break;case 12:wb.offset+=2u;w=&wb;break;
        case 13:out=f.x.view;break;case 14:out=f.w.view;break;case 15:g_pso_f16_gemv_bf16[7]=nil;break;default:REQUIRE(0);}
    uint64_t of=fnv(myds_metal_buffer_data(f.out.owner),f.out.bytes+f.out.offset+64u),wf=fnv(myds_metal_buffer_data(f.w.owner),f.w.bytes+f.w.offset+64u),xf=fnv(myds_metal_buffer_data(f.x.owner),f.x.bytes+f.x.offset+64u);
    myds_metal_f16_gemv_bf16_fuse_counters(NULL,true);REQUIRE(myds_metal_begin_batch());uint64_t before=myds_metal_dispatch_count();
    int ret=myds_metal_v41_gemv_f16_bf16(out,w,x,rows,width);g_pso_f16_gemv_bf16[7]=saved;
    uint64_t dispatches=myds_metal_dispatch_count()-before;REQUIRE(myds_metal_commit_batch());
    myds_metal_f16_gemv_bf16_fuse_counters_t count;myds_metal_f16_gemv_bf16_fuse_counters(&count,false);
    bool unchanged=of==fnv(myds_metal_buffer_data(f.out.owner),f.out.bytes+f.out.offset+64u)&&wf==fnv(myds_metal_buffer_data(f.w.owner),f.w.bytes+f.w.offset+64u)&&xf==fnv(myds_metal_buffer_data(f.x.owner),f.x.bytes+f.x.offset+64u);
    printf("{" SCHEMA ",\"kind\":\"rejected_call\",\"case\":%u,\"returncode\":%d,\"dispatches\":%" PRIu64 ",\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"failures\":%" PRIu64 ",\"all_buffers_unchanged\":%s,\"no_reference_fallback\":true,\"files\":[",case_id,ret,dispatches,count.requests,count.encoded,count.failures,unchanged?"true":"false");
    char stem[64];snprintf(stem,sizeof(stem),"reject%u",case_id);file_item(stem,"unchanged-output","f32",f.out);printf(",");file_item(stem,"unchanged-input","f32",f.x);printf(",");file_item(stem,"unchanged-weight","f16",f.w);puts("]}");fflush(stdout);
    REQUIRE(!ret&&!dispatches&&count.requests==1u&&!count.encoded&&count.failures==1u&&unchanged);destroy(&f);
}
static void completion(void) {
    policy(1,0);struct fixture f=create(512,5120,103u,0u);raw_reference(&f);
    myds_metal_f16_gemv_bf16_fuse_counters(NULL,true);uint32_t next=myds_metal_get_commit_call_count()+1u;
    REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;REQUIRE(projected(&f,f.out,true));consume(f.out,f.out_copy);
    myds_metal_inject_commit_failure(next);int ret=myds_metal_commit_batch();myds_metal_inject_commit_failure(0);
    REQUIRE(cb.status==MTLCommandBufferStatusCompleted);myds_metal_f16_gemv_bf16_fuse_counters_t count;myds_metal_f16_gemv_bf16_fuse_counters(&count,false);
    bool same=full_equal(&f,f.out,f.out_copy);
    printf("{" SCHEMA ",\"kind\":\"completion_contract\",\"injected_host_commit_failure\":true,\"real_gpu_failure\":false,\"outer_returncode\":%d,\"cb_status\":4,\"completed_cb\":true,\"encoded_is_not_completion_success\":true,\"full_f32_bit_exact\":%s,\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"failures\":%" PRIu64,ret,same?"true":"false",count.requests,count.encoded,count.failures);shape_fields(&f);files("completion","completion",&f,f.out,f.out_copy);puts("}");fflush(stdout);
    REQUIRE(!ret&&same&&count.requests==1u&&count.encoded==1u&&!count.failures);destroy(&f);
}
static void timing(unsigned group,unsigned shape,unsigned c,unsigned a,unsigned reps) {
    policy(c,a);struct fixture f=create(shapes[shape][0],shapes[shape][1],1001u,0u);
    struct guarded input[6],raw[6],ref[6],copied[6];
    /* Complete all reference PSO warmup/oracles before any timed sample. */
    for(unsigned s=0;s<=samples;++s){fill_x(f.x,1001u+s,0u);raw_reference(&f);input[s]=alloc(f.x.bytes,80u);raw[s]=alloc(f.raw.bytes,96u);ref[s]=alloc(f.raw.bytes,112u);copied[s]=alloc(f.raw.bytes,144u);memcpy(myds_metal_buffer_data(input[s].view),myds_metal_buffer_data(f.x.view),f.x.bytes);memcpy(myds_metal_buffer_data(raw[s].view),myds_metal_buffer_data(f.raw.view),f.raw.bytes);memcpy(myds_metal_buffer_data(ref[s].view),myds_metal_buffer_data(f.ref.view),f.ref.bytes);memcpy(myds_metal_buffer_data(copied[s].view),myds_metal_buffer_data(f.ref_copy.view),f.ref_copy.bytes);}
    const char *names[]={"A1","B","A2"};
    for(unsigned arm=0;arm<3u;++arm)for(unsigned s=0;s<=samples;++s){
        memcpy(myds_metal_buffer_data(f.x.view),myds_metal_buffer_data(input[s].view),f.x.bytes);poison(f.out);poison(f.out_copy);
        myds_metal_f16_gemv_bf16_fuse_counters(NULL,true);uint64_t before=myds_metal_dispatch_count();
        id<MTLCommandBuffer> cb=graph(&f,f.out,f.out_copy,arm==1u,reps);uint64_t dispatches=myds_metal_dispatch_count()-before;
        myds_metal_f16_gemv_bf16_fuse_counters_t count;myds_metal_f16_gemv_bf16_fuse_counters(&count,false);
        struct fixture snap=f;snap.x=input[s];snap.raw=raw[s];snap.ref=ref[s];snap.ref_copy=copied[s];snap.seed=1001u+s;
        bool equal=full_equal(&snap,f.out,f.out_copy);double start=cb.GPUStartTime,end=cb.GPUEndTime;
        printf("{" SCHEMA ",\"kind\":\"gpu_timing\",\"group\":%u,\"shape\":%u,\"arm\":\"%s\",\"sample\":%u,\"warmup\":%s,\"repetitions\":%u,\"scope\":\"project_round_output_copy_graph\",\"candidate_api_used\":%s,\"actual_kernel\":\"%s\",\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"failures\":%" PRIu64 ",\"dispatches\":%" PRIu64 ",\"gpu_start_s\":%.17g,\"gpu_end_s\":%.17g,\"gpu_us_per_call\":%.17g,\"completed_cb\":true,\"cb_status\":4,\"encoder_dispatch_type_verified\":true,\"caller_output_join\":true,\"output_copy_consumer\":true,\"consumer_war_join\":true,\"oracle_outside_timed_cb\":true,\"full_f32_bit_exact\":%s,\"finite\":true,\"canaries\":true,\"dispatch_hist_disabled\":true,\"fixture_not_fullmodel\":true,\"cold_dram_not_proven\":true",group,shape,names[arm],s,s?"false":"true",reps,arm==1u?"true":"false",arm==1u?"kernel_mul_mv_f16_f32_4_bf16_nr2":"kernel_mul_mv_f16_f32_4",count.requests,count.encoded,count.failures,dispatches,start,end,(end-start)*1e6/reps,equal?"true":"false");
        shape_fields(&snap);char asset[64],measured[64];snprintf(asset,sizeof(asset),"timing-g%u-s%u",group,s);snprintf(measured,sizeof(measured),"timing-g%u-%s-s%u",group,names[arm],s);files(asset,measured,&snap,f.out,f.out_copy);puts("}");fflush(stdout);
        REQUIRE(equal&&dispatches==(arm==1u?2u:3u)*reps&&count.requests==(arm==1u?reps:0u)&&count.encoded==count.requests&&!count.failures);
        REQUIRE(isfinite(start)&&isfinite(end)&&start>0&&end>start);finite(f.out);finite(f.out_copy);
    }
    for(unsigned s=0;s<=samples;++s){release(input[s]);release(raw[s]);release(ref[s]);release(copied[s]);}destroy(&f);
}
static void host_config(void) {
    REQUIRE(!g_initialized&&!g_device&&myds_metal_f16_gemv_bf16_fuse_enabled()==0);
    unsetenv("MYDS_F16_GEMV_BF16_FUSE");REQUIRE(myds_metal_f16_gemv_bf16_fuse_parse_mode()==0);
    setenv("MYDS_F16_GEMV_BF16_FUSE","0",1);REQUIRE(myds_metal_f16_gemv_bf16_fuse_parse_mode()==0);
    setenv("MYDS_F16_GEMV_BF16_FUSE","1",1);REQUIRE(myds_metal_f16_gemv_bf16_fuse_parse_mode()==1);
    const char *bad[]={"","2","01","+1","-1","true"," 1","1 ","1\n","0x1"};
    for(unsigned i=0;i<10u;++i){setenv("MYDS_F16_GEMV_BF16_FUSE",bad[i],1);REQUIRE(!myds_metal_init("/invalid/f16-bf16/env"));REQUIRE(!g_initialized&&!g_device&&!g_queue&&!g_library&&!myds_metal_f16_gemv_bf16_fuse_enabled());}
    unsigned canonical=0;for(uint32_t i=0;i<65536u;++i){uint32_t bits=i<<16u;REQUIRE(bf16(bits)==bits);++canonical;}
    for(unsigned i=0;i<12u;++i){REQUIRE(bf16(bf16(tie_bits[i]))==bf16(tie_bits[i]));}
    REQUIRE(bf16(0x7f800000u)==0x7f800000u&&bf16(0xff800000u)==0xff800000u&&bf16(0x7fc12345u)==0x7fc10000u);
    unsetenv("MYDS_F16_GEMV_BF16_FUSE");myds_metal_f16_gemv_bf16_fuse_counters(NULL,true);
    REQUIRE(!myds_metal_v41_gemv_f16_bf16(NULL,NULL,NULL,32,5120));myds_metal_f16_gemv_bf16_fuse_counters_t ct;myds_metal_f16_gemv_bf16_fuse_counters(&ct,true);REQUIRE(ct.requests==1u&&!ct.encoded&&ct.failures==1u);myds_metal_f16_gemv_bf16_fuse_counters(&ct,false);REQUIRE(!ct.requests&&!ct.encoded&&!ct.failures);
    printf("{" SCHEMA ",\"kind\":\"host_config\",\"pass\":true,\"no_gpu_initialization\":true,\"strict_invalid_values\":10,\"canonical_bf16_words\":%u,\"round_tie_boundary_words\":12,\"nan_inf_integer_round_only\":true,\"counter_reset_and_failure\":true}\n",canonical);
}
static void init_record(unsigned sequence,int requested,bool success) {
    printf("{" SCHEMA ",\"kind\":\"initialization\",\"sequence\":%u,\"requested_mode\":%d,\"success\":%s,\"actual_mode\":%d,\"cleanup_verified\":true}\n",sequence,requested,success?"true":"false",myds_metal_f16_gemv_bf16_fuse_enabled());
}
static unsigned positive(const char *s){char *end;errno=0;unsigned long n=strtoul(s,&end,10);REQUIRE(!errno&&end!=s&&!*end&&n>0&&n<=10000u);return (unsigned)n;}
int main(int argc,char **argv) {
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"--host-only"))host_only=1;else if(!strcmp(argv[i],"--expect-missing-candidate"))expect_missing=1;else if(!strcmp(argv[i],"--no-timing"))no_timing=1;else if(!strcmp(argv[i],"--shader-dir")&&i+1<argc)shader_dir=argv[++i];else if(!strcmp(argv[i],"--output-dir")&&i+1<argc)output_dir=argv[++i];else if(!strcmp(argv[i],"--samples")&&i+1<argc)samples=positive(argv[++i]);else if(!strcmp(argv[i],"--repetitions")&&i+1<argc)repetitions=positive(argv[++i]);else {fprintf(stderr,"invalid fixture argument\n");return 2;}}
    REQUIRE(samples==5u&&repetitions==32u);host_config();if(host_only)return 0;
    REQUIRE(output_dir&&mkdir(output_dir,0700)==0);unsetenv("MYDS_METAL_CONCURRENT");unsetenv("MYDS_DISPATCH_HIST");unsetenv("MYDS_PROF");unsetenv("MYDS_PHASE_PROFILE");
    if(expect_missing){
        setenv("MYDS_F16_GEMV_BF16_FUSE","1",1);REQUIRE(!myds_metal_init(shader_dir));REQUIRE(!g_device&&!g_library&&!g_queue&&!g_initialized);init_record(0,1,false);
        unsetenv("MYDS_F16_GEMV_BF16_FUSE");REQUIRE(myds_metal_init(shader_dir));REQUIRE(!myds_metal_f16_gemv_bf16_fuse_enabled());
        id<MTLFunction> missing=[g_library newFunctionWithName:@"kernel_mul_mv_f16_f32_4_bf16_nr2"];REQUIRE(!missing);id<MTLFunction> old=[g_library newFunctionWithName:@"kernel_mul_mv_f16_f32_4"];REQUIRE(old);
        init_record(1,-1,true);myds_metal_cleanup();REQUIRE(!g_device&&!g_library&&!g_queue&&!g_initialized);
        setenv("MYDS_F16_GEMV_BF16_FUSE","1",1);REQUIRE(!myds_metal_init(shader_dir));REQUIRE(!g_device&&!g_library&&!g_queue&&!g_initialized);init_record(2,1,false);
        puts("{" SCHEMA ",\"kind\":\"missing_complete\",\"pass\":true,\"candidate_symbol_missing\":true,\"reference_symbol_available\":true,\"explicit_one_failed_before_dispatch\":true,\"default_zero_retry_succeeded\":true,\"native_files\":0}");return 0;
    }
    setenv("MYDS_F16_GEMV_BF16_FUSE","0",1);REQUIRE(myds_metal_init(shader_dir));REQUIRE(!myds_metal_f16_gemv_bf16_fuse_enabled());
    setenv("MYDS_F16_GEMV_BF16_FUSE","1",1);REQUIRE(!myds_metal_f16_gemv_bf16_fuse_enabled());init_record(0,0,true);myds_metal_cleanup();
    REQUIRE(!g_initialized&&!g_device&&!g_library&&!g_queue);REQUIRE(myds_metal_init(shader_dir));REQUIRE(myds_metal_f16_gemv_bf16_fuse_enabled()==1);init_record(1,1,true);
    setenv("MYDS_F16_GEMV_BF16_FUSE","0",1);REQUIRE(myds_metal_f16_gemv_bf16_fuse_enabled()==1);setenv("MYDS_F16_GEMV_BF16_FUSE","1",1);
    printf("{" SCHEMA ",\"kind\":\"identity\",\"actual_mode\":1,\"reference_kernel\":\"kernel_mul_mv_f16_f32_4\",\"candidate_kernel\":\"kernel_mul_mv_f16_f32_4_bf16_nr2\",\"round_kernel\":\"kernel_dsv41_bf16_linear\",\"production_shapes\":[[4096,1280],[32,5120],[512,5120],[128,512]],\"nsg_psos\":8,\"immutable_mode\":true,\"reference_uses_legacy_api\":true,\"fixture_not_fullmodel\":true,\"cold_dram_not_proven\":true,\"ambient_encoder_override_cleared\":true,\"component_env_is_fixed_one\":true}\n");
    unsigned case_id=0;for(unsigned sh=0;sh<15u;++sh)for(unsigned p=0;p<2u;++p)for(unsigned c=0;c<2u;++c)for(unsigned a=0;a<2u;++a)numeric(case_id++,sh,p,c,a);
    for(unsigned i=0;i<4u;++i)reuse(i);for(unsigned i=0;i<16u;++i)rejected(i);completion();
    unsigned groups=0;if(!no_timing)for(unsigned sh=0;sh<4u;++sh)for(unsigned c=0;c<2u;++c)for(unsigned a=0;a<2u;++a)for(unsigned r=0;r<2u;++r)timing(groups++,sh,c,a,r?32u:1u);
    myds_metal_cleanup();REQUIRE(!g_initialized&&!g_device&&!g_library&&!g_queue&&!myds_metal_f16_gemv_bf16_fuse_enabled());unsetenv("MYDS_F16_GEMV_BF16_FUSE");REQUIRE(myds_metal_init(shader_dir));init_record(2,-1,true);myds_metal_cleanup();
    printf("{" SCHEMA ",\"kind\":\"suite_complete\",\"pass\":true,\"parity_cases\":120,\"reuse_cases\":4,\"rejected_calls\":16,\"completion_contracts\":1,\"timing_groups\":%u,\"timing_records\":%u,\"samples\":5,\"repetitions\":32,\"native_files\":%u,\"cleanup_verified\":true}\n",groups,groups*18u,native_files);return 0;
}
