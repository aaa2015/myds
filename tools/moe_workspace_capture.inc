/* Diagnostic input/output observer, compiled only into the capture v41 object.
 * It observes the existing batch call; it never changes the route or provider.
 * The extra completion points make this capture correctness evidence only. */
#include <CommonCrypto/CommonDigest.h>
#include <math.h>

static struct {
    const char *directory;
    FILE *records;
    uint32_t calls,cases,mm_layers,tail_layers;
    int failed,pending,selected;
    uint32_t layer,start,rows,capacity,batch;
    int gate_type,up_type,down_type,actual_mm;
    float clamp;
    char xsha[65],idsha[65],wsha[65];
} mw_capture;

static void mw_capture_sha(const void *data,size_t bytes,char result[65]) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256_CTX c;CC_SHA256_Init(&c);
    const unsigned char *p=data;
    while(bytes){size_t n=bytes>1048576u?1048576u:bytes;CC_SHA256_Update(&c,p,(CC_LONG)n);p+=n;bytes-=n;}
    CC_SHA256_Final(digest,&c);
    for(unsigned i=0;i<CC_SHA256_DIGEST_LENGTH;i++)snprintf(result+i*2u,3,"%02x",digest[i]);
    result[64]=0;
}

int myds_moe_workspace_capture_begin(const char *directory) {
    if(!directory || directory[0]!='/' || mw_capture.directory || myds_trace_active())return 0;
    char path[4096];int n=snprintf(path,sizeof(path),"%s/capture-records.jsonl",directory);
    if(n<0 || (size_t)n>=sizeof(path))return 0;
    FILE *f=fopen(path,"wx");if(!f)return 0;
    mw_capture.directory=directory;mw_capture.records=f;return 1;
}

static int mw_capture_dump(const char *kind,const myds_metal_buffer_t *buffer,size_t bytes,char sha[65]) {
    if(!buffer || myds_metal_buffer_size(buffer)<bytes)return 0;
    const void *data=myds_metal_buffer_data((myds_metal_buffer_t *)buffer);if(!data)return 0;
    char path[4096];int n=snprintf(path,sizeof(path),"%s/case-%03u-%s.bin",mw_capture.directory,mw_capture.cases,kind);
    if(n<0 || (size_t)n>=sizeof(path))return 0;
    FILE *f=fopen(path,"wx");if(!f)return 0;
    int ok=fwrite(data,1,bytes,f)==bytes;if(fclose(f)!=0)ok=0;
    if(ok)mw_capture_sha(data,bytes,sha);return ok;
}

static int mw_capture_complete_boundary(void) {
    if(!myds_metal_batch_active())return 0;
    if(!myds_metal_get_auto_barrier())myds_metal_barrier();
    return myds_metal_flush_batch() && myds_metal_commit_batch();
}

static int myds_moe_workspace_capture_before(uint32_t layer,uint32_t absolute_start,uint32_t rows,
    uint32_t capacity,uint32_t batch,const myds_metal_buffer_t *x,const myds_metal_buffer_t *ids,
    const myds_metal_buffer_t *weights,int gate_type,int up_type,int down_type,float clamp,int actual_mm) {
    if(!mw_capture.directory)return 1;
    mw_capture.calls++;
    if(mw_capture.pending || mw_capture.failed)return 0;
    mw_capture.selected=layer==0u || layer==19u || layer==39u;
    if(!mw_capture.selected)return 1;
    if(!rows || rows>batch || capacity>batch || batch>65536u || mw_capture.cases>=64u ||
       gate_type!=16 || up_type!=16 || down_type!=10 || !isfinite(clamp) || clamp<=0 ||
       (actual_mm!=0 && actual_mm!=1) || !mw_capture_complete_boundary()) {
        mw_capture.failed=1;return 0;
    }
    mw_capture.layer=layer;mw_capture.start=absolute_start;mw_capture.rows=rows;
    mw_capture.capacity=capacity;mw_capture.batch=batch;mw_capture.gate_type=gate_type;
    mw_capture.up_type=up_type;mw_capture.down_type=down_type;mw_capture.clamp=clamp;mw_capture.actual_mm=actual_mm;
    const size_t xb=(size_t)rows*5120u*4u,rb=(size_t)rows*6u*4u;
    const float *xp=myds_metal_buffer_data((myds_metal_buffer_t *)x);
    const int32_t *ip=myds_metal_buffer_data((myds_metal_buffer_t *)ids);
    const float *wp=myds_metal_buffer_data((myds_metal_buffer_t *)weights);
    int ok=xp && ip && wp && myds_metal_buffer_size(x)>=xb && myds_metal_buffer_size(ids)>=rb && myds_metal_buffer_size(weights)>=rb;
    for(size_t i=0;ok && i<(size_t)rows*5120u;i++)if(!isfinite(xp[i]))ok=0;
    for(uint32_t row=0;ok && row<rows;row++)for(unsigned j=0;ok && j<6;j++) {
        int32_t id=ip[(size_t)row*6u+j];float weight=wp[(size_t)row*6u+j];
        if(id<0 || id>=384 || !isfinite(weight) || weight<0)ok=0;
        for(unsigned k=0;k<j;k++)if(ip[(size_t)row*6u+k]==id)ok=0;
    }
    if(ok)ok=mw_capture_dump("input-f32",x,xb,mw_capture.xsha) &&
        mw_capture_dump("ids-i32",ids,rb,mw_capture.idsha) && mw_capture_dump("routeweights-f32",weights,rb,mw_capture.wsha);
    const int resumed=myds_metal_begin_batch();
    if(!ok || !resumed){mw_capture.failed=1;return 0;}
    mw_capture.pending=1;return 1;
}

static int myds_moe_workspace_capture_after(const myds_metal_buffer_t *output) {
    if(!mw_capture.directory || !mw_capture.selected)return 1;
    if(!mw_capture.pending || mw_capture.failed || !mw_capture_complete_boundary()){mw_capture.failed=1;return 0;}
    const size_t bytes=(size_t)mw_capture.rows*5120u*4u;
    const float *p=output?myds_metal_buffer_data((myds_metal_buffer_t *)output):NULL;
    int ok=p && myds_metal_buffer_size(output)>=bytes;
    for(size_t i=0;ok && i<bytes/4u;i++)if(!isfinite(p[i]))ok=0;
    char digest[65]={0};if(ok)ok=mw_capture_dump("native-output-f32",output,bytes,digest);
    const int resumed=myds_metal_begin_batch();
    if(ok && resumed) {
        const unsigned i=mw_capture.cases;
        ok=fprintf(mw_capture.records,
          "{\"case\":%u,\"layer\":%u,\"absolute_start\":%u,\"rows\":%u,\"capacity\":%u,\"moe_batch_capacity\":%u,"
          "\"dim\":5120,\"expert_dim\":2304,\"routed_experts\":384,\"top_k\":6,"
          "\"gate_type\":%d,\"up_type\":%d,\"down_type\":%d,\"clamp\":%.9g,\"actual_mm\":%s,"
          "\"input_f32\":{\"path\":\"case-%03u-input-f32.bin\",\"bytes\":%zu,\"sha256\":\"%s\"},"
          "\"ids_i32\":{\"path\":\"case-%03u-ids-i32.bin\",\"bytes\":%zu,\"sha256\":\"%s\"},"
          "\"routeweights_f32\":{\"path\":\"case-%03u-routeweights-f32.bin\",\"bytes\":%zu,\"sha256\":\"%s\"},"
          "\"native_output_f32\":{\"path\":\"case-%03u-native-output-f32.bin\",\"bytes\":%zu,\"sha256\":\"%s\"},"
          "\"all_finite\":true,\"top6_ids_valid_unique\":true,\"trace_on\":false,"
          "\"boundary\":\"actual_prefill_routed_moe_batch\",\"observer_commit_wait\":true,"
          "\"observer_batch_restored\":true,\"performance_from_capture\":false}\n",
          i,mw_capture.layer,mw_capture.start,mw_capture.rows,mw_capture.capacity,mw_capture.batch,
          mw_capture.gate_type,mw_capture.up_type,mw_capture.down_type,mw_capture.clamp,mw_capture.actual_mm?"true":"false",
          i,bytes,mw_capture.xsha,i,(size_t)mw_capture.rows*24u,mw_capture.idsha,
          i,(size_t)mw_capture.rows*24u,mw_capture.wsha,i,bytes,digest)>0 && fflush(mw_capture.records)==0;
    }
    if(!ok || !resumed){mw_capture.failed=1;return 0;}
    unsigned bit=1u<<(mw_capture.layer==0u?0u:mw_capture.layer==19u?1u:2u);
    if(mw_capture.actual_mm)mw_capture.mm_layers|=bit;else mw_capture.tail_layers|=bit;
    mw_capture.pending=0;mw_capture.cases++;return 1;
}

int myds_moe_workspace_capture_finish(void) {
    int ok=mw_capture.directory && mw_capture.records && !mw_capture.failed && !mw_capture.pending &&
        (mw_capture.mm_layers|mw_capture.tail_layers)==7u && mw_capture.cases>=3u;
    if(mw_capture.records && fclose(mw_capture.records)!=0)ok=0;mw_capture.records=NULL;
    char path[4096];int n=snprintf(path,sizeof(path),"%s/capture-observer-summary.json",mw_capture.directory?mw_capture.directory:"");
    FILE *f=n>0 && (size_t)n<sizeof(path)?fopen(path,"wx"):NULL;
    if(!f)return 0;
    if(fprintf(f,"{\"schema_version\":1,\"capture_kind\":\"myds_prefill_moe_workspace\",\"status\":\"%s\","
       "\"actual_batch_calls\":%u,\"cases\":%u,\"mm_layers_mask\":%u,\"nonmm_layers_mask\":%u,"
       "\"real_prefill_tokens\":16401,\"performance_from_capture\":false}\n",
       ok?"complete":"failed",mw_capture.calls,mw_capture.cases,mw_capture.mm_layers,mw_capture.tail_layers)<0)ok=0;
    if(fclose(f)!=0)ok=0;return ok;
}
