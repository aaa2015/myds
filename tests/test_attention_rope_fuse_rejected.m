/* CPU-only production opt-in/input rejection contracts. Uses the real getter,
 * but never initializes Metal, creates a command buffer, or submits GPU work.
 * The private diagnostic allowance must be absent in this translation unit. */
#ifdef MYDS_ATTN_ROPE_FUSE_PRIVATE_PROBE
#error "Production contract fixture must not define a private candidate macro"
#endif
#define main attention_packed_unused_main
#include "test_attention_packed.m"
#undef main

static void rejected_cpu_only(void) {
    REQUIRE(!g_device && !g_queue && !g_library && !g_batch_cb);
    REQUIRE(myds_metal_dispatch_count()==0);
}

int main(void) {
    setenv("MYDS_ATTN_PACKED_REGISTER","1",1);
    unsetenv("MYDS_ATTN_PACKED_NSG");unsetenv("MYDS_ATTN_PACKED_REDUCE");
    unsetenv("MYDS_ATTN_ROPE_FUSE");REQUIRE(myds_v41_attention_rope_fuse_enabled()==0);
    setenv("MYDS_ATTN_ROPE_FUSE","0",1);REQUIRE(myds_v41_attention_rope_fuse_enabled()==0);
    setenv("MYDS_ATTN_ROPE_FUSE","1",1);REQUIRE(myds_v41_attention_rope_fuse_enabled()==1);
    setenv("MYDS_ATTN_PACKED_NSG","8",1);setenv("MYDS_ATTN_PACKED_REDUCE","1",1);
    REQUIRE(myds_v41_attention_rope_fuse_enabled()==1);
    unsetenv("MYDS_ATTN_PACKED_NSG");unsetenv("MYDS_ATTN_PACKED_REDUCE");
    const char *invalid[]={"","2","01","true","1 ","-1"};
    for(unsigned i=0;i<sizeof(invalid)/sizeof(*invalid);i++) {
        setenv("MYDS_ATTN_ROPE_FUSE",invalid[i],1);
        REQUIRE(myds_v41_attention_rope_fuse_enabled()==-1);
        REQUIRE(myds_v41_gpu_scratch_create(128)==NULL);
        rejected_cpu_only();
    }
    setenv("MYDS_ATTN_ROPE_FUSE","1",1);
    const char *bad_keys[]={"MYDS_ATTN_PACKED_REGISTER","MYDS_ATTN_PACKED_REGISTER",
        "MYDS_ATTN_PACKED_NSG","MYDS_ATTN_PACKED_NSG","MYDS_ATTN_PACKED_REDUCE","MYDS_ATTN_PACKED_REDUCE"};
    const char *bad_values[]={"0","2","4","08","0","01"};
    for(unsigned i=0;i<sizeof(bad_keys)/sizeof(*bad_keys);i++) {
        setenv(bad_keys[i],bad_values[i],1);
        REQUIRE(myds_v41_attention_rope_fuse_enabled()==-1 && myds_v41_gpu_scratch_create(128)==NULL);
        rejected_cpu_only();
        setenv("MYDS_ATTN_PACKED_REGISTER","1",1);unsetenv("MYDS_ATTN_PACKED_NSG");unsetenv("MYDS_ATTN_PACKED_REDUCE");
    }
    REQUIRE(myds_v41_attention_rope_fuse_enabled()==1);
    myds_v41_attention_rope_fuse_counters(NULL,true);
    /* Fake CPU-owned wrappers deliberately have no MTLBuffer. Range rejection
     * must precede staging, barriers, or writes. Preserve the
     * entire backing storage and scratch structure, including guard bytes. */
    uint8_t storage[6][160],snapshot[sizeof(storage)];memset(storage,0xa5,sizeof(storage));
    memcpy(snapshot,storage,sizeof(storage));
    myds_metal_buffer_t b[6]={0};
    for(unsigned i=0;i<6;i++){b[i].ptr=storage[i]+16;b[i].size=128;}
    myds_v41_gpu_scratch_t scratch={0},scratch_before;
    scratch.kv_f16=&b[4];scratch.mask=&b[5];scratch.pad=&b[4];scratch.partial=&b[5];
    memcpy(&scratch_before,&scratch,sizeof(scratch));
    for(unsigned direct=0;direct<2;direct++)for(unsigned invalid_input=0;invalid_input<4;invalid_input++) {
        bool consumed=true;
        myds_v41_gpu_scratch_t *selected_scratch=invalid_input==1?NULL:&scratch;
        const uint32_t pos=invalid_input==2?1048576u:127u,nraw=invalid_input==3?0u:1u;
        int ok=direct?myds_v41_gpu_attention_direct_rope_fuse(&b[0],&b[1],&b[2],&b[3],nraw,0,
            &b[4],&b[5],1,0,selected_scratch,pos,false,true,false,&consumed):
            myds_v41_gpu_attention_rope_fuse(&b[0],&b[1],&b[2],&b[3],nraw,0,&b[4],0,
            selected_scratch,pos,false,true,false,&consumed);
        REQUIRE(!ok && !consumed && !memcmp(storage,snapshot,sizeof(storage)));
        REQUIRE(!memcmp(&scratch,&scratch_before,sizeof(scratch)));rejected_cpu_only();
    }
    REQUIRE(!myds_v41_gpu_attention_rope_fuse(&b[0],&b[1],&b[2],&b[3],1,0,&b[4],0,
        &scratch,127,false,true,false,NULL));
    REQUIRE(!myds_v41_gpu_attention_direct_rope_fuse(&b[0],&b[1],&b[2],&b[3],1,0,&b[4],&b[5],
        1,0,&scratch,127,false,true,false,NULL));
    myds_metal_decode_candidate_counters_t count;myds_v41_attention_rope_fuse_counters(&count,false);
    REQUIRE(count.requests==10 && count.encoded==0 && count.failures==10);
    REQUIRE(!memcmp(storage,snapshot,sizeof(storage)) && !memcmp(&scratch,&scratch_before,sizeof(scratch)));
    rejected_cpu_only();
    setenv("MYDS_ATTN_ROPE_FUSE","0",1);REQUIRE(myds_v41_attention_rope_fuse_enabled()==0);
    puts("{\"kind\":\"production_rf_contract\",\"pass\":true,\"cpu_only\":true,\"GPU_work_submitted\":false,\"strict_absent_zero\":true,\"legal_one_enabled\":true,\"invalid_flags_and_configs_reject_scratch\":true,\"scoped_calls\":10,\"encoded\":0,\"dispatch_delta\":0,\"complete_poison_bytes_same\":true,\"scratch_bytes_same\":true}");
    return 0;
}
