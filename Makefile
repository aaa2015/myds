CC ?= clang
# -Ithird_party: 视觉用的 iris 单头解码器（JPEG/PNG）
CFLAGS ?= -O3 -Wall -Wextra -std=c11 -Iinclude -Ithird_party
OBJCFLAGS ?= -O3 -Wall -Wextra -Iinclude -fobjc-arc
LDFLAGS ?= -framework Foundation -framework Metal -lm -lpthread

BIN_DIR = bin
BUILD_DIR = build

# ★ 视觉相关四个文件别漏：漏了不会报错 —— 只是这些代码【没被编进二进制】，
#   直到有人引用它们才在链接期暴露。我第一版就漏了三个（vispre/visw/visfwd），
#   而当时所有测试仍然全过，因为没人用到它们。
SRCS_C = src/myds_cpu.c src/myds_log.c src/myds_gguf.c src/myds_tokenizer.c \
         src/myds_engine.c src/myds_v41.c src/myds_v41_gpu.c src/myds_v41_moe.c src/myds_v41_decode.c \
         src/myds_engram.c src/myds_dspark.c \
         src/myds_vispre.c src/myds_visw.c src/myds_visfwd.c src/myds_vjob.c \
           src/myds_kvdsk.c \
         src/myds_server.c
SRCS_M = src/myds_metal.m
OBJS_C = $(patsubst src/%.c, $(BUILD_DIR)/%.o, $(SRCS_C))
OBJS_M = $(patsubst src/%.m, $(BUILD_DIR)/%.o, $(SRCS_M))

TARGET = $(BIN_DIR)/myds
TEST_METAL = $(BIN_DIR)/test_metal
TEST_V41_GPU = $(BIN_DIR)/test_v41_gpu
TEST_ATTN_POLICY = $(BIN_DIR)/test_attention_policy
TEST_ATTN_VEC = $(BIN_DIR)/test_attention_vec
TEST_QKV_FUSION = $(BIN_DIR)/test_qkv_fusion
TEST_DECODE_Q8 = $(BIN_DIR)/test_decode_q8
TEST_MOE_DOWN_NSG = $(BIN_DIR)/test_moe_down_nsg
TEST_MOE_IQ2_LUT = $(BIN_DIR)/test_moe_iq2_lut
TEST_ATTN_PACKED = $(BIN_DIR)/test_attention_packed
TEST_TOPK_NAN_FUSE = $(BIN_DIR)/test_topk_nan_fuse
TEST_TOPK_NAN_SHUFFLE = $(BIN_DIR)/test_topk_nan_shuffle
TEST_INDEXER_HEAD_SLOTS = $(BIN_DIR)/test_indexer_head_slots
TEST_TOPK_THRESHOLD = $(BIN_DIR)/test_topk_threshold
TEST_F16_BF16_FUSE = $(BIN_DIR)/test_f16_bf16_fuse
TEST_INDEXER_ROPE_QUANT = $(BIN_DIR)/test_indexer_rope_quant
TEST_BATCH2_Q8_SHARED = $(BIN_DIR)/test_batch2_q8_shared
TEST_LM_HEAD_Q8 = $(BIN_DIR)/test_lm_head_q8
TEST_ENGRAM_PROJECT_ROUND = $(BIN_DIR)/test_engram_project_round
TEST_MOE_Q2_SUM_CACHE = $(BIN_DIR)/test_moe_q2_sum_cache
Q2_SUM_CACHE_ORACLE = $(BIN_DIR)/q2_sum_cache_oracle.metallib
TEST_MOE_Q2_SUM_CACHE_FUSED = $(BIN_DIR)/test_moe_q2_sum_cache_fused
Q2_SUM_CACHE_FUSED_ORACLE = $(BIN_DIR)/q2_sum_cache_fused_oracle.metallib
TEST_DEEPER_SSD = $(BIN_DIR)/test_deeper_ssd
TEST_GGUF = $(BIN_DIR)/test_gguf
TEST_TOKENIZER = $(BIN_DIR)/test_tokenizer
TEST_SERVER = $(BIN_DIR)/test_server
TEST_LOG = $(BIN_DIR)/test_log
TEST_SERVER_API = $(BIN_DIR)/test_server_api
TEST_ENGINE = $(BIN_DIR)/test_engine
TEST_CPU = $(BIN_DIR)/test_cpu
TEST_ENGRAM = $(BIN_DIR)/test_engram
TEST_ALIGN = $(BIN_DIR)/test_align
TEST_PLD = $(BIN_DIR)/test_pld
TEST_KVDSK = $(BIN_DIR)/test_kvdsk
TEST_P1 = $(BIN_DIR)/test_p1
TEST_DSPARK_ROLLBACK = $(BIN_DIR)/test_dspark_rollback
TEST_DSPARK_CAPTURE = $(BIN_DIR)/test_dspark_capture
TEST_BATCH2_PARITY = $(BIN_DIR)/test_batch2_parity
TEST_BATCH2_E2E = $(BIN_DIR)/test_batch2_e2e
TEST_TREE_SPEC = $(BIN_DIR)/test_tree_spec
BENCH_SYNTHETIC = $(BIN_DIR)/bench_synthetic
BENCH_METAL = $(BIN_DIR)/bench_metal
BENCH_PHASE = $(BIN_DIR)/bench_phase
BENCH_ENGRAM_IO = $(BIN_DIR)/bench_engram_io
DUMP_TENSORS = $(BIN_DIR)/dump_tensors
METALLIB = $(BIN_DIR)/myds.metallib
METAL_SRCS = $(wildcard metal/*.metal)

ALL_TESTS = $(TEST_METAL) $(TEST_V41_GPU) $(TEST_ATTN_POLICY) $(TEST_ATTN_VEC) $(TEST_QKV_FUSION) $(TEST_DECODE_Q8) $(TEST_MOE_DOWN_NSG) $(TEST_DEEPER_SSD) $(TEST_GGUF) $(TEST_TOKENIZER) $(TEST_SERVER) $(TEST_LOG) $(TEST_SERVER_API) $(TEST_ENGINE) $(TEST_CPU) $(TEST_ENGRAM) $(TEST_ALIGN) $(TEST_PLD) $(TEST_KVDSK) $(TEST_P1) $(TEST_DSPARK_ROLLBACK) $(TEST_DSPARK_CAPTURE) $(TEST_BATCH2_PARITY) $(TEST_BATCH2_E2E) $(TEST_TREE_SPEC)

.PHONY: all test clean bench metallib build-tail-init test-tail-init

# Rebuild consumers when public or internal layouts change.
-include $(wildcard $(BUILD_DIR)/*.d)
.DEFAULT_GOAL := all

all: $(TARGET) $(ALL_TESTS) $(DUMP_TENSORS) $(METALLIB)

metallib: $(METALLIB)

# Optional real-library init/failure/retry regression. The separate build target
# compiles only; test-tail-init executes GPU work and is scheduled explicitly.
build-tail-init:
	python3 tools/build_tail_init_fixture.py --source "$(CURDIR)" --output "$(abspath $(BUILD_DIR)/tail-init)"

test-tail-init: $(METALLIB) build-tail-init
	"$(abspath $(BUILD_DIR)/tail-init/test_tail_init)" "$(abspath $(BUILD_DIR)/tail-init/missing-tail)" "$(abspath $(BIN_DIR))"

$(METALLIB): $(METAL_SRCS) tools/build_metallib.py | $(BIN_DIR) $(BUILD_DIR)
	python3 tools/build_metallib.py --bin-dir $(BIN_DIR) --build-dir $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: src/%.m | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/main.o: src/main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_metal.o: tests/test_metal.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_v41_gpu.o: tests/test_v41_gpu.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_attention_policy.o: tests/test_attention_policy.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_attention_vec.o: tests/test_attention_vec.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_qkv_fusion.o: tests/test_qkv_fusion.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/bench_phase.o: tools/bench_phase.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_gguf.o: tests/test_gguf.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_tokenizer.o: tests/test_tokenizer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_server.o: tests/test_server.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_log.o: tests/test_log.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_server_api.o: tests/test_server_api.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_engine.o: tests/test_engine.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(TARGET): $(OBJS_C) $(OBJS_M) $(BUILD_DIR)/main.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(TEST_METAL): $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_metal.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(TEST_V41_GPU): $(BUILD_DIR)/myds_v41_gpu.o $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_v41_gpu.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(TEST_ATTN_POLICY): $(BUILD_DIR)/myds_v41_gpu.o $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_attention_policy.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(TEST_ATTN_VEC): $(BUILD_DIR)/myds_v41_gpu.o $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_attention_vec.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(TEST_QKV_FUSION): $(BUILD_DIR)/myds_v41_gpu.o $(BUILD_DIR)/myds_v41_decode.o $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_qkv_fusion.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

# Build-only from all; GPU execution occurs only in the explicit test suite.
# The fixture includes the unchanged runtime to inject a fused-PSO failure.
$(TEST_DECODE_Q8): tests/test_decode_q8.m src/myds_metal.m $(wildcard include/*.h) $(BUILD_DIR)/myds_v41_decode.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc tests/test_decode_q8.m $(BUILD_DIR)/myds_v41_decode.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

# Includes the actual runtime to check both Q2 PSOs and initialization state.
# Building submits no GPU work; the Python driver runs isolated component arms.
$(TEST_MOE_DOWN_NSG): tests/test_moe_down_nsg.m src/myds_metal.m $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc tests/test_moe_down_nsg.m $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

.PHONY: test-moe-down-nsg
test-moe-down-nsg: $(TEST_MOE_DOWN_NSG)
	python3 tests/test_moe_down_nsg.py

# Optional decode candidates: build submits no GPU work. Run their explicit
# fixtures only in a controlled validation window, separately from benchmarks.
$(TEST_MOE_IQ2_LUT): tests/test_moe_iq2_lut.m src/myds_metal.m $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc tests/test_moe_iq2_lut.m $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

$(TEST_ATTN_PACKED): tests/test_attention_packed.m src/myds_metal.m src/myds_v41_gpu.c $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc tests/test_attention_packed.m $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

$(TEST_TOPK_NAN_FUSE): tests/test_topk_nan_fuse.m src/myds_metal.m src/myds_v41_gpu.c $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc tests/test_topk_nan_fuse.m $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

# Build only. The fixture submits GPU work and belongs to an exclusive window.
$(TEST_TOPK_NAN_SHUFFLE): tests/test_topk_nan_shuffle.m src/myds_metal.m src/myds_v41_gpu.c $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc $< $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

.PHONY: build-topk-nan-shuffle-candidate
build-topk-nan-shuffle-candidate: $(TEST_TOPK_NAN_SHUFFLE)

# Explicit BUILD ONLY: this native component submits GPU work when run, and
# must remain outside ALL_TESTS and the ordinary make test recipe.
$(TEST_INDEXER_HEAD_SLOTS): tests/test_indexer_head_slots.m src/myds_metal.m src/myds_v41_gpu.c $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc $< $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

.PHONY: build-indexer-head-slots-candidate
build-indexer-head-slots-candidate: $(TEST_INDEXER_HEAD_SLOTS)

# Explicit build-only threshold fixture; never part of ALL_TESTS.
$(TEST_TOPK_THRESHOLD): tests/test_topk_threshold.m src/myds_metal.m src/myds_v41_gpu.c $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc $< $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

.PHONY: build-topk-threshold-candidate
build-topk-threshold-candidate: $(TEST_TOPK_THRESHOLD)

# Explicit build only; the component submits GPU work in an exclusive window.
$(TEST_F16_BF16_FUSE): tests/test_f16_bf16_fuse.m src/myds_metal.m $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc $< $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

.PHONY: build-f16-bf16-fuse-candidate
build-f16-bf16-fuse-candidate: $(TEST_F16_BF16_FUSE)

# Explicit build only; run the component in an exclusive validation window.
$(TEST_INDEXER_ROPE_QUANT): tests/test_indexer_rope_quant.m src/myds_metal.m src/myds_v41_gpu.c $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc $< $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

.PHONY: build-indexer-rope-quant-candidate
build-indexer-rope-quant-candidate: $(TEST_INDEXER_ROPE_QUANT)

$(TEST_BATCH2_Q8_SHARED): tests/test_batch2_q8_shared.m src/myds_metal.m $(wildcard include/*.h) $(OBJS_C) $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc tests/test_batch2_q8_shared.m $(OBJS_C) $(LDFLAGS) -o $@

.PHONY: build-decode-candidates
build-decode-candidates: $(TEST_MOE_IQ2_LUT) $(TEST_ATTN_PACKED) $(TEST_TOPK_NAN_FUSE) $(TEST_BATCH2_Q8_SHARED)

$(TEST_LM_HEAD_Q8): tests/test_lm_head_q8.m src/myds_metal.m $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc $< $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

.PHONY: build-lm-head-candidate
build-lm-head-candidate: $(TEST_LM_HEAD_Q8)

$(TEST_ENGRAM_PROJECT_ROUND): tests/test_engram_project_round.m src/myds_metal.m $(wildcard include/*.h) $(OBJS_C) $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc $< $(OBJS_C) $(LDFLAGS) -o $@

.PHONY: build-engram-project-round-candidate
build-engram-project-round-candidate: $(TEST_ENGRAM_PROJECT_ROUND)

# Q2 cache component is opt-in and BUILD ONLY: not included in ALL_TESTS.
# The fixture includes the real runtime; link only C helpers, not myds_metal.o.
$(TEST_MOE_Q2_SUM_CACHE): tests/test_moe_q2_sum_cache.m src/myds_metal.m $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc $< $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

# Independent original-order sumy oracle uses the identical production compiler
# flags. Emitting the source is CPU-only; neither build recipe submits GPU work.
$(Q2_SUM_CACHE_ORACLE): $(TEST_MOE_Q2_SUM_CACHE) | $(BUILD_DIR)
	$(TEST_MOE_Q2_SUM_CACHE) --write-oracle-source $(BUILD_DIR)/q2_sum_cache_oracle.metal
	xcrun -sdk macosx metal -O3 -ffast-math -c $(BUILD_DIR)/q2_sum_cache_oracle.metal -o $(BUILD_DIR)/q2_sum_cache_oracle.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/q2_sum_cache_oracle.air -o $@

.PHONY: build-moe-q2-sum-cache-candidate
build-moe-q2-sum-cache-candidate: $(TEST_MOE_Q2_SUM_CACHE) $(Q2_SUM_CACHE_ORACLE)

# Fused IQ2/cache component is explicit BUILD ONLY, never ALL_TESTS.
$(TEST_MOE_Q2_SUM_CACHE_FUSED): tests/test_moe_q2_sum_cache_fused.m src/myds_metal.m $(wildcard include/*.h) $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(OBJCFLAGS) -Isrc $< $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(LDFLAGS) -o $@

$(Q2_SUM_CACHE_FUSED_ORACLE): $(TEST_MOE_Q2_SUM_CACHE_FUSED) | $(BUILD_DIR)
	$(TEST_MOE_Q2_SUM_CACHE_FUSED) --write-oracle-source $(BUILD_DIR)/q2_sum_cache_fused_oracle.metal
	xcrun -sdk macosx metal -O3 -ffast-math -c $(BUILD_DIR)/q2_sum_cache_fused_oracle.metal -o $(BUILD_DIR)/q2_sum_cache_fused_oracle.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/q2_sum_cache_fused_oracle.air -o $@

.PHONY: build-moe-q2-sum-cache-fused-candidate
build-moe-q2-sum-cache-fused-candidate: $(TEST_MOE_Q2_SUM_CACHE_FUSED) $(Q2_SUM_CACHE_FUSED_ORACLE)

$(TEST_DEEPER_SSD): tests/test_deeper_ssd.c tests/test_engine.c tests/test_p1_regressions.c $(OBJS_C) $(OBJS_M) $(METALLIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) tests/test_deeper_ssd.c $(OBJS_C) $(OBJS_M) $(LDFLAGS) -o $@

# Opt-in full-model harness: build only; never run from all/test.
$(BENCH_PHASE): $(OBJS_C) $(OBJS_M) $(BUILD_DIR)/bench_phase.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

# Metadata/tokenizer-only input audit. No Metal framework, model loader or GPU.
$(BIN_DIR)/count_chat_tokens: tools/count_chat_tokens.c src/myds_server.c src/myds_gguf.c src/myds_tokenizer.c src/myds_log.c $(wildcard include/*.h) src/myds_tokenizer_unicode.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(filter %.c,$^) -Wl,-dead_strip -lpthread -lm -o $@

$(TEST_GGUF): $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/test_gguf.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(TEST_TOKENIZER): $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_tokenizer.o $(BUILD_DIR)/test_tokenizer.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(TEST_SERVER): $(BUILD_DIR)/myds_server.o $(BUILD_DIR)/myds_vjob.o $(BUILD_DIR)/myds_vispre.o $(BUILD_DIR)/myds_visw.o $(BUILD_DIR)/myds_visfwd.o $(BUILD_DIR)/myds_kvdsk.o $(BUILD_DIR)/myds_dspark.o $(BUILD_DIR)/myds_engine.o $(BUILD_DIR)/myds_v41.o $(BUILD_DIR)/myds_v41_gpu.o $(BUILD_DIR)/myds_v41_moe.o $(BUILD_DIR)/myds_v41_decode.o $(BUILD_DIR)/myds_engram.o $(BUILD_DIR)/myds_cpu.o $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_tokenizer.o $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_server.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

# 图片块区间对齐：纯头文件（myds_align.h），不需要任何 .c。
$(TEST_ALIGN): tests/test_align.c include/myds_align.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_align.c

# Prompt-Lookup Decoding：纯头文件（myds_pld.h），纳秒级 N-gram 匹配，零外部依赖。
$(TEST_PLD): tests/test_pld.c include/myds_pld.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_pld.c

# 低位区 KV 落盘：只需要存储层与日志，不需要模型（也就不必占着跑生产的机器开第二个实例）。
$(TEST_KVDSK): tests/test_kvdsk.c include/myds_kvdsk.h $(BUILD_DIR)/myds_kvdsk.o $(BUILD_DIR)/myds_log.o | $(BIN_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_kvdsk.c $(BUILD_DIR)/myds_kvdsk.o $(BUILD_DIR)/myds_log.o

$(TEST_P1): tests/test_p1_regressions.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(METALLIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_p1_regressions.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(LDFLAGS)

$(TEST_DSPARK_ROLLBACK): tests/test_dspark_rollback.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(METALLIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_dspark_rollback.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(LDFLAGS)

$(TEST_DSPARK_CAPTURE): tests/test_dspark_capture.c src/myds_dspark.c $(wildcard include/*.h) $(filter-out $(BUILD_DIR)/myds_dspark.o $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ tests/test_dspark_capture.c $(filter-out $(BUILD_DIR)/myds_dspark.o $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(LDFLAGS)

$(TEST_BATCH2_PARITY): tests/test_batch2_parity.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(METALLIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_batch2_parity.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(LDFLAGS)

$(TEST_BATCH2_E2E): tests/test_batch2_e2e.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(METALLIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_batch2_e2e.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(LDFLAGS)

$(TEST_TREE_SPEC): tests/test_tree_spec.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(METALLIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_tree_spec.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(LDFLAGS)

$(TEST_LOG): $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_log.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(TEST_SERVER_API): $(BUILD_DIR)/myds_server.o $(BUILD_DIR)/myds_vjob.o $(BUILD_DIR)/myds_vispre.o $(BUILD_DIR)/myds_visw.o $(BUILD_DIR)/myds_visfwd.o $(BUILD_DIR)/myds_kvdsk.o $(BUILD_DIR)/myds_dspark.o $(BUILD_DIR)/myds_engine.o $(BUILD_DIR)/myds_v41.o $(BUILD_DIR)/myds_v41_gpu.o $(BUILD_DIR)/myds_v41_moe.o $(BUILD_DIR)/myds_v41_decode.o $(BUILD_DIR)/myds_engram.o $(BUILD_DIR)/myds_cpu.o $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_tokenizer.o $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_server_api.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(TEST_ENGINE): $(BUILD_DIR)/myds_kvdsk.o $(BUILD_DIR)/myds_dspark.o $(BUILD_DIR)/myds_engine.o $(BUILD_DIR)/myds_v41.o $(BUILD_DIR)/myds_v41_gpu.o $(BUILD_DIR)/myds_v41_moe.o $(BUILD_DIR)/myds_v41_decode.o $(BUILD_DIR)/myds_engram.o $(BUILD_DIR)/myds_cpu.o $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_tokenizer.o $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_engine.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(BUILD_DIR)/dump_tensors.o: tools/dump_tensors.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(DUMP_TENSORS): $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/dump_tensors.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD_DIR)/trace_model.o: tools/trace_model.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BIN_DIR)/trace_model: $(filter-out $(BUILD_DIR)/myds_server.o,$(OBJS_C)) $(OBJS_M) $(BUILD_DIR)/trace_model.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

# Explicit full-model correctness gate; never part of the model-free test suite.
$(BUILD_DIR)/test_v41_decode_state.o: tools/test_v41_decode_state.c tools/reference/trace_input.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BIN_DIR)/test_v41_decode_state: $(filter-out $(BUILD_DIR)/myds_server.o,$(OBJS_C)) $(OBJS_M) $(BUILD_DIR)/test_v41_decode_state.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

# Explicit real-support Engine lifecycle gate; never run by make test.
$(BUILD_DIR)/test_engine_capture_state.o: tools/test_engine_capture_state.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BIN_DIR)/test_engine_capture_state: $(OBJS_C) $(OBJS_M) $(BUILD_DIR)/test_engine_capture_state.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(BUILD_DIR)/test_engram.o: tests/test_engram.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(TEST_ENGRAM): $(BUILD_DIR)/test_engram.o $(BUILD_DIR)/myds_engram.o $(BUILD_DIR)/myds_gguf.o | $(BIN_DIR)
	$(CC) $^ -lm -lpthread -o $@

# CPU-only real-token Engram I/O harness; does not load GPU/backbone weights.
$(BENCH_ENGRAM_IO): tools/bench_engram_io.c $(BUILD_DIR)/myds_engram.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_tokenizer.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -lm -lpthread -o $@

$(BUILD_DIR)/test_cpu.o: tests/test_cpu.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(TEST_CPU): $(BUILD_DIR)/test_cpu.o $(BUILD_DIR)/myds_cpu.o | $(BIN_DIR)
	$(CC) $^ -lm -o $@

$(BUILD_DIR)/bench_synthetic.o: tools/bench_synthetic.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BENCH_SYNTHETIC): $(BUILD_DIR)/bench_synthetic.o $(OBJS_C) $(OBJS_M) $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(BUILD_DIR)/bench_metal.o: tools/bench_metal.m | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -MMD -MP -c $< -o $@

$(BENCH_METAL): $(BUILD_DIR)/bench_metal.o $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

bench: $(TEST_CPU) $(BENCH_METAL) $(BENCH_SYNTHETIC)
	./$(TEST_CPU) --bench
	./$(BENCH_METAL)
	./$(BENCH_SYNTHETIC)

# 视觉权重加载校验。需要指定真实的视觉 GGUF 路径：
#   make test-visw VIS_GGUF=/path/to/Vision.gguf [VIS_LOAD=1]
# ★ 默认【不加 --load】—— 全量 load 要 1.81 GiB，不该在随便一台机器上就跑起来。
VIS_GGUF ?=
VIS_LOAD ?=

test-visw:
	@if [ -z "$(VIS_GGUF)" ]; then \
		echo "用法: make test-visw VIS_GGUF=/path/to/DeepSeek-V4.1-Flash-Vision.gguf [VIS_LOAD=1]"; \
		exit 2; \
	fi
	$(CC) $(CFLAGS) -Iinclude -o $(BIN_DIR)/test_visw tests/test_visw.c \
		src/myds_visw.c src/myds_gguf.c src/myds_log.c $(BUILD_DIR)/myds_metal.o \
		-framework Foundation -framework Metal -lm
	@$(BIN_DIR)/test_visw "$(VIS_GGUF)" $(if $(VIS_LOAD),--load,)

.PHONY: test-visw

# 视觉预处理/布局 与 ds4 的逐字节对拍（开发期门禁，需要 ds4 源码在侧）
# 不并入 test: —— 它依赖外部仓库路径，缺了会跳过而不是失败。
vis-crosscheck:
	@tools/vis_crosscheck.sh

.PHONY: vis-crosscheck

test: $(ALL_TESTS) $(METALLIB)
	@echo "\n========================================================"
	@echo "               RUNNING ALL UNIT TESTS                   "
	@echo "========================================================"
	@./$(TEST_GGUF)
	@echo ""
	@./$(TEST_TOKENIZER)
	@echo ""
	@./$(TEST_METAL)
	@./$(TEST_V41_GPU)
	@./$(TEST_V41_GPU) --slab
	@./$(TEST_V41_GPU) --threshold
	@./$(TEST_ATTN_POLICY)
	@./$(TEST_ATTN_VEC)
	@MYDS_QKV_NR0=4 ./$(TEST_QKV_FUSION)
	@MYDS_QKV_NR0=2 ./$(TEST_QKV_FUSION)
	@python3 tests/test_moe_down_nsg.py
	@echo ""
	@./$(TEST_SERVER)
	@echo ""
	@./$(TEST_LOG)
	@echo ""
	@./$(TEST_SERVER_API)
	@echo ""
	@./$(TEST_ENGINE)
	@echo ""
	@./$(TEST_CPU)
	@echo ""
	@./$(TEST_ENGRAM)
	@echo ""
	@./$(TEST_ALIGN)
	@echo ""
	@./$(TEST_PLD)
	@echo ""
	@./$(TEST_KVDSK)
	@echo ""
	@./$(TEST_P1)
	@echo ""
	@./$(TEST_DSPARK_ROLLBACK)
	@./$(TEST_DSPARK_CAPTURE)
	@./$(TEST_DSPARK_CAPTURE) --markov-only
	@echo ""
	@./$(TEST_BATCH2_PARITY)
	@echo ""
	@./$(TEST_BATCH2_E2E)
	@echo ""
	@./$(TEST_TREE_SPEC)
	@echo ""
	@MYDS_QKV_NR0=4 ./$(TEST_DECODE_Q8) bin
	@MYDS_QKV_NR0=2 ./$(TEST_DECODE_Q8) bin
	@MYDS_DECODE_Q8_FUSE_BF16=0 ./$(TEST_DECODE_Q8) bin
	@echo ""
	@./$(TEST_DEEPER_SSD)
	@./$(TEST_DEEPER_SSD) --legacy-selection
	@python3 tests/test_decode_q8_contract.py
	@echo ""
	@python3 tests/test_http_bytes.py
	@python3 tests/test_bench_reproduce.py
	@python3 tools/test_engram_pool_failures.py
	@python3 tests/test_engram_io_tools.py
	@python3 tests/test_engram_io_window.py
	@python3 docs/benchmarks/2026-10-05-engram-io/scripts/test-full-model-summary.py
	@python3 tests/test_v41_engram_tool.py
	@python3 tests/test_slim_dense.py
	@echo "\n========================================================"
	@echo "      ALL TEST SUITES PASSED CLEANLY & SUCCESSFULLY!    "
	@echo "========================================================"

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) tests/verify_metal tests/test_metal /tmp/test_myds*
