# Separate normal-API reuse diagnostic. Standard binary and core objects stay untouched.
include Makefile
$(BUILD_DIR)/moe_workspace_reuse_diag.c: src/myds_v41.c tools/moe_workspace_fullmodel_getters.inc | $(BUILD_DIR)
	cat src/myds_v41.c tools/moe_workspace_fullmodel_getters.inc > $@
$(BUILD_DIR)/moe_workspace_reuse_diag.o: $(BUILD_DIR)/moe_workspace_reuse_diag.c
	$(CC) $(CFLAGS) -Isrc -DMYDS_BENCH_MOE_WS_API=1 -c $< -o $@
$(BUILD_DIR)/bench_moe_workspace_reuse.o: tools/bench_moe_workspace_reuse.c tools/bench_moe_workspace_fullmodel.c tools/moe_workspace_memory.inc | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Isrc -DMYDS_BENCH_MOE_WS_API=1 -c $< -o $@
$(BIN_DIR)/bench_moe_workspace_reuse: $(filter-out $(BUILD_DIR)/main.o $(BUILD_DIR)/myds_v41.o,$(OBJS_C)) $(OBJS_M) $(BUILD_DIR)/moe_workspace_reuse_diag.o $(BUILD_DIR)/bench_moe_workspace_reuse.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
