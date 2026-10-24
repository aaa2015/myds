# Explicit isolated fullmodel harness; standard binary/core objects unchanged.
include Makefile
ifeq ($(FULLMODEL_MOE_WS_API),1)
MOE_WS_API_FLAG := -DMYDS_BENCH_MOE_WS_API=1
endif
$(BUILD_DIR)/moe_workspace_fullmodel_diag.c: src/myds_v41.c tools/moe_workspace_fullmodel_getters.inc | $(BUILD_DIR)
	cat src/myds_v41.c tools/moe_workspace_fullmodel_getters.inc > $@
$(BUILD_DIR)/moe_workspace_fullmodel_diag.o: $(BUILD_DIR)/moe_workspace_fullmodel_diag.c
	$(CC) $(CFLAGS) -Isrc $(MOE_WS_API_FLAG) -c $< -o $@
$(BUILD_DIR)/bench_moe_workspace_fullmodel.o: tools/bench_moe_workspace_fullmodel.c tools/moe_workspace_memory.inc | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Isrc $(MOE_WS_API_FLAG) -c $< -o $@
$(BIN_DIR)/bench_moe_workspace_fullmodel: $(filter-out $(BUILD_DIR)/main.o $(BUILD_DIR)/myds_v41.o,$(OBJS_C)) $(OBJS_M) $(BUILD_DIR)/moe_workspace_fullmodel_diag.o $(BUILD_DIR)/bench_moe_workspace_fullmodel.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
