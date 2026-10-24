# Explicit normal-API sequence harness; no diagnostic/fault core objects.
include Makefile
ifeq ($(LOCALGROW_API),1)
LOCALGROW_API_FLAG := -DMYDS_BENCH_MOE_WS_API=1
endif
$(BUILD_DIR)/moe_workspace_fullmodel_localgrow_diag.c: src/myds_v41.c tools/moe_workspace_fullmodel_getters.inc | $(BUILD_DIR)
	cat src/myds_v41.c tools/moe_workspace_fullmodel_getters.inc > $@
$(BUILD_DIR)/moe_workspace_fullmodel_localgrow_diag.o: $(BUILD_DIR)/moe_workspace_fullmodel_localgrow_diag.c
	$(CC) $(CFLAGS) -Isrc $(LOCALGROW_API_FLAG) -c $< -o $@
$(BUILD_DIR)/bench_moe_workspace_fullmodel_localgrow.o: tools/bench_moe_workspace_fullmodel_localgrow.c tools/bench_moe_workspace_fullmodel.c tools/moe_workspace_memory.inc tools/moe_workspace_sequence_state_localgrow.inc tools/moe_workspace_layout_localgrow.inc | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Isrc $(LOCALGROW_API_FLAG) -c $< -o $@
$(BIN_DIR)/bench_moe_workspace_fullmodel_localgrow: $(filter-out $(BUILD_DIR)/main.o $(BUILD_DIR)/myds_v41.o,$(OBJS_C)) $(OBJS_M) $(BUILD_DIR)/moe_workspace_fullmodel_localgrow_diag.o $(BUILD_DIR)/bench_moe_workspace_fullmodel_localgrow.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
