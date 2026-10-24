# Explicit new localgrow correctness diagnostics; never a default/production tool.
include Makefile
include tools/moe_workspace_probes.mk
.PHONY: moe-workspace-recovery-promptclear-tools
moe-workspace-recovery-promptclear-tools: $(BIN_DIR)/test_moe_workspace_cancel_promptclear $(BIN_DIR)/test_moe_workspace_recovery_promptclear
$(BUILD_DIR)/test_moe_workspace_cancel_promptclear.o: tools/test_moe_workspace_cancel_promptclear.c tools/bench_moe_workspace_fullmodel.c tools/moe_workspace_layout_promptclear.inc tools/moe_workspace_memory.inc tools/moe_prompt_snapshot_promptclear.inc include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -DMYDS_BENCH_MOE_WS_API=1 -MMD -MP -c $< -o $@
$(BIN_DIR)/test_moe_workspace_cancel_promptclear: $(MOE_WS_TEST_OBJECTS) $(BUILD_DIR)/myds_server.o $(BUILD_DIR)/test_moe_workspace_cancel_promptclear.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
$(BUILD_DIR)/test_moe_workspace_recovery_promptclear.o: tools/test_moe_workspace_recovery_promptclear.c tools/test_moe_workspace_cancel_promptclear.c tools/bench_moe_workspace_fullmodel.c tools/moe_workspace_layout_promptclear.inc tools/moe_workspace_memory.inc tools/moe_prompt_snapshot_promptclear.inc include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -DMYDS_BENCH_MOE_WS_API=1 -MMD -MP -c $< -o $@
$(BIN_DIR)/test_moe_workspace_recovery_promptclear: $(MOE_WS_TEST_OBJECTS) $(BUILD_DIR)/myds_server.o $(BUILD_DIR)/test_moe_workspace_recovery_promptclear.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
