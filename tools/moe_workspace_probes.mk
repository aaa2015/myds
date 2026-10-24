# Explicit isolated tools only. Usage:
# make -f Makefile -f tools/moe_workspace_probes.mk moe-workspace-tools
# No regular engine object, default target or production binary is replaced.
MOE_WS_COMMON_C = $(filter-out $(BUILD_DIR)/myds_server.o $(BUILD_DIR)/myds_v41.o,$(OBJS_C))
MOE_WS_TEST_OBJECTS = $(MOE_WS_COMMON_C) $(BUILD_DIR)/myds_v41_moe_workspace_test.o $(BUILD_DIR)/myds_metal_moe_workspace_test.o

.PHONY: moe-workspace-tools
moe-workspace-tools: $(BIN_DIR)/capture_moe_workspace $(BIN_DIR)/bench_moe_workspace $(BIN_DIR)/test_moe_workspace_allocations $(BIN_DIR)/bench_moe_workspace_failures

$(BUILD_DIR)/myds_v41_moe_workspace_test.o: src/myds_v41.c include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BUILD_DIR)/myds_metal_moe_workspace_test.o: src/myds_metal.m include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BUILD_DIR)/myds_v41_moe_workspace_capture.o: src/myds_v41.c tools/moe_workspace_capture.inc include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_CAPTURE=1 -MMD -MP -c $< -o $@

$(BUILD_DIR)/capture_moe_workspace.o: tools/capture_moe_workspace.c tools/reference/trace_input.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BIN_DIR)/capture_moe_workspace: $(MOE_WS_COMMON_C) $(BUILD_DIR)/myds_v41_moe_workspace_capture.o $(OBJS_M) $(BUILD_DIR)/capture_moe_workspace.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(BUILD_DIR)/bench_moe_workspace.o: tools/bench_moe_workspace.m include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BIN_DIR)/bench_moe_workspace: $(MOE_WS_TEST_OBJECTS) $(BUILD_DIR)/bench_moe_workspace.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(BUILD_DIR)/test_moe_workspace_allocations.o: tools/test_moe_workspace_allocations.c include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BIN_DIR)/test_moe_workspace_allocations: $(MOE_WS_TEST_OBJECTS) $(BUILD_DIR)/test_moe_workspace_allocations.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(BUILD_DIR)/bench_moe_workspace_failures.o: tools/bench_moe_workspace_failures.m tools/bench_moe_workspace.m include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BIN_DIR)/bench_moe_workspace_failures: $(MOE_WS_TEST_OBJECTS) $(BUILD_DIR)/bench_moe_workspace_failures.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

# Future explicit recovery tools. Compilation does not invoke either program.
.PHONY: moe-workspace-recovery-tools
moe-workspace-recovery-tools: $(BIN_DIR)/test_moe_workspace_cancel $(BIN_DIR)/test_moe_workspace_recovery

$(BUILD_DIR)/test_moe_workspace_cancel.o: tools/test_moe_workspace_cancel.c tools/bench_moe_workspace_fullmodel.c tools/moe_workspace_memory.inc include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -DMYDS_BENCH_MOE_WS_API=1 -MMD -MP -c $< -o $@

$(BIN_DIR)/test_moe_workspace_cancel: $(MOE_WS_TEST_OBJECTS) $(BUILD_DIR)/myds_server.o $(BUILD_DIR)/test_moe_workspace_cancel.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(BUILD_DIR)/test_moe_workspace_recovery.o: tools/test_moe_workspace_recovery.c tools/test_moe_workspace_cancel.c tools/bench_moe_workspace_fullmodel.c tools/moe_workspace_memory.inc include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -DMYDS_BENCH_MOE_WS_API=1 -MMD -MP -c $< -o $@

$(BIN_DIR)/test_moe_workspace_recovery: $(MOE_WS_TEST_OBJECTS) $(BUILD_DIR)/myds_server.o $(BUILD_DIR)/test_moe_workspace_recovery.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
