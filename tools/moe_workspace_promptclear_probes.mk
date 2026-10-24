# Explicit isolated tools only. Usage:
# make -f Makefile -f tools/moe_workspace_promptclear_probes.mk moe-promptclear-tools
# No regular engine object, default target or production binary is replaced.
MOE_PC_COMMON_C = $(filter-out $(BUILD_DIR)/myds_server.o $(BUILD_DIR)/myds_v41.o,$(OBJS_C))
MOE_PC_TEST_OBJECTS = $(MOE_PC_COMMON_C) $(BUILD_DIR)/myds_v41_moe_pc_component_test.o $(BUILD_DIR)/myds_metal_moe_pc_component_test.o

.PHONY: moe-promptclear-tools
moe-promptclear-tools: $(BIN_DIR)/capture_moe_workspace $(BIN_DIR)/bench_moe_workspace_promptclear $(BIN_DIR)/test_moe_workspace_promptclear_allocations $(BIN_DIR)/bench_moe_workspace_failures $(BIN_DIR)/bench_moe_workspace_promptclear_transactions

$(BUILD_DIR)/myds_v41_moe_pc_component_test.o: src/myds_v41.c include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BUILD_DIR)/myds_metal_moe_pc_component_test.o: src/myds_metal.m include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BUILD_DIR)/myds_v41_moe_pc_component_capture.o: src/myds_v41.c tools/moe_workspace_capture.inc include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_CAPTURE=1 -MMD -MP -c $< -o $@

$(BUILD_DIR)/capture_moe_workspace.o: tools/capture_moe_workspace.c tools/reference/trace_input.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BIN_DIR)/capture_moe_workspace: $(MOE_PC_COMMON_C) $(BUILD_DIR)/myds_v41_moe_pc_component_capture.o $(OBJS_M) $(BUILD_DIR)/capture_moe_workspace.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(BUILD_DIR)/bench_moe_workspace_promptclear.o: tools/bench_moe_workspace_promptclear.m tools/moe_workspace_probe_layout_promptclear.inc include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BIN_DIR)/bench_moe_workspace_promptclear: $(MOE_PC_TEST_OBJECTS) $(BUILD_DIR)/bench_moe_workspace_promptclear.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(BUILD_DIR)/test_moe_workspace_promptclear_allocations.o: tools/test_moe_workspace_promptclear_allocations.c tools/test_moe_workspace_allocations.c tools/moe_workspace_probe_layout_promptclear.inc include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BIN_DIR)/test_moe_workspace_promptclear_allocations: $(MOE_PC_TEST_OBJECTS) $(BUILD_DIR)/test_moe_workspace_promptclear_allocations.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@

$(BUILD_DIR)/bench_moe_workspace_failures.o: tools/bench_moe_workspace_failures.m tools/bench_moe_workspace.m include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BIN_DIR)/bench_moe_workspace_failures: $(MOE_PC_TEST_OBJECTS) $(BUILD_DIR)/bench_moe_workspace_failures.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@


$(BUILD_DIR)/bench_moe_workspace_promptclear_transactions.o: tools/bench_moe_workspace_promptclear_transactions.m tools/bench_moe_workspace_promptclear.m tools/moe_workspace_probe_layout_promptclear.inc include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@

$(BIN_DIR)/bench_moe_workspace_promptclear_transactions: $(MOE_PC_TEST_OBJECTS) $(BUILD_DIR)/bench_moe_workspace_promptclear_transactions.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
