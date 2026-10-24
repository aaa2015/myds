# Isolated TESTING diagnostic; compile-only does not invoke the program.
PROMPT_WS_COMMON_C = $(filter-out $(BUILD_DIR)/myds_v41.o,$(OBJS_C))
PROMPT_WS_COMMON_M = $(filter-out $(BUILD_DIR)/myds_metal.o,$(OBJS_M))
.PHONY: prompt-workspace-tools
prompt-workspace-tools: $(BIN_DIR)/test_prompt_workspace_allocations
$(BUILD_DIR)/myds_v41_prompt_workspace_test.o: src/myds_v41.c include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@
$(BUILD_DIR)/myds_metal_prompt_workspace_test.o: src/myds_metal.m include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@
$(BUILD_DIR)/test_prompt_workspace_allocations.o: tools/test_prompt_workspace_allocations.c include/myds_internal.h include/myds_metal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DMYDS_MOE_WORKSPACE_TESTING=1 -MMD -MP -c $< -o $@
$(BIN_DIR)/test_prompt_workspace_allocations: $(PROMPT_WS_COMMON_C) $(PROMPT_WS_COMMON_M) $(BUILD_DIR)/myds_v41_prompt_workspace_test.o $(BUILD_DIR)/myds_metal_prompt_workspace_test.o $(BUILD_DIR)/test_prompt_workspace_allocations.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
