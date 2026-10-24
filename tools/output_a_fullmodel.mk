# Isolated fullmodel harness; explicit build-only target. No production target changes.
include Makefile
ifeq ($(FULLMODEL_OUTA_API),1)
FULLMODEL_API_FLAG := -DMYDS_BENCH_OUTA_FIXED_API=1
endif
$(BUILD_DIR)/bench_output_a_fullmodel.o: tools/bench_output_a_fullmodel.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FULLMODEL_API_FLAG) -c $< -o $@
$(BIN_DIR)/bench_output_a_fullmodel: $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(BUILD_DIR)/bench_output_a_fullmodel.o $(METALLIB) | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
