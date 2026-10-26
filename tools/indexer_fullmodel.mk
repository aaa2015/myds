# Explicit CPU build/link only. No production target or shader build is invoked.
# Example: make -f tools/indexer_fullmodel.mk BUILD_DIR=/tmp/idx/build \
#   BIN_DIR=/tmp/idx/bin /tmp/idx/bin/bench_indexer_fullmodel
include Makefile
.DEFAULT_GOAL := $(BIN_DIR)/bench_indexer_fullmodel

INDEXER_BASE_INCLUDE := $(abspath $(BUILD_DIR)/bench_indexer_fullmodel_base.inc)
INDEXER_ADAPTER_FLAG :=
ifneq ($(INDEXER_ADAPTER_HEADER),)
INDEXER_ADAPTER_FLAG := -DMYDS_BENCH_INDEXER_ADAPTER_HEADER='"$(abspath $(INDEXER_ADAPTER_HEADER))"'
endif

# Adapter selection changes compile-time behavior. Recompile this small wrapper
# on every explicit build so reusing an object cannot silently drop its API.
.PHONY: indexer-fullmodel-driver-force
indexer-fullmodel-driver-force:

$(INDEXER_BASE_INCLUDE): tools/bench_output_a_fullmodel.c tools/prepare_indexer_fullmodel.py tools/indexer_fullmodel_state.inc | $(BUILD_DIR)
	python3 tools/prepare_indexer_fullmodel.py --base tools/bench_output_a_fullmodel.c --out "$@"

$(BUILD_DIR)/bench_indexer_fullmodel.o: tools/bench_indexer_fullmodel.c $(INDEXER_BASE_INCLUDE) $(wildcard include/*.h) $(INDEXER_ADAPTER_HEADER) indexer-fullmodel-driver-force | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I"$(abspath tools)" -DMYDS_BENCH_INDEXER_BASE_INCLUDE='"$(INDEXER_BASE_INCLUDE)"' $(INDEXER_ADAPTER_FLAG) -c $< -o $@

$(BIN_DIR)/bench_indexer_fullmodel: $(filter-out $(BUILD_DIR)/main.o,$(OBJS_C)) $(OBJS_M) $(BUILD_DIR)/bench_indexer_fullmodel.o | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
