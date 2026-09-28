CC ?= clang
CFLAGS ?= -O3 -Wall -Wextra -std=c11 -Iinclude
OBJCFLAGS ?= -O3 -Wall -Wextra -Iinclude -fobjc-arc
LDFLAGS ?= -framework Foundation -framework Metal -lm -lpthread

BIN_DIR = bin
BUILD_DIR = build

SRCS_C = src/myds_log.c src/myds_gguf.c src/myds_tokenizer.c src/myds_engine.c src/myds_server.c
SRCS_M = src/myds_metal.m
OBJS_C = $(patsubst src/%.c, $(BUILD_DIR)/%.o, $(SRCS_C))
OBJS_M = $(patsubst src/%.m, $(BUILD_DIR)/%.o, $(SRCS_M))

TARGET = $(BIN_DIR)/myds
TEST_METAL = $(BIN_DIR)/test_metal
TEST_GGUF = $(BIN_DIR)/test_gguf
TEST_TOKENIZER = $(BIN_DIR)/test_tokenizer
TEST_SERVER = $(BIN_DIR)/test_server
TEST_LOG = $(BIN_DIR)/test_log
TEST_SERVER_API = $(BIN_DIR)/test_server_api
TEST_ENGINE = $(BIN_DIR)/test_engine
DUMP_TENSORS = $(BIN_DIR)/dump_tensors

ALL_TESTS = $(TEST_METAL) $(TEST_GGUF) $(TEST_TOKENIZER) $(TEST_SERVER) $(TEST_LOG) $(TEST_SERVER_API) $(TEST_ENGINE)

.PHONY: all test clean

all: $(TARGET) $(ALL_TESTS) $(DUMP_TENSORS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.m | $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: src/main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/test_metal.o: tests/test_metal.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/test_gguf.o: tests/test_gguf.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/test_tokenizer.o: tests/test_tokenizer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/test_server.o: tests/test_server.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/test_log.o: tests/test_log.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/test_server_api.o: tests/test_server_api.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/test_engine.o: tests/test_engine.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS_C) $(OBJS_M) $(BUILD_DIR)/main.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(TEST_METAL): $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_metal.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(TEST_GGUF): $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/test_gguf.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(TEST_TOKENIZER): $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_tokenizer.o $(BUILD_DIR)/test_tokenizer.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(TEST_SERVER): $(BUILD_DIR)/test_server.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(TEST_LOG): $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_log.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(TEST_SERVER_API): $(BUILD_DIR)/test_server_api.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(TEST_ENGINE): $(BUILD_DIR)/myds_engine.o $(BUILD_DIR)/myds_metal.o $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/myds_tokenizer.o $(BUILD_DIR)/myds_log.o $(BUILD_DIR)/test_engine.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD_DIR)/dump_tensors.o: tools/dump_tensors.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(DUMP_TENSORS): $(BUILD_DIR)/myds_gguf.o $(BUILD_DIR)/dump_tensors.o | $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@

test: $(ALL_TESTS)
	@echo "\n========================================================"
	@echo "               RUNNING ALL UNIT TESTS                   "
	@echo "========================================================"
	@./$(TEST_GGUF)
	@echo ""
	@./$(TEST_TOKENIZER)
	@echo ""
	@./$(TEST_METAL)
	@echo ""
	@./$(TEST_SERVER)
	@echo ""
	@./$(TEST_LOG)
	@echo ""
	@./$(TEST_SERVER_API)
	@echo ""
	@./$(TEST_ENGINE)
	@echo "\n========================================================"
	@echo "      ALL TEST SUITES PASSED CLEANLY & SUCCESSFULLY!    "
	@echo "========================================================"

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) tests/verify_metal tests/test_metal /tmp/test_myds*
