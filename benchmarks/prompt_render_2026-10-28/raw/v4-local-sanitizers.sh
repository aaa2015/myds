#!/bin/sh
# Reproduce the targeted macOS ASan/UBSan checks from the repository root.
# Server and server API test are instrumented; linked engine/Metal build/*.o
# come from make bin/test_server_api and are not sanitizer-instrumented.
set -eu
work=$(mktemp -d /tmp/myds-prompt-sanitize.XXXXXX)
flags='-O1 -g -Wall -Wextra -std=c11 -Iinclude -Ithird_party -mmacosx-version-min=26.0 -fsanitize=address,undefined -fno-omit-frame-pointer'
clang $flags -c src/myds_server.c -o "$work/server.o"
clang $flags -c tests/test_server_api.c -o "$work/test.o"
clang -fsanitize=address,undefined -mmacosx-version-min=26.0 "$work/server.o" \
  build/myds_vjob.o build/myds_vispre.o build/myds_visw.o build/myds_visfwd.o \
  build/myds_kvdsk.o build/myds_dspark.o build/myds_engine.o build/myds_v41.o \
  build/myds_v41_gpu.o build/myds_v41_moe.o build/myds_v41_decode.o \
  build/myds_engram.o build/myds_engram_cpu.o build/myds_cpu.o build/myds_metal.o \
  build/myds_gguf.o build/myds_tokenizer.o build/myds_log.o "$work/test.o" \
  -framework Foundation -framework Metal -lm -lpthread -o "$work/server-test"
"$work/server-test"
clang $flags tests/test_prompt_render_bounds.c src/myds_log.c \
  -Wl,-dead_strip -lpthread -lm -o "$work/bounds-test"
"$work/bounds-test"
