/* Directly exercise the two internal renderers without a model or GPU.
 * A protected page makes scanning beyond the exhausted output capacity fail
 * immediately, without a machine-dependent wall-clock threshold. */
#define _DARWIN_C_SOURCE
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../src/myds_server.c"

int main(void) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    char *input = mmap(NULL, page * 2u, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(input != MAP_FAILED);
    memset(input, 'a', page);
    input[page] = 0;
    assert(mprotect(input + page, page, PROT_NONE) == 0);

    unsigned char guarded[66];
    for (int json = 0; json < 2; ++json) {
        memset(guarded, 0xa5, sizeof(guarded));
        prompt_buffer_t out = {(char *)guarded + 1, 64, 0, false};
        const bool ok = json ? prompt_dump_json_string(&out, input)
                             : prompt_tool_text(&out, input);
        assert(!ok && out.overflow && out.length == 63);
        assert(guarded[0] == 0xa5 && guarded[65] == 0xa5 && guarded[64] == 0);
    }
    assert(munmap(input, page * 2u) == 0);
    puts("PASS: exhausted prompt buffers do not scan into a protected input page");
    return 0;
}
