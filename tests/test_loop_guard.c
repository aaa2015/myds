#include "myds_loop_guard.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    myds_loop_guard_t lg;
    myds_loop_guard_init(&lg, true);

    // Test 1: Non-repeating sequence
    for (int32_t i = 1; i <= 50; ++i) {
        bool detected = myds_loop_guard_feed(&lg, i);
        assert(!detected);
    }
    printf("[PASS] Non-repeating sequence\n");

    // Test 2: Repeating 4-token sequence 4 times
    // Period = 4, tokens = [100, 101, 102, 103]
    myds_loop_guard_init(&lg, true);
    int32_t seq4[] = {100, 101, 102, 103};
    bool detected4 = false;
    for (int rep = 0; rep < 4; ++rep) {
        for (int i = 0; i < 4; ++i) {
            detected4 = myds_loop_guard_feed(&lg, seq4[i]);
            if (rep < 3) {
                assert(!detected4);
            }
        }
    }
    assert(detected4);
    assert(lg.detected_period == 4);
    assert(lg.detected_reps == 4);
    printf("[PASS] Period 4 loop detected (period=%u, reps=%u)\n", lg.detected_period, lg.detected_reps);

    // Test 3: Repeating 20-token sequence 3 times
    myds_loop_guard_init(&lg, true);
    int32_t seq20[20];
    for (int i = 0; i < 20; ++i) seq20[i] = 200 + i;
    bool detected20 = false;
    for (int rep = 0; rep < 3; ++rep) {
        for (int i = 0; i < 20; ++i) {
            detected20 = myds_loop_guard_feed(&lg, seq20[i]);
            if (rep < 2) {
                assert(!detected20);
            }
        }
    }
    assert(detected20);
    assert(lg.detected_period == 20);
    assert(lg.detected_reps == 3);
    printf("[PASS] Period 20 loop detected (period=%u, reps=%u)\n", lg.detected_period, lg.detected_reps);

    // Test 4: Disabled flag
    myds_loop_guard_init(&lg, false);
    for (int rep = 0; rep < 10; ++rep) {
        for (int i = 0; i < 4; ++i) {
            assert(!myds_loop_guard_feed(&lg, seq4[i]));
        }
    }
    printf("[PASS] Loop guard disabled flag works\n");

    printf("\nALL LOOP GUARD TESTS PASSED!\n");
    return 0;
}

