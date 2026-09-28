#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

static int g_captured_count = 0;
static char g_last_captured_msg[4096] = {0};
static myds_log_level_t g_last_captured_level = MYDS_LOG_INFO;

static void test_log_callback(myds_log_level_t level, const char *msg, void *user_data) {
    (void)user_data;
    g_captured_count++;
    g_last_captured_level = level;
    strncpy(g_last_captured_msg, msg, sizeof(g_last_captured_msg) - 1);
}

#define NUM_THREADS 4
#define LOGS_PER_THREAD 25

static void *threaded_log_worker(void *arg) {
    long id = (long)arg;
    for (int i = 0; i < LOGS_PER_THREAD; i++) {
        MYDS_LOG_I("Thread %ld log message #%d", id, i);
    }
    return NULL;
}

int main(void) {
    printf("====================================================\n");
    printf("          Running myds Logging Unit Tests           \n");
    printf("====================================================\n");

    /* 1. Level string conversion */
    assert(myds_log_level_from_string("DEBUG") == MYDS_LOG_DEBUG);
    assert(myds_log_level_from_string("info") == MYDS_LOG_INFO);
    assert(myds_log_level_from_string("WARN") == MYDS_LOG_WARN);
    assert(myds_log_level_from_string("warning") == MYDS_LOG_WARN);
    assert(myds_log_level_from_string("error") == MYDS_LOG_ERROR);
    assert(myds_log_level_from_string("none") == MYDS_LOG_NONE);
    assert(strcmp(myds_log_level_name(MYDS_LOG_DEBUG), "DEBUG") == 0);
    assert(strcmp(myds_log_level_name(MYDS_LOG_INFO), "INFO") == 0);
    assert(strcmp(myds_log_level_name(MYDS_LOG_WARN), "WARN") == 0);
    assert(strcmp(myds_log_level_name(MYDS_LOG_ERROR), "ERROR") == 0);
    printf("  [PASS] Log level string conversions and names\n");

    /* 2. Custom log callback and message capture */
    myds_log_set_callback(test_log_callback, NULL);
    myds_log_set_color(false);

    myds_log_set_level(MYDS_LOG_INFO);
    g_captured_count = 0;

    MYDS_LOG_D("This debug log should be filtered out");
    assert(g_captured_count == 0);

    MYDS_LOG_I("Hello %s %d", "world", 42);
    assert(g_captured_count == 1);
    assert(g_last_captured_level == MYDS_LOG_INFO);
    assert(strcmp(g_last_captured_msg, "Hello world 42") == 0);
    printf("  [PASS] Log level filtering (INFO filters out DEBUG)\n");

    /* 3. Higher level threshold */
    myds_log_set_level(MYDS_LOG_WARN);
    MYDS_LOG_I("This info log should be filtered out");
    assert(g_captured_count == 1);

    MYDS_LOG_W("Warning alert: disk space low");
    assert(g_captured_count == 2);
    assert(g_last_captured_level == MYDS_LOG_WARN);
    assert(strcmp(g_last_captured_msg, "Warning alert: disk space low") == 0);

    MYDS_LOG_E("Error failure code %d", 500);
    assert(g_captured_count == 3);
    assert(g_last_captured_level == MYDS_LOG_ERROR);
    assert(strcmp(g_last_captured_msg, "Error failure code 500") == 0);
    printf("  [PASS] WARN/ERROR log capture and threshold checking\n");

    /* 4. Multi-threaded logging safety */
    myds_log_set_level(MYDS_LOG_INFO);
    g_captured_count = 0;

    pthread_t threads[NUM_THREADS];
    for (long i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, threaded_log_worker, (void *)i);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    assert(g_captured_count == NUM_THREADS * LOGS_PER_THREAD);
    printf("  [PASS] Multi-threaded concurrent logging safety (%d threads, %d logs)\n",
           NUM_THREADS, NUM_THREADS * LOGS_PER_THREAD);

    /* Clean up callback */
    myds_log_set_callback(NULL, NULL);

    printf("====================================================\n");
    printf("        ALL LOGGING UNIT TESTS PASSED!              \n");
    printf("====================================================\n");
    return 0;
}
