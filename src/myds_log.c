#include "myds_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

static pthread_mutex_t   g_log_mutex = PTHREAD_MUTEX_INITIALIZER;
static myds_log_level_t  g_log_level = MYDS_LOG_INFO;
static int               g_log_color = -1; /* -1 = auto detect, 0 = off, 1 = on */
static bool              g_initialized = false;
static myds_log_callback_t g_custom_cb = NULL;
static void             *g_custom_cb_data = NULL;

static void init_log_settings(void) {
    if (g_initialized) return;
    g_initialized = true;

    const char *lvl_env = getenv("MYDS_LOG_LEVEL");
    if (lvl_env) {
        g_log_level = myds_log_level_from_string(lvl_env);
    }

    if (g_log_color == -1) {
        if (getenv("NO_COLOR") != NULL) {
            g_log_color = 0;
        } else if (isatty(fileno(stdout))) {
            g_log_color = 1;
        } else {
            g_log_color = 0;
        }
    }
}

void myds_log_set_level(myds_log_level_t level) {
    pthread_mutex_lock(&g_log_mutex);
    g_log_level = level;
    g_initialized = true;
    pthread_mutex_unlock(&g_log_mutex);
}

myds_log_level_t myds_log_get_level(void) {
    if (!g_initialized) init_log_settings();
    return g_log_level;
}

void myds_log_set_color(bool enable) {
    pthread_mutex_lock(&g_log_mutex);
    g_log_color = enable ? 1 : 0;
    pthread_mutex_unlock(&g_log_mutex);
}

void myds_log_set_callback(myds_log_callback_t cb, void *user_data) {
    pthread_mutex_lock(&g_log_mutex);
    g_custom_cb = cb;
    g_custom_cb_data = user_data;
    pthread_mutex_unlock(&g_log_mutex);
}

const char *myds_log_level_name(myds_log_level_t level) {
    switch (level) {
        case MYDS_LOG_DEBUG: return "DEBUG";
        case MYDS_LOG_INFO:  return "INFO";
        case MYDS_LOG_WARN:  return "WARN";
        case MYDS_LOG_ERROR: return "ERROR";
        default:             return "UNKNOWN";
    }
}

myds_log_level_t myds_log_level_from_string(const char *name) {
    if (!name) return MYDS_LOG_INFO;
    if (strcasecmp(name, "DEBUG") == 0) return MYDS_LOG_DEBUG;
    if (strcasecmp(name, "INFO") == 0)  return MYDS_LOG_INFO;
    if (strcasecmp(name, "WARN") == 0 || strcasecmp(name, "WARNING") == 0) return MYDS_LOG_WARN;
    if (strcasecmp(name, "ERROR") == 0) return MYDS_LOG_ERROR;
    if (strcasecmp(name, "NONE") == 0 || strcasecmp(name, "OFF") == 0) return MYDS_LOG_NONE;
    return MYDS_LOG_INFO;
}

void myds_log(myds_log_level_t level, const char *fmt, ...) {
    if (!g_initialized) init_log_settings();
    if (level < g_log_level || level >= MYDS_LOG_NONE) return;

    /* Format timestamp */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm_buf;
    localtime_r(&tv.tv_sec, &tm_buf);

    char time_str[32];
    snprintf(time_str, sizeof(time_str), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
             tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
             tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
             (int)(tv.tv_usec / 1000));

    /* Format message body */
    char msg_buf[4096];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
    va_end(args);

    pthread_mutex_lock(&g_log_mutex);

    /* If custom callback is registered, invoke it */
    if (g_custom_cb) {
        g_custom_cb(level, msg_buf, g_custom_cb_data);
    }

    /* Print to console */
    const char *color_start = "";
    const char *color_end = "";
    if (g_log_color == 1) {
        switch (level) {
            case MYDS_LOG_DEBUG: color_start = "\033[36m"; break; /* Cyan */
            case MYDS_LOG_INFO:  color_start = "\033[32m"; break; /* Green */
            case MYDS_LOG_WARN:  color_start = "\033[33m"; break; /* Yellow */
            case MYDS_LOG_ERROR: color_start = "\033[31m"; break; /* Red */
            default: break;
        }
        color_end = "\033[0m";
    }

    FILE *out = (level >= MYDS_LOG_WARN) ? stderr : stdout;
    fprintf(out, "[%s] [%s%s%s] %s\n",
            time_str, color_start, myds_log_level_name(level), color_end, msg_buf);
    fflush(out);

    pthread_mutex_unlock(&g_log_mutex);
}
