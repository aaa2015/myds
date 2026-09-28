#ifndef MYDS_LOG_H
#define MYDS_LOG_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MYDS_LOG_DEBUG = 0,
    MYDS_LOG_INFO  = 1,
    MYDS_LOG_WARN  = 2,
    MYDS_LOG_ERROR = 3,
    MYDS_LOG_NONE  = 4
} myds_log_level_t;

typedef void (*myds_log_callback_t)(myds_log_level_t level, const char *msg, void *user_data);

/* Configuration */
void              myds_log_set_level(myds_log_level_t level);
myds_log_level_t  myds_log_get_level(void);
void              myds_log_set_color(bool enable);
void              myds_log_set_callback(myds_log_callback_t cb, void *user_data);
const char       *myds_log_level_name(myds_log_level_t level);
myds_log_level_t  myds_log_level_from_string(const char *name);

/* Print functions */
void myds_log(myds_log_level_t level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

#define MYDS_LOG_D(...) myds_log(MYDS_LOG_DEBUG, __VA_ARGS__)
#define MYDS_LOG_I(...) myds_log(MYDS_LOG_INFO,  __VA_ARGS__)
#define MYDS_LOG_W(...) myds_log(MYDS_LOG_WARN,  __VA_ARGS__)
#define MYDS_LOG_E(...) myds_log(MYDS_LOG_ERROR, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* MYDS_LOG_H */
