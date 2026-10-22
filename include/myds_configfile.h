/* myds_configfile.h — 启动参数的单一来源
 *
 * 为什么需要它：生产启动参数分散在三处（fleet.conf 的列、start_fleet.sh 的
 * SERVICE_ENV、代码里的 getenv 默认值），实际造成过这些问题：
 *   · 180 个 MYDS_ 变量里，start_fleet.sh 只设了 55 个，132 个走代码默认值，
 *     而没人知道那些默认值是什么
 *   · start_fleet.sh 在 macmini 与 macstudio 各一份，改一边忘另一边
 *   · 参数被多处钳制（如 moe_chunks 被 MOE_CHUNKS / guard/chunk / macroblock/chunk
 *     三方取 min），配置写的 9 实际是 4，且没有任何日志暴露
 *
 * 做法：把配置文件读成环境变量，放在 main() 最早期。这样 180 处 getenv 一行不改
 * 就能吃上配置；同时保留环境变量与命令行的覆盖能力（优先级见下）。
 *
 * 优先级（高 → 低）：
 *   1. 命令行显式参数（--ctx-size 等）
 *   2. 真实环境变量（例如一次性诊断 `MYDS_PROFILE_PREFILL=1 ./myds ...`）
 *   3. 配置文件
 *   4. 代码内 getenv 默认值
 *
 * 即：配置文件**不覆盖**已存在的环境变量，只填补空缺。
 * 这样既有单一来源，又保住了「临时覆盖一次」的能力。
 */
#ifndef MYDS_CONFIGFILE_H
#define MYDS_CONFIGFILE_H

#include <stdbool.h>
#include <stddef.h>

/* 从 path 加载配置。规则：
 *   - 每行 `KEY=VALUE`；`#` 开头或空行忽略；行尾 `#` 注释需前置空格（` # `）
 *   - 只接受 KEY 以 MYDS_ 开头（避免误设 PATH 等）
 *   - 默认不覆盖已存在的环境变量（override_env=false）
 *   - 值为空字符串时表示「显式置空」，会覆盖
 * 返回：成功设置的条数；<0 表示打不开。
 * 副作用：会打印一行汇总日志（条数 / 覆盖了几条已有变量 / 被跳过的非法行数）。 */
int myds_configfile_load(const char *path, bool override_env);

/* 把当前进程环境里所有 MYDS_* 变量按 KEY=VALUE 写到 path。
 * 用于「从正在跑的生产进程导出权威配置」——这正是解决参数漂移的手段。
 * 返回写入条数；<0 失败。 */
int myds_configfile_dump_env(const char *path);

/* 比对：配置文件中、与实际环境变量不同的项。
 * 只报告，不修改。返回差异条数；<0 表示打不开。 */
int myds_configfile_diff(const char *path);

/* 配置文件里 KEY 的个数（不含注释）。<0 = 打不开。 */
int myds_configfile_count(const char *path);

#endif /* MYDS_CONFIGFILE_H */
