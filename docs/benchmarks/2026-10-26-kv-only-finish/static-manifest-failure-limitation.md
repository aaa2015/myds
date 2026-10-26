只读静态提醒，不属于本次 int 声明类型修复，也不改变本次 M4 结果。

v2 reader 的 joint gate 根据独立 raw 数值/guards/window/finiteBF16 和342个GPU completions计算，并要求 manifest 的 all_complete_outputs_and_guards_pass/status 与 joint 一致；manifest.failure 当前仅保留到报告，没有额外作为拒收条件。因此，如果人为构造“failure 非 null，同时 status 为成功且 allpass=true，其他所有 raw/completion 完整通过”的自相矛盾 manifest，现有 reader 不会单独因 failure 字段拒收。这是静态完整性限制，不是已复现的 GPU 数值问题。

冻结 harness 的 need(false, reason) 会同时设置 failure、finish("failed") 并 exit(1)，本次真实 M4 manifest 没有 failure，wrapper exit0，harness status/allpass 与独立 raw 门均通过。因此该静态限制不影响 retained M4 的 v2 正确性结论。

后续另行冻结的目标 M2 wrapper 可明确联合要求 failure absent/null、status=synthetic_correctness_complete_not_adoption、allpass=true、exit0，再由同一 reader 独立审 raw 数值/guards/completions。此建议不回写旧窗口、旧协议、旧 reader 或当前已冻结 7f42684e… v2。

源码位置：/tmp/myds-kv-only-finish-reader-v2-20261026/review_kv_only_finish_v2.py 的 joint/status 检查和 harness_failure 报告；冻结 harness /tmp/myds-kv-only-finish-probe-final-v1-20261025/source/tools/bench_kv_only_finish_probe.m 的 need 函数。
