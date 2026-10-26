本次新增 v2 reader，仅修正原 producer 声明字段 `finite_bf16_required` 的重算类型：冻结 Objective-C harness 的 `@(r!=0)` 将 C 比较表达式的 int 结果输出为 JSON 整数 0/1；旧 reader 将同字段计算为 Python bool，再用严格类型比较拒收了真实声明。

原 reader db555ce6…、其 CPU freeze、旧 `evidence_invalid` 报告 49261e15…、组件协议和 M4 执行窗口全部保留。本次没有 GPU 重跑。v2 的唯一 gate 语义变更为 `finite_bf16_required=int(role != "dummy_q")`，其他新增代码只报告 old/repaired reader SHA 和修复范围；源码逆变换可逐 bytes 恢复旧 reader。compressed、guards、finite_bf16、current slot、completion 等原 BOOL 类型门均保持不变。bool/float 冒充该整数声明仍被拒收。

v2 对同一份 retained M4 raw 的独立重审通过：114 cases、342 completions、1026 complete payloads、2052 guards、3535 retained files，完整 Q/KV/window、before/args/provenance、finite BF16、window 全槽、显式 library/PSO、geometry 和 GPU completion 门全部通过。该结果是 Apple M4 上的 synthetic correctness-only；未评估时间、完整 decode graph、真实模型、运行时接入、端到端性能或生产采用。旧 reader 的类型错误不能报告为 GPU 数值失败。

额外只读审阅提到 `manifest.failure` 与成功 status 自相矛盾时的完整性检查属于后续改进，不在本次单字段授权修复中；本次实际 retained manifest 没有 failure，未改变既有门。
