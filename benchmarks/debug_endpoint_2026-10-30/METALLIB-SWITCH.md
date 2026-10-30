# 生产切到可复现的 metallib（账本 §132）

## 结果

| 项 | 切换前 | 切换后 |
|---|---|---|
| metallib | `fe557933`（旧工具链，**已无人能复现**） | **`719242c9`（两机逐字节复现）** |
| binary | `3c9c858b` | `3c9c858b`（未变） |
| config | `735e0dc7` | `735e0dc7`（未变） |
| PID / nice | — | 44285 / 5 |
| verify | — | **PASS** |

## 执行前拦下两件事

**① 工作树里有人正在写新功能。** 查 `git status` 发现 `myds_steer` 一套未提交文件
（3 分钟内 mtime），且 `build_metallib.py` 已把它加进 `SHADER_FILES`。

⇒ 我本地那份 `bin/myds.metallib` 是 `787f9d7d`，**含对方的 steer shader，不是目标产物**。
直接部署会把别人未完成的功能带上生产。
**改用 `git archive HEAD` 的干净树重建**，得 `719242c9`，并确认树内无 steer。

**② 部署侧工作树长期不是干净检出。** macstudio 停在 `5437df0`（旧提交），
8,572 删除 / 65 修改 / 336 未跟踪。`sync_myds.sh` 只 rsync 源码、不同步 git 历史。
**在那种树上断言"工作树必须干净"毫无意义。**

## 因此简化：不需要重新构建

两边 `bin/myds` 已相同 ⇒ 只需换一个文件：

1. 干净树构建 metallib → `719242c9`
2. 拷到 macstudio
3. **单文件原子替换**（旧进程持有旧 inode，不受影响）
4. 备份到 `.deploy-prev/metallib/`
5. 停服 → 等内存回收 → `start_fleet.sh` + `renice 5`
6. **发哑请求并丢弃**
7. 验证

## 预热的实测代价

| 请求 | 耗时 |
|---|---|
| 第 1 次（JIT 编译路径） | **0.955 s** |
| 第 2 次（缓存路径，稳态） | **0.202 s** |

**JIT 慢约 4.7 倍。** 这不只是数值不同——`§128` 的结论有了工程侧的第二个理由。

## 回滚

```bash
ssh macstudio 'cd ~/Projects/myds && cp .deploy-prev/metallib/myds.metallib bin/myds.metallib'
```

## 未做（已知缺口）

**未跑全模型 logits/state 门。** 按 §0.4，换 shader 属数值路径变更，严格应过 0-ULP。
§118/§124 的"1644 位置逐位相同"覆盖的是 `score` 逐 token 路径，**不是全模型**。
