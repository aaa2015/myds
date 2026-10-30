# 跨机完全可复现（账本 §131）

## 触发

§130 装完 macmini 的 Metal 后复查，发现 Xcode 已升 27.2 —— 默认工具链已对齐。

| | macmini | macstudio |
|---|---|---|
| `xcode-select` | `/Applications/Xcode.app`（27.2） | `/Library/Developer/CommandLineTools` |
| clang | `2100.4.1.1` | `2100.4.1.1` |
| SDK | 27.2 | 27.2 |

**默认路径就一致，不需要再指定 `DEVELOPER_DIR`。**

## 实测

| 产物 | macmini | macstudio | 一致 |
|---|---|---|---|
| `bin/myds`（干净构建） | `3c9c858b` | `3c9c858b` | ✅ |
| `myds.metallib` | `719242c9` | `719242c9` | ✅ |
| 生产在用 | — | `fe557933` | — |

## 失效的两条旧结论

**§117「需显式 `DEVELOPER_DIR=CLT`」** → 现在默认路径一致，操作要求撤销。

**§117/§124「metallib 不跨工具链可复现」** → 那是旧 vs 新工具链的事实。
准确表述：**同一工具链下可复现，跨工具链不可复现** —— 与 binary 规律相同。

## macmini 现在能真正自验证

```
rm -rf bin/myds.metallib && make bin/myds.metallib
→ Successfully built (4,283,287 bytes), 719242c9
make test → EXIT=0
```

此前它依赖仓里预置的 metallib，通过与否取决于那份文件是否与源码匹配。隐患消除。

## 待裁决：生产是否切到 `719242c9`

| | 身份 | 可复现性 |
|---|---|---|
| 生产 `fe557933` | 旧工具链 | **已无人能复现** |
| 新 `719242c9` | 新工具链 | **两台机器逐字节复现** |

从身份可验证看 `719242c9` 更好：哈希可被两个独立机器重算。
但换库的代价是 §126–§128 已证的：**缓存失效 ⇒ 部署后首请求走 JIT，
logits 差 0.1%、argmax 差 0.49%**（§128 规则：先发哑请求丢弃结果）。

**本轮未执行，等裁决。**
