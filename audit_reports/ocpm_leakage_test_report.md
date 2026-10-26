# OCPM 最小防泄漏测试报告

## 运行结论

本轮在元数据方案上更新了 OCPM 行为代码。动态测试因当前环境缺少 PyTorch 而未执行；版本漂移静态审计仍失败，新增 provenance/unknown-mask 静态审计通过。

运行命令：

```text
python -m pytest rf-detr/tests/marn/test_ocpm_leakage.py rf-detr/tests/marn/test_ocpm_leakage_static.py -q -rs --confcutdir=rf-detr/tests/marn -o addopts=''
```

结果：`1 failed, 2 passed, 1 skipped`。

- 动态测试文件覆盖第 1–8 项，共 9 个参数化用例；整体因 `ModuleNotFoundError: No module named 'torch'` 跳过。
- provenance 和 unknown-mask 静态检查通过。
- 静态版本漂移测试执行失败：当前 `PrototypeBank` 只有 `update_count` 原型更新次数，没有模型/投影版本或 `drift_possible` 标记。
- Python 编译检查、`compileall` 和 `git diff --check` 通过。

## 测试映射

| 编号 | 检查内容 | 测试 | 当前结果 |
|---|---|---|---|
| 1 | 当前样本不能读取自己刚写入的原型 | `test_current_sample_cannot_read_its_pending_write` | 未执行，缺少 PyTorch |
| 2 | optimizer step 前只能读取旧 snapshot | `test_optimizer_window_reads_only_old_snapshot` | 未执行，缺少 PyTorch |
| 3 | optimizer step 后才提交并可读取新原型 | `test_commit_makes_prototype_visible_only_after_optimizer_boundary` | 未执行，缺少 PyTorch |
| 4 | 验证阶段 bank 不变 | `test_validation_and_test_reads_do_not_mutate_bank[validation]` | 未执行，缺少 PyTorch |
| 5 | 测试阶段 bank 不变 | `test_validation_and_test_reads_do_not_mutate_bank[test]` | 未执行，缺少 PyTorch |
| 6 | 同 batch 写入顺序不影响读取 | `test_batch_write_order_cannot_change_snapshot_reads` | 未执行，缺少 PyTorch |
| 7 | 旧 checkpoint 缺失 conf/IoU 不伪造 | `test_old_checkpoint_missing_reliability_is_unknown_not_fabricated` | 未执行，缺少 PyTorch |
| 8 | NaN、Inf、空槽位不进入正常读取 | `test_nonfinite_and_empty_slots_are_not_read` | 未执行，缺少 PyTorch |
| 9 | 版本变化后旧原型标记可能漂移 | `test_bank_exposes_model_or_projection_version_for_drift_marking` | **失败** |
| 10 | provenance 和未知字段位掩码持久化 | `test_bank_persists_provenance_and_unknown_mask` | **通过（静态）** |
| 11 | `quality_diag` 不得作为维护回退 | `test_quality_diag_cannot_be_used_as_maintenance_fallback` | **通过（静态）** |

## 已观察到的时序证据

静态检查确认以下实现存在：

1. `MARNAdapter.forward()` 训练阶段在窗口开始创建 `bank.snapshot(step)`，后续 microbatch 复用该 snapshot。
2. `MARNAdapter.commit()` 要求训练状态、capture 已消费、snapshot 存在且已经 `observe()`；提交后才清空 snapshot 和 pending。
3. 评估分支只调用 `bank.snapshot(step)`，不进入 `observe()` 或 `commit()`。
4. `PrototypeReader` 用 snapshot 的 `valid`、有限值特征和有限值几何共同过滤槽位。
5. 旧 checkpoint 迁移对缺失 `conf` 和 `iou` 使用 NaN，对 `reliability_known` 使用 False；没有伪造 0 或 1。
6. bank 现在持久化 `feature_layer`、`match_source` 和 `unknown_mask`；缺失字段由 unknown 哨兵和位掩码表示。
7. `quality_diag` 不再作为重复候选或维护排序的回退分数。

## 失败样例

失败断言为：

```text
PrototypeBank has update_count only; it cannot mark prototypes stale after a model/projection version change
```

这不是测试误报。`update_count` 只能表示槽位被更新的次数，不能判断特征提取模型、投影层或数据域版本是否变化。因此当前实现无法满足第 9 项。

## Smoke 准入

当前**不允许进入 OCPM smoke test**：

- 第 1–8 项尚未在真实 PyTorch 环境执行；
- 第 9 项已明确失败，版本漂移保护缺失。

进入 smoke test 前至少需要在安装 PyTorch 的环境重新运行第 1–8 项，并对版本漂移字段作出明确选择。若本阶段坚持“不扩展功能”，应把第 9 项标记为已知限制，而不能宣称防泄漏测试全部通过。
