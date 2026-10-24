# C3B：EMA、原型库、权重导出与恢复短测

前提：C2 和 C3A 已通过。本阶段使用真实 COCO 数据管线，验证新增模块的训练状态能够被正确平均、保存和恢复。它不是正式训练，不输出部分验证集 AP，也不把带断言和频繁同步的耗时当作吞吐结果。

## 现在执行什么

将本包的 `marn_c3b` 文件夹上传到：

`/root/autodl-tmp/project_marn-detr/experiments/marn_c3b/`

```bash
conda activate marn-detr
cd /root/autodl-tmp/project_marn-detr/experiments/marn_c3b
python install_c3b.py && python -u check_c3b.py
```

安装只执行一次。如果已经显示 `C3B_INSTALLED`，需要重跑检查时只执行：

```bash
python -u check_c3b.py
```

无需 tee：检查脚本自动保存日志和报告，失败也尽量生成报告 zip。末尾会打印实际文件位置：

```text
C3B_EMA_CHECKPOINT_CHECK_COMPLETED
C3B_REPORT_ZIP .../c3b_report_时间戳.zip
```

把最后的报告 zip 上传即可，不要上传测试权重。失败时上传生成的同一份报告，不必自行重装或连续重跑。

## 工作量与资源

- 先跑 8 个真实训练微批次，每批 8 张图，梯度累计 2 次：共 4 次优化器更新，有效 batch=16。
- 在原生 epoch 边界保存完整状态；进行 2 个验证批次的 regular 和 EMA 前向只读检查。
- 导出并重新构建 regular、EMA 两份推理模型，逐项检查权重及同一输入的 logits/boxes 完全一致。
- 从完整 `.ckpt` 恢复 epoch=1/global_step=4，再跑 2 个训练微批次，确认恢复后第一个 batch_idx=0、实际累计 2 次反传、最终 step=5。恢复阶段也可能触发有界验证。
- 总共训练读取 80 张图。`max_epochs=2` 是用于检验跨 epoch 恢复的短测配置；每个诊断 epoch 限制为 8 个微批次，不会遍历整个 COCO。
- 不重新下载权重，初始化使用本地已校验的官方 Small 权重。恢复和重建均关闭额外 encoder 预训练下载。
- 使用 4 个 workers、BF16、单卡，沿用 baseline 数据增强配置。临时权重通常合计不到约 1 GiB；启动前要求磁盘至少剩余 3 GiB。准确大小写入报告。

## EMA 策略

策略名：`online_bank_exact_copy_v1`。

1. 检测器和新增可学习参数继续使用 RF-DETR 原版 EMA 算法及 baseline 的 decay/tau。
2. 原型特征、几何、质量、类别、valid、更新时间、来源 ID、提交步数，以及 reader 的类别查找表都作为离散状态整体复制，不做参数式平均。特别是槽位替换后，不能把不同来源的槽位内容再混成一份 EMA 缓冲区。
3. 复制发生在优化器更新和原型库 commit 完成之后；梯度累计的中间微批次不会更新 EMA。
4. EMA 模型与 regular 模型在验证时读取各自冻结的库副本。两份库内容在同步时完全相同，但不存在共享存储的可变写入。
5. EMA 模型读取的是在线训练模型生成的历史特征，**没有额外维护由 EMA encoder/decoder 产生的专属原型库**。同层位置保持一致，权重版本有差异；这是成本与特征一致性之间的显式选择，不声称二者完全等价。后续主要机制比较先看 regular 模型；EMA 单独报告并保持各实验同一策略。
6. 训练结束时保留 regular 参数，不默认用 EMA 覆盖它；两类权重明确分别导出。

仅支持单卡、eager、BF16/FP32、普通检测。组合式 fused optimizer+EMA 被禁用，因为它会绕过原型库提交边界；普通 fused AdamW 是否启用仍由原版条件决定。

## 文件用途与管理

本包五个文件都放在 `experiments/marn_c3b/`：

| 文件 | 用途与覆盖行为 | 删除时机 |
|---|---|---|
| `install_c3b.py` | 校验源码哈希、备份 C2 的 `integration.py`，仅修改 EMA 接入保护条件；新增 `src/rfdetr/marn/lifecycle.py`。C2 类仍禁止直接使用普通 EMA。 | 建议归档；确认通过后实验目录中的安装器可删 |
| `lifecycle.py` | 新增训练类、EMA 回调、完整 checkpoint 元数据，以及专用推理导出/重建函数；安装器将其复制到源码目录 | **源码目录中的副本必须保留**；实验目录内副本可在归档本包后删 |
| `check_c3b.py` | 有界真实数据短测；生成日志、报告及三份临时权重 | 建议保留供复现，文件很小 |
| `source_guard.json` | 防止补丁应用到未审阅的另一版源码 | 与安装器一起保留 |
| `README.md` | 当前步骤、策略和文件管理说明 | 建议保留 |

新增运行产物：

| 文件或目录 | 用途 | 删除时机 |
|---|---|---|
| `check_outputs/时间戳/c3b_check.log` | 本次完整日志/失败堆栈 | 保留通过版本 |
| 同目录 `c3b_result.json` | 验收结论、提交统计、显存与临时权重大小 | 保留通过版本 |
| 同目录 `protocol.json` | 实际短测配置与 EMA 策略 | 与结果一起保留 |
| 同目录 `c3b_epoch_boundary.ckpt` | 仅用于验证完整 optimizer/scheduler/EMA/bank 恢复 | **我确认本次通过后可删除**，不用于正式训练起点 |
| 同目录 `regular_test.pth`、`ema_test.pth` | 仅用于推理导出与重建的相等性检查 | **我确认本次通过后可删除**，不用于正式训练起点 |
| `c3b_report_时间戳.zip` | 上传用的小报告包，不含权重或数据集 | 我读取后且原始日志/结果保留，可删 |
| `runs/source_backups/marn_c3b_时间戳/` | 修改前源码和前后哈希 | 目前保留用于回滚 |

当前 C3A 已确认通过，因此旧的 `c3a_report_*.zip` 上传副本、`c3_context.zip`、`collect_c3_context.py` 可以删除；保留 C3A 的通过日志、JSON、源码模块以及源码备份。

官方预训练权重、formal baseline 的整个目录、`src/rfdetr/marn/` 和源码备份都继续保留。不删除 `runs/` 或 `experiments/` 整个目录。

## 能证明什么、还没证明什么

检查覆盖：真实数据的标注 ID 到达 GPU、累计期间库冻结、每次 commit 后 EMA 库精确同步、参数确实按 EMA 更新、验证不写库、regular/EMA 权重独立、完整状态恢复、恢复后两次反传、推理权重的严格重建与输出一致性。

这里的 `.ckpt` 用于完整续训；两个 `.pth` 是 MARN 专用推理格式，需要 `rfdetr.marn.lifecycle.load_inference`，不能直接交给 `RFDETRSmall(pretrain_weights=...)` 或原版轻量权重加载器。没有偷偷把缺失的 MARN 权重用随机值补齐。

本阶段不验证 mid-epoch 数据采样器/RNG 的逐样本完全重放，也没有完成正式 best-checkpoint 选择和全量 COCO AP/APs/APm/APl 评价接入。本次 warmup_steps=1 和短学习率日程只为工程验收，不能直接当正式实验方案。

本地完成了脚本语法检查、基于你上传源码的安装模拟、精确备份检查、限定修改范围的 AST 检查和重复安装拒绝检查；还以 NumPy 替身检查 EMA 分组中按共享存储识别缓冲区的逻辑。本地无 PyTorch/CUDA，这些不能代替服务器的实际数值验收。

通过后进入正式运行前的性能与实验协议阶段：测量开启 memory 后的吞吐、补齐 best/last 保存与评价，再准备 prototype 与参数匹配的 query-only 对照。暂不进入 UAMR 或 MA-NAS。
