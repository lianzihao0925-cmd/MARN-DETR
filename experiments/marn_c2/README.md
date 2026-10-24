# C2：RF-DETR 内部接入与步末写库短测

## 使用范围

这是阶段 C 的接入检查包，不是正式训练入口。仅支持单 GPU、eager、BF16/FP32、普通目标检测；compile、CUDA graphs、激活重计算、FP16 GradScaler、EMA 暂不支持。使用两张 COCO train2017 图像，重复构成六个 batch，每两批累积一次梯度，运行三个更新步，再从完整 epoch 边界 checkpoint 恢复一个更新步。该数据重复仅用于工程验收，不用于吞吐、精度或 epoch 训练量声明。

## 安装与运行

将包中文件放在 /root/autodl-tmp/project_marn-detr/experiments/marn_c2，激活 marn-detr 环境：

```bash
python install_c2.py
set -o pipefail
python -u check_c2.py 2>&1 | tee c2_check.log
```

安装器先验证全部源码锚点和 Python 语法，再备份三个原文件到 runs/source_backups/marn_c2_时间戳，并保存 before/after SHA256。任何源码锚点不匹配都会在修改前停止。安装失败会恢复备份；重复安装会拒绝覆盖。首次成功安装后，如果短测失败，只重跑短测，不要重复安装。

安装文件：src/rfdetr/marn/{__init__,core,integration}.py。

修改文件：models/transformer.py、models/lwdetr.py、models/criterion.py。未安装 adapter 的普通模型保持原计算路径。Git 中的其他改动不重置。

## 实现语义

- 插入点在第 1 层 decoder 原有 reference 更新及辅助输出记录之后、第 2 层之前。
- 原始隐藏特征用于同层原型读写，分类和几何先验来自第 1 层检测头；先验停止梯度，reader/refiner 正常学习。
- criterion 在官方批量匹配完成后观察已有索引，不替换匹配、不增加匹配调用、不改变损失归一化。
- 每组 300 查询中随机选择 30 个，重复训练组使用相同相对索引。专用 RNG 不消耗检测器 RNG。只是阶段 C 的随机对照路线，尚非 UAMR。
- 写入候选来自训练图像匹配对象，置信度>=0.2、IoU>=0.5，质量为 sqrt(confidence*IoU)。使用真实 COCO annotation ID，以 (image_id, annotation_id) 在整个优化器窗口内去重，保留最高质量匹配。
- 每类 4 槽；同类余弦相似度>=0.9 时以 0.9 动量更新；否则先填空槽；无空槽时按 (1-quality)+0.1*age/1000+0.1*redundancy 选择替换槽。是待验收的明确规则，不是已证实最优超参。
- bank 的 source_image/source_object 表示最近贡献者；EMA 原型不是该对象的原样特征。完整贡献历史记录和覆盖/拒绝按类统计仍需正式训练日志补齐。
- 同一优化器窗口复用同一 clone 快照，所有写入暂存至官方 optimizer_step 返回后提交，梯度累积期间不提交。
- 验证只读；完整 .ckpt 保存 adapter 参数、bank 数值/元数据、提交步数及配置，并拒绝在尚未提交的窗口内保存。
- 短测冷启动积累一步；正式积累日程尚未确定，不能直接拿它跑十轮比较。

## 验收信号

- C2_DISABLED_FORWARD_LOSS_IDENTICAL
- C2_EMPTY_BANK_FORWARD_LOSS_IDENTICAL
- C2_NONEMPTY_ZERO_INIT_FORWARD_LOSS_IDENTICAL
- C2_COMMIT（每次更新一次，microbatches=2，重复匹配只产生唯一对象写入）
- C2_EVAL_BANK_FROZEN
- C2_RESTORE_STATE_OK（bank、模块参数、优化器、调度器状态核对）
- C2_INTEGRATION_CHECK_COMPLETED

结果保存在 check_outputs/c2_result.json，临时完整 checkpoint 保存在 check_outputs/c2_boundary.ckpt。恢复测试位于这一合成短测 epoch 的边界，不声称实现任意真实数据采样位置/RNG 的精确续跑。

## 尚未完成

正式增强管线必须同步维护 annotation ID 的过滤与排序；当前 checker 使用自定义的固定 resize 数据集直接携带原始 annotation ID，没有修改官方数据增强代码。没有该字段时 adapter 明确拒绝写库。

EMA 不可直接平均 bank 的年龄、来源 ID 或有效标记；正式训练前须明确 bank 与 EMA detector 的对应策略并测验证恢复。query-only 参数匹配对照、原型干预、APs/APm/APl、开销验收和正式训练入口也尚未完成。

当前本地验证：全部文件语法；使用用户提供的源码片段演练安装、锚点定位、修改后语法、重复安装保护。没有本地 PyTorch/RF-DETR，未声称短测已运行。必须以服务器日志验收。
