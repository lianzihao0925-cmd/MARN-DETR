# UAMR 配对干预收益标签审计与实现记录

## 结论

已新增 `rf-detr/src/rfdetr/marn/uamr.py`，实现训练期、固定匹配的配对收益标签。该模块只产生离线监督和审计统计，不接入 `MARNAdapter` 的随机读取，也不改变 OCPM bank。

当前 UAMR 推理路由仍关闭。模块没有根据相关性自动打开路由；后续只有在独立的留出实验确认收益预测有效后，才应由外部实验配置决定是否接入。

## 标签定义

对同一输入的两次输出使用同一个最终层 Hungarian `indices`：

- `without_memory`：关闭记忆；
- `with_memory`：打开记忆；
- `full_image_benefit = full_image_loss_without - full_image_loss_with`；
- 正值表示官方检测损失下降，表示干预有利。

每个分支分别记录加权的 `loss_ce`、`loss_bbox` 和 `loss_giou`。分类损失、定位损失、整图总损失和局部总损失均单独保存。

局部标签可以通过 `selected_query_indices` 指定查询；如果未指定，则对该图全部固定匹配对象的局部损失取平均。整图标签始终保留，用于检查多查询和上下文交互。未匹配的指定查询标记为 `local_valid=False`，不会伪造局部收益。

## 防泄漏与协议约束

`PairedInterventionContext` 要求调用者显式确认：

- 训练或离线训练阶段；
- 同一输入；
- 同一模型状态；
- 同一 bank snapshot；
- 同一 Hungarian 匹配；
- 同一随机状态。

任一条件不满足都会拒绝生成标签。验证和测试阶段会被阶段检查拒绝；该模块不向 bank 写入，也不产生在线路由决策。

置信度、单对象 IoU、查询与原型相似度、`quality_diag` 和 `Q_write` 不参与收益标签计算。它们只能作为后续路由模型的独立输入或诊断字段，不能替代配对干预收益。

## 统计与相关性审计

`summarize_benefits` 和 `summarize_paired_labels` 记录有限样本中的正、零、负收益数量及比例，并单独记录无效数量、均值、标准差和分位数。零判断默认采用严格零；如实验需要容差，必须作为显式诊断配置记录。

`audit_benefit_predictions` 只报告预测收益与实际收益的样本数、Pearson 相关、MSE、MAE、符号一致率和实际正/零/负比例。它没有内置通过阈值，不会自动接入 UAMR；相关性不足时应保持现有随机读取基线。

## 未完成事项

- 尚未在当前环境运行真实张量 smoke test，因为环境缺少 PyTorch；
- 尚未把标签生成器接入训练主循环，避免改变当前 OCPM 时序和检测损失；
- 尚未使用验证或测试标签训练、校准或打开 UAMR 路由；
- 需要在有 PyTorch 的训练环境中确认 criterion 的实际损失配置、空匹配和固定尺寸 target padding 路径。
