# C4：正式保存与完整评价接入

用途：在长训练前验证独立 best/last、原子保存、原生 epoch 边界恢复、可重建的 MARN 导出，以及完整 COCO 评价。远程放置目录 `experiments/marn_c4/`。由助手执行，不需要手动重复运行。

| 文件 | 用途、覆盖行为 | 保留 / 删除 |
|---|---|---|
| `formal_tools.py` | 实验级回调，保留原生训练/评价；替换本次 Trainer 的 EMA、指标捕获、best/last 回调，不改安装源码 | 正式运行依赖，保留 |
| `check_formal.py` | 三个短 epoch（每轮 2 微批）验证 best/last，再恢复 2 微批；随后完整评价 5000 张 val 图像 | 保留作为验收入口；不要当精度训练脚本 |
| `query_only.py` | 同参数量 query-only 对照及两种机制的显式导出/重载；不读取跨图原型。计算量另报，不声称 FLOPs 完全相同 | 正式运行依赖，保留 |
| `check_query_only.py` | 检查参数量、零启动、后续梯度、真实数据累计/EMA、独立重载与预测；失败发生在导出后时复用已有短测 checkpoint 完成检查 | 保留；通过后 `query_only_check/` 中临时 `.ckpt/.pth` 可删 |
| `run_arm.py` | 单个机制分支的正式入口，仅由控制器调用；使用 10-epoch 学习率日程，在指定轮数结束 | 保留；不覆盖已有运行目录 |
| `run_pair.py` | 显式 GPU 小时上限的配对筛选控制器；依次运行 MARN 和 query-only | 保留；正式 GPU 预算确定前不启动训练 |
| `README.md` | 本说明 | 保留 |
| `check_outputs/时间戳/` | 每次新建，含配置/结果/评价 JSON 和短测权重 | 通过验收后临时权重可删；JSON 与日志保留 |
| `best_regular.ckpt`、`best_ema.ckpt`、`last.ckpt` | 完整训练状态，包括 regular 模型、optimizer、scheduler、EMA、bank、回调与主进程 RNG；独立更新 | 此处短测可删；正式训练版本保留 |
| `last_regular_test.pth`、`last_ema_test.pth` | MARN 格式推理导出，保存检测参数、冻结 bank 和配置 | 此处仅测试，验证通过可删 |
| `coco_epoch_编号_regular.json` / `_ema.json` | 同一官方评价链路的 AP/AP50/AP75/APs/APm/APl/AR100 | 保留 |
| `epoch_audit.json` | loss 有限、步数、学习率、bank、显存、含验证/保存的时间 | 保留；短测 epoch 时间不代表全量 epoch |

best_regular 与 best_ema 按各自验证 AP 选取，last 每个 epoch 独立更新。best_ema 的完整 checkpoint 内仍含 regular 模型及单独 EMA 状态，恢复训练应加载整个 checkpoint；要导出 best EMA，需显式使用里面的 EMA 参数和 bank，不能把 regular state_dict 冒充 EMA。

保存采用同目录临时文件再原子替换，需预留临时文件空间。所有保存位于完成 optimizer 与 bank 提交的原生 epoch 边界。主进程 Python/NumPy/PyTorch/CUDA RNG 保存和恢复；持久 DataLoader worker 的增强随机流、epoch 中途采样器进度尚未精确序列化，不声称逐位恢复后续数据轨迹。

工程检查使用专门的 `engineering/regular`、`engineering/ema` 下降分数来制造“无改善”情形，仅替换此检查的 checkpoint monitor；真实 COCO 结果仍单独保存，不能把工程分数报告为 AP。正式运行使用真实 `val/mAP_50_95` 与 `val/ema_mAP_50_95`。

全量评价针对官方权重 + 零输出/空 bank 的 MARN，验证初始恒等与评价链路。它不是训练后 MARN 的精度证据。评价 maxDets=100、固定 512、BF16、原始 COCO 类别 ID；指标按 0–1 保存，转百分数时乘 100。

## 已准备的正式配对筛选

待新增 GPU 时间预算确定后由助手启动，不需要用户手动运行。4 GPU 小时方案为每分支 4 个完整 epoch；8 GPU 小时方案为每分支 8 个完整 epoch。两者均保留原 Baseline 的 10-epoch 学习率总日程，在预定 epoch 边界停止，不把学习率日程压缩成 4/8 轮。控制器对整个配对任务使用墙钟上限（单卡时是保守的 GPU 小时上限）；超时停止并保留已保存的 last，不把不等更新量结果作为配对结论。

两分支使用官方同一初始化、seed42、同数据增强与顺序、有效 batch16、BF16、原始检测损失、EMA 与每轮完整 COCO 评价。每组 warmup_steps=500，只积累记忆而不启用修正；第 500 次更新后按每组 30 个查询开始相同预算的修正。query-only 完全去掉原型读写及有关辅助监督，用当前查询的 tied key 投影提供条件向量。两者总参数 32,194,630，新增 adapter 参数 83,460；query-only 的 bank 状态保持空且不作为输入。

正式输出在 `runs/mechanism_pair_seed42_时间戳/{marn,query_only}/`：各分支保留三份完整 best_regular/best_ema/last checkpoint，以及四份显式机制格式 best/last regular/EMA 推理权重；配置、逐轮完整指标、数据哈希、时间和结果 JSON 保留。每轮核对目标元信息哈希与前两批图像哈希。只进行一个配对种子，属于机制筛选，不足以证明充分收敛或多种子收益。UAMR 和 NAS 尚未启动。

正式推理导出格式为 `marn_mechanism_inference_v1`，通过 `query_only.load_experiment` 读取；不要用原生 RFDETR 或旧版 `lifecycle.load_inference` 读取这些文件。完整 `.ckpt` 用于恢复训练，`.pth` 用于冻结推理，二者用途不同。

本阶段工程验收已完成，报告为本机 `c4_report_20261016_225200.zip`。短测目录中可丢弃的 `.ckpt/.pth` 已清理，配置、验收 JSON、日志保留；正式配对训练尚未启动。更新此脚本包只覆盖同名脚本和说明，不包含模型源码安装操作或正式实验权重。
