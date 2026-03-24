# 两阶段训练方案说明

## 方案概述

本方案支持在训练过程中动态调整 batch size 和 GPU 数量，同时自动处理学习率缩放。

| 阶段 | Epochs | Batch Size | GPUs | 有效 Batch Size | 学习率 |
|------|--------|------------|------|-----------------|--------|
| Phase 1 | 1-8 | 24 | 6 | 144 | 0.001 |
| Phase 2 | 9+ | 64 | 8 | 512 | 0.00267 (自动缩放) |

## 学习率缩放原理

当 batch size 增大时，需要相应调整学习率以保持训练稳定性。本方案使用**线性缩放**策略：

```
LR_new = LR_old × (Batch_new / Batch_old)
        = 0.001 × (64 / 24)
        ≈ 0.00267
```

### 为什么需要缩放？

1. **梯度估计**：更大的 batch size 提供更准确的梯度估计
2. **噪声水平**：大 batch 的梯度噪声更小，可以使用更大的学习率
3. **收敛速度**：适当缩放可以维持相似的收敛动态

### Warmup 策略

切换到 Phase 2 后，前 2 个 epoch 会进行学习率 warmup：
- Epoch 9: LR = 0.00267 × 0.5 = 0.0013
- Epoch 10: LR = 0.00267 × 1.0 = 0.00267

这有助于模型平稳适应新的 batch size。

## 使用方法

### 方式一：自动两阶段脚本（推荐）

```bash
chmod +x run_streaming_two_phase.sh
./run_streaming_two_phase.sh
```

脚本会自动：
1. 检测当前 checkpoint 的 epoch
2. 执行 Phase 1（如果需要）
3. 自动缩放学习率
4. 执行 Phase 2

### 方式二：手动分阶段

**Phase 1:**
```bash
python3 train_streaming_adaptive.py experiments/streaming_model \
    --train_h5 "/path/to/train.h5" \
    --val_h5 "/path/to/valid.h5" \
    --gpus 0 1 2 3 4 5 \
    --epochs 8 \
    --batch_size 24 \
    --lr 0.001 \
    --phase "1"
```

**Phase 2:**
```bash
python3 train_streaming_adaptive.py experiments/streaming_model \
    --train_h5 "/path/to/train.h5" \
    --val_h5 "/path/to/valid.h5" \
    --resume_ckpt experiments/streaming_model/checkpoints/last.ckpt \
    --gpus 0 1 2 3 4 5 6 7 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --old_batch_size 24 \
    --warmup_epochs 2 \
    --phase "2"
```

## 技术细节

### 1. Checkpoint 兼容性

PyTorch Lightning 的 checkpoint 包含：
- Model state dict
- Optimizer state (包括学习率)
- LR Scheduler state
- Epoch number
- Global step

恢复训练时，所有状态都会自动加载。

### 2. 学习率缩放实现

通过自定义 `LRScalingCallback` 实现：

```python
class LRScalingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        scale_factor = new_batch / old_batch
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= scale_factor
```

这种方式比修改 checkpoint 文件更可靠。

### 3. DDP 注意事项

- Phase 1 (6 GPUs) 和 Phase 2 (8 GPUs) 都使用 DDP
- 切换 GPU 数量后，模型会被重新包装为 DDP
- 梯度同步机制会自动适应新的 world size

## 常见问题

### Q1: 切换 batch size 会影响模型收敛吗？

**A:** 理论上不会，只要学习率按比例缩放。实践表明：
- Batch size 增大 2-3 倍通常是安全的
- 配合 warmup 可以进一步降低风险
- 建议监控验证 loss，如有异常可调整缩放因子

### Q2: 可以从 Phase 2 退回到 Phase 1 吗？

**A:** 可以，但不推荐。如果必须这样做：
```bash
--batch_size 24 --old_batch_size 64 --lr 0.00267
# 学习率会自动缩放为: 0.00267 × (24/64) ≈ 0.001
```

### Q3: 如果 Phase 1 只训练了 5 个 epoch 就中断？

**A:** 脚本会自动检测当前 epoch，继续完成 Phase 1：
```
Current checkpoint epoch: 5
Starting/Resuming Phase 1...
# 会从 epoch 5 继续训练到 epoch 8
```

### Q4: 显存够用吗？

**A:** Phase 2 (batch=64, 8 GPUs) 的显存需求：
- 单 GPU batch = 64 / 8 = 8
- 与 Phase 1 (batch=24, 6 GPUs) 的单 GPU batch = 4 相比
- 显存需求约增加 2 倍

**估算**：如果 Phase 1 使用 12GB 显存，Phase 2 需要约 24GB。如果显存不足，可以：
- 减小 batch size 到 48
- 启用梯度累积
- 使用梯度检查点

### Q5: 训练速度对比？

| 指标 | Phase 1 | Phase 2 | 提升 |
|------|---------|---------|------|
| 单步时间 | ~200ms | ~250ms | - |
| 每秒样本数 | 144 | 512 | **3.5x** |
| 每 epoch 时间 | ~30min | ~10min | **3x** |

*注：实际数值取决于硬件和数据 I/O*

## 监控建议

在 TensorBoard 中监控以下指标：

1. **train_loss**: 应该在切换后保持稳定或继续下降
2. **val_loss**: 无明显上升
3. **learning_rate**: 确认缩放和 warmup 正确应用
4. **epoch**: 确认连续性（应显示 epoch 8 → 9 平滑过渡）

如果切换后出现：
- Loss 突然增大 → 可能是 LR 过大，减小缩放因子或增加 warmup
- Loss 不再下降 → 可能是 LR 过小，检查缩放计算
- OOM 错误 → 减小 batch size 或启用梯度累积

## 回滚方案

如果 Phase 2 训练出现问题，可以回滚到 Phase 1 的 checkpoint：

```bash
# 找到 Phase 1 的最佳 checkpoint
ls experiments/streaming_model/checkpoints/*.ckpt

# 回滚并降低学习率继续训练
python3 train_streaming_adaptive.py experiments/streaming_model \
    --resume_ckpt experiments/streaming_model/checkpoints/epoch=0007-xxx.ckpt \
    --batch_size 24 \
    --gpus 0 1 2 3 4 5 \
    --lr 0.0005  # 降低学习率
```
