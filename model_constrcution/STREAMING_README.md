# Streaming Complex MTASS 模型说明

## 概述

本修改将 `self.apply_self_attn = True` 的模型改造为支持**流式解码 (Streaming Inference)**，使模型在训练完成后可以逐帧处理音频，适用于实时应用场景。

## 核心问题与解决方案

### 1. 原始问题

当 `apply_self_attn = True` 时，原始代码使用标准的自注意力机制：

```python
scores = torch.matmul(query, key) / math.sqrt(d_k)  # [batch, sen_len, sen_len]
```

这种实现需要访问**整个序列**的 key 来计算注意力分数，这在流式场景下是不可能的（无法看到未来帧）。

### 2. 解决方案

#### (1) 因果掩码（Causal Masking）

在训练时使用**因果掩码**，确保当前帧只能关注当前及之前的帧：

```python
def self_attention(query, key, value, mask=None, dropout=None, causal=True):
    ...
    if causal:
        sen_len = scores.size(-1)
        causal_mask = torch.tril(torch.ones(sen_len, sen_len, device=scores.device))
        scores = scores.masked_fill(causal_mask.unsqueeze(0) == 0, -1e9)
```

#### (2) 流式自注意力状态管理

为推理阶段添加了 `StreamingSelfAttention` 类，维护 key 和 value 的历史状态：

```python
class StreamingSelfAttention:
    def __init__(self, max_history_len=1000):
        self.key_history = None
        self.value_history = None
    
    def forward(self, query, key, value):
        # 将当前 key/value 与历史状态拼接
        full_key = torch.cat([self.key_history, key], dim=-1)
        full_value = torch.cat([self.value_history, value], dim=-1)
        # 更新历史状态
        self.key_history = full_key.detach()
        self.value_history = full_value.detach()
```

#### (3) 流式 GLU 模块

创建了 `StreamingGLU` 类，支持两种模式：
- `forward()`: 非流式前向传播（用于训练）
- `streaming_forward()`: 流式前向传播（用于推理，维护状态）

#### (4) 流式 GTCN 模块

创建了 `StreamingGTCN` 类，用于残差修复模块的流式处理。

## 文件列表

| 文件 | 说明 |
|------|------|
| `DNN_models/Complex_MTASS_Streaming.py` | 流式模型核心实现，包含 `StreamingComplexMTASS` 类 |
| `DNN_models/Complex_MTASS_Solver_Streaming.py` | 流式求解器，包含损失计算和流式推理工具 |
| `DNN_models/Complex_MTASS_model_Streaming.py` | PyTorch Lightning 封装，支持训练和推理 |
| `train_streaming.py` | 训练脚本 |
| `test_streaming.py` | 流式测试脚本，支持对比测试 |

## 使用方法

### 训练

```bash
python train_streaming.py ./model_parameters_streaming \
    --train_h5 /path/to/train.h5 \
    --val_h5 /path/to/dev.h5 \
    --gpus 0 1 2 3 \
    --epochs 100 \
    --batch_size 32
```

### 流式推理测试

```bash
python test_streaming.py \
    --test_h5 /path/to/test.h5 \
    --ckpt_path ./model_parameters_streaming/checkpoints/last.ckpt \
    --output_dir ./test_streaming_results \
    --num_sources 3 \
    --chunk_size 1 \
    --test_non_streaming
```

参数说明：
- `--chunk_size`: 每次处理的帧数，默认 1（逐帧处理），可增大以提高效率
- `--test_non_streaming`: 同时测试非流式推理用于对比

## 流式推理 API

### 方法 1: 使用工具函数（推荐）

```python
from DNN_models.Complex_MTASS_Streaming import StreamingComplexMTASS
from DNN_models.Complex_MTASS_model_Streaming import ComplexMTASSStreamingLightning
from DNN_models.Complex_MTASS_Solver_Streaming import Complex_MTASS_model_Streaming

# 加载模型
model = ComplexMTASSStreamingLightning.load_from_checkpoint(
    'path/to/checkpoint.ckpt',
    model_class=StreamingComplexMTASS,
    loss_class=Complex_MTASS_model_Streaming,
)
model.eval()

# 时域信号流式分离
speech, music, others = Complex_MTASS_model_Streaming.streaming_separation_with_overlap_add(
    model.model, 
    mixture_signal,  # [batch, num_samples]
    win_len=512,
    win_inc=256,
    fft_len=512,
    chunk_size=1,  # 逐帧处理
    device='cuda'
)
```

### 方法 2: 手动控制流式推理

```python
# 重置状态（每个新 utterance 开始前调用）
model.reset_streaming_state()

# 逐帧处理
for t in range(num_frames):
    x_frame = input_frames[:, :, t:t+1]  # 取单帧
    y1, y2, y3 = model.streaming_forward(x_frame)
    # y1, y2, y3 是当前帧的分离结果
```

## 关键技术细节

### 1. 因果卷积

代码中已设置 `is_causal = True`，确保卷积操作不会访问未来信息：

```python
self.is_causal = True
if self.is_causal:
    pad = nn.ConstantPad1d((2*dilation, 0), value=0.)  # 只在左侧填充
```

### 2. 状态缓存限制

为防止内存无限增长，历史状态有最大长度限制：

```python
self.max_history_len = max_history_len  # 默认 1000 帧

# 超过限制时裁剪
if self.history_len > self.max_history_len:
    self.key_history = self.key_history[:, :, -self.max_history_len:]
```

### 3. 训练与推理的一致性

训练时使用因果掩码确保模型学习到因果行为，推理时通过状态管理实现真正的流式处理，两者逻辑一致。

## 性能注意事项

1. **chunk_size**: 增大 chunk_size 可以提高 GPU 利用率，但会增加延迟
   - `chunk_size=1`: 最低延迟，适合实时应用
   - `chunk_size=10+`: 更高吞吐量，适合离线处理

2. **max_history_len**: 自注意力历史长度
   - 默认值 1000 帧（约 16 秒@16kHz）
   - 可根据实际应用调整

3. **状态重置**: 每个新 utterance 前必须调用 `reset_streaming_state()`

## 验证

运行测试脚本时会自动验证：
- 流式和非流式输出的一致性
- 分离性能指标（SI-SDR）
- 推理时间

预期结果：流式和非流式的 SI-SDR 差异应小于 0.1 dB。
