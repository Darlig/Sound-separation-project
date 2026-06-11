# Online Mix 三类声音分离实验流程

本项目当前主流程是 speech / concert / bird 三类声音分离。训练采用 online mix：训练时从单源 CSV 中动态抽样并混合；测试采用固定测试集：先生成固定混合 wav，再推理和统计指标。

## 一、数据准备

### 1. Online Mix 训练数据

用途：用于训练。

首先，切分单源数据：

```text
dataset/3class_speech_concert_bird_20260424/data_process/split_3class_speech_concert_bird.py
```

该脚本用于把 speech / concert / bird 单源数据切分成 train / valid / test。

输入：原始单源数据清单，包括 speech manifests，以及 concert / bird 的原始 CSV。

输出：

```text
speech_train.csv / speech_valid.csv / speech_test.csv
concert_train.csv / concert_valid.csv / concert_test.csv
bird_train.csv / bird_valid.csv / bird_test.csv
split_summary.json
```


然后，将同一 split 下的三类单源 CSV 合并成 online mix 原始单类数据：

```text
train_sources.csv = speech_train.csv + concert_train.csv + bird_train.csv
valid_sources.csv = speech_valid.csv + concert_valid.csv + bird_valid.csv
```


训练数据示例：

```text
dataset/3class_speech_concert_bird_20260424/dataset/train_sources.csv
dataset/3class_speech_concert_bird_20260424/dataset/valid_sources.csv
```

### 2. 固定测试数据

用途：用于测试和指标统计。

第一步，生成固定测试组合 metadata：

```text
dataset/create_csv_speech_concert_bird_v2.py
```

该脚本从 `speech_test.csv / concert_test.csv / bird_test.csv` 中抽样，生成类似下面格式的测试组合 CSV：

```text
s1_path,s1_label,s1_snr,s2_path,s2_label,s2_snr,...
```

常见输出例如：

```text
test_2mix.csv
test_3mix_each_class.csv
```

第二步，根据测试 metadata 生成 wav 测试集：

```text
model_constrcution/generate_test_wavs_speech_concert_bird.py
```

该脚本读取 `test_2mix.csv` 等 metadata，生成推理评测用 wav 目录：

```text
sample0/
  mixture.wav
  speech_gt.wav
  concert_gt.wav
  bird_gt.wav
```

生成后的目录作为推理脚本的 `--wav_dir`。

## 二、模型代码

核心网络：

```text
model_constrcution/DNN_models/Complex_MTASS.py
```

训练封装和 loss：

```text
model_constrcution/DNN_models/Complex_MTASS_model.py
model_constrcution/DNN_models/Complex_MTASS_Solver.py
```

`Complex_MTASS.py` 定义模型主体 `Complex_MTASS`。 相比原版MTASSNet，默认设置卷积结构为因果的，注意力开启，且也只看左侧上下文。测试时会使用其中的 `forward_streaming()` 和 `reset_streaming_state()` 做流式推理（通过模型内状态缓存实现）。

## 三、训练

### 1. 训练入口

脚本：

```text
model_constrcution/train.py
```

`train.py` 是主训练入口。使用 online mix 时指定：

```text
--data_mode online_csv
--train_source_csv .../train_sources.csv
--val_source_csv .../valid_sources.csv
```

它负责构建数据集、DataLoader、Lightning 模型、checkpoint 和 logger，并启动训练。

### 2. Online Mix 实现

脚本：

```text
model_constrcution/online_mix_dataset.py
```

该代码实现训练时的动态混音逻辑：从 `train_sources.csv` / `valid_sources.csv` 中按类别抽样，加载 wav，裁剪或补零到 10 秒，按随机 SNR 混成 2mix/3mix，并返回训练需要的 `X1, Y1..Y3, R1..R3`，mix音频不保存磁盘。

主要读取字段：

```text
filename 或 path 或 audio_path
category
source
```

### 3. 训练 run 示例

示例脚本：

```text
model_constrcution/run_train_online_mix_0424_720k_l1_0.1_sisdr.sh
```

该脚本给出一条完整 online mix 训练命令，包含原始单类训练/验证数据、mix数量、不同loss权重、batch size 和 GPU 配置。

关键参数示例：

```text
--online_num_sources 2 3
--train_samples_per_epoch 720000
--val_samples_per_epoch 1000
--magnitude_l1_loss_weight 0.1
--sisdr_loss_weight 1
```

通常在 `model_constrcution/` 目录下运行该 shell 脚本。

## 四、测试

### 1. 推理与 SDR/SI-SDR 评测

脚本：

```text
model_constrcution/test_wav_streaming_offline_model_speech_concert_bird.py
```

该脚本加载训练好的 checkpoint，对 `sampleN/mixture.wav` 做流式分离，输出：

```text
speech_es.wav
concert_es.wav
bird_es.wav
```

同时统计：

```text
SDR
SI-SDR
SDRi
```

主要输入：

```text
--wav_dir      固定测试 wav 根目录
--ckpt_path    训练得到的 checkpoint
--output_dir   推理结果输出目录
```
