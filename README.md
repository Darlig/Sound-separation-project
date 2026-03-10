# 固定类别数量的声音分离（第一阶段——speech,music,others）

## 目录结构
```plaintext
├── VGGSound\                                # VGGSound数据的csv文件(分为train和test)
├── dataset\                                 # 训练及测试数据的csv文件
    ├── create_csv.py                        # 数据的csv文件生成代码
├── model_constrcution\
    ├── DNN_models\                          # 模型代码
         ├──  Complex_MTASS.py               
         ├──  Complex_MTASS_model.py         
         ├──  Complex_MTASS_Solver.py
    ├── extract_features.py                  # 提取特征
    ├── run.sh                               # 训练脚本
    ├── eval.sh                              # 测试脚本
    ├── train.py                             # 训练代码
    ├── test.py                              # 测试代码
├── environment.yaml                         # 环境依赖
└── README.md                                # 本文件
```

## 安装依赖
```bash
conda env create -f environment.yml
```

## 数据准备：
使用VGGSound数据集来混合生成2~5mix的训练集、验证集和测试集，其中VGGSound数据集路径在/work107/luoxiaoxue/data/VGGSound。后续需要添加华为提供的部分数据，其路径在/work107/luoxiaoxue/data/Huawei。
```bash
python create_csv.py  --input_csv /dataset/3class_data_nosing/train.csv  --output_csv dataset/3class_data_nosing/metadata/train_3mix.csv  --num_sources 3  --dataset_type train
```

## 特征提取：
```bash
python extract_features.py --input_csv_list dataset/3class_data_nosing/metadata/train_2mix.csv dataset/3class_data_nosing/metadata/train_3mix.csv dataset/3class_data_nosing/metadata/train_4mix.csv dataset/3class_data_nosing/metadata/train_5mix.csv --output_dir /processed_data/3class_nosing --split train
```
## 模型训练：
```bash
# 编辑run.sh文件来设置实验目录及训练、验证数据
bash run.sh
```
## 测试及推理：
```bash
bash eval.sh
```
