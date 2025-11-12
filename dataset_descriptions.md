# 数据集描述文档

本文档基于 `dataset_readme.md` 中的描述，对所有涉及的数据集进行了更完整和明确的整理。每个数据集的介绍包括类型、路径、文件结构以及数据描述，旨在提供规范、一目了然的参考。

## AffectNet

### 类型
人脸表情数据集，包含图像和对应的三类标注（表情、效价、唤醒度）。

### 路径
- 整体文件路径：`/home/remote1/lvshuyang/Datasets/AffectNet`
- 训练集路径：`/home/remote1/lvshuyang/Datasets/AffectNet/train_set`
- 训练集标注文件：`/home/remote1/lvshuyang/Datasets/AffectNet/train_set/annotations`
- 训练集图片文件：`/home/remote1/lvshuyang/Datasets/AffectNet/train_set/images`
- 验证集路径：`/home/remote1/lvshuyang/Datasets/AffectNet/val_set`
- 验证集图片文件：`/home/remote1/lvshuyang/Datasets/AffectNet/val_set/images`
- 验证集标注文件：`/home/remote1/lvshuyang/Datasets/AffectNet/val_set/annotations`

### 文件结构
```
AffectNet/
├── train_set/
│   ├── annotations/
│   │   ├── 0_aro.npy
│   │   ├── 0_exp.npy
│   │   ├── 0_lnd.npy
│   │   ├── 0_val.npy
│   │   └── ... (其他标注文件，以图片ID命名)
│   └── images/
│       └── ... (图像文件，如 .jpg 或 .png)
└── val_set/
    ├── annotations/
    │   └── ... (类似训练集标注文件)
    └── images/
        └── ... (图像文件)
```

### 数据描述
- **annotations/**: 标注文件，使用 Python 的 `numpy` 库读取。每个图片对应四个文件（以图片ID命名）：
  - `{ID}_exp.npy`: 面部表情ID，范围 0-7（对应不同表情类别）。
  - `{ID}_val.npy`: 效价值，范围 [-1, +1]（对于'不确定'和'无人脸'类别，该值为 -2）。
  - `{ID}_aro.npy`: 唤醒度值，范围 [-1, +1]（对于'不确定'和'无人脸'类别，该值为 -2）。
  - `{ID}_lnd.npy`: 面部关键点坐标（可能用于对齐）。
- **images/**: 图像文件，包含人脸图片，用于训练和验证。

## Affwild2

### 类型
多模态数据集，包含视频、音频和标注，用于情感分析（效价/唤醒度、表情识别、动作单元检测）。

### 路径
- 数据集总体路径：`/home/remote1/lvshuyang/Datasets/Affwild2`
- 训练集AU标注文件路径：`/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/AU_Detection_Challenge/Train_Set`
- 验证集AU标注文件路径：`/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/AU_Detection_Challenge/Validation_Set`
- 训练集EXP标注文件路径：`/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/EXPR_Recognition_Challenge/Train_Set`
- 验证集EXP标注文件路径：`/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set`
- 训练集VA标注文件路径：`/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/VA_Estimation_Challenge/Train_Set`
- 验证集VA标注文件路径：`/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/VA_Estimation_Challenge/Validation_Set`
- 视频存放路径：
  - `/home/remote1/lvshuyang/Datasets/Affwild2/batch1`
  - `/home/remote1/lvshuyang/Datasets/Affwild2/batch2`
  - `/home/remote1/lvshuyang/Datasets/Affwild2/batch3`

### 文件结构
```
Affwild2/
├── ABAW Annotations/
│   ├── AU_Detection_Challenge/
│   │   ├── Train_Set/
│   │   │   └── ... (标注文件，如 10-60-1280x720_right.txt)
│   │   └── Validation_Set/
│   │       └── ... (标注文件)
│   ├── EXPR_Recognition_Challenge/
│   │   ├── Train_Set/
│   │   │   └── ... (标注文件)
│   │   └── Validation_Set/
│   │       └── ... (标注文件)
│   └── VA_Estimation_Challenge/
│       ├── Train_Set/
│       │   └── ... (标注文件)
│       └── Validation_Set/
│           └── ... (标注文件)
├── batch1/
│   └── ... (视频文件)
├── batch2/
│   └── ... (视频文件)
└── batch3/
    └── ... (视频文件)
```

### 数据描述
- **VA_Estimation_Challenge/**: 效价和唤醒度估计标注。
  - 每个标注文件为 `.txt` 文件，以视频命名。
  - 第一行：`valence,arousal`
  - 后续每行：逗号分隔的效价和唤醒度值，对应每个视频帧。范围 [-1, 1]。值为 -5 的帧应忽略。
- **EXPR_Recognition_Challenge/**: 表情识别标注。
  - 每个标注文件为 `.txt` 文件，以视频命名。
  - 第一行：`Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other`
  - 后续每行：整数值 (0-7)，对应表情类别。每行对应一个视频帧。值为 -1 的帧应忽略。
- **AU_Detection_Challenge/**: 动作单元检测标注（可选）。
- **batch1/、batch2/、batch3/**: 视频文件，对应标注中的视频。

## Emotiontalk

### 类型
三模态数据集，包含视频、音频和文本，用于情感分析。

### 路径
- 数据集总体路径：`/home/remote1/lvshuyang/Datasets/Emotiontalk`
- 文本模态及标注路径：`/home/remote1/lvshuyang/Datasets/Emotiontalk/Text/json`
- 音频模态标注路径：`/home/remote1/lvshuyang/Datasets/Emotiontalk/Audio/json`
- 视频模态标注路径：`/home/remote1/lvshuyang/Datasets/Emotiontalk/Video/json`
- 音频存放路径：`/home/remote1/lvshuyang/Datasets/Emotiontalk/Audio/wav`
- 视频存放路径：`/home/remote1/lvshuyang/Datasets/Emotiontalk/Multimodal/mp4`

### 文件结构
```
Emotiontalk/
├── Audio/
│   ├── json/
│   │   ├── G00001/
│   │   │   ├── G00001_02/
│   │   │   │   └── ... (json 文件，如 G00001_02_01_002.json)
│   │   │   └── ... (其他子文件夹)
│   │   └── ... (其他组文件夹)
│   └── wav/
│       ├── G00001/
│       │   └── ... (wav 文件)
│       └── ... (其他组文件夹)
├── Multimodal/
│   ├── json/
│   │   └── ... (类似 Audio/json)
│   └── mp4/
│       └── ... (mp4 视频文件)
├── Text/
│   └── json/
│       └── ... (类似 Audio/json)
└── Video/
    └── json/
        └── ... (类似 Audio/json)
```

### 数据描述
- **Text/json/**: 文本模态标注文件（JSON 格式）。
  - 字段：
    - `data`: 每位标注者的标注结果。
    - `emotion_result`: 情感结果。
    - `speaker_id`: 说话者ID。
    - `file_name`: 文件路径（相对路径）。
    - `content`: 转写文本。
- **Audio/json/**: 音频模态标注文件（JSON 格式）。
  - 字段：
    - `data`: 每位标注者的标注结果。
    - `emotion_result`: 情感结果。
    - `speaker_id`: 说话者ID。
    - `file_name`: 文件路径（相对路径）。
- **Video/json/**: 视频模态标注文件（JSON 格式）。
  - 字段：
    - `data`: 每位标注者的标注结果。
    - `emotion_result`: 情感结果。
    - `speaker_id`: 说话者ID。
    - `file_name`: 文件路径（相对路径）。
- **Audio/wav/**: 音频文件（.wav 格式），对应标注。
- **Multimodal/mp4/**: 视频文件（.mp4 格式），对应标注。

## MELD

### 类型
视频数据集，来自《老友记》，包含视频、文本和情感标注。

### 路径
- 数据集总体路径：`/home/remote1/lvshuyang/Datasets/MELD`
- 训练集标注文件路径：`/home/remote1/lvshuyang/Datasets/MELD/train_sent_emo.csv`
- 测试集标注文件路径：`/home/remote1/lvshuyang/Datasets/MELD/test_sent_emo.csv`
- 验证集标注文件路径：`/home/remote1/lvshuyang/Datasets/MELD/dev_sent_emo.csv`
- 训练集视频路径：`/home/remote1/lvshuyang/Datasets/MELD/train_splits`
- 测试集视频路径：`/home/remote1/lvshuyang/Datasets/MELD/output_repeated_splits_test`
- 验证集视频路径：`/home/remote1/lvshuyang/Datasets/MELD/dev_splits_complete`

### 文件结构
```
MELD/
├── train_sent_emo.csv
├── test_sent_emo.csv
├── dev_sent_emo.csv
├── train_splits/
│   └── ... (视频文件，如 dia0_utt0.mp4)
├── output_repeated_splits_test/
│   └── ... (视频文件)
└── dev_splits_complete/
    └── ... (视频文件)
```

### 数据描述
- **train_sent_emo.csv、test_sent_emo.csv、dev_sent_emo.csv**: CSV 标注文件，包含所有标注信息。
  - 列：
    - `Sr No.`: 话语序列号。
    - `Utterance`: 话语文本。
    - `Speaker`: 说话者姓名。
    - `Emotion`: 情感类别（中性、高兴、悲伤、愤怒、惊讶、恐惧、厌恶）。
    - `Sentiment`: 情感极性（正面、中性、负面）。
    - `Dialogue_ID`: 对话索引。
    - `Utterance_ID`: 话语在对话中的索引。
    - `Season`: 季度编号。
    - `Episode`: 剧集编号。
    - `StartTime`: 开始时间（格式：时:分:秒,毫秒）。
    - `EndTime`: 结束时间（格式：时:分:秒,毫秒）。
- **train_splits/、output_repeated_splits_test/、dev_splits_complete/**: 视频文件（.mp4），命名规则 `dia{DIALOGUE_ID}_utt{UTTERANCE_ID}.mp4`。

## MEMO-Bench

### 类型
人脸图像数据集（AI生成），用于情感分析。

### 路径
- 数据集总体路径：`/home/remote1/lvshuyang/Datasets/MEMO-Bench`
- 图片路径：`/home/remote1/lvshuyang/Datasets/MEMO-Bench/dataset`

### 文件结构
```
MEMO-Bench/
├── annotation_emo.csv
├── annotation_quality.csv
└── dataset/
    └── ... (图像文件)
```

### 数据描述
- **annotation_emo.csv**: 情感标注文件（CSV 格式），包含情感类别标注。
- **annotation_quality.csv**: 质量标注文件（CSV 格式）。
- **dataset/**: 图像文件，情感类别包括 happy、sad、angry、surprise、worry、neutral。

## CH-SIMS

### 类型
视频数据集，用于情感分析，包含原始视频和处理后的数据。

### 路径
- 数据集总体路径：`/home/remote1/lvshuyang/Datasets/CH-SIMS`
- 标注文件路径：`/home/remote1/lvshuyang/Datasets/CH-SIMS/label.csv`
- 处理后数据路径：`/home/remote1/lvshuyang/Datasets/CH-SIMS/Processed`
- 原始视频路径：`/home/remote1/lvshuyang/Datasets/CH-SIMS/Raw`
- 示例视频文件：`/home/remote1/lvshuyang/Datasets/CH-SIMS/Raw/video_0001/0001.mp4`

### 文件结构
```
CH-SIMS/
├── label.csv
├── Processed/
└── Raw/
    ├── video_0001/
    ├── video_0002/
    ├── video_0003/
    ├── video_0004/
    ├── video_0005/
    ├── video_0006/
    ├── video_0007/
    ├── video_0008/
    ├── video_0009/
    ├── video_0010/
    ├── video_0011/
    ├── video_0012/
    ├── video_0013/
    ├── video_0014/
    ├── video_0015/
    ├── video_0016/
    ├── video_0017/
    ├── video_0018/
    ├── video_0019/
    ├── video_0020/
    ├── video_0021/
    ├── video_0022/
    ├── video_0023/
    ├── video_0024/
    ├── video_0025/
    ├── video_0026/
    ├── video_0027/
    ├── video_0028/
    ├── video_0029/
    ├── video_0030/
    ├── video_0031/
    ├── video_0032/
    ├── video_0033/
    ├── video_0034/
    ├── video_0035/
    ├── video_0036/
    ├── video_0037/
    ├── video_0038/
    ├── video_0039/
    ├── video_0040/
    ├── video_0041/
    ├── video_0042/
    ├── video_0043/
    ├── video_0044/
    ├── video_0045/
    ├── video_0046/
    ├── video_0047/
    ├── video_0048/
    ├── video_0049/
    ├── video_0050/
    ├── video_0051/
    ├── video_0052/
    ├── video_0053/
    ├── video_0054/
    ├── video_0055/
    ├── video_0056/
    ├── video_0057/
    ├── video_0058/
    ├── video_0059/
    └── video_0060/
```

### 数据描述
- **label.csv**: CSV 标注文件，包含情感标注信息。
- **Processed/**: 处理后的数据文件夹，可能包含提取的特征或预处理视频。
- **Raw/**: 原始视频文件夹，每个子文件夹对应一个视频样本（video_0001/ 到 video_0060/），包含原始视频文件。

## CH-SIMS v2

### 类型
视频数据集，用于情感分析，分为监督（s）和无监督（u）版本。

### 路径
- 数据集总体路径：`/home/remote1/lvshuyang/Datasets/CH-SIMS v2`
- 监督版本路径：`/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(s)`
- 无监督版本路径：`/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(u)`
- 监督版本标注：`/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(s)/meta.csv`
- 无监督版本标注：`/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(u)/meta.csv`
- 监督版本示例视频：`/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(s)/Raw/aqgy3_0001/00000.mp4`
- 无监督版本示例视频：`/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(u)/Raw/aqgy1_0001/00000.mp4`

### 文件结构
```
CH-SIMS v2/
├── CH-SIMS v2(s)/
│   ├── meta.csv
│   ├── Processed/
│   └── Raw/
│       ├── aqgy3_0001/
│       ├── aqgy3_0002/
│       ├── aqgy3_0003/
│       ├── aqgy3_0004/
│       ├── aqgy3_0005/
│       ├── aqgy3_0006/
│       ├── aqgy3_0007/
│       ├── aqgy3_0008/
│       ├── aqgy4_0001/
│       ├── aqgy4_0002/
│       ├── aqgy4_0003/
│       ├── aqgy4_0004/
│       ├── aqgy4_0005/
│       ├── aqgy4_0006/
│       ├── aqgy4_0007/
│       ├── aqgy4_0008/
│       ├── aqgy4_0009/
│       ├── aqgy4_0010/
│       ├── aqgy4_0011/
│       ├── aqgy4_0012/
│       ├── aqgy4_0014/
│       ├── aqgy4_0015/
│       ├── aqgy4_0016/
│       ├── aqgy4_0017/
│       ├── aqgy4_0018/
│       ├── aqgy4_0019/
│       ├── aqgy4_0020/
│       ├── aqgy4_0021/
│       ├── aqgy4_0022/
│       ├── aqgy4_0023/
│       ├── aqgy4_0024/
│       ├── aqgy5_0001/
│       ├── aqgy5_0002/
│       ├── aqgy5_0003/
│       ├── aqgy5_0004/
│       ├── aqgy5_0005/
│       ├── aqgy5_0006/
│       ├── aqgy5_0007/
│       ├── aqgy5_0008/
│       ├── aqgy5_0009/
│       ├── aqgy5_0010/
│       ├── aqgy5_0011/
│       ├── aqgy5_0012/
│       ├── aqgy5_0013/
│       ├── aqgy5_0014/
│       ├── aqgy5_0015/
│       ├── aqgy5_0016/
│       ├── aqgy5_0017/
│       ├── aqgy5_0018/
│       ├── aqgy5_0019/
│       ├── aqgy5_0020/
│       ├── aqgy5_0021/
│       ├── aqgy5_0022/
│       ├── aqgy5_0023/
│       ├── aqgy5_0024/
│       ├── aqgy5_0025/
│       ├── aqgy5_0026/
│       ├── aqgy5_0027/
│       ├── aqgy5_0028/
│       ├── aqgy5_0029/
│       ├── aqgy5_0030/
│       ├── aqgy5_0031/
│       ├── aqgy5_0032/
│       ├── aqgy5_0033/
│       ├── aqgy5_0034/
│       ├── aqgy5_0036/
│       ├── csbgs2_0001/
│       ├── csbgs2_0002/
│       ├── csbgs2_0003/
│       ├── test1/
│       ├── test10/
│       ├── test11/
│       ├── test12/
│       ├── test13/
│       ├── test14/
│       ├── test15/
│       ├── test2/
│       ├── test3/
│       ├── test4/
│       ├── test5/
│       ├── test6/
│       ├── test7/
│       ├── test8/
│       ├── test9/
│       ├── video_0001/
│       ├── video_0002/
│       ├── video_0003/
│       ├── video_0004/
│       ├── video_0005/
│       ├── video_0006/
│       ├── video_0007/
│       ├── video_0008/
│       ├── video_0009/
│       ├── video_0010/
│       ├── video_0011/
│       ├── video_0012/
│       ├── video_0013/
│       ├── video_0014/
│       ├── video_0015/
│       ├── video_0016/
│       ├── video_0017/
│       ├── video_0018/
│       ├── video_0019/
│       ├── video_0020/
│       ├── video_0021/
│       ├── video_0022/
│       ├── video_0023/
│       ├── video_0024/
│       ├── video_0025/
│       ├── video_0026/
│       ├── video_0027/
│       ├── video_0028/
│       ├── video_0029/
│       ├── video_0030/
│       ├── video_0031/
│       ├── video_0032/
│       ├── video_0033/
│       ├── video_0034/
│       ├── video_0035/
│       ├── video_0036/
│       ├── video_0037/
│       ├── video_0038/
│       ├── video_0039/
│       ├── video_0040/
│       ├── video_0041/
│       ├── video_0042/
│       ├── video_0043/
│       ├── video_0044/
│       ├── video_0045/
│       ├── video_0046/
│       ├── video_0047/
│       ├── video_0048/
│       ├── video_0049/
│       ├── video_0050/
│       ├── video_0051/
│       ├── video_0052/
│       ├── video_0053/
│       ├── video_0054/
│       ├── video_0055/
│       ├── video_0056/
│       ├── video_0057/
│       ├── video_0058/
│       ├── video_0059/
│       └── ... (更多文件夹)
└── CH-SIMS v2(u)/
    ├── meta.csv
    ├── Processed/
    └── Raw/
        └── ... (类似监督版本的文件夹)
```

### 数据描述
- **CH-SIMS v2(s)/ 和 CH-SIMS v2(u)/**: 监督和无监督版本的数据集。
  - **meta.csv**: 元数据文件，包含数据集的元信息。
  - **Processed/**: 处理后的数据文件夹。
  - **Raw/**: 原始视频文件夹，每个子文件夹对应一个视频样本，包含原始视频文件。

## CMU-MOSEI

### 类型
多模态情感数据集，包含视频、音频和文本，用于情感分析。

### 路径
- 数据集总体路径：`/home/remote1/lvshuyang/Datasets/CMU-MOSEI`
- 标注文件路径：`/home/remote1/lvshuyang/Datasets/CMU-MOSEI/label.csv`
- 处理后数据路径：`/home/remote1/lvshuyang/Datasets/CMU-MOSEI/Processed`
- 原始视频路径：`/home/remote1/lvshuyang/Datasets/CMU-MOSEI/Raw`

### 文件结构
```
CMU-MOSEI/
├── label.csv
├── Processed/
└── Raw/
    ├── _0efYOjQYRc/
    ├── _1nvuNk7EFY/
    ├── _26JmJnPKfM/
    ├── _2u0MkRqpjA/
    ├── _4K620KW_Is/
    ├── _4PNh8dIILI/
    ├── _7HVhnSYX1Y/
    ├── _8pvMpMdGM4/
    ├── _aJghSQmxD8/
    ├── _aZDaIfGfPo/
    ├── _bIJOxiIJFk/
    ├── _BMdEKNF5Js/
    ├── _BtdwH6mfWg/
    ├── _BvNNdEkhZA/
    ├── _BXmhCPTQmA/
    ├── _gr7o5ynhnw/
    ├── _gzYkdjNvPc/
    ├── _HabvL0VnrQ/
    └── ... (更多文件夹，如 _HanACduNJk/ 等)
```

### 数据描述
- **label.csv**: CSV 标注文件，包含情感标注信息。
- **Processed/**: 处理后的数据文件夹，可能包含提取的多模态特征。
- **Raw/**: 原始视频文件夹，每个子文件夹对应一个视频样本，包含原始视频、音频和文本文件。

## CMU-MOSI

### 类型
多模态情感数据集，包含视频、音频和文本，用于情感分析。

### 路径
- 数据集总体路径：`/home/remote1/lvshuyang/Datasets/CMU-MOSI`
- 标注文件路径：`/home/remote1/lvshuyang/Datasets/CMU-MOSI/label.csv`
- 处理后数据路径：`/home/remote1/lvshuyang/Datasets/CMU-MOSI/Processed`
- 原始视频路径：`/home/remote1/lvshuyang/Datasets/CMU-MOSI/Raw`

### 文件结构
```
CMU-MOSI/
├── label.csv
├── Processed/
└── Raw/
    ├── _dI--eQ6qVU/
    ├── 03bSnISJMiM/
    ├── 0h-zjBukYpk/
    ├── 1DmNV9C1hbY/
    ├── 1iG0909rllw/
    ├── 2iD-tVS8NPw/
    ├── 2WGyTLYerpo/
    ├── 5W7Z1C_fDaE/
    ├── 6_0THN4chvY/
    ├── 6Egk_28TtTM/
    ├── 73jzhE8R1TQ/
    ├── 7JsX8y1ysxY/
    ├── 8d-gEyoeBzc/
    ├── 8OtFthrtaJM/
    ├── 8qrpnFRGt2A/
    ├── 9c67fiY0wGQ/
    ├── 9J25DZhivz8/
    ├── 9qR7uwkblbs/
    ├── 9T9Hf74oK10/
    ├── Af8D0E4ZXaw/
    ├── aiEXnCPZubE/
    ├── atnd_PF-Lbs/
    ├── Bfr499ggo-0/
    ├── BI97DNYfe5I/
    ├── BioHAh1qJAQ/
    ├── bOL9jKpeJRs/
    ├── bvLlb-M3UXU/
    ├── BvYR0L6f2Ig/
    ├── BXuRRbG0Ugk/
    ├── c5xsKMxpXnc/
    ├── c7UH_rxdZv4/
    ├── Ci-AH39fi3Y/
    ├── Clx4VXItLTE/
    ├── cM3Yna7AavY/
    ├── cW1FSBF59ik/
    ├── cXypl4FnoZo/
    ├── d3_k5Xpfmik/
    ├── d6hH302o4v8/
    ├── Dg_0XKD0Mf4/
    ├── dq3Nf_lMPnE/
    ├── etzxEpPuc6I/
    ├── f_pcplsH_V0/
    ├── f9O3YtZ2VfI/
    ├── fvVhgmXxadc/
    ├── G-xst2euQUc/
    ├── G6GlGvlkxAQ/
    ├── GWuJjcEuzt8/
    ├── HEsqda8_d0Q/
    ├── I5y0__X72p0/
    ├── iiK8YX8oH1E/
    ├── Iu2PFX3z_1s/
    ├── IumbAb8q2dM/
    ├── Jkswaaud0hk/
    ├── jUzDDGyPkXU/
    ├── k5Y_838nuGo/
    ├── LSi-o-IrDMs/
    ├── lXPQBPVc5Cw/
    ├── MLal-t_vJPM/
    ├── nbWiPyCm4g0/
    ├── Njd1F0vZSm4/
    ├── nzpVDcQ0ywM/
    ├── Nzq88NnDkEk/
    ├── ob23OKe5a9Q/
    ├── OQvJTdtJ2H4/
    ├── OtBXNcAL_lE/
    ├── Oz06ZWiO20M/
    ├── phBUpBr1hSo/
    ├── pLTX3ipuDJI/
    ├── POKffnXeBds/
    ├── PZ-lDQFboO8/
    ├── QN9ZIUWUXsY/
    ├── Qr1Ca94K55A/
    ├── rnaNMUZpvvg/
    ├── Sqr0AcuoNnk/
    ├── tIrG4oNLFzE/
    ├── tmZoasNr4rU/
    ├── tStelxIAHjw/
    ├── TvyZBvOMOTc/
    ├── v0zCBqDeKcE/
    ├── VbQk4H8hgr0/
    ├── VCslbP0mgZI/
    ├── Vj1wYRQjB-o/
    ├── vvZ4IcEtiZc/
    ├── vyB00TXsimI/
    ├── W8NXH0Djyww/
    ├── WKA5OygbEKI/
    ├── wMbj6ajWbic/
    ├── X3j2zQgwYgE/
    ├── yDtzw_Y-7RU/
    ├── yvsjCA6Y5Fc/
    ├── ZAIRrfG22O0/
    ├── zhpQhgha_KU/
    ├── ZUXBRvtny7o/
    └── ... (更多文件夹)
```

### 数据描述
- **label.csv**: CSV 标注文件，包含情感标注信息。
- **Processed/**: 处理后的数据文件夹，可能包含提取的多模态特征。
- **Raw/**: 原始视频文件夹，每个子文件夹对应一个视频样本，包含原始视频、音频和文本文件。

## DFEW

### 类型
人脸表情视频数据集，用于情感分析，包含视频片段和标注。

### 路径
- 数据集总体路径：`/home/remote1/lvshuyang/Datasets/DFEW`
- 说明文档：`/home/remote1/lvshuyang/Datasets/DFEW/README.md`
- 标注文件：`/home/remote1/lvshuyang/Datasets/DFEW/Annotation/annotation.xlsx`
- 数据分割目录：`/home/remote1/lvshuyang/Datasets/DFEW/EmoLabel_DataSplit`
- 原始视频目录：`/home/remote1/lvshuyang/Datasets/DFEW/Clip/original`
- 224×224 帧目录：`/home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224`
- 224×224（16 帧）目录：`/home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224_16f`
- 224×224（AVI）目录：`/home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224_avi`

### 文件结构
```
DFEW/
├── README.md
├── Annotation/
├── Clip/
│   ├── clip_224x224/
│   │   ├── 00001/
│   │   ├── 00002/
│   │   ├── 00003/
│   │   ├── 00004/
│   │   ├── 00005/
│   │   ├── 00006/
│   │   ├── 00007/
│   │   ├── 00008/
│   │   ├── 00010/
│   │   ├── 00011/
│   │   ├── 00012/
│   │   ├── 00013/
│   │   ├── 00014/
│   │   ├── 00015/
│   │   ├── 00016/
│   │   ├── 00017/
│   │   ├── 00018/
│   │   ├── 00019/
│   │   ├── 00020/
│   │   ├── 00021/
│   │   ├── 00022/
│   │   ├── 00023/
│   │   ├── 00024/
│   │   ├── 00025/
│   │   ├── 00026/
│   │   ├── 00027/
│   │   ├── 00028/
│   │   ├── 00029/
│   │   ├── 00030/
│   │   ├── 00031/
│   │   ├── 00032/
│   │   ├── 00033/
│   │   ├── 00034/
│   │   ├── 00035/
│   │   ├── 00036/
│   │   ├── 00037/
│   │   ├── 00039/
│   │   ├── 00040/
│   │   ├── 00041/
│   │   ├── 00042/
│   │   ├── 00043/
│   │   ├── 00044/
│   │   ├── 00045/
│   │   ├── 00046/
│   │   ├── 00047/
│   │   ├── 00048/
│   │   ├── 00049/
│   │   ├── 00050/
│   │   ├── 00051/
│   │   ├── 00052/
│   │   ├── 00053/
│   │   ├── 00054/
│   │   ├── 00055/
│   │   ├── 00056/
│   │   ├── 00057/
│   │   ├── 00058/
│   │   ├── 00059/
│   │   ├── 00060/
│   │   ├── 00061/
│   │   ├── 00062/
│   │   ├── 00063/
│   │   ├── 00064/
│   │   ├── 00065/
│   │   ├── 00066/
│   │   ├── 00067/
│   │   ├── 00068/
│   │   ├── 00070/
│   │   ├── 00071/
│   │   ├── 00072/
│   │   ├── 00073/
│   │   ├── 00074/
│   │   ├── 00075/
│   │   ├── 00076/
│   │   ├── 00077/
│   │   ├── 00078/
│   │   ├── 00079/
│   │   ├── 00080/
│   │   ├── 00082/
│   │   ├── 00083/
│   │   ├── 00084/
│   │   ├── 00085/
│   │   ├── 00086/
│   │   ├── 00087/
│   │   ├── 00088/
│   │   ├── 00089/
│   │   ├── 00090/
│   │   ├── 00091/
│   │   ├── 00092/
│   │   ├── 00093/
│   │   ├── 00094/
│   │   ├── 00095/
│   │   ├── 00096/
│   │   ├── 00097/
│   │   ├── 00098/
│   │   ├── 00099/
│   │   ├── 00100/
│   │   ├── 00101/
│   │   ├── 00102/
│   │   ├── 00103/
│   │   ├── 00104/
│   │   ├── 00105/
│   │   ├── 00106/
│   │   ├── 00107/
│   │   ├── 00108/
│   │   ├── 00109/
│   │   ├── 00110/
│   │   ├── 00111/
│   │   ├── 00112/
│   │   ├── 00113/
│   │   ├── 00114/
│   │   ├── 00115/
│   │   ├── 00116/
│   │   ├── 00117/
│   │   ├── 00118/
│   │   ├── 00119/
│   │   ├── 00120/
│   │   ├── 00121/
│   │   ├── 00122/
│   │   ├── 00124/
│   │   ├── 00125/
│   │   ├── 00126/
│   │   ├── 00127/
│   │   ├── 00128/
│   │   ├── 00129/
│   │   ├── 00130/
│   │   ├── 00131/
│   │   ├── 00132/
│   │   ├── 00133/
│   │   ├── 00134/
│   │   ├── 00135/
│   │   ├── 00136/
│   │   ├── 00137/
│   │   ├── 00138/
│   │   ├── 00139/
│   │   ├── 00140/
│   │   ├── 00141/
│   │   ├── 00142/
│   │   ├── 00143/
│   │   ├── 00144/
│   │   ├── 00145/
│   │   ├── 00146/
│   │   ├── 00147/
│   │   ├── 00148/
│   │   ├── 00149/
│   │   ├── 00150/
│   │   ├── 00151/
│   │   ├── 00152/
│   │   ├── 00153/
│   │   ├── 00154/
│   │   ├── 00155/
│   │   ├── 00156/
│   │   ├── 00157/
│   │   ├── 00158/
│   │   ├── 00159/
│   │   ├── 00160/
│   │   ├── 00161/
│   │   ├── 00162/
│   │   ├── 00164/
│   │   ├── 00165/
│   │   ├── 00166/
│   │   ├── 00167/
│   │   ├── 00168/
│   │   ├── 00169/
│   │   ├── 00170/
│   │   ├── 00171/
│   │   ├── 00172/
│   │   ├── 00173/
│   │   ├── 00174/
│   │   ├── 00175/
│   │   ├── 00176/
│   │   ├── 00177/
│   │   ├── 00178/
│   │   ├── 00179/
│   │   ├── 00180/
│   │   ├── 00182/
│   │   ├── 00183/
│   │   ├── 00184/
│   │   ├── 00185/
│   │   ├── 00186/
│   │   ├── 00187/
│   │   ├── 00188/
│   │   ├── 00189/
│   │   ├── 00190/
│   │   ├── 00191/
│   │   ├── 00192/
│   │   ├── 00193/
│   │   ├── 00194/
│   │   ├── 00195/
│   │   ├── 00197/
│   │   ├── 00198/
│   │   ├── 00199/
│   │   ├── 00200/
│   │   ├── 00201/
│   │   ├── 00202/
│   │   ├── 00203/
│   │   ├── 00204/
│   │   └── ... (更多文件夹)
│   ├── clip_224x224_16f/
│   │   └── ... (类似 clip_224x224)
│   ├── clip_224x224_avi/
│   │   └── ... (类似 clip_224x224)
│   └── original/
│       └── ... (原始视频)
├── ECSTFL/
│   └── checkpoint/
├── EmoLabel_DataSplit/
│   ├── test(single-labeled)/
│   ├── train(single-labeled)/
└── pretrained_weight/
    ├── i3d/
    ├── mc3_18/
    ├── r3d_18/
```

### 数据描述
- **README.md**: 数据集说明文档。
- **Annotation/**: 标注文件夹，包含情感标注信息。
- **Clip/**: 视频片段文件夹。
  - **clip_224x224/**: 224x224 分辨率的视频片段，每个子文件夹对应一个视频ID。
  - **clip_224x224_16f/**: 16帧版本的片段。
  - **clip_224x224_avi/**: AVI 格式的片段。
  - **original/**: 原始视频。
- **ECSTFL/**: 可能包含训练检查点。
- **EmoLabel_DataSplit/**: 数据分割文件夹，包含训练和测试集的单标签数据。
- **pretrained_weight/**: 预训练权重文件夹，包含 i3d、mc3_18、r3d_18 等模型权重。
