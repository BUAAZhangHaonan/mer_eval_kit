# 已有数据集统计
## AffectNet
### 类型
人脸表情数据集

包含**图像**和对应的三类**标注**

### 路径
```
整体文件路径：/home/remote1/lvshuyang/Datasets/AffectNet
训练集路径：/home/remote1/lvshuyang/Datasets/AffectNet/train_set
训练集标注文件：/home/remote1/lvshuyang/Datasets/AffectNet/train_set/annotations
训练集图片文件：/home/remote1/lvshuyang/Datasets/AffectNet/train_set/images
验证集路径：/home/remote1/lvshuyang/Datasets/AffectNet/val_set
验证集图片文件：/home/remote1/lvshuyang/Datasets/AffectNet/val_set/images
验证集标注文件：/home/remote1/lvshuyang/Datasets/AffectNet/val_set/annotations
```


### 介绍
`annotations`下为标注文件,需要利用 `Python` 的 `numpy` 库来读取这些文件

命名为在描述的图片名称后加相应的后缀

| 列名 | 描述 | 数据类型 |
|------|------|----------|
| Expression | 面部表情ID (0-7) | 整数 |
| Valence | 效价值 [-1,+1] | 浮点数 |
| Arousal | 唤醒度值 [-1,+1] | 浮点数 |


    _aro.npy：人脸坐标（x 和 y）信息

    _exp.npy：面部的表情ID
    0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt

    _lnd.npy：表情的效价值，范围在 [-1, +1] 之间（对于'不确定'和'无人脸'类别，该值为 -2）

    _val.npy：表情的唤醒度值，范围在 [-1, +1] 之间（对于'不确定'和'无人脸'类别，该值为 -2）

`images`下为图像文件

AffectNet8Labels_Documentation_OneDrive_March2021.pdf下是官方数据集介绍


## Affwild2
### 类型
包含视频，音频

### 路径
```
数据集总体路径：/home/remote1/lvshuyang/Datasets/Affwild2

训练集AU标注文件路径：/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/AU_Detection_Challenge/Train_Set

验证集AU标注文件路径：/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/AU_Detection_Challenge/Validation_Set

训练集EXP标注文件路径：/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/EXPR_Recognition_Challenge/Train_Set

验证集EXP标注文件路径：/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set

训练集VA标注文件路径：/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/VA_Estimation_Challenge/Train_Set

验证集VA标注文件路径：/home/remote1/lvshuyang/Datasets/Affwild2/ABAW Annotations/VA_Estimation_Challenge/Validation_Set

视频存放路径：
/home/remote1/lvshuyang/Datasets/Affwild2/batch1
/home/remote1/lvshuyang/Datasets/Affwild2/batch2
/home/remote1/lvshuyang/Datasets/Affwild2/batch3

```
### 介绍

#### VA_Estimation_Challenge

它包含两个子文件夹：`Train_Set`、`Validation_Set`。每个标注文件为txt文件，以其对应的视频命名。

每个标注文件的第一行是：
`valence`,`arousal`

此后的每一行显示与每个视频帧对应的逗号分隔的效价和唤醒度值，即，第一个效价-唤醒度值对应第一个视频帧，第二个效价-唤醒度值对应第二个视频帧，依此类推。

请注意，效价和唤醒度的取值范围为：[-1,1]。对于某些帧，效价和/或唤醒度的值可能为-5；此类帧应被忽略。

#### EXPR_Recognition_Challenge

它包含两个子文件夹：Train_Set、Validation_Set。每个标注文件为txt文件，以其对应的视频命名。

每个标注文件的第一行总是：
`Neutral`,`Anger`,`Disgust`,`Fear`,`Happiness`,`Sadness`,`Surprise`,`Other`

此后的每一行显示一个在 {0,1,2,3,4,5,6,7} 范围内的标注值；这些值对应于情绪：{中性, 愤怒, 厌恶, 恐惧, 快乐, 悲伤, 惊讶, 其他}。每个值对应于每个视频帧，即第一个值对应第一个视频帧，第二个值对应第二个视频帧，依此类推。

对于某些帧，标注值可能为-1；此类帧应被忽略。

#### AU_Detection_Challenge

包含动作单元检测的标注（应该用不上）


## Emotiontalk

### 类型
视频，音频，文本三模态的数据集

### 路径
```
根目录：

数据集总体路径：/home/remote1/lvshuyang/Datasets/Emotiontalk

文本模态及标注路径：/home/remote1/lvshuyang/Datasets/Emotiontalk/Text/json

音频模态标注路径：/home/remote1/lvshuyang/Datasets/Emotiontalk/Audio/json

视频模态标注路径：/home/remote1/lvshuyang/Datasets/Emotiontalk/Video/json

音频存放路径：/home/remote1/lvshuyang/Datasets/Emotiontalk/Audio/wav

视频存放路径：/home/remote1/lvshuyang/Datasets/Emotiontalk/Multimodal/mp4
```
具体的json文件路径存在多个层级
如文件`G00001_02_01_002.json`存放在
/home/remote1/lvshuyang/Datasets/Emotiontalk/Audio/json/G00001/G00001_02/G00001_02_01下

在json文件中，
`file_name`（文件路径）中存放的是从上面的根目录算起的相对文件路径

### 介绍
标注介绍：
  * **文本模态：** `data`（每位标注者的标注结果）、`emotion_result`（情感结果）、`speaker_id`（说话者ID）、`file_name`（文件路径）、`content`（转写文本）。
  * **音频模态：** `data`（每位标注者的标注结果）、`emotion_result`（情感结果）、`speaker_id`（说话者ID）、`paragraphs`（时间戳）、`sourceAttr`（描述）、`file_name`（文件路径）、`content`（转写文本）。
  * **视频模态：** `data`（每位标注者的标注结果）、`emotion_result`（情感结果）、`speaker_id`（说话者ID）、`file_name`（文件路径）。
  * **多模态：** `data`（每位标注者的标注结果）、`emotion_result`（情感结果）、`Continuous label_result`（连续标签结果）、`speaker_id`（说话者ID）、`file_name`（文件路径）。


## MELD
### 类型
来自<老友记>的视频数据集
包含视频，文本，情感标注

### 路径
```
数据集总体路径：/home/remote1/lvshuyang/Datasets/MELD
训练集标注文件路径：/home/remote1/lvshuyang/Datasets/MELD/train_sent_emo.csv
测试集标注文件路径：/home/remote1/lvshuyang/Datasets/MELD/test_sent_emo.csv
验证集标注文件路径：/home/remote1/lvshuyang/Datasets/MELD/dev_sent_emo.csv
训练集视频路径：/home/remote1/lvshuyang/Datasets/MELD/train_splits
测试集视频路径：/home/remote1/lvshuyang/Datasets/MELD/output_repeated_splits_test
验证集视频路径：/home/remote1/lvshuyang/Datasets/MELD/dev_splits_complete
```
### 介绍
### 视频文件
文件命名规则为`diax_utty.mp4`，其中x为Dialogue_ID，对话索引数，y为Utterance_ID，特定话语在对话中的索引
#### csv文件

`dev_sent_emo.csv`，`test_sent_emo.csv`，`train_sent_emo.csv`包含了所有标注信息，具体如下表

**列说明**

| 列名| 描述 |
|-------------|------|
| Sr No. | 话语的序列号，主要用于在不同版本或具有不同子集的多个副本中引用话语。 |
| Utterance | 来自 EmotionLines 的单个话语字符串。 |
| Speaker | 与话语关联的说话者姓名。 |
| Emotion | 说话者在该话语中表达的情感（中性、高兴、悲伤、愤怒、惊讶、恐惧、厌恶）。 |
| Sentiment | 说话者在该话语中表达的情感极性（正面、中性、负面）。 |
| Dialogue_ID | 对话的索引，从 0 开始。 |
| Utterance_ID | 特定话语在对话中的索引，从 0 开始。 |
| Season | 话语所属的《老友记》季度编号。 |
| Episode | 话语所属的特定季度中的剧集编号。 |
| StartTime | 话语在给定剧集中的开始时间，格式为'时:分:秒,毫秒'。 |
| EndTime | 话语在给定剧集中的结束时间，格式为'时:分:秒,毫秒'。 |



## MEMO-Bench
### 类型
人脸图像（AI生成）

### 路径
```
数据集总体路径：/home/remote1/lvshuyang/Datasets/MEMO-Bench

图片路径：/home/remote1/lvshuyang/Datasets/MEMO-Bench/dataset
```
### 介绍
情感类别：六种（happy、sad、angry、surprise、worry、neutral）

* annotation_emo.csv：每张 AI 生成人像图像的主观标注数据，包括情感类型和情感程度；
  
  包含`Image`，`Score`，`Distortions`三列，分别为图片名称，情感程度，图片情感标注

* annotation_quality.csv：每张 AI 生成人像图像的主观标注数据，包括图像质量。
  包含`Image`，`Quality`两列，分别为图片名称，图像质量
  

## CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity)

### 类型
多模态 (视频, 音频, 文本), 情感与情绪标签

### 路径
```
数据集总体路径：/home/remote1/lvshuyang/Datasets/CMU-MOSEI

# --- 标注 & 文本数据 ---

# 主标签文件，包含文本、情感标注和数据划分
/home/remote1/lvshuyang/Datasets/CMU-MOSEI/label.csv


# --- 原始多媒体数据 ---

# 包含原始视频片段的文件夹 (以video_id命名)
# 示例: /home/remote1/lvshuyang/Datasets/CMU-MOSEI/Raw/_0efYOjQYRc
/home/remote1/lvshuyang/Datasets/CMU-MOSEI/Raw/


# --- 预处理对齐特征 ---

# 包含【对齐的】多模态特征
/home/remote1/lvshuyang/Datasets/CMU-MOSEI/Processed/aligned.pkl

# 包含【非对齐的】多模态特征
/home/remote1/lvshuyang/Datasets/CMU-MOSEI/Processed/unaligned.pkl
```


### 介绍

#### 1. 主标签文件 (`label.csv`)

该文件是数据集的核心，提供了每个视频片段的文本、情感/情绪标注以及数据划分信息。

文件各列描述如下 (`video_id,clip_id,text,label,annotation,mode,label_T,label_A,label_V`)：
- **video_id**: 原始 YouTube 视频的唯一标识符。
- **clip_id**: 在该 `video_id` 内的片段编号。
- **text**: 从视频片段中转录出的人声文本内容。
- **label**: **主要的情感极性分数 (Sentiment Score)**。这是一个**连续值**，通常范围在 `[-3, 3]` 之间，表示情感的强度。负数代表负面情感，正数代表正面情感。
- **annotation**: 基于 `label` 分数生成的**分类标签** (`Positive`, `Neutral`, `Negative`)。
- **mode**: 数据划分，用于标识该样本属于 `train` (训练集), `validation` (验证集) 还是 `test` (测试集)。
- **label_T, label_A, label_V**: 用于其他研究任务的标签（如纯文本情感、唤醒度Arousal、效价Valence），在主要的情感分析任务中可能为空。

#### 2. 原始视频数据 (`Raw/`)

此文件夹存放了从 YouTube 下载的原始视频片段。每个文件名对应 `label.csv` 中的一个 `video_id`。

#### 3. 预处理特征 (`Processed/`)

该文件夹包含从原始数据中提取并处理好的多模态特征，以 Python Pickle (`.pkl`) 格式存储，可以直接加载用于模型训练，**极大地方便了研究工作**。

- **`aligned.pkl` (对齐的特征)**
  - **含义**: "对齐" 指的是来自文本、音频、视频三种模态的特征序列**在时间上被强制同步**，并被处理成了**相同的长度**。序列中的第 `i` 个元素对应着所有模态在同一时刻的特征。
  - **优势**: 简化了需要在每个时间步进行特征融合的模型（早期融合）的设计，便于学习细粒度的跨模态瞬时交互。
  - **类比**: 就像一部带有完美同步字幕的电影，字幕、画面和声音在每一帧都是对齐的。

- **`unaligned.pkl` (非对齐的特征)**
  - **含义**: "非对齐" 指的是数据保留了其**原始的、独立的时间序列**。三种模态的特征序列有着**各自不同的长度**，没有进行同步处理。例如，文本特征的长度是单词的数量，而视频特征的长度是视频的总帧数。
  - **优势**: 保留了每种模态最原始的时间信息，没有因对齐而产生信息损失，更接近真实世界的场景。适用于先独立处理各模态再进行融合的模型（晚期融合）。
  - **类比**: 就像一本书和它对应的有声读物，内容相同但你拥有的是两个独立的文件，没有直接的时间戳关联。


## CMU-MOSI (Multimodal Opinion-level Sentiment Intensity)

### 类型
多模态 (视频, 音频, 文本), 情感标签

### 路径
```
数据集总体路径：/home/remote1/lvshuyang/Datasets/CMU-MOSI

# --- 标注 & 文本数据 ---

# 主标签文件
/home/remote1/lvshuyang/Datasets/CMU-MOSI/label.csv


# --- 原始多媒体数据 ---

# 包含原始视频片段的文件夹 (以video_id命名)
# 示例: /home/remote1/lvshuyang/Datasets/CMU-MOSI/Raw/_dI--eQ6qVU
/home/remote1/lvshuyang/Datasets/CMU-MOSI/Raw/


# --- 预处理对齐特征 ---

# 包含【对齐的】多模态特征
/home/remote1/lvshuyang/Datasets/CMU-MOSI/Processed/aligned.pkl

# 包含【非对齐的】多模态特征
/home/remote1/lvshuyang/Datasets/CMU-MOSI/Processed/unaligned.pkl
```
### 介绍

#### 1. 主标签文件 (`label.csv`)

作为专注于情感强度分析的经典数据集，其结构与 MOSEI 类似。

文件各列描述如下 (`video_id,clip_id,text,label,label_T,label_A,label_V,annotation,mode`)：
- **video_id**: 原始 YouTube 视频的唯一标识符。
- **clip_id**: 在该 `video_id` 内的片段编号。
- **text**: 从视频片段中转录出的人声文本内容。
- **label**: **情感极性分数 (Sentiment Score)**。这是一个范围在 `[-3, 3]` 的**连续值**。-3 代表非常负面，+3 代表非常正面。
- **label_T, label_A, label_V**: 用于其他研究任务的标签，在主要情感分析任务中可能为空。
- **annotation**: 基于 `label` 分数生成的**分类标签** (`Positive`, `Neutral`, `Negative`)。
- **mode**: 数据划分 (`train`, `validation`, `test`)。

#### 2. 原始视频数据 (`Raw/`)

此文件夹存放了原始的视频片段，与 `label.csv` 中的条目一一对应。

#### 3. 预处理特征 (`Processed/`)

该文件夹包含可以直接用于模型开发的 `.pkl` 特征文件。

- **`aligned.pkl` (对齐的特征)**
  - **含义**: "对齐" 指的是来自文本、音频、视频三种模态的特征序列**在时间上被强制同步**，并被处理成了**相同的长度**。序列中的第 `i` 个元素对应着所有模态在同一时刻的特征。
  - **优势**: 简化了需要在每个时间步进行特征融合的模型（早期融合）的设计，便于学习细粒度的跨模态瞬时交互。
  - **类比**: 就像一部带有完美同步字幕的电影，字幕、画面和声音在每一帧都是对齐的。

- **`unaligned.pkl` (非对齐的特征)**
  - **含义**: "非对齐" 指的是数据保留了其**原始的、独立的时间序列**。三种模态的特征序列有着**各自不同的长度**，没有进行同步处理。例如，文本特征的长度是单词的数量，而视频特征的长度是视频的总帧数。
  - **优势**: 保留了每种模态最原始的时间信息，没有因对齐而产生信息损失，更接近真实世界的场景。适用于先独立处理各模态再进行融合的模型（晚期融合）。
  - **类比**: 就像一本书和它对应的有声读物，内容相同但你拥有的是两个独立的文件，没有直接的时间戳关联。


## DFEW (Dynamic Facial Expression in-the-Wild)

### 类型
包含视频、预处理帧、模型权重

### 路径
```
数据集总体路径：/home/remote1/lvshuyang/Datasets/DFEW

# --- 标注文件 & 数据划分 ---

# 七维情感分布标注路径 (包含所有16372个视频的完整标注)
/home/remote1/lvshuyang/Datasets/DFEW/Annotation/annotation.xlsx

# 训练集划分标注文件路径 (五折交叉验证)
/home/remote1/lvshuyang/Datasets/DFEW/EmoLabel_DataSplit/train(single-labeled)/set_1.csv
# ...以此类推至 set_5.csv

# 验证集划分标注文件路径 (五折交叉验证)
/home/remote1/lvshuyang/Datasets/DFEW/EmoLabel_DataSplit/test(single-labeled)/set_1.csv
# ...以此类推至 set_5.csv


# --- 视频 & 图像帧数据 ---

# 原始视频片段路径 (mp4格式)
# 示例: /home/remote1/lvshuyang/Datasets/DFEW/Clip/original/1.mp4
/home/remote1/lvshuyang/Datasets/DFEW/Clip/original/

# 预处理后的视频路径 (avi格式, 224x224)
# 示例: /home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224_avi/00001.avi
/home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224_avi/

# 预处理后的图像帧路径 (无时序插值, 224x224)
# 示例: /home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224/00001/00001_00001.jpg
/home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224/

# 预处理后的图像帧路径 (带时序插值, 16帧, 224x224)
# 示例: /home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224_16f/00001/1.jpg
/home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224_16f/


# --- 模型权重 ---

# EC-STFL 模型权重
/home/remote1/lvshuyang/Datasets/DFEW/ECSTFL/checkpoint/r3d_dfew.pth

# 预训练I3D模型权重
/home/remote1/lvshuyang/Datasets/DFEW/pretrained_weight/i3d/i3d_fold1_epo398_UAR48.44_WAR59.63.pth

```
### 介绍

#### 1. 七维情感分布标注 (`annotation.xlsx`)

该文件为数据集中所有 16372 个视频片段提供了完整的情感分布标注。

文件中的每一行格式如下：
`(vote_happy) (vote_sad) (vote_neutral) (vote_angry) (vote_surprise) (vote_disgust) (vote_fear) (clip_name) (annotation)`

- **vote_...**: 代表10位标注者中有多少人将该片段标注为对应的情绪。
- **clip_name**: 视频片段的文件名。
- **annotation**: 根据投票结果确定的单标签类别。

`annotation` 列的标签定义如下：
- **0**: 非单标签 (没有任何一个情绪的票数超过阈值6)
- **1**: Happy (高兴)
- **2**: Sad (悲伤)
- **3**: Neutral (中性)
- **4**: Angry (愤怒)
- **5**: Surprise (惊讶)
- **6**: Disgust (厌恶)
- **7**: Fear (恐惧)

#### 2. 单标签五折交叉验证划分 (`EmoLabel_DataSplit`)

该目录包含了论文中进行模型评估所使用的五折交叉验证数据。这些数据是从所有片段中筛选出的 11697 个具有明确单标签的视频片段（原始共有12059个，其中362个因无法检测到人脸而被剔除）。

- **train(single-labeled)/**: 包含5个 `csv` 文件 (`set_1.csv` 到 `set_5.csv`)，分别对应五次交叉验证中的训练集。
- **test(single-labeled)/**: 包含5个 `csv` 文件 (`set_1.csv` 到 `set_5.csv`)，分别对应五次交叉验证中的测试集。

每个 `csv` 文件包含视频片段名和对应的情感标签。标签定义如下：
- **1**: Happy (高兴)
- **2**: Sad (悲伤)
- **3**: Neutral (中性)
- **4**: Angry (愤怒)
- **5**: Surprise (惊讶)
- **6**: Disgust (厌恶)
- **7**: Fear (恐惧)


## CH-SIMS
### 类型
每个视频片段提供了独立的单模态标注（文本、音频和视觉）标注，包含 2,281 个视频片段。

### 路径
```
数据集整体路径：/home/remote1/lvshuyang/Datasets/CH-SIMS
标注文件：/home/remote1/lvshuyang/Datasets/CH-SIMS/label.csv
视频文件根目录：/home/remote1/lvshuyang/Datasets/CH-SIMS/Raw
示例视频文件：/home/remote1/lvshuyang/Datasets/CH-SIMS/Raw/video_0001/0001.mp4
未对齐的多模态数据：/home/remote1/lvshuyang/Datasets/CH-SIMS/Processed/unaligned_39.pkl
```
### 介绍
标注介绍
| 列名| 描述 |
|-------------|------|
| video_id | 视频的id，同时也是存储该视频的文件夹名称 |
| clip_id | 切片id，对应在该视频的文件夹下的视频名称 |
| text | 文本 |
| label | 情感标签，值为{-1.0，-0.8，-0.6，-0.4，-0.2，0.0，0.2，0.4，0.6，0.8，1.0} |
| label_T | 文本单模态的情感标注，{-1.0，-0.8，-0.6，-0.4，-0.2，0.0，0.2，0.4，0.6，0.8，1.0} |
| label_A | 音频的单模态情感标注，{-1.0，-0.8，-0.6，-0.4，-0.2，0.0，0.2，0.4，0.6，0.8，1.0} |
| label_V | 视频的单模态情感标注，{-1.0，-0.8，-0.6，-0.4，-0.2，0.0，0.2，0.4，0.6，0.8，1.0} |
| annotation | 情感极性标注，值为[Negative，Positive，Neutral] |
| mode | 所属的mode，分为训练集train，测试集test和验证集valid |

## CH-SIMS v2(s)
### 类型
4,402 个高质量、完整标注的视频片段，用于监督学习。

包含视频，文本，音频和多模态四种标注信息
### 路径
```
数据集整体路径：/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(s)
标注文件：/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(s)/meta.csv
视频文件根目录：/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(s)/Raw
示例视频文件：/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(s)/Raw/aqgy3_0001/00000.mp4
未对齐的多模态数据：/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(s)/Processed/unaligned.pkl
```
### 介绍
标注介绍
| 列名| 描述 |
|-------------|------|
| video_id | 视频的id，同时也是存储该视频的文件夹名称 |
| clip_id | 切片id，对应在该视频的文件夹下的视频名称 |
| text | 文本 |
| label | 情感标签，值为{-1.0，-0.8，-0.6，-0.4，-0.2，0.0，0.2，0.4，0.6，0.8，1.0} |
| label_T | 文本单模态的情感标注，{-1.0，-0.8，-0.6，-0.4，-0.2，0.0，0.2，0.4，0.6，0.8，1.0} |
| label_A | 音频的单模态情感标注，{-1.0，-0.8，-0.6，-0.4，-0.2，0.0，0.2，0.4，0.6，0.8，1.0} |
| label_V | 视频的单模态情感标注，{-1.0，-0.8，-0.6，-0.4，-0.2，0.0，0.2，0.4，0.6，0.8，1.0} |
| annotation | 情感极性标注，值为[Negative，Positive，Neutral] |
| mode | 所属的mode，分为训练集train，测试集test和验证集valid |

## CH-SIMS v2(u)
### 类型
10,161 个未标注的原始视频片段，用于半监督学习、无监督学习等

### 路径
```
数据集整体路径：/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(u)
标注文件（标注列为空）：/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(u)/meta.csv
视频文件根目录：/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(u)/Raw
示例视频文件：/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(u)/Raw/aqgy1_0001/00000.mp4
未对齐的多模态数据：/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(u)/Processed/unaligned.pkl
```
### 介绍
为未标注的数据。


