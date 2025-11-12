
# 多模态情感识别通用测评套件（MER Eval Kit）

> **覆盖数据集**：AffectNet、Aff-Wild2（VA/EXPR）、CH-SIMS、CMU-MOSEI、DFEW、EmotionTalk、MELD、MEMO-Bench  
> **覆盖指标**：Accuracy、Macro/Weighted F1、UAR/WAR、MAE、Pearson Corr、**CCC**、Acc-2（两种定义）、Acc-7（就地取整）  
> **适配模型**：Qwen3-Omni 等任意本地/远端多模态模型（通过 *Adapter* 接口接入）；内置随机基线 *Stub Adapter* 可直接跑通

---

## 1. 快速开始

1）**准备环境**：Python 3.9+，建议安装 `opencv-python`（视频抽帧）与 `imageio`（回退读取）。本套件默认只用标准库+`math`，如无 `opencv-python` 会自动降级到 `imageio`。  
2）**数据已本地下载**：将 `configs/*.json` 中的路径改成你的本地绝对路径或相对路径。  
3）**试跑一个数据集（以 AffectNet 分类为例）**：
```bash
python evaluate.py \
  --dataset affectnet_cls \
  --split val \
  --config configs/affectnet.json \
  --adapter adapters/qwen3_omni_stub.py \
  --output outputs/affectnet_cls_val_qwen3_stub.json
```
> 首次运行会使用内置的随机输出适配器，仅用于验证流程。你可以基于 `adapters/qwen3_omni_stub.py` 改成真实模型调用。

---

## 2. 目录结构

```
mer_eval_kit/
  evaluate.py                # 统一评测入口（命令行）
  metrics.py                 # 各类指标（含 CCC/Acc-2/Acc-7/UAR/WAR 等）
  label_spaces.py            # 各数据集的标准标签空间（7/8 类表情、MELD 情绪等）
  adapters/
    base.py                  # 适配器接口（需实现 predict(...)）
    qwen3_omni_stub.py       # 示例适配器（随机输出/读取预存文件）
  datasets/
    __init__.py
    utils.py                 # 视频抽帧/音频占位/通用工具
    affectnet.py             # AffectNet 分类 + VA 回归
    affwild2.py              # Aff-Wild2: VA（CCC）与 EXPR（F1）
    ch_sims.py               # CH-SIMS: 连续强度 + 二分类（Acc-2/F1/MAE/Corr）
    cmu_mosei.py             # CMU-MOSEI: 连续强度 + Acc-2/Acc-7/F1/MAE/Corr（需 manifest 或 MMSDK 处理过的CSV）
    dfew.py                  # DFEW: UAR/WAR + Top-1
    emotiontalk.py           # EmotionTalk（中文对话）：分类/强度 + 可选 caption（本评测不计）
    meld.py                  # MELD：Weighted-F1 + Accuracy
    memo_bench.py            # MEMO-Bench：合成画像的情绪分类
  configs/                   # 每个数据集一份配置（填写你本地路径即可）
    affectnet.json
    affwild2_va.json
    affwild2_expr.json
    ch_sims.json
    cmu_mosei.json
    dfew.json
    emotiontalk.json
    meld.json
    memo_bench.json
  outputs/                   # 评测结果（自动创建）
  README_zh.md               # 本说明
```

---

## 3. 统一适配器接口（*Adapter*）

- 所有模型只需实现一个类 `Adapter`（见 `adapters/base.py` 的抽象定义），并提供：
  ```python
  def predict(self, item: dict, task: str) -> dict:
      ...
  ```
- `task` 取值：
  - `image_emotion_class`（AffectNet/MEMO-Bench）
  - `image_va_reg`（AffectNet）
  - `video_emotion_class`（Aff-Wild2 EXPR、DFEW）
  - `video_va_reg`（Aff-Wild2 VA）
  - `mosei_sentiment`（连续强度）
  - `chsims_sentiment`（连续强度 + 二分类）
  - `meld_dialog_emotion`（对话情绪分类）
  - `emotiontalk_dialog_emotion`（中文对话情绪）
- 你可以在 `predict(...)` 里调用本地推理引擎/远端接口/预计算缓存文件。示例参见 `adapters/qwen3_omni_stub.py`。

---

## 4. 数据集读取与“官方结构”

- **AffectNet**：默认读取 `training.csv` / `validation.csv`，图像目录见配置中的 `train_img_dir/val_img_dir`。  
- **Aff-Wild2**：
  - **VA**：默认读取 `annotations/VA_Estimation_Challenge/*/valence/<video>.txt` 与 `arousal/<video>.txt`，视频或帧目录可在配置中指定（`video_dir` 或 `frames_dir`）。
  - **EXPR**：默认读取 `annotations/EXPR_Recognition_Challenge/Train_Set/labels.csv` / `Val_Set/labels.csv`，或等价的 `manifest.jsonl`（每行一个样本：`{"video_path": "...", "label": "happy"}`）。
- **CH-SIMS**：默认读取官方提供的 `train/dev/test.csv`（或 `jsonl`），包含 `text`、`video_path`（可选）、`label`（[-1,1]）。
- **CMU-MOSEI**：建议使用已由 MMSDK/社区脚本导出的 `manifest.csv/jsonl`（列含 `video_path,text,label`）；若直接使用 `.csd`，请先用 MMSDK 生成 CSV（本套件预留钩子，不强制 MMSDK 依赖）。
- **DFEW**：默认读取官方 `annotations/DFEW_*.csv` 或 5-Fold 的 `fold{1..5}.csv`（两列：`video_path,label`）。
- **MELD**：默认读取 `train_sent_emo.csv/dev_sent_emo.csv/test_sent_emo.csv`（列含 `Emotion, Utterance, Dialogue_ID, Utterance_ID`），视频/音频相对路径按官方 `MELD.Raw/` 组织推导。
- **EmotionTalk**：默认读取 `train/dev/test.jsonl`（每行含 `text/audio_path/video_path/label` 的任意子集）。
- **MEMO-Bench**：读取 `metadata.csv`（`image_path,emotion`），用于 6 类情绪识别。

> **若你的本地结构与以上略有差异**：请直接在对应 `configs/*.json` 中修改路径或指明 `manifest` 文件。**评测脚本不修改任何原始文件**。

---

## 5. 报告与指标

- 按论文/竞赛的通行做法生成：
  - AffectNet：**7/8 类 Acc** + **V/A：CCC + RMSE**；
  - Aff-Wild2：**V/A：CCC（V/A 平均）**；**EXPR：Macro-F1**；
  - CH-SIMS：**Acc-2（neg-vs-nonneg / neg-vs-pos）+ F1 + MAE + Corr**；
  - MOSEI：**Acc-2（双定义）+ F1 + MAE + Corr + Acc-7**；
  - DFEW：**UAR + WAR + Top-1 Acc**；
  - MELD/EmotionTalk：**Weighted-F1 + Accuracy**；
  - MEMO-Bench：**Accuracy + Macro-F1**。

- 结果将写入 `--output` 指定的 JSON，并在控制台打印人类可读的汇总表。

---

## 6. 小贴士

- 视频类任务如你的模型只支持“整段视频”输入，评测器会把 `video_path` 直接交给适配器；
  如模型仅支持“逐帧图片”，可在 `Adapter.predict` 里用 `datasets/utils.py` 的 `sample_video_frames(...)` 抽帧后逐帧推理并在适配器端做**多数投票/平均概率**聚合。
- 如果你已有**预计算预测**（如批量推理写成了 JSONL），可在 `qwen3_omni_stub.py` 中切换到 `PrecomputedAdapter`，无需再次调用模型。

---

## 7. 许可证与引用

- 本评测脚本仅用于研究用途。数据集请遵循各自许可证与使用条款。
- 论文/竞赛指标定义参考：AffectNet、Aff-Wild2（ABAW）、CH-SIMS、CMU-MOSEI、DFEW、MELD、EmotionTalk、MEMO-Bench 的官方与后续标准实践。
