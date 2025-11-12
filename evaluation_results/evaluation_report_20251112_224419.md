# Qwen3-Omni 多模态情感分析评估报告

**评估时间**: 2025年11月12日 22:44:19

**模型**: qwen3-omni

**服务地址**: http://localhost:8080/v1

**最大样本数**: 10

## 总体统计

- 总数据集数: 12
- 完全成功: 12
- 部分成功: 0
- 失败: 0
- 成功率: 100.0%

## 数据集类型统计

- **分类任务**: 6 个数据集
- **回归任务**: 2 个数据集
- **多模态任务**: 4 个数据集

## 详细评估结果

### ✅ AffectNet_Classifier

**状态**: success

**样本数**: 10

**评估指标**:

- 准确率指标: acc=0.2000
- F1指标: macro_f1=0.0556, weighted_f1=0.0667
- 其他指标: total_samples=10

---

### ✅ AffectNet_VA

**状态**: success

**样本数**: 10

**评估指标**:

- CCC指标: valence_ccc=0.0000, arousal_ccc=0.0000, mean_ccc=0.0000
- MAE指标: valence_mae=0.3224, arousal_mae=0.4549
- 相关系数: valence_corr=0.0000, arousal_corr=0.0000
- 其他指标: total_samples=10

---

### ✅ AffWild2_VA

**状态**: success

**样本数**: 0

**评估指标**:

- CCC指标: valence_ccc=0.0000, arousal_ccc=0.0000, mean_ccc=0.0000
- MAE指标: valence_mae=0.4107, arousal_mae=0.4055
- 相关系数: valence_corr=0.0000, arousal_corr=0.0000
- 其他指标: total_frames=35243

---

### ✅ AffWild2_EXPR

**状态**: success

**样本数**: 10

**评估指标**:

- 准确率指标: acc=0.3000
- F1指标: macro_f1=0.0923, weighted_f1=0.1385
- 其他指标: total_samples=10

---

### ✅ EmotionTalk

**状态**: success

**样本数**: 10

**评估指标**:

- 准确率指标: acc=1.0000
- F1指标: weighted_f1=1.0000, macro_f1=1.0000
- 其他指标: total_samples=10

---

### ✅ MELD

**状态**: success

**样本数**: 10

**评估指标**:

- 准确率指标: acc=0.2000
- F1指标: weighted_f1=0.0667, macro_f1=0.0833
- 其他指标: total_samples=10, emotion_distribution={'surprise': 1, 'anger': 1, 'neutral': 2, 'happy': 6}, prediction_distribution={'neutral': 10}

---

### ✅ MEMO_Bench

**状态**: success

**样本数**: 10

**评估指标**:

- 准确率指标: acc=0.2000
- F1指标: macro_f1=0.0556, weighted_f1=0.0667
- 其他指标: total_samples=10, emotion_distribution={'sad': 1, 'neutral': 2, 'happy': 2, 'anger': 3, 'surprise': 1, 'fear': 1}, prediction_distribution={'neutral': 10}

---

### ✅ CH_SIMS

**状态**: success

**样本数**: 10

**评估指标**:

- 准确率指标: acc2_neg_nonneg=0.6000, acc2_neg_pos=0.5000, acc_binary=0.6000
- F1指标: f1_binary_macro=0.3750
- MAE指标: mae=0.6600
- 相关系数: corr=0.0000
- 其他指标: total_samples=10

---

### ✅ CH_SIMS_V2

**状态**: success

**样本数**: 10

**评估指标**:

- 准确率指标: acc2_neg_nonneg=0.6000, acc2_neg_pos=0.5000, acc_binary=0.6000
- F1指标: f1_binary_macro=0.3750
- MAE指标: mae=0.6600
- 相关系数: corr=0.0000
- 其他指标: total_samples=10

---

### ✅ CMU_MOSEI

**状态**: success

**样本数**: 10

**评估指标**:

- 准确率指标: acc2_neg_nonneg=0.8000, acc2_neg_pos=0.2500, acc7=0.6000, acc_binary=0.8000
- F1指标: f1_binary_macro=0.4444
- MAE指标: mae=0.6000
- 相关系数: corr=0.0000
- 其他指标: total_samples=10

---

### ✅ CMU_MOSI

**状态**: success

**样本数**: 10

**评估指标**:

- 准确率指标: acc2_neg_nonneg=0.2000, acc2_neg_pos=0.8889, acc7=0.1000, acc_binary=0.2000
- F1指标: f1_binary_macro=0.1667
- MAE指标: mae=1.3200
- 相关系数: corr=0.0000
- 其他指标: total_samples=10

---

### ✅ DFEW

**状态**: success

**样本数**: 10

**评估指标**:

- 准确率指标: top1_acc=0.2000
- F1指标: macro_f1=0.0667, weighted_f1=0.0667
- 其他指标: war=0.2, uar=0.14285714285714285, total_samples=10, fold=1, emotion_distribution={'sad': 3, 'neutral': 2, 'anger': 2, 'fear': 1, 'surprise': 2}, prediction_distribution={'neutral': 10}, label_space=['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

---

## 性能汇总

### 分类任务准确率

| 数据集 | 最佳准确率 |
|--------|------------|
| AffectNet_Classifier | 0.2000 |
| AffWild2_EXPR | 0.3000 |
| EmotionTalk | 1.0000 |
| MELD | 0.2000 |
| MEMO_Bench | 0.2000 |
| DFEW | 0.2000 |

### 回归任务平均CCC

| 数据集 | 平均CCC |
|--------|---------|
| AffectNet_VA | 0.0000 |
| AffWild2_VA | 0.0000 |

## 问题与建议

### 改进建议

1. **视频处理优化**: 考虑使用更智能的视频帧采样策略，如关键帧检测
2. **错误处理**: 改进类型错误处理，特别是字符串和整数的混合运算
3. **内存管理**: 对于大规模评估，考虑分批处理以避免内存溢出
4. **并行处理**: 可以考虑并行化多个数据集的评估过程
