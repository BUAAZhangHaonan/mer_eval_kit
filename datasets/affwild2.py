# -*- coding: utf-8 -*-
"""
datasets/affwild2.py
—— Aff-Wild2 的 VA（逐帧 CCC）与 EXPR（8类表情 Macro-F1）。
根据实际数据集格式，读取TXT格式的标注文件和对应视频文件。
"""
import os
import glob
import logging
from typing import Dict, Iterator, List
from dataclasses import dataclass

from datasets.base import BaseDataset, ClassificationDataset, RegressionDataset, DatasetConfig
from datasets.label_spaces import AFFWILD2_EXPR_7, AFFWILD2_EXPR_8, normalize_label
from datasets.metrics import precision_recall_f1, accuracy, ccc, mae, pearsonr

logger = logging.getLogger(__name__)

# 表情标签映射
EXPR_LABELS = ["Neutral", "Anger", "Disgust", "Fear",
               "Happiness", "Sadness", "Surprise", "Other"]


@dataclass
class AffWild2VAConfig(DatasetConfig):
    """AffWild2 VA数据集配置"""
    annotation_dir: str = "ABAW Annotations/VA_Estimation_Challenge"
    video_dirs: List[str] = None  # 视频存放目录列表
    split: str = "val"
    num_frames: int = 16

    def __post_init__(self):
        if self.video_dirs is None:
            self.video_dirs = ["batch1", "batch2", "batch3"]


@dataclass
class AffWild2ExprConfig(DatasetConfig):
    """AffWild2 EXPR数据集配置"""
    annotation_dir: str = "ABAW Annotations/EXPR_Recognition_Challenge"
    video_dirs: List[str] = None  # 视频存放目录列表
    split: str = "val"
    label_space_8: bool = True  # True: 8类（含 Other），False: 7类

    def __post_init__(self):
        if self.video_dirs is None:
            self.video_dirs = ["batch1", "batch2", "batch3"]


class AffWild2VA(RegressionDataset):
    """Aff-Wild2 的 Valence/Arousal 逐帧 CCC 评测。"""

    @property
    def config_class(self):
        return AffWild2VAConfig

    def _validate_sample(self, sample: Dict) -> bool:
        """允许逐帧序列样本通过校验。"""
        if not BaseDataset._validate_sample(self, sample):
            return False
        if "valence_seq" in sample and "arousal_seq" in sample:
            return True
        return super()._validate_sample(sample)

    def _get_required_paths(self) -> List[str]:
        """返回AffWild2 VA数据集必要的路径"""
        paths = [
            self.config.annotation_dir,
            f"{self.config.annotation_dir}/Train_Set",
            f"{self.config.annotation_dir}/Validation_Set"
        ]
        paths.extend(self.config.video_dirs)
        return paths

    def _find_video_file(self, video_name: str) -> str:
        """查找视频文件"""
        for video_dir in self.config.video_dirs:
            # 尝试不同的扩展名
            for ext in ['.mp4', '.avi', '.mov']:
                video_path = os.path.join(
                    self.root, video_dir, f"{video_name}{ext}")
                if os.path.exists(video_path):
                    return video_path
        return ""

    def _load_va_annotations(self, annotation_file: str) -> tuple:
        """加载VA标注文件"""
        valence_vals = []
        arousal_vals = []

        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()

            # 跳过第一行（标题行）
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        val = float(parts[0])
                        aro = float(parts[1])
                        # 跳过无效值（-5）
                        if val != -5 and aro != -5:
                            valence_vals.append(max(-1.0, min(1.0, val)))
                            arousal_vals.append(max(-1.0, min(1.0, aro)))
                except (ValueError, IndexError):
                    continue

        except Exception as e:
            logger.warning(f"读取VA标注文件失败 {annotation_file}: {e}")

        return valence_vals, arousal_vals

    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代VA样本"""
        split_dir = "Train_Set" if split == "train" else "Validation_Set"
        annotation_root = os.path.join(
            self.root, self.config.annotation_dir, split_dir)

        if not os.path.exists(annotation_root):
            logger.error(f"标注目录不存在: {annotation_root}")
            return

        # 获取所有标注文件
        annotation_files = glob.glob(os.path.join(annotation_root, "*.txt"))

        sample_count = 0
        for ann_file in annotation_files:
            video_name = os.path.splitext(os.path.basename(ann_file))[0]

            # 加载VA标注
            valence_vals, arousal_vals = self._load_va_annotations(ann_file)

            if not valence_vals or not arousal_vals:
                continue

            # 查找视频文件
            video_path = self._find_video_file(video_name)

            yield {
                "id": f"affwild2_va_{split}_{sample_count}",
                "video_name": video_name,
                "video_path": video_path,
                "valence_seq": valence_vals,
                "arousal_seq": arousal_vals,
                "num_frames": len(valence_vals)
            }

            sample_count += 1

    def evaluate(self, adapter, split: str = "val") -> Dict:
        """评估VA性能"""
        v_true_all, v_pred_all = [], []
        a_true_all, a_pred_all = [], []

        for item in self._safe_iter_samples(split):
            # 构造输入项
            inp = {
                "id": item["id"],
                "video_path": item["video_path"],
                "video_name": item["video_name"],
                "num_frames": self.config.num_frames
            }

            pred = adapter.predict(inp, task="video_va_reg")

            # 如果预测是序列，使用序列；否则重复单值
            v_pred = pred.get("valence", 0.0)
            a_pred = pred.get("arousal", 0.0)

            if isinstance(v_pred, list):
                v_pred_list = v_pred[:len(item["valence_seq"])]
                a_pred_list = a_pred[:len(item["arousal_seq"])] if isinstance(
                    a_pred, list) else [a_pred] * len(v_pred_list)
            else:
                # 单值重复到序列长度
                seq_len = min(len(item["valence_seq"]),
                              len(item["arousal_seq"]))
                v_pred_list = [v_pred] * seq_len
                a_pred_list = [a_pred] * seq_len

            v_true_all.extend(item["valence_seq"][:len(v_pred_list)])
            a_true_all.extend(item["arousal_seq"][:len(a_pred_list)])
            v_pred_all.extend(v_pred_list)
            a_pred_all.extend(a_pred_list)

        # 计算指标
        out = {
            "valence_ccc": ccc(v_true_all, v_pred_all),
            "arousal_ccc": ccc(a_true_all, a_pred_all),
            "valence_mae": mae(v_true_all, v_pred_all),
            "arousal_mae": mae(a_true_all, a_pred_all),
            "valence_corr": pearsonr(v_true_all, v_pred_all),
            "arousal_corr": pearsonr(a_true_all, a_pred_all),
            "total_frames": len(v_true_all)
        }

        out["mean_ccc"] = 0.5 * (out["valence_ccc"] + out["arousal_ccc"])
        return out


class AffWild2EXPR(ClassificationDataset):
    """Aff-Wild2 的表情分类评测。"""

    @property
    def config_class(self):
        return AffWild2ExprConfig

    def _get_required_paths(self) -> List[str]:
        """返回AffWild2 EXPR数据集必要的路径"""
        paths = [
            self.config.annotation_dir,
            f"{self.config.annotation_dir}/Train_Set",
            f"{self.config.annotation_dir}/Validation_Set"
        ]
        paths.extend(self.config.video_dirs)
        return paths

    def _find_video_file(self, video_name: str) -> str:
        """查找视频文件"""
        for video_dir in self.config.video_dirs:
            for ext in ['.mp4', '.avi', '.mov']:
                video_path = os.path.join(
                    self.root, video_dir, f"{video_name}{ext}")
                if os.path.exists(video_path):
                    return video_path
        return ""

    def _load_expr_annotations(self, annotation_file: str) -> List[int]:
        """加载表情标注文件"""
        labels = []

        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()

            # 跳过第一行（标题行）
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue

                try:
                    label_idx = int(line)
                    # 跳过无效值（-1）
                    if 0 <= label_idx < len(EXPR_LABELS):
                        labels.append(label_idx)
                except ValueError:
                    continue

        except Exception as e:
            logger.warning(f"读取表情标注文件失败 {annotation_file}: {e}")

        return labels

    def _get_majority_label(self, labels: List[int]) -> str:
        """获取多数投票标签"""
        if not labels:
            return "Neutral"

        # 统计每个标签的票数
        from collections import Counter
        label_counts = Counter(labels)
        majority_idx = label_counts.most_common(1)[0][0]

        return EXPR_LABELS[majority_idx]

    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代表情分类样本"""
        split_dir = "Train_Set" if split == "train" else "Validation_Set"
        annotation_root = os.path.join(
            self.root, self.config.annotation_dir, split_dir)

        if not os.path.exists(annotation_root):
            logger.error(f"标注目录不存在: {annotation_root}")
            return

        # 获取所有标注文件
        annotation_files = glob.glob(os.path.join(annotation_root, "*.txt"))

        label_space = AFFWILD2_EXPR_8 if self.config.label_space_8 else AFFWILD2_EXPR_7
        sample_count = 0

        for ann_file in annotation_files:
            video_name = os.path.splitext(os.path.basename(ann_file))[0]

            # 加载表情标注
            labels = self._load_expr_annotations(ann_file)
            if not labels:
                continue

            # 获取多数投票标签
            majority_label = self._get_majority_label(labels)

            # 映射到标准标签空间
            if majority_label == "Other":
                if not self.config.label_space_8:
                    continue  # 7类模式跳过"Other"
                label = "other"
            else:
                label = normalize_label(majority_label)

            # 查找视频文件
            video_path = self._find_video_file(video_name)

            yield {
                "id": f"affwild2_expr_{split}_{sample_count}",
                "video_name": video_name,
                "video_path": video_path,
                "label": label,
                "label_space": label_space,
                "frame_labels": labels,
                "majority_label": majority_label
            }

            sample_count += 1

    def evaluate(self, adapter, split: str = "val") -> Dict:
        """评估表情分类性能"""
        y_true, y_pred = [], []

        for item in self._safe_iter_samples(split):
            pred = adapter.predict(item, task="video_emotion_class")
            y_true.append(item["label"])
            y_pred.append(pred.get("label", ""))

        # 计算指标
        acc = accuracy(y_true, y_pred)
        macro = precision_recall_f1(y_true, y_pred, average="macro")["f1"]
        weighted = precision_recall_f1(
            y_true, y_pred, average="weighted")["f1"]

        return {
            "acc": acc,
            "macro_f1": macro,
            "weighted_f1": weighted,
            "total_samples": len(y_true)
        }
