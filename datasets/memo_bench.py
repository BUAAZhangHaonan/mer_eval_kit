# -*- coding: utf-8 -*-
"""
datasets/memo_bench.py
—— MEMO-Bench：AI生成人像情绪分类（6类）。主指标：Acc + Macro-F1。
读取 annotation_emo.csv 和图像文件，处理6种情感类别。
"""
import os
import re
import csv
import glob
import logging
from typing import Dict, Iterator, List
from dataclasses import dataclass

from datasets.base import ClassificationDataset, DatasetConfig
from datasets.label_spaces import MEMO_6, normalize_label
from datasets.metrics import accuracy, precision_recall_f1

logger = logging.getLogger(__name__)


@dataclass
class MEMOConfig(DatasetConfig):
    """MEMO-Bench数据集配置"""
    # 标注文件
    emotion_csv: str = "annotation_emo.csv"
    quality_csv: str = "annotation_quality.csv"

    # 图像目录
    image_dir: str = "dataset"

    # 处理选项
    use_quality_filter: bool = False  # 是否使用质量过滤
    min_quality_score: float = 3.0   # 最低质量分数
    image_extensions: List[str] = None

    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']


class MEMOBench(ClassificationDataset):
    """MEMO-Bench AI生成人像情感分类评测器"""

    @property
    def config_class(self):
        return MEMOConfig

    def _get_required_paths(self) -> List[str]:
        """返回MEMO-Bench数据集必要的路径"""
        return [
            self.config.emotion_csv,
            self.config.quality_csv,
            self.config.image_dir
        ]

    def _load_emotion_annotations(self) -> Dict[str, Dict]:
        """加载情感标注"""
        emotion_file = os.path.join(self.root, self.config.emotion_csv)
        annotations = {}

        if not os.path.exists(emotion_file):
            logger.error(f"情感标注文件不存在: {emotion_file}")
            return annotations

        try:
            with open(emotion_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_name = row.get("Image", "").strip()
                    score = row.get("Score", "").strip()
                    distortions = row.get("Distortions", "").strip()

                    if image_name:
                        annotations[image_name] = {
                            "score": score,
                            "distortions": distortions
                        }
        except Exception as e:
            logger.error(f"读取情感标注文件失败: {e}")

        return annotations

    def _load_quality_annotations(self) -> Dict[str, float]:
        """加载质量标注"""
        quality_file = os.path.join(self.root, self.config.quality_csv)
        quality_scores = {}

        if not os.path.exists(quality_file):
            logger.warning(f"质量标注文件不存在: {quality_file}")
            return quality_scores

        try:
            with open(quality_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_name = row.get("Image", "").strip()
                    quality = row.get("Quality", "").strip()

                    if image_name and quality:
                        try:
                            quality_scores[image_name] = float(quality)
                        except ValueError:
                            continue
        except Exception as e:
            logger.warning(f"读取质量标注文件失败: {e}")

        return quality_scores

    def _find_image_file(self, image_name: str) -> str:
        """查找图像文件"""
        image_dir = os.path.join(self.root, self.config.image_dir)

        if not os.path.exists(image_dir):
            return ""

        # 尝试不同的扩展名
        for ext in self.config.image_extensions:
            image_path = os.path.join(image_dir, image_name)
            if not image_path.endswith(ext):
                image_path = image_path + ext

            if os.path.exists(image_path):
                return image_path

        # 直接查找文件名（不管扩展名）
        for ext in self.config.image_extensions:
            pattern = os.path.join(image_dir, f"{image_name}*{ext}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]

        return ""

    def _extract_emotion_from_distortions(self, distortions: str) -> str:
        """从Distortions字段提取情感标签"""
        if not distortions:
            return ""

        # 以非字母字符分割，尝试匹配标准标签
        tokens = [tok for tok in re.split(
            r"[^a-zA-Z]+", distortions.lower()) if tok]
        for token in tokens:
            label = normalize_label(token)
            if label in MEMO_6:
                return label

        # 兜底：直接标准化整个字符串
        label = normalize_label(distortions)
        return label if label in MEMO_6 else ""

    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代MEMO-Bench样本"""
        # 加载标注
        emotion_annotations = self._load_emotion_annotations()
        quality_scores = self._load_quality_annotations()

        sample_count = 0

        for image_name, emotion_data in emotion_annotations.items():
            # 查找图像文件
            image_path = self._find_image_file(image_name)
            if not image_path or not os.path.exists(image_path):
                logger.warning(f"找不到图像文件: {image_name}")
                continue

            # 质量过滤
            if self.config.use_quality_filter:
                quality = quality_scores.get(image_name, 0.0)
                if quality < self.config.min_quality_score:
                    continue

            # 提取情感标签
            distortions = emotion_data.get("distortions", "")
            label = self._extract_emotion_from_distortions(distortions)

            # 确保标签在标准空间中
            if label not in MEMO_6:
                logger.warning(f"未知情感标签: {label} (图像: {image_name})")
                continue

            yield {
                "id": f"memo_bench_{sample_count}",
                "image_name": image_name,
                "image_path": image_path,
                "label": label,
                "label_space": MEMO_6,
                "emotion_score": emotion_data.get("score", ""),
                "distortions": distortions,
                "quality_score": quality_scores.get(image_name, 0.0)
            }

            sample_count += 1

    def evaluate(self, adapter, split: str = "test") -> Dict:
        """评估MEMO-Bench性能"""
        y_true, y_pred = [], []

        for item in self._safe_iter_samples(split):
            pred = adapter.predict(item, task="image_emotion_class")
            y_true.append(item["label"])
            y_pred.append(pred.get("label", ""))

        # 防止除零错误
        if not y_true:
            return {
                "acc": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "total_samples": 0,
                "emotion_distribution": {},
                "prediction_distribution": {}
            }

        # 计算指标
        acc = accuracy(y_true, y_pred)
        macro_f1 = precision_recall_f1(y_true, y_pred, average="macro")["f1"]
        weighted_f1 = precision_recall_f1(
            y_true, y_pred, average="weighted")["f1"]

        # 按情感类别的统计
        from collections import Counter
        true_dist = Counter(y_true)
        pred_dist = Counter(y_pred)

        return {
            "acc": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "total_samples": len(y_true),
            "emotion_distribution": dict(true_dist),
            "prediction_distribution": dict(pred_dist)
        }
