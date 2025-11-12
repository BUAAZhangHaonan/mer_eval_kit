# -*- coding: utf-8 -*-
"""
datasets/cmu_mosei.py
—— CMU-MOSEI 评测器（连续强度 + Acc-2/Acc-7/F1/MAE/Corr）。
读取 label.csv 和预处理特征，支持对齐和非对齐特征。
"""
import os
import csv
import pickle
import logging
from typing import Dict, Iterator, List
from dataclasses import dataclass

from datasets.base import MultiModalDataset, DatasetConfig
from datasets.metrics import mae, pearsonr, acc2_neg_nonneg, acc2_neg_pos, acc7_from_continuous, precision_recall_f1, accuracy

logger = logging.getLogger(__name__)


@dataclass
class MOSEIConfig(DatasetConfig):
    """CMU-MOSEI数据集配置"""
    # 标注文件
    label_file: str = "label.csv"

    # 数据目录
    raw_dir: str = "Raw"
    processed_dir: str = "Processed"

    # 处理选项
    use_processed_features: bool = False  # 是否使用预处理特征
    use_aligned_features: bool = True   # 是否使用对齐特征
    aligned_feature_file: str = "aligned.pkl"
    unaligned_feature_file: str = "unaligned.pkl"

    # 字段映射
    text_key: str = "text"
    label_key: str = "label"
    video_key: str = "video_path"
    audio_key: str = "audio_path"


class CMUMOSEI(MultiModalDataset):
    """CMU-MOSEI 多模态情感强度评测器"""

    @property
    def config_class(self):
        return MOSEIConfig

    def _get_required_paths(self) -> List[str]:
        """返回CMU-MOSEI数据集必要的路径"""
        paths = [
            self.config.label_file,
            self.config.raw_dir,
            self.config.processed_dir
        ]
        return paths

    def _load_label_csv(self) -> List[Dict]:
        """加载label.csv文件"""
        label_file = os.path.join(self.root, self.config.label_file)

        if not os.path.exists(label_file):
            logger.error(f"标注文件不存在: {label_file}")
            return []

        try:
            with open(label_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logger.error(f"读取标注文件失败: {e}")
            return []

    def _load_processed_features(self) -> Dict:
        """加载预处理特征"""
        if not self.config.use_processed_features:
            return {}

        # 选择特征文件
        if self.config.use_aligned_features:
            feature_file = os.path.join(
                self.root, self.config.processed_dir, self.config.aligned_feature_file)
        else:
            feature_file = os.path.join(
                self.root, self.config.processed_dir, self.config.unaligned_feature_file)

        if not os.path.exists(feature_file):
            logger.warning(f"特征文件不存在: {feature_file}")
            return {}

        try:
            with open(feature_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"加载特征文件失败: {e}")
            return {}

    def _find_media_file(self, video_id: str, media_type: str) -> str:
        """查找媒体文件"""
        raw_dir = os.path.join(self.root, self.config.raw_dir, video_id)

        if not os.path.exists(raw_dir):
            return ""

        # 查找对应类型的文件
        extensions = ['.mp4', '.avi', '.mov'] if media_type == 'video' else [
            '.wav', '.mp3', '.m4a']

        for ext in extensions:
            media_path = os.path.join(raw_dir, f"{video_id}{ext}")
            if os.path.exists(media_path):
                return media_path

        return ""

    def _parse_label_row(self, row: Dict) -> Dict:
        """解析标注行"""
        try:
            video_id = row.get("video_id", "").strip()
            clip_id = row.get("clip_id", "").strip()
            text = row.get(self.config.text_key, "").strip()

            # 解析情感标签
            label = self._safe_float(row.get(self.config.label_key, 0.0))
            label = max(-3.0, min(3.0, label))  # 限制在[-3,3]范围内

            # 解析其他标签
            label_t = self._safe_float(
                row.get("label_T", label), default=label)
            label_a = self._safe_float(
                row.get("label_A", label), default=label)
            label_v = self._safe_float(
                row.get("label_V", label), default=label)

            # 解析其他字段
            annotation = row.get("annotation", "")
            mode = (row.get("mode", "") or "").strip().lower()

            return {
                "video_id": video_id,
                "clip_id": clip_id,
                "text": text,
                "label": label,
                "label_T": label_t,
                "label_A": label_a,
                "label_V": label_v,
                "annotation": annotation,
                "mode": mode
            }

        except Exception as e:
            logger.warning(f"解析标注行失败: {e}")
            return {}

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """安全地将值转换为浮点数，异常时返回默认值"""
        try:
            if value in (None, ""):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代CMU-MOSEI样本"""
        # 加载标注数据
        label_data = self._load_label_csv()
        processed_features = self._load_processed_features()

        sample_count = 0
        split_key = (split or "test").lower()

        for row in label_data:
            parsed_data = self._parse_label_row(row)
            if not parsed_data:
                continue

            # 按模式过滤
            mode = parsed_data.get("mode")
            if split_key == "train" and mode != "train":
                continue
            elif split_key in {"dev", "valid", "val", "validation"} and mode not in {"dev", "valid", "val", "validation"}:
                continue
            elif split_key in {"test", "eval", "evaluation"} and mode not in {"test", "eval", "evaluation"}:
                continue

            # 查找媒体文件
            video_path = self._find_media_file(
                parsed_data["video_id"], "video")
            audio_path = self._find_media_file(
                parsed_data["video_id"], "audio")

            # 查找预处理特征
            feature_key = f"{parsed_data['video_id']}_{parsed_data['clip_id']}"
            features = processed_features.get(feature_key, {})

            yield {
                "id": f"cmu_mosei_{split}_{sample_count}",
                "video_id": parsed_data["video_id"],
                "clip_id": parsed_data["clip_id"],
                "text": parsed_data["text"],
                "label": parsed_data["label"],
                "label_T": parsed_data["label_T"],
                "label_A": parsed_data["label_A"],
                "label_V": parsed_data["label_V"],
                "annotation": parsed_data["annotation"],
                "mode": parsed_data["mode"],
                "video_path": video_path,
                "audio_path": audio_path,
                "features": features
            }

            sample_count += 1

    def evaluate(self, adapter, split: str = "test") -> Dict:
        """评估CMU-MOSEI性能"""
        y_true, y_pred = [], []

        for item in self._safe_iter_samples(split):
            pred = adapter.predict(item, task="mosei_sentiment")
            p = float(pred.get("polarity", 0.0))
            p = max(-3.0, min(3.0, p))
            y_true.append(item["label"])
            y_pred.append(p)

        # 计算回归指标
        out = {
            "mae": mae(y_true, y_pred),
            "corr": pearsonr(y_true, y_pred),
            "acc2_neg_nonneg": acc2_neg_nonneg(y_true, y_pred, zero=0.0),
            "acc2_neg_pos": acc2_neg_pos(y_true, y_pred, zero=0.0),
            "acc7": acc7_from_continuous(y_true, y_pred),
            "total_samples": len(y_true)
        }

        # 计算二分类指标
        y_true_bin = ["nonneg" if v >= 0 else "neg" for v in y_true]
        y_pred_bin = ["nonneg" if v >= 0 else "neg" for v in y_pred]
        out["f1_binary_macro"] = precision_recall_f1(
            y_true_bin, y_pred_bin, average="macro")["f1"]
        out["acc_binary"] = accuracy(y_true_bin, y_pred_bin)

        return out


class CMUMOSI(CMUMOSEI):
    """CMU-MOSI 多模态情感强度评测器（继承CMU-MOSEI的基础功能）"""

    def _get_required_paths(self) -> List[str]:
        """返回CMU-MOSI数据集必要的路径"""
        # CMU-MOSI的结构与CMU-MOSEI相同
        paths = [
            self.config.label_file,
            self.config.raw_dir,
            self.config.processed_dir
        ]
        return paths

    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代CMU-MOSI样本"""
        # 加载标注数据
        label_data = self._load_label_csv()
        processed_features = self._load_processed_features()

        sample_count = 0
        split_key = (split or "test").lower()

        for row in label_data:
            parsed_data = self._parse_label_row(row)
            if not parsed_data:
                continue

            # 按模式过滤
            mode = parsed_data.get("mode")
            if split_key == "train" and mode != "train":
                continue
            elif split_key in {"dev", "valid", "val", "validation"} and mode not in {"dev", "valid", "val", "validation"}:
                continue
            elif split_key in {"test", "eval", "evaluation"} and mode not in {"test", "eval", "evaluation"}:
                continue

            # 查找媒体文件
            video_path = self._find_media_file(
                parsed_data["video_id"], "video")
            audio_path = self._find_media_file(
                parsed_data["video_id"], "audio")

            # 查找预处理特征
            feature_key = f"{parsed_data['video_id']}_{parsed_data['clip_id']}"
            features = processed_features.get(feature_key, {})

            yield {
                "id": f"cmu_mosi_{split}_{sample_count}",
                "video_id": parsed_data["video_id"],
                "clip_id": parsed_data["clip_id"],
                "text": parsed_data["text"],
                "label": parsed_data["label"],
                "label_T": parsed_data["label_T"],
                "label_A": parsed_data["label_A"],
                "label_V": parsed_data["label_V"],
                "annotation": parsed_data["annotation"],
                "mode": parsed_data["mode"],
                "video_path": video_path,
                "audio_path": audio_path,
                "features": features
            }

            sample_count += 1
