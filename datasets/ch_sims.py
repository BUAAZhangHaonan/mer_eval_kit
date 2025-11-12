# -*- coding: utf-8 -*-
"""
datasets/ch_sims.py
—— CH-SIMS（中文多模态情感强度）评测器。
读取 label.csv 和对应的多模态数据，处理连续情感强度标签。
"""
import os
import csv
import pickle
import glob
import logging
from typing import Dict, Iterator, List
from dataclasses import dataclass

from datasets.base import MultiModalDataset, DatasetConfig
from datasets.metrics import mae, pearsonr, acc2_neg_nonneg, acc2_neg_pos, precision_recall_f1, accuracy

logger = logging.getLogger(__name__)


@dataclass
class CHSIMSConfig(DatasetConfig):
    """CH-SIMS数据集配置"""
    # 标注文件
    label_file: str = "label.csv"

    # 数据目录
    raw_dir: str = "Raw"
    processed_dir: str = "Processed"

    # 处理选项
    use_processed_features: bool = False  # 是否使用预处理的特征
    feature_file: str = "unaligned_39.pkl"  # 特征文件名
    video_extensions: List[str] = None

    def __post_init__(self):
        if self.video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov']


class CHSIMS(MultiModalDataset):
    """CH-SIMS 中文多模态情感强度评测器"""

    @property
    def config_class(self):
        return CHSIMSConfig

    def _get_required_paths(self) -> List[str]:
        """返回CH-SIMS数据集必要的路径"""
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
            # CH-SIMS的label.csv没有header行，需要手动指定列名
            column_names = [
                "video_id", "clip_id", "text",
                "label", "label_T", "label_A", "label_V",
                "emotion", "mode"
            ]

            with open(label_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, fieldnames=column_names)
                # 跳过第一行（如果是header的话）
                data = list(reader)
                if data and data[0].get("video_id") == "video_id":
                    data = data[1:]
                return data
        except Exception as e:
            logger.error(f"读取标注文件失败: {e}")
            return []

    def _find_video_file(self, video_id: str, clip_id: str) -> str:
        """查找视频文件"""
        raw_dir = os.path.join(self.root, self.config.raw_dir, video_id)

        if not os.path.exists(raw_dir):
            return ""

        # 尝试不同的文件名格式
        possible_names = [
            f"{clip_id}.mp4",
            f"{clip_id}.avi",
            f"{clip_id}.mov",
            f"{video_id}_{clip_id}.mp4",
            f"{video_id}_{clip_id}.avi",
            f"{video_id}_{clip_id}.mov",
        ]

        for name in possible_names:
            video_path = os.path.join(raw_dir, name)
            if os.path.exists(video_path):
                return video_path

        # 递归查找
        for ext in self.config.video_extensions:
            pattern = os.path.join(raw_dir, f"**/*{clip_id}*{ext}")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]

        return ""

    def _load_processed_features(self) -> Dict:
        """加载预处理特征"""
        if not self.config.use_processed_features:
            return {}

        feature_file = os.path.join(
            self.root, self.config.processed_dir, self.config.feature_file)

        if not os.path.exists(feature_file):
            logger.warning(f"特征文件不存在: {feature_file}")
            return {}

        try:
            with open(feature_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"加载特征文件失败: {e}")
            return {}

    def _parse_label_row(self, row: Dict) -> Dict:
        """解析标注行"""
        try:
            video_id = row.get("video_id", "").strip()
            clip_id = row.get("clip_id", "").strip()
            text = row.get("text", "").strip()

            # 解析情感标签
            label = self._safe_float(row.get("label", 0.0))
            label = max(-1.0, min(1.0, label))  # 限制在[-1,1]范围内

            # 解析单模态标签
            label_t = self._safe_float(
                row.get("label_T", label), default=label)
            label_a = self._safe_float(
                row.get("label_A", label), default=label)
            label_v = self._safe_float(
                row.get("label_V", label), default=label)

            # 解析其他字段 (CH-SIMS中使用"emotion"而不是"annotation")
            emotion = row.get("emotion", "")
            mode = (row.get("mode", "") or "").strip().lower()

            annotation = row.get("annotation") or emotion

            return {
                "video_id": video_id,
                "clip_id": clip_id,
                "text": text,
                "label": label,
                "label_T": max(-1.0, min(1.0, label_t)),
                "label_A": max(-1.0, min(1.0, label_a)),
                "label_V": max(-1.0, min(1.0, label_v)),
                "emotion": emotion,
                "annotation": annotation,
                "mode": mode
            }

        except Exception as e:
            logger.warning(f"解析标注行失败: {e}")
            return {}

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """安全地将值转换为浮点数，遇到缺失或非法值时返回默认值"""
        try:
            if value in (None, ""):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代CH-SIMS样本"""
        # 加载标注数据
        label_data = self._load_label_csv()
        processed_features = self._load_processed_features()

        sample_count = 0

        for row in label_data:
            parsed_data = self._parse_label_row(row)
            if not parsed_data:
                continue

            # 按模式过滤
            mode = (parsed_data.get("mode") or "").lower()
            split_key = split.lower()

            if split_key == "train" and mode != "train":
                continue
            elif split_key in {"dev", "valid", "val", "validation"} and mode not in {"dev", "valid", "val", "validation"}:
                continue
            elif split_key in {"test", "eval", "evaluation"} and mode not in {"test", "eval", "evaluation"}:
                continue

            # 查找视频文件
            video_path = self._find_video_file(
                parsed_data["video_id"],
                parsed_data["clip_id"]
            )

            # 查找预处理特征
            feature_key = f"{parsed_data['video_id']}_{parsed_data['clip_id']}"
            features = processed_features.get(feature_key, {})

            yield {
                "id": f"chsims_{split}_{sample_count}",
                "video_id": parsed_data["video_id"],
                "clip_id": parsed_data["clip_id"],
                "text": parsed_data["text"],
                "label": parsed_data["label"],
                "label_T": parsed_data["label_T"],
                "label_A": parsed_data["label_A"],
                "label_V": parsed_data["label_V"],
                "annotation": parsed_data.get("annotation", ""),
                "emotion": parsed_data.get("emotion", ""),
                "mode": parsed_data["mode"],
                "video_path": video_path,
                "features": features
            }

            sample_count += 1

    def evaluate(self, adapter, split: str = "test") -> Dict:
        """评估CH-SIMS性能"""
        y_true, y_pred = [], []

        for item in self._safe_iter_samples(split):
            pred = adapter.predict(item, task="chsims_sentiment")
            p = float(pred.get("polarity", 0.0))
            p = max(-1.0, min(1.0, p))  # clip到[-1,1]
            y_true.append(item["label"])
            y_pred.append(p)

        # 计算回归指标
        out = {
            "mae": mae(y_true, y_pred),
            "corr": pearsonr(y_true, y_pred),
            "acc2_neg_nonneg": acc2_neg_nonneg(y_true, y_pred, zero=0.0),
            "acc2_neg_pos": acc2_neg_pos(y_true, y_pred, zero=0.0),
            "total_samples": len(y_true)
        }

        # 计算二分类指标
        y_true_bin = ["nonneg" if v >= 0 else "neg" for v in y_true]
        y_pred_bin = ["nonneg" if v >= 0 else "neg" for v in y_pred]
        out["f1_binary_macro"] = precision_recall_f1(
            y_true_bin, y_pred_bin, average="macro")["f1"]
        out["acc_binary"] = accuracy(y_true_bin, y_pred_bin)

        return out


class CHSIMSV2(CHSIMS):
    """CH-SIMS v2 数据集处理器（继承CH-SIMS的基础功能）"""

    def _get_required_paths(self) -> List[str]:
        """返回CH-SIMS v2数据集必要的路径"""
        # CH-SIMS v2有监督和无监督两个版本
        s_version_paths = [
            f"CH-SIMS v2(s)/{self.config.label_file}",
            f"CH-SIMS v2(s)/{self.config.raw_dir}",
            f"CH-SIMS v2(s)/{self.config.processed_dir}"
        ]

        u_version_paths = [
            f"CH-SIMS v2(u)/{self.config.label_file}",
            f"CH-SIMS v2(u)/{self.config.raw_dir}",
            f"CH-SIMS v2(u)/{self.config.processed_dir}"
        ]

        return s_version_paths + u_version_paths

    def _load_label_csv(self, version: str = "s") -> List[Dict]:
        """加载指定版本的meta.csv文件"""
        meta_file = os.path.join(self.root, f"CH-SIMS v2({version})/meta.csv")

        if not os.path.exists(meta_file):
            logger.error(f"标注文件不存在: {meta_file}")
            return []

        try:
            with open(meta_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logger.error(f"读取标注文件失败: {e}")
            return []

    def _find_video_file(self, video_id: str, clip_id: str, version: str = "s") -> str:
        """查找v2版本的视频文件"""
        raw_dir = os.path.join(
            self.root, f"CH-SIMS v2({version})/Raw", video_id)

        if not os.path.exists(raw_dir):
            return ""

        # v2版本的视频文件名可能不同，尝试多种格式
        possible_names = [
            f"{clip_id}.mp4",
            f"{clip_id}.avi",
            f"{clip_id}.mov",
            f"{video_id}_{clip_id}.mp4",
            f"{video_id}_{clip_id}.avi",
            f"{video_id}_{clip_id}.mov",
        ]

        for name in possible_names:
            video_path = os.path.join(raw_dir, name)
            if os.path.exists(video_path):
                return video_path

        return ""

    def _load_processed_features(self, version: str = "s") -> Dict:
        """加载v2版本的预处理特征"""
        if not self.config.use_processed_features:
            return {}

        feature_file = os.path.join(
            self.root, f"CH-SIMS v2({version})/Processed", self.config.feature_file)

        if not os.path.exists(feature_file):
            return {}

        try:
            with open(feature_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"加载特征文件失败: {e}")
            return {}

    def iter_samples(self, split: str = "test", version: str = "s") -> Iterator[Dict]:
        """迭代CH-SIMS v2样本"""
        # 加载指定版本的数据
        label_data = self._load_label_csv(version)
        processed_features = self._load_processed_features(version)

        sample_count = 0

        for row in label_data:
            parsed_data = self._parse_label_row(row)
            if not parsed_data:
                continue

            # 按模式过滤
            mode = (parsed_data.get("mode") or "").lower()
            split_key = split.lower()

            if split_key == "train" and mode != "train":
                continue
            elif split_key in {"dev", "valid", "val", "validation"} and mode not in {"dev", "valid", "val", "validation"}:
                continue
            elif split_key in {"test", "eval", "evaluation"} and mode not in {"test", "eval", "evaluation"}:
                continue

            # 查找视频文件
            video_path = self._find_video_file(
                parsed_data["video_id"],
                parsed_data["clip_id"],
                version
            )

            # 查找预处理特征
            feature_key = f"{parsed_data['video_id']}_{parsed_data['clip_id']}"
            features = processed_features.get(feature_key, {})

            yield {
                "id": f"chsims_v2_{version}_{split}_{sample_count}",
                "version": version,
                "video_id": parsed_data["video_id"],
                "clip_id": parsed_data["clip_id"],
                "text": parsed_data["text"],
                "label": parsed_data["label"],
                "label_T": parsed_data["label_T"],
                "label_A": parsed_data["label_A"],
                "label_V": parsed_data["label_V"],
                "annotation": parsed_data.get("annotation", parsed_data.get("emotion", "")),
                "mode": parsed_data["mode"],
                "video_path": video_path,
                "features": features
            }

            sample_count += 1
