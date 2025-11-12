# -*- coding: utf-8 -*-
"""
datasets/meld.py
—— MELD：多说话人对话情绪分类。主指标：Weighted-F1 + Accuracy。
读取 train_sent_emo.csv/dev_sent_emo.csv/test_sent_emo.csv，正确解析视频路径。
"""
import os
import csv
import logging
from typing import Dict, Iterator, List
from dataclasses import dataclass

from datasets.base import MultiModalDataset, DatasetConfig
from datasets.label_spaces import MELD_7, normalize_label
from datasets.metrics import accuracy, precision_recall_f1

logger = logging.getLogger(__name__)


@dataclass
class MELDConfig(DatasetConfig):
    """MELD数据集配置"""
    # CSV文件名
    train_csv: str = "train_sent_emo.csv"
    dev_csv: str = "dev_sent_emo.csv"
    test_csv: str = "test_sent_emo.csv"

    # 视频目录
    train_video_dir: str = "train_splits"
    dev_video_dir: str = "dev_splits_complete"
    test_video_dir: str = "output_repeated_splits_test"

    # 处理选项
    use_video: bool = True  # 是否包含视频路径
    video_extensions: List[str] = None

    def __post_init__(self):
        if self.video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov']


class MELD(MultiModalDataset):
    """MELD 多模态对话情感分类评测器"""

    @property
    def config_class(self):
        return MELDConfig

    def _get_required_paths(self) -> List[str]:
        """返回MELD数据集必要的路径"""
        paths = [
            self.config.train_csv,
            self.config.dev_csv,
            self.config.test_csv,
            self.config.train_video_dir,
            self.config.dev_video_dir,
            self.config.test_video_dir
        ]
        return paths

    def _get_csv_file(self, split: str) -> str:
        """获取对应的CSV文件"""
        if split == "train":
            return self.config.train_csv
        elif split == "dev":
            return self.config.dev_csv
        else:
            return self.config.test_csv

    def _get_video_dir(self, split: str) -> str:
        """获取对应的视频目录"""
        if split == "train":
            return self.config.train_video_dir
        elif split == "dev":
            return self.config.dev_video_dir
        else:
            return self.config.test_video_dir

    def _find_video_file(self, dialogue_id: str, utterance_id: str, split: str) -> str:
        """查找视频文件"""
        if not self.config.use_video:
            return ""

        video_dir = self._get_video_dir(split)
        full_video_dir = os.path.join(self.root, video_dir)

        if not os.path.exists(full_video_dir):
            return ""

        # 尝试不同的文件名格式
        possible_names = [
            f"dia{dialogue_id}_utt{utterance_id}.mp4",
            f"dia{dialogue_id}_utt{utterance_id}.avi",
            f"dia{dialogue_id}_utt{utterance_id}.mov",
            f"dia{dialogue_id}_utt{utterance_id}",  # 无扩展名
        ]

        for name in possible_names:
            for ext in self.config.video_extensions:
                if not name.endswith(ext):
                    candidate = name + ext
                else:
                    candidate = name

                video_path = os.path.join(full_video_dir, candidate)
                if os.path.exists(video_path):
                    return video_path

        return ""

    def _parse_csv_row(self, row: Dict, split: str) -> Dict:
        """解析CSV行数据"""
        # 提取基本信息
        dialogue_id = row.get("Dialogue_ID", row.get("DialogueID", ""))
        utterance_id = row.get("Utterance_ID", row.get("UtteranceID", ""))
        utterance = row.get("Utterance", row.get("text", ""))
        speaker = row.get("Speaker", "")
        emotion = row.get("Emotion", row.get("emotion", ""))
        sentiment = row.get("Sentiment", row.get("sentiment", ""))

        # 处理时间信息
        start_time = row.get("StartTime", "")
        end_time = row.get("EndTime", "")
        season = row.get("Season", "")
        episode = row.get("Episode", "")

        # 标准化情感标签
        normalized_emotion = normalize_label(emotion)
        if normalized_emotion not in MELD_7:
            normalized_emotion = "neutral"

        # 查找视频文件
        video_path = self._find_video_file(dialogue_id, utterance_id, split)

        return {
            "id": f"meld_{dialogue_id}_{utterance_id}",
            "dialogue_id": dialogue_id,
            "utterance_id": utterance_id,
            "text": utterance,
            "speaker": speaker,
            "label": normalized_emotion,
            "label_space": MELD_7,
            "video_path": video_path,
            "sentiment": sentiment,
            "start_time": start_time,
            "end_time": end_time,
            "season": season,
            "episode": episode,
        }

    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代MELD样本"""
        csv_file = self._get_csv_file(split)
        csv_path = os.path.join(self.root, csv_file)

        if not os.path.exists(csv_path):
            logger.error(f"CSV文件不存在: {csv_path}")
            return

        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    try:
                        sample = self._parse_csv_row(row, split)
                        yield sample
                    except Exception as e:
                        logger.warning(f"解析CSV行失败 (行{i+1}): {e}")
                        continue

        except Exception as e:
            logger.error(f"读取CSV文件失败 {csv_path}: {e}")

    def evaluate(self, adapter, split: str = "test") -> Dict:
        """评估MELD性能"""
        y_true, y_pred = [], []

        for item in self._safe_iter_samples(split):
            pred = adapter.predict(item, task="meld_dialog_emotion")
            y_true.append(item["label"])
            y_pred.append(pred.get("label", ""))

        # 计算指标
        acc = accuracy(y_true, y_pred)
        weighted_f1 = precision_recall_f1(
            y_true, y_pred, average="weighted")["f1"]
        macro_f1 = precision_recall_f1(y_true, y_pred, average="macro")["f1"]

        # 按情感类别的详细统计
        from collections import Counter
        true_dist = Counter(y_true)
        pred_dist = Counter(y_pred)

        return {
            "acc": acc,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "total_samples": len(y_true),
            "emotion_distribution": dict(true_dist),
            "prediction_distribution": dict(pred_dist)
        }
