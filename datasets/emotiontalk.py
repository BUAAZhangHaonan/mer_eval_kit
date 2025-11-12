# -*- coding: utf-8 -*-
"""
datasets/emotiontalk.py
—— EmotionTalk（中文对话）评测器。
处理复杂的多层级JSON结构，支持文本、音频、视频三种模态。
"""
import os
import json
import glob
import logging
from typing import Dict, Iterator, List
from dataclasses import dataclass

from datasets.base import MultiModalDataset, DatasetConfig
from datasets.label_spaces import EMOTIONTALK_7, normalize_label
from datasets.metrics import accuracy, precision_recall_f1, mae, pearsonr

logger = logging.getLogger(__name__)

# EmotionTalk 支持的标签空间（复用统一的 7 类情感集合）
EMOTIONTALK_LABELS = EMOTIONTALK_7


@dataclass
class EmotionTalkConfig(DatasetConfig):
    """EmotionTalk数据集配置"""
    # 模态路径
    text_json_dir: str = "Text/json"
    audio_json_dir: str = "Audio/json"
    video_json_dir: str = "Video/json"
    multimodal_json_dir: str = "Multimodal/json"

    # 文件路径
    audio_wav_dir: str = "Audio/wav"
    video_mp4_dir: str = "Multimodal/mp4"

    # 处理选项
    include_multimodal: bool = True  # 是否包含多模态标注
    merge_modalities: bool = True   # 是否合并不同模态的标注


class EmotionTalk(MultiModalDataset):
    """EmotionTalk 多模态情感对话评测器"""

    @property
    def config_class(self):
        return EmotionTalkConfig

    def _get_required_paths(self) -> List[str]:
        """返回EmotionTalk数据集必要的路径"""
        paths = [
            self.config.text_json_dir,
            self.config.audio_json_dir,
            self.config.video_json_dir,
            self.config.multimodal_json_dir,
            self.config.audio_wav_dir,
            self.config.video_mp4_dir
        ]
        return paths

    def _collect_json_files(self, json_dir: str) -> List[str]:
        """递归收集JSON文件"""
        full_path = os.path.join(self.root, json_dir)
        if not os.path.exists(full_path):
            logger.warning(f"目录不存在: {full_path}")
            return []

        json_files = glob.glob(os.path.join(
            full_path, "**/*.json"), recursive=True)
        return sorted(json_files)

    def _load_json_annotation(self, json_file: str) -> Dict:
        """加载单个JSON标注文件"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.warning(f"加载JSON文件失败 {json_file}: {e}")
            return {}

    def _extract_emotion_from_data(self, data_field: List) -> str:
        """从data字段提取情感标签"""
        if not data_field or not isinstance(data_field, list):
            return "neutral"

        # 收集所有标注者的情感结果
        emotions = []
        for annotator in data_field:
            if isinstance(annotator, dict) and "emotion_result" in annotator:
                emotion = annotator["emotion_result"]
                if emotion and isinstance(emotion, str):
                    emotions.append(normalize_label(emotion))

        if not emotions:
            return "neutral"

        # 返回多数投票结果
        from collections import Counter
        emotion_counts = Counter(emotions)
        return emotion_counts.most_common(1)[0][0]

    def _resolve_media_path(self, file_name: str, base_dir: str) -> str:
        """解析媒体文件路径"""
        if not file_name:
            return ""

        # 如果是绝对路径，直接返回
        if os.path.isabs(file_name):
            return file_name

        # 构建完整路径
        full_path = os.path.join(self.root, base_dir, file_name)
        return full_path if os.path.exists(full_path) else ""

    def _process_modality_annotation(self, json_file: str, modality: str) -> List[Dict]:
        """处理单一模态的标注"""
        data = self._load_json_annotation(json_file)
        if not data:
            return []

        samples = []

        # 基础信息
        speaker_id = data.get("speaker_id", "")
        file_name = data.get("file_name", "")

        # 根据模态提取内容
        content = ""
        if modality == "text":
            content = data.get("content", "")

        # 提取情感标签
        emotion = self._extract_emotion_from_data(data.get("data", []))
        if emotion not in EMOTIONTALK_LABELS:
            emotion = "neutral"

        # 解析媒体文件路径
        media_path = ""
        if modality == "audio":
            media_path = self._resolve_media_path(
                file_name, self.config.audio_wav_dir)
        elif modality == "video":
            media_path = self._resolve_media_path(
                file_name, self.config.video_mp4_dir)

        sample = {
            "id": f"emotiontalk_{modality}_{os.path.basename(json_file).replace('.json', '')}",
            "modality": modality,
            "speaker_id": speaker_id,
            "file_name": file_name,
            "content": content,
            "label": emotion,
            "label_space": EMOTIONTALK_LABELS,
            "text": content if modality == "text" else "",
            "audio_path": media_path if modality == "audio" else "",
            "video_path": media_path if modality == "video" else "",
            "json_path": json_file
        }

        samples.append(sample)
        return samples

    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代EmotionTalk样本"""
        all_samples = []

        # 处理各模态的标注
        modalities = [
            ("text", self.config.text_json_dir),
            ("audio", self.config.audio_json_dir),
            ("video", self.config.video_json_dir)
        ]

        if self.config.include_multimodal:
            modalities.append(("multimodal", self.config.multimodal_json_dir))

        for modality, json_dir in modalities:
            json_files = self._collect_json_files(json_dir)

            for json_file in json_files:
                samples = self._process_modality_annotation(
                    json_file, modality)
                all_samples.extend(samples)

        # 如果启用了模态合并，按文件名合并不同模态的样本
        if self.config.merge_modalities:
            merged_samples = self._merge_modalities(all_samples)
            for sample in merged_samples:
                yield sample
        else:
            for sample in all_samples:
                yield sample

    def _merge_modalities(self, samples: List[Dict]) -> List[Dict]:
        """合并不同模态的样本"""
        # 按基础文件名分组
        groups = {}
        for sample in samples:
            base_key = sample["speaker_id"] + "_" + \
                os.path.basename(sample["file_name"]).split('.')[0]
            if base_key not in groups:
                groups[base_key] = {}
            groups[base_key][sample["modality"]] = sample

        merged_samples = []
        for i, (key, group) in enumerate(groups.items()):
            # 使用第一个样本的基础信息
            first_sample = next(iter(group.values()))

            # 合并所有模态的信息
            merged_sample = {
                "id": f"emotiontalk_merged_{i}",
                "speaker_id": first_sample["speaker_id"],
                "file_name": first_sample["file_name"],
                "label": first_sample.get("label") if first_sample.get("label") in EMOTIONTALK_LABELS else "neutral",
                "label_space": EMOTIONTALK_LABELS,
            }

            # 合并文本内容
            if "text" in group:
                merged_sample["text"] = group["text"]["content"]

            # 合并媒体路径
            if "audio" in group:
                merged_sample["audio_path"] = group["audio"]["audio_path"]
            else:
                merged_sample["audio_path"] = ""

            if "video" in group:
                merged_sample["video_path"] = group["video"]["video_path"]
            else:
                merged_sample["video_path"] = ""

            # 保存各模态的原始标签
            for modality, sample in group.items():
                merged_sample[f"{modality}_label"] = sample["label"]

            merged_samples.append(merged_sample)

        return merged_samples

    def evaluate(self, adapter, split: str = "test") -> Dict:
        """评估EmotionTalk性能"""
        y_true, y_pred = [], []

        for item in self._safe_iter_samples(split):
            pred = adapter.predict(item, task="emotiontalk_dialog_emotion")
            y_true.append(item["label"])
            y_pred.append(pred.get("label", ""))

        # 计算指标
        acc = accuracy(y_true, y_pred)
        weighted_f1 = precision_recall_f1(
            y_true, y_pred, average="weighted")["f1"]
        macro_f1 = precision_recall_f1(y_true, y_pred, average="macro")["f1"]

        return {
            "acc": acc,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "total_samples": len(y_true)
        }
