# -*- coding: utf-8 -*-
"""
datasets/dfew.py
—— DFEW：电影片段的动态表情分类。主指标：UAR + WAR（Top-1 = WAR）。
读取 annotation.xlsx 和五折交叉验证数据，支持多种视频格式。
"""
import os
import csv
import logging
import pandas as pd
from typing import Dict, Iterator, List
from dataclasses import dataclass
import cv2
from PIL import Image
import torch

from datasets.base import ClassificationDataset, DatasetConfig
from datasets.label_spaces import DFEW_7, normalize_label
from datasets.metrics import uar, war, accuracy, precision_recall_f1

logger = logging.getLogger(__name__)

# DFEW标签映射
DFEW_LABEL_MAPPING = {
    0: "non_single_label",
    1: "happy",
    2: "sad",
    3: "neutral",
    4: "anger",
    5: "surprise",
    6: "disgust",
    7: "fear",
}


@dataclass
class DFEWConfig(DatasetConfig):
    """DFEW数据集配置"""
    # 标注文件
    annotation_xlsx: str = "Annotation/annotation.xlsx"

    # 数据分割目录
    data_split_dir: str = "EmoLabel_DataSplit"

    # 视频目录
    clip_dir: str = "Clip/clip_224x224"
    clip_16f_dir: str = "Clip/clip_224x224_16f"
    clip_avi_dir: str = "Clip/clip_224x224_avi"
    original_dir: str = "Clip/original"

    # 处理选项
    use_preprocessed: bool = True   # 是否使用预处理视频
    use_16_frames: bool = False     # 是否使用16帧版本
    video_extensions: List[str] = None

    def __post_init__(self):
        if self.video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov']


class DFEW(ClassificationDataset):
    """DFEW 动态表情分类评测器"""

    @property
    def config_class(self):
        return DFEWConfig

    def _get_required_paths(self) -> List[str]:
        """返回DFEW数据集必要的路径"""
        return [
            self.config.annotation_xlsx,
            self.config.data_split_dir,
            self.config.clip_dir,
            self.config.clip_16f_dir,
            self.config.clip_avi_dir,
            self.config.original_dir
        ]

    def _load_annotation_xlsx(self) -> Dict:
        """加载annotation.xlsx文件。如果文件不包含clip名称列，则返回空映射。"""
        annotation_file = os.path.join(self.root, self.config.annotation_xlsx)

        if not os.path.exists(annotation_file):
            logger.warning(f"标注文件不存在，跳过: {annotation_file}")
            return {}

        try:
            df = pd.read_excel(annotation_file)

            # 寻找包含clip名称的列
            candidate_cols = [
                col for col in df.columns
                if any(key in str(col).lower() for key in ("clip", "video", "name"))
            ]

            if not candidate_cols:
                logger.warning("annotation.xlsx不包含clip名称列，将使用分割文件中的标签")
                return {}

            clip_col = candidate_cols[0]
            label_col = None
            for key in ("label", "annotation", "emotion"):
                if key in (c.lower() for c in df.columns):
                    label_col = next(
                        col for col in df.columns if col.lower() == key
                    )
                    break

            if label_col is None:
                logger.warning("annotation.xlsx未找到标签列，将使用分割文件中的标签")
                return {}

            annotations = {}
            for _, row in df.iterrows():
                clip_name = str(row.get(clip_col, "")).strip()
                if not clip_name:
                    continue
                try:
                    annotation = int(row.get(label_col, 0))
                except (TypeError, ValueError):
                    continue
                annotations[clip_name] = annotation

            return annotations
        except Exception as e:
            logger.error(f"读取标注文件失败: {e}")
            return {}

    def _load_split_csv(self, split: str, fold: int = 1) -> List[Dict[str, str]]:
        """加载分割CSV文件，返回包含clip名称和标签的字典列表"""
        if split == "train":
            csv_file = os.path.join(
                self.root, self.config.data_split_dir, f"train(single-labeled)/set_{fold}.csv"
            )
        else:
            csv_file = os.path.join(
                self.root, self.config.data_split_dir, f"test(single-labeled)/set_{fold}.csv"
            )

        if not os.path.exists(csv_file):
            logger.error(f"分割文件不存在: {csv_file}")
            return []

        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                fieldnames = [fn.lower() for fn in reader.fieldnames or []]

                clip_key = None
                for key in ("clip_name", "video_name", "video", "clip"):
                    if key in fieldnames:
                        clip_key = reader.fieldnames[fieldnames.index(key)]
                        break

                if clip_key is None:
                    logger.error(f"分割文件缺少clip名称列: {csv_file}")
                    return []

                label_key = None
                for key in ("label", "annotation", "emotion"):
                    if key in fieldnames:
                        label_key = reader.fieldnames[fieldnames.index(key)]
                        break

                samples = []
                for row in reader:
                    clip_name = (row.get(clip_key) or "").strip()
                    if not clip_name:
                        continue
                    annotation_raw = row.get(
                        label_key, "0") if label_key else "0"
                    try:
                        annotation = int(float(annotation_raw))
                    except (TypeError, ValueError):
                        annotation = 0
                    samples.append({
                        "clip_name": clip_name,
                        "annotation": annotation
                    })

                return samples
        except Exception as e:
            logger.error(f"读取分割文件失败: {e}")
            return []

    def _find_video_file(self, clip_name: str) -> str:
        """查找视频文件，优先使用预处理目录，找不到时回退到原始目录"""
        search_dirs = []

        if self.config.use_preprocessed:
            if self.config.use_16_frames:
                search_dirs.append(os.path.join(
                    self.root, self.config.clip_16f_dir))
            search_dirs.append(os.path.join(self.root, self.config.clip_dir))

        # 始终尝试原始目录作为兜底
        search_dirs.append(os.path.join(self.root, self.config.original_dir))

        for video_dir in search_dirs:
            if not os.path.exists(video_dir):
                continue

            for ext in self.config.video_extensions:
                video_path = os.path.join(video_dir, f"{clip_name}{ext}")
                if os.path.exists(video_path):
                    return video_path

            frame_dir = os.path.join(video_dir, clip_name)
            if os.path.isdir(frame_dir):
                return frame_dir

        return ""

    def _map_annotation_to_label(self, annotation: int) -> str:
        """将标注映射到标签"""
        label = DFEW_LABEL_MAPPING.get(annotation, "neutral")

        # 跳过非单标签样本
        if label == "non_single_label":
            return ""

        return normalize_label(label)

    def iter_samples(self, split: str = "test", fold: int = 1) -> Iterator[Dict]:
        """迭代DFEW样本"""
        # 加载标注
        annotations = self._load_annotation_xlsx()

        # 加载分割
        split_items = self._load_split_csv(split, fold)

        sample_count = 0

        for item in split_items:
            clip_name = item["clip_name"]

            annotation = annotations.get(clip_name, item.get("annotation", 0))
            label = self._map_annotation_to_label(annotation)

            if not label:  # 跳过非单标签样本
                continue

            # 查找视频文件
            video_path = self._find_video_file(clip_name)

            # 优化视频帧加载：限制为8帧以适应32K上下文
            frames = self._load_frames_from_path(video_path, num_frames=8)

            if not frames and not video_path:
                logger.warning(f"找不到视频文件: {clip_name}")
            elif not frames and video_path:
                logger.warning(f"视频文件存在但无法加载帧: {clip_name}")

            yield {
                "id": f"dfew_{split}_{fold}_{sample_count}",
                "clip_name": clip_name,
                "video_path": video_path,
                "label": label,
                "label_space": DFEW_7,
                "annotation": annotation,
                "fold": fold,
                "split": split,
                "frames": frames
            }

            sample_count += 1

    def evaluate(self, adapter, split: str = "test", fold: int = 1) -> Dict:
        """评估DFEW性能"""
        y_true, y_pred = [], []

        for item in self._safe_iter_samples(split):
            pred = adapter.predict(item, task="video_emotion_class")
            y_true.append(item["label"])
            y_pred.append(pred.get("label", ""))

        # 计算指标
        label_space = DFEW_7
        out = {
            "war": war(y_true, y_pred),
            "uar": uar(y_true, y_pred, labels=label_space),
            "top1_acc": accuracy(y_true, y_pred),
            "macro_f1": precision_recall_f1(y_true, y_pred, average="macro")["f1"],
            "weighted_f1": precision_recall_f1(y_true, y_pred, average="weighted")["f1"],
            "total_samples": len(y_true),
            "fold": fold
        }

        # 按情感类别的统计
        from collections import Counter
        true_dist = Counter(y_true)
        pred_dist = Counter(y_pred)
        out["emotion_distribution"] = dict(true_dist)
        out["prediction_distribution"] = dict(pred_dist)
        out["label_space"] = label_space

        return out

    def evaluate_cross_validation(self, adapter, folds: int = 5) -> Dict:
        """进行五折交叉验证评估"""
        all_results = []

        for fold in range(1, folds + 1):
            logger.info(f"评估第 {fold} 折...")
            result = self.evaluate(adapter, split="test", fold=fold)
            all_results.append(result)

        # 计算平均结果
        avg_result = {
            "avg_war": sum(r["war"] for r in all_results) / len(all_results),
            "avg_uar": sum(r["uar"] for r in all_results) / len(all_results),
            "avg_top1_acc": sum(r["top1_acc"] for r in all_results) / len(all_results),
            "avg_macro_f1": sum(r["macro_f1"] for r in all_results) / len(all_results),
            "avg_weighted_f1": sum(r["weighted_f1"] for r in all_results) / len(all_results),
            "total_samples": sum(r["total_samples"] for r in all_results),
            "fold_results": all_results
        }

        # 计算标准差
        import statistics
        avg_result["war_std"] = statistics.stdev(
            [r["war"] for r in all_results])
        avg_result["uar_std"] = statistics.stdev(
            [r["uar"] for r in all_results])
        avg_result["top1_acc_std"] = statistics.stdev(
            [r["top1_acc"] for r in all_results])
        avg_result["macro_f1_std"] = statistics.stdev(
            [r["macro_f1"] for r in all_results])

        return avg_result
    

    def _load_frames_from_path(self, path: str, num_frames: int = 8) -> List[Image.Image]:
        """从路径加载帧，支持视频文件和帧目录
        
        Args:
            path: 视频文件路径或帧目录路径
            num_frames: 要加载的帧数，默认8帧以适应32K上下文限制
            
        Returns:
            List[Image.Image]: PIL图像列表
        """
        frames = []
        if not path or not os.path.exists(path):
            logger.warning(f"路径不存在: {path}")
            return frames

        try:
            if os.path.isdir(path):
                # 处理帧目录
                frame_files = sorted([os.path.join(path, f) for f in os.listdir(path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                if not frame_files:
                    logger.warning(f"帧目录为空: {path}")
                    return frames
                
                # 均匀采样帧
                if len(frame_files) <= num_frames:
                    selected_files = frame_files
                else:
                    indices = torch.linspace(0, len(frame_files) - 1, num_frames, dtype=torch.long)
                    selected_files = [frame_files[i] for i in indices]
                
                for file_path in selected_files:
                    try:
                        img = Image.open(file_path).convert('RGB')
                        frames.append(img)
                    except Exception as e:
                        logger.warning(f"加载帧失败 {file_path}: {e}")
                        continue
                        
            else:
                # 处理视频文件
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    logger.warning(f"无法打开视频文件: {path}")
                    return frames
                
                try:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames == 0:
                        logger.warning(f"视频文件无帧: {path}")
                        return frames
                    
                    # 均匀采样帧
                    if total_frames <= num_frames:
                        indices = list(range(total_frames))
                    else:
                        indices = torch.linspace(0, total_frames - 1, num_frames, dtype=torch.long)
                        indices = indices.tolist()
                    
                    for i in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(Image.fromarray(frame_rgb))
                        else:
                            logger.warning(f"读取第{i}帧失败: {path}")
                            
                finally:
                    cap.release()
                    
        except Exception as e:
            logger.error(f"加载帧时发生错误 {path}: {e}")
            
        logger.debug(f"从 {path} 加载了 {len(frames)} 帧")
        return frames
