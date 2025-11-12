# -*- coding: utf-8 -*-
"""
datasets/affectnet.py
—— AffectNet 的分类与 V/A 读取器。
根据实际数据集格式，读取.npy格式的标注文件和对应的图像文件。
"""
import os
import numpy as np
import glob
import logging
from typing import Dict, Iterator, List
from dataclasses import dataclass

from datasets.base import ClassificationDataset, RegressionDataset, DatasetConfig
from datasets.label_spaces import AFFECTNET_8, AFFECTNET_7
from datasets.metrics import accuracy, precision_recall_f1, ccc, mae, pearsonr

logger = logging.getLogger(__name__)


@dataclass
class AffectNetConfig(DatasetConfig):
    """AffectNet数据集配置"""
    train_dir: str = "train_set"
    val_dir: str = "val_set"
    drop_contempt: bool = False  # 是否7类（移除 contempt）

    # 图像文件扩展名
    image_extensions: List[str] = None

    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']


class AffectNetClassifier(ClassificationDataset):
    """AffectNet 7/8 类表情分类评测。"""

    @property
    def config_class(self):
        return AffectNetConfig

    def _get_required_paths(self) -> List[str]:
        """返回AffectNet数据集必要的路径"""
        paths = []
        for split_dir in [self.config.train_dir, self.config.val_dir]:
            paths.extend([
                f"{split_dir}/annotations",
                f"{split_dir}/images"
            ])
        return paths

    def _get_annotation_files(self, split_dir: str) -> List[str]:
        """获取标注文件列表"""
        ann_dir = os.path.join(self.root, split_dir, "annotations")
        exp_files = glob.glob(os.path.join(ann_dir, "*_exp.npy"))
        return sorted(exp_files)

    def _get_image_file(self, split_dir: str, image_id: str) -> str:
        """根据图片ID查找对应的图像文件"""
        img_dir = os.path.join(self.root, split_dir, "images")

        for ext in self.config.image_extensions:
            # 尝试不同的文件名格式
            possible_names = [
                f"{image_id}{ext}",
                f"{image_id}.{ext[1:]}",  # 去掉点号
            ]

            for name in possible_names:
                img_path = os.path.join(img_dir, name)
                if os.path.exists(img_path):
                    return img_path

        return ""

    def _load_annotations(self, exp_file: str) -> Dict:
        """加载单个样本的所有标注"""
        try:
            # 提取图片ID
            base_name = os.path.basename(exp_file)
            image_id = base_name.replace("_exp.npy", "")

            # 加载所有标注文件
            ann_dir = os.path.dirname(exp_file)
            annotations = {}

            # 表情标注
            annotations['expression'] = int(np.load(exp_file).item())

            # 效价标注
            val_file = os.path.join(ann_dir, f"{image_id}_val.npy")
            if os.path.exists(val_file):
                annotations['valence'] = float(np.load(val_file).item())

            # 唤醒度标注
            aro_file = os.path.join(ann_dir, f"{image_id}_aro.npy")
            if os.path.exists(aro_file):
                annotations['arousal'] = float(np.load(aro_file).item())

            # 关键点标注（可选）
            lnd_file = os.path.join(ann_dir, f"{image_id}_lnd.npy")
            if os.path.exists(lnd_file):
                annotations['landmarks'] = np.load(lnd_file).tolist()

            return image_id, annotations

        except Exception as e:
            logger.warning(f"加载标注文件 {exp_file} 失败: {e}")
            return None, {}

    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代数据样本"""
        if split == "train":
            split_dir = self.config.train_dir
        elif split == "val":
            split_dir = self.config.val_dir
        else:
            split_dir = self.config.val_dir  # 默认使用验证集

        label_space = AFFECTNET_7 if self.config.drop_contempt else AFFECTNET_8
        exp_files = self._get_annotation_files(split_dir)

        sample_count = 0
        for exp_file in exp_files:
            image_id, annotations = self._load_annotations(exp_file)
            if image_id is None:
                continue

            # 获取标签
            exp_idx = annotations.get('expression', -1)
            if exp_idx < 0 or exp_idx >= len(AFFECTNET_8):
                continue

            label = AFFECTNET_8[exp_idx]

            # 如果是7类模式且标签为contempt，跳过
            if self.config.drop_contempt and label == "contempt":
                continue

            # 查找图像文件
            img_path = self._get_image_file(split_dir, image_id)
            if not img_path or not os.path.exists(img_path):
                logger.warning(f"找不到图像文件: {image_id}")
                continue

            yield {
                "id": f"affectnet_{split}_{sample_count}",
                "image_id": image_id,
                "image_path": img_path,
                "label": label,
                "label_space": label_space,
                "expression_idx": exp_idx,
                **{k: v for k, v in annotations.items() if k != 'expression'}
            }

            sample_count += 1

    def evaluate(self, adapter, split: str = "val") -> Dict:
        """评估分类性能"""
        y_true, y_pred = [], []

        for item in self._safe_iter_samples(split):
            pred = adapter.predict(item, task="image_emotion_class")
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


class AffectNetVA(RegressionDataset):
    """AffectNet 的 V/A 回归评测。"""

    @property
    def config_class(self):
        return AffectNetConfig

    def _get_required_paths(self) -> List[str]:
        """返回AffectNet数据集必要的路径"""
        return AffectNetClassifier._get_required_paths(self)

    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代VA回归样本"""
        # 复用分类器的样本生成逻辑
        classifier = AffectNetClassifier(self.config.__dict__)

        sample_count = 0
        for item in classifier._safe_iter_samples(split):
            # 检查是否有VA标注
            if 'valence' not in item or 'arousal' not in item:
                continue

            # 检查VA值是否有效（-2表示无效）
            if item['valence'] == -2 or item['arousal'] == -2:
                continue

            yield {
                "id": f"affectnet_va_{split}_{sample_count}",
                "image_id": item["image_id"],
                "image_path": item["image_path"],
                "valence": max(-1.0, min(1.0, item['valence'])),
                "arousal": max(-1.0, min(1.0, item['arousal'])),
                "expression_idx": item.get("expression_idx", -1)
            }

            sample_count += 1

    def evaluate(self, adapter, split: str = "val") -> Dict:
        """评估VA回归性能"""
        yv_true, yv_pred = [], []
        ya_true, ya_pred = [], []

        for item in self._safe_iter_samples(split):
            pred = adapter.predict(item, task="image_va_reg")
            v = float(pred.get("valence", 0.0))
            a = float(pred.get("arousal", 0.0))

            yv_true.append(item["valence"])
            yv_pred.append(v)
            ya_true.append(item["arousal"])
            ya_pred.append(a)

        # 计算指标
        out = {
            "valence_ccc": ccc(yv_true, yv_pred),
            "arousal_ccc": ccc(ya_true, ya_pred),
            "valence_mae": mae(yv_true, yv_pred),
            "arousal_mae": mae(ya_true, ya_pred),
            "valence_corr": pearsonr(yv_true, yv_pred),
            "arousal_corr": pearsonr(ya_true, ya_pred),
            "total_samples": len(yv_true)
        }

        out["mean_ccc"] = 0.5 * (out["valence_ccc"] + out["arousal_ccc"])
        return out
