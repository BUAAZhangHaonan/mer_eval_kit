# -*- coding: utf-8 -*-
"""
datasets/base.py
—— 统一的数据集基类，定义标准接口和通用功能。
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Any, Optional
from dataclasses import dataclass

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """数据集配置基类"""
    root: str
    name: str
    # 通用配置
    verify_files: bool = True  # 是否验证文件存在性
    strict_mode: bool = False  # 严格模式：遇到错误时抛出异常
    max_samples: Optional[int] = None  # 最大样本数限制（用于调试）


class BaseDataset(ABC):
    """数据集基类，定义统一接口"""

    def __init__(self, config: Dict):
        """初始化数据集

        Args:
            config: 数据集配置字典
        """
        self.config = self.config_class(**config)
        self.name = self.config.name
        self.root = self.config.root

        # 验证根目录
        if not os.path.exists(self.root):
            error_msg = f"数据集根目录不存在: {self.root}"
            if self.config.strict_mode:
                raise FileNotFoundError(error_msg)
            else:
                logger.error(error_msg)

        # 验证数据集
        if self.config.verify_files:
            self._verify_dataset()

    @property
    @abstractmethod
    def config_class(self):
        """返回对应的配置类"""
        pass

    @abstractmethod
    def iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """迭代数据样本

        Args:
            split: 数据集分割 ("train", "val", "test")

        Yields:
            Dict: 样本数据，必须包含id字段
        """
        pass

    @abstractmethod
    def evaluate(self, adapter, split: str = "test") -> Dict:
        """评估模型性能

        Args:
            adapter: 模型适配器
            split: 数据集分割

        Returns:
            Dict: 评估指标
        """
        pass

    def _verify_dataset(self) -> bool:
        """验证数据集完整性

        Returns:
            bool: 验证是否通过
        """
        logger.info(f"验证数据集 {self.name}...")
        try:
            # 检查必要文件/目录
            required_paths = self._get_required_paths()
            missing_paths = []

            for path in required_paths:
                full_path = os.path.join(
                    self.root, path) if not os.path.isabs(path) else path
                if not os.path.exists(full_path):
                    missing_paths.append(full_path)

            if missing_paths:
                error_msg = f"数据集 {self.name} 缺少必要文件/目录: {missing_paths}"
                if self.config.strict_mode:
                    raise FileNotFoundError(error_msg)
                else:
                    logger.warning(error_msg)
                    return False

            logger.info(f"数据集 {self.name} 验证通过")
            return True

        except Exception as e:
            logger.error(f"验证数据集 {self.name} 时出错: {e}")
            if self.config.strict_mode:
                raise
            return False

    def _get_required_paths(self) -> List[str]:
        """返回数据集必要的文件/目录路径列表

        子类应该重写此方法来指定必要的路径
        """
        return []

    def _safe_iter_samples(self, split: str = "test") -> Iterator[Dict]:
        """安全地迭代样本，包含错误处理

        Args:
            split: 数据集分割

        Yields:
            Dict: 样本数据
        """
        sample_count = 0
        try:
            for sample in self.iter_samples(split):
                # 验证样本格式
                if not self._validate_sample(sample):
                    if self.config.strict_mode:
                        raise ValueError(f"样本格式无效: {sample}")
                    else:
                        logger.warning(f"跳过无效样本: {sample}")
                        continue

                yield sample
                sample_count += 1

                # 限制样本数量
                if self.config.max_samples and sample_count >= self.config.max_samples:
                    logger.info(f"达到最大样本数量限制: {self.config.max_samples}")
                    break

        except Exception as e:
            logger.error(f"迭代样本时出错: {e}")
            if self.config.strict_mode:
                raise

    def _validate_sample(self, sample: Dict) -> bool:
        """验证单个样本的格式

        Args:
            sample: 样本数据

        Returns:
            bool: 是否有效
        """
        # 检查必要字段
        if "id" not in sample:
            return False

        # 子类可以重写此方法添加更多验证
        return True

    def get_stats(self, split: str = "test") -> Dict[str, Any]:
        """获取数据集统计信息

        Args:
            split: 数据集分割

        Returns:
            Dict: 统计信息
        """
        stats = {
            "name": self.name,
            "split": split,
            "total_samples": 0,
            "valid_samples": 0,
            "errors": 0
        }

        for sample in self._safe_iter_samples(split):
            stats["total_samples"] += 1
            if self._validate_sample(sample):
                stats["valid_samples"] += 1
            else:
                stats["errors"] += 1

        return stats

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', root='{self.root}')"


class MultiModalDataset(BaseDataset):
    """多模态数据集基类"""

    def _validate_sample(self, sample: Dict) -> bool:
        """验证多模态样本"""
        if not super()._validate_sample(sample):
            return False

        # 检查至少有一种模态
        modalities = ["text", "image", "image_path",
                      "video", "video_path", "audio", "audio_path"]
        has_modality = any(key in sample for key in modalities)

        return has_modality


class ClassificationDataset(BaseDataset):
    """分类任务数据集基类"""

    def _validate_sample(self, sample: Dict) -> bool:
        """验证分类样本"""
        if not super()._validate_sample(sample):
            return False

        # 检查标签字段
        if "label" not in sample:
            return False

        return True


class RegressionDataset(BaseDataset):
    """回归任务数据集基类"""

    def _validate_sample(self, sample: Dict) -> bool:
        """验证回归样本"""
        if not super()._validate_sample(sample):
            return False

        # 检查至少有一个数值字段
        numeric_fields = ["valence", "arousal", "polarity", "label"]
        has_numeric = any(key in sample for key in numeric_fields)

        return has_numeric
