#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_datasets.py
—— 数据集测试脚本，用于验证所有数据集的加载和处理功能。

vllm serve /home/remote1/lvshuyang/Models/Qwen/Qwen3-Omni-30B-A3B-Thinking --host 0.0.0.0 --port 8080 --served-model-name qwen3-omni --api-key g203 --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.95 --enable-prefix-caching --generation-config vllm --trust-remote-code --revision main
"""
from adapters.base import BaseAdapter
# from adapters.qwen_adapter import QwenAdapter
from adapters.qwen_adapter_vllm import Adapter
from datasets import *
import os
import sys
import json
import logging
import itertools
from typing import Dict


# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestAdapter(BaseAdapter):
    """测试用的模拟适配器"""

    def predict(self, item: Dict, task: str) -> Dict:
        """返回固定的预测结果用于测试"""
        if task in ["image_emotion_class", "video_emotion_class", "meld_dialog_emotion", "emotiontalk_dialog_emotion"]:
            # 分类任务
            if hasattr(item, 'get') and 'label_space' in item:
                label_space = item['label_space']
                label = label_space[0] if label_space else "neutral"
            else:
                label = "neutral"
            return {"label": label}

        elif task in ["image_va_reg", "video_va_reg"]:
            # VA回归任务
            return {"valence": 0.0, "arousal": 0.0}

        elif task in ["mosei_sentiment", "chsims_sentiment"]:
            # 情感强度任务
            return {"polarity": 0.0}

        return {}


class DatasetTester:
    """数据集测试器"""

    def __init__(self, base_url="http://localhost:8080/v1", model_name="qwen3-omni", max_samples=None, max_frames=8):
        """
        初始化数据集测试器

        Args:
            base_url: vLLM服务地址
            model_name: 模型名称
            max_samples: 每个数据集最大测试样本数，None表示使用配置文件中的设置
            max_frames: 视频最大帧数（考虑到32K上下文限制）
        """
        self.adapter = Adapter(
            base_url=base_url,
            model_name=model_name,
            max_frames=max_frames,
            video_strategy="uniform"  # 均匀采样策略
        )
        self.max_samples = max_samples
        self.results = {}

    def test_dataset(self, dataset_name: str, dataset_class, config_file: str) -> Dict:
        """测试单个数据集"""
        logger.info(f"开始测试数据集: {dataset_name}")

        try:
            # 加载配置
            config_path = os.path.join("configs", config_file)
            if not os.path.exists(config_path):
                return {"status": "error", "message": f"配置文件不存在: {config_path}"}

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 设置最大样本数（优先使用命令行参数，然后是配置文件，最后是默认值）
            if self.max_samples is not None:
                config["max_samples"] = self.max_samples
            elif config.get("max_samples") in (None, 0):
                config["max_samples"] = max(5, config.get("probe_samples", 5))

            # 创建数据集实例
            dataset = dataset_class(config)

            # 测试基本属性
            result = {
                "dataset_name": dataset_name,
                "status": "success",
                "config_loaded": True,
                "paths_verified": False,
                "samples_loaded": False,
                "evaluation_passed": False,
                "sample_count": 0,
                "errors": []
            }

            # 验证路径
            try:
                required_paths = dataset._get_required_paths()
                missing_paths = []
                for path in required_paths:
                    full_path = os.path.join(dataset.root, path)
                    if not os.path.exists(full_path):
                        missing_paths.append(full_path)

                if missing_paths:
                    result["errors"].append(f"缺失路径: {missing_paths}")
                else:
                    result["paths_verified"] = True
                    logger.info(f"路径验证通过: {len(required_paths)} 个路径")
            except Exception as e:
                result["errors"].append(f"路径验证失败: {e}")

            # 测试样本加载
            test_split = config.get(
                "test_split") or config.get("split") or "test"
            probe_size = int(config.get("probe_samples", 5))

            try:
                samples = list(itertools.islice(
                    dataset.iter_samples(test_split), probe_size))
                result["sample_count"] = len(samples)
                result["samples_loaded"] = True

                # 检查样本格式
                if samples:
                    sample = samples[0]
                    required_fields = ["id"]
                    missing_fields = [
                        f for f in required_fields if f not in sample]
                    if missing_fields:
                        result["errors"].append(f"样本缺少字段: {missing_fields}")

                logger.info(f"成功加载 {len(samples)} 个样本")
            except Exception as e:
                result["errors"].append(f"样本加载失败: {e}")

            # 测试评估
            try:
                if result["samples_loaded"] and result["sample_count"] > 0:
                    eval_result = dataset.evaluate(self.adapter, test_split)
                    result["evaluation_passed"] = True
                    result["evaluation_metrics"] = eval_result
                    logger.info(f"评估通过，指标: {list(eval_result.keys())}")
                    if "top1_acc" in eval_result:
                        accuracy = eval_result["top1_acc"]
                        # 在少量样本测试中，我们只要求它不是0，证明至少对了一个
                        # 在正式的大规模测试中，可以设置一个更高的门槛，比如 0.1 (10%)
                        min_accuracy_threshold = 0.001

                        logger.info(f"检查模型准确率: {accuracy:.4f}")
                        if accuracy < min_accuracy_threshold:
                            # 如果准确率低于门槛（这里是几乎为0），则添加一个错误信息
                            error_msg = (
                                f"性能警报: 模型准确率 ({accuracy:.4f}) "
                                f"低于设定的最低门槛 ({min_accuracy_threshold}). "
                                f"模型可能完全没有预测对任何样本。"
                            )
                            result["errors"].append(error_msg)
                            logger.warning(error_msg)
                elif result["sample_count"] == 0:
                    result["errors"].append("样本数量为0，跳过评估")
            except Exception as e:
                result["errors"].append(f"评估失败: {e}")

            # 如果有错误，标记为失败
            if result["errors"]:
                result["status"] = "partial_failure"

            return result

        except Exception as e:
            return {
                "dataset_name": dataset_name,
                "status": "error",
                "message": str(e),
                "errors": [str(e)]
            }

    def test_all_datasets(self) -> Dict:
        """测试所有数据集"""
        # 定义所有要测试的数据集
        datasets_to_test = [
            ("AffectNet_Classifier", AffectNetClassifier, "affectnet.json"),
            ("AffectNet_VA", AffectNetVA, "affectnet.json"),
            ("AffWild2_VA", AffWild2VA, "affwild2_va.json"),
            ("AffWild2_EXPR", AffWild2EXPR, "affwild2_expr.json"),
            ("EmotionTalk", EmotionTalk, "emotiontalk.json"),
            ("MELD", MELD, "meld.json"),
            ("MEMO_Bench", MEMOBench, "memo_bench.json"),
            ("CH_SIMS", CHSIMS, "ch_sims.json"),
            ("CH_SIMS_V2", CHSIMSV2, "ch_sims_v2.json"),
            ("CMU_MOSEI", CMUMOSEI, "cmu_mosei.json"),
            ("CMU_MOSI", CMUMOSI, "cmu_mosi.json"),
            ("DFEW", DFEW, "dfew.json"),
        ]

        results = {}

        for dataset_name, dataset_class, config_file in datasets_to_test:
            try:
                result = self.test_dataset(
                    dataset_name, dataset_class, config_file)
                results[dataset_name] = result

                # 打印简要结果
                status = result["status"]
                sample_count = result.get("sample_count", 0)
                error_count = len(result.get("errors", []))

                logger.info(
                    f"{dataset_name}: {status}, 样本数: {sample_count}, 错误数: {error_count}")

                if result["errors"]:
                    for error in result["errors"]:
                        logger.warning(f"  {error}")

            except Exception as e:
                logger.error(f"测试 {dataset_name} 时发生异常: {e}")
                results[dataset_name] = {
                    "dataset_name": dataset_name,
                    "status": "error",
                    "message": str(e)
                }

        return results

    def generate_report(self, results: Dict) -> str:
        """生成测试报告"""
        report = []
        report.append("# 数据集测试报告\n")
        report.append(f"测试时间: {os.popen('date').read().strip()}\n")

        # 统计信息
        total_datasets = len(results)
        success_count = sum(1 for r in results.values()
                            if r.get("status") == "success")
        partial_count = sum(1 for r in results.values()
                            if r.get("status") == "partial_failure")
        error_count = sum(1 for r in results.values()
                          if r.get("status") == "error")

        report.append("## 总体统计\n")
        report.append(f"- 总数据集数: {total_datasets}")
        report.append(f"- 完全成功: {success_count}")
        report.append(f"- 部分成功: {partial_count}")
        report.append(f"- 失败: {error_count}")
        report.append(f"- 成功率: {success_count/total_datasets*100:.1f}%\n")

        # 详细结果
        report.append("## 详细结果\n")

        for dataset_name, result in results.items():
            status = result.get("status", "unknown")
            status_emoji = {
                "success": "✅", "partial_failure": "⚠️", "error": "❌"}.get(status, "❓")

            report.append(f"### {status_emoji} {dataset_name}\n")
            report.append(f"**状态**: {status}\n")

            if status == "success":
                sample_count = result.get("sample_count", 0)
                report.append(f"**样本数**: {sample_count}\n")

                if "evaluation_metrics" in result:
                    metrics = result["evaluation_metrics"]
                    report.append(f"**评估指标**: {', '.join(metrics.keys())}\n")

            elif status == "partial_failure":
                sample_count = result.get("sample_count", 0)
                report.append(f"**样本数**: {sample_count}\n")
                report.append("**错误**:\n")
                for error in result.get("errors", []):
                    report.append(f"- {error}\n")

            else:  # error
                report.append(f"**错误**: {result.get('message', '未知错误')}\n")

            report.append("---\n")

        # 问题汇总
        report.append("## 问题汇总\n")
        all_errors = []
        for result in results.values():
            all_errors.extend(result.get("errors", []))

        if all_errors:
            error_summary = {}
            for error in all_errors:
                error_type = error.split(':')[0] if ':' in error else 'other'
                error_summary[error_type] = error_summary.get(
                    error_type, 0) + 1

            for error_type, count in sorted(error_summary.items()):
                report.append(f"- {error_type}: {count} 次")

        return "\n".join(report)

    def run_tests(self, save_report: bool = True):
        """运行所有测试"""
        logger.info("开始运行数据集测试...")

        results = self.test_all_datasets()

        # 生成报告
        report = self.generate_report(results)

        if save_report:
            report_file = "dataset_test_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"测试报告已保存到: {report_file}")

        # 打印报告
        print("\n" + "="*50)
        print("测试完成！")
        print("="*50)
        print(report)

        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试所有数据集的加载和处理功能")
    parser.add_argument("--save-report", action="store_true", default=True,
                        help="保存测试报告到文件")
    parser.add_argument("--dataset", type=str,
                        help="测试特定数据集（可选）")
    parser.add_argument("--base-url", type=str, default="http://localhost:8080/v1",
                        help="vLLM服务地址")
    parser.add_argument("--model-name", type=str, default="qwen3-omni",
                        help="模型名称")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="每个数据集最大测试样本数")
    parser.add_argument("--dry-run", action="store_true",
                        help="只测试数据加载，不进行模型推理")

    args = parser.parse_args()

    tester = DatasetTester(
        base_url=args.base_url,
        model_name=args.model_name,
        max_samples=args.max_samples
    )

    if args.dataset:
        # 测试特定数据集
        logger.info(f"测试指定数据集: {args.dataset}")
        # 查找对应的数据集配置
        dataset_map = {
            "AffectNet_Classifier": (AffectNetClassifier, "affectnet.json"),
            "AffectNet_VA": (AffectNetVA, "affectnet.json"),
            "AffWild2_VA": (AffWild2VA, "affwild2_va.json"),
            "AffWild2_EXPR": (AffWild2EXPR, "affwild2_expr.json"),
            "EmotionTalk": (EmotionTalk, "emotiontalk.json"),
            "MELD": (MELD, "meld.json"),
            "MEMO_Bench": (MEMOBench, "memo_bench.json"),
            "CH_SIMS": (CHSIMS, "ch_sims.json"),
            "CH_SIMS_V2": (CHSIMSV2, "ch_sims_v2.json"),
            "CMU_MOSEI": (CMUMOSEI, "cmu_mosei.json"),
            "CMU_MOSI": (CMUMOSI, "cmu_mosi.json"),
            "DFEW": (DFEW, "dfew.json"),
        }

        if args.dataset in dataset_map:
            dataset_class, config_file = dataset_map[args.dataset]
            result = tester.test_dataset(
                args.dataset, dataset_class, config_file)

            # 生成报告
            report = tester.generate_report({args.dataset: result})

            if args.save_report:
                report_file = f"dataset_test_{args.dataset.lower()}_report.md"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"测试报告已保存到: {report_file}")

            print(report)
        else:
            logger.error(f"未知数据集: {args.dataset}")
            logger.info(f"可用数据集: {', '.join(dataset_map.keys())}")
    else:
        # 测试所有数据集
        tester.run_tests(args.save_report)


if __name__ == "__main__":
    main()
