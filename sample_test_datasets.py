#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_test_datasets.py
—— 从所有数据集中挑选样本进行测试的评估脚本。

使用示例:
    # 从每个数据集挑选5个样本进行测试
    python sample_test_datasets.py --samples-per-dataset 5
    
    # 使用不同的vLLM服务地址
    python sample_test_datasets.py --base-url http://192.168.1.100:8080/v1 --samples-per-dataset 3

vLLM服务启动命令:
    vllm serve /path/to/Qwen3-Omni-30B-A3B-Thinking \
        --host 0.0.0.0 --port 8080 \
        --served-model-name qwen3-omni \
        --api-key g203 \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --max-num-seqs 4 \
        --max-num-batched-tokens 32768 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        --generation-config vllm \
        --trust-remote-code \
        --revision main
"""
from adapters.qwen_adapter_vllm import Adapter
from datasets import *
import os
import sys
import json
import logging
import itertools
from typing import Dict
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SampleTester:
    """样本测试器"""

    def __init__(self, base_url: str = "http://localhost:8080/v1",
                 model_name: str = "qwen3-omni",
                 samples_per_dataset: int = 5,
                 output_dir: str = "sample_test_results",
                 max_frames: int = 8,
                 dataset_filter: str = None):
        """
        初始化样本测试器

        Args:
            base_url: vLLM服务地址
            model_name: 模型名称
            samples_per_dataset: 每个数据集测试的样本数
            output_dir: 结果输出目录
            max_frames: 视频最大帧数
        """
        self.base_url = base_url
        self.model_name = model_name
        self.samples_per_dataset = samples_per_dataset
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.dataset_filter = dataset_filter

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化适配器
        self.adapter = Adapter(
            base_url=base_url,
            model_name=model_name,
            max_frames=max_frames,
            video_strategy="uniform"
        )

        # 定义所有数据集
        self.datasets_to_test = [
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

    def test_dataset_samples(self, dataset_name: str, dataset_class, config_file: str) -> Dict:
        """测试单个数据集的样本"""
        logger.info(f"开始测试数据集: {dataset_name}")

        try:
            # 加载配置
            config_path = os.path.join("configs", config_file)
            if not os.path.exists(config_path):
                return {"status": "error", "message": f"配置文件不存在: {config_path}"}

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 设置最大样本数为测试样本数
            config["max_samples"] = self.samples_per_dataset

            # 创建数据集实例
            dataset = dataset_class(config)

            # 获取测试样本
            test_split = config.get(
                "test_split") or config.get("split") or "test"
            samples = list(itertools.islice(
                dataset.iter_samples(test_split), self.samples_per_dataset))

            if not samples:
                return {"status": "error", "message": "无法获取样本"}

            results = {
                "dataset_name": dataset_name,
                "status": "success",
                "samples_tested": len(samples),
                "predictions": []
            }

            # 对每个样本进行预测
            for i, sample in enumerate(samples):
                logger.info(
                    f"测试样本 {i+1}/{len(samples)}: {sample.get('id', f'sample_{i}')}")

                # 确定任务类型
                task = self._determine_task_type(dataset_name)

                # 进行预测
                try:
                    prediction = self.adapter.predict(sample, task=task)
                    results["predictions"].append({
                        "sample_id": sample.get("id", f"sample_{i}"),
                        "true_label": sample.get("label"),
                        "prediction": prediction,
                        "task": task
                    })
                    logger.info(f"  真实标签: {sample.get('label')}")
                    logger.info(f"  预测结果: {prediction}")
                except Exception as e:
                    logger.error(f"  预测失败: {e}")
                    results["predictions"].append({
                        "sample_id": sample.get("id", f"sample_{i}"),
                        "true_label": sample.get("label"),
                        "prediction": {"error": str(e)},
                        "task": task
                    })

            return results

        except Exception as e:
            logger.error(f"测试数据集 {dataset_name} 失败: {e}")
            return {
                "dataset_name": dataset_name,
                "status": "error",
                "message": str(e)
            }

    def _determine_task_type(self, dataset_name: str) -> str:
        """根据数据集名称确定任务类型"""
        if "VA" in dataset_name:
            return "video_va_reg" if "video" in dataset_name.lower() else "image_va_reg"
        elif "EXPR" in dataset_name or "Classifier" in dataset_name:
            return "video_emotion_class" if "video" in dataset_name.lower() else "image_emotion_class"
        elif "EmotionTalk" in dataset_name or "MELD" in dataset_name:
            return "meld_dialog_emotion"
        elif "CH_SIMS" in dataset_name:
            return "chsims_sentiment"
        elif "MOSEI" in dataset_name or "MOSI" in dataset_name:
            return "mosei_sentiment"
        else:
            # 默认使用视频情感分类
            return "video_emotion_class"

    def run_sample_tests(self) -> Dict:
        """运行所有数据集的样本测试"""
        logger.info("开始样本测试...")

        all_results = {}

        for dataset_name, dataset_class, config_file in self.datasets_to_test:
            if self.dataset_filter and dataset_name != self.dataset_filter:
                continue
            try:
                result = self.test_dataset_samples(
                    dataset_name, dataset_class, config_file)
                all_results[dataset_name] = result

                status = result["status"]
                samples_tested = result.get("samples_tested", 0)

                logger.info(
                    f"{dataset_name}: {status}, 测试样本数: {samples_tested}")

            except Exception as e:
                logger.error(f"测试 {dataset_name} 时发生异常: {e}")
                all_results[dataset_name] = {
                    "dataset_name": dataset_name,
                    "status": "error",
                    "message": str(e)
                }

        return all_results

    def generate_report(self, results: Dict) -> str:
        """生成测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = []
        report.append("# Qwen3-Omni 样本测试报告\n")
        report.append(
            f"**测试时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        report.append(f"**模型**: {self.model_name}\n")
        report.append(f"**服务地址**: {self.base_url}\n")
        report.append(f"**每数据集样本数**: {self.samples_per_dataset}\n")

        # 总体统计
        total_datasets = len(results)
        success_count = sum(1 for r in results.values()
                            if r.get("status") == "success")
        error_count = total_datasets - success_count
        total_samples = sum(r.get("samples_tested", 0)
                            for r in results.values() if r.get("status") == "success")

        report.append("## 总体统计\n")
        report.append(f"- 总数据集数: {total_datasets}")
        report.append(f"- 成功数据集: {success_count}")
        report.append(f"- 失败数据集: {error_count}")
        report.append(f"- 总测试样本数: {total_samples}")
        report.append(f"- 成功率: {success_count/total_datasets*100:.1f}%\n")

        # 详细结果
        report.append("## 详细测试结果\n")

        for dataset_name, result in results.items():
            status = result.get("status", "unknown")
            status_emoji = {"success": "✅", "error": "❌"}.get(status, "❓")

            report.append(f"### {status_emoji} {dataset_name}\n")
            report.append(f"**状态**: {status}\n")

            if status == "success":
                samples_tested = result.get("samples_tested", 0)
                report.append(f"**测试样本数**: {samples_tested}\n")

                predictions = result.get("predictions", [])
                if predictions:
                    report.append("**预测结果**:\n")
                    for pred in predictions:
                        sample_id = pred.get("sample_id", "unknown")
                        true_label = pred.get("true_label", "unknown")
                        prediction = pred.get("prediction", {})
                        task = pred.get("task", "unknown")

                        report.append(f"- **样本**: {sample_id}\n")
                        report.append(f"  - 任务类型: {task}\n")
                        report.append(f"  - 真实标签: {true_label}\n")
                        report.append(f"  - 预测结果: {prediction}\n")
            else:
                message = result.get("message", "未知错误")
                report.append(f"**错误**: {message}\n")

            report.append("---\n")

        return "\n".join(report)

    def save_results(self, results: Dict, report: str):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果JSON
        results_file = os.path.join(
            self.output_dir, f"sample_test_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"详细结果已保存到: {results_file}")

        # 保存报告Markdown
        report_file = os.path.join(
            self.output_dir, f"sample_test_report_{timestamp}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"测试报告已保存到: {report_file}")

        return results_file, report_file

    def run(self):
        """运行完整测试"""
        logger.info("开始Qwen3-Omni样本测试...")

        results = self.run_sample_tests()
        report = self.generate_report(results)

        # 保存结果
        results_file, report_file = self.save_results(results, report)

        # 打印摘要
        print("\n" + "="*60)
        print("Qwen3-Omni 样本测试完成！")
        print("="*60)
        print(f"详细结果: {results_file}")
        print(f"测试报告: {report_file}")
        print("="*60)

        return results, report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen3-Omni样本测试 - 从每个数据集挑选样本进行测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                           # 每个数据集测试5个样本
  %(prog)s --samples-per-dataset 10   # 每个数据集测试10个样本
  %(prog)s --base-url http://192.168.1.100:8080/v1  # 使用远程服务
        """
    )

    parser.add_argument("--base-url", type=str, default="http://localhost:8080/v1",
                        help="vLLM服务地址 (默认: http://localhost:8080/v1)")

    parser.add_argument("--model-name", type=str, default="qwen3-omni",
                        help="模型名称 (默认: qwen3-omni)")

    parser.add_argument("--samples-per-dataset", type=int, default=5,
                        help="每个数据集测试的样本数 (默认: 5)")

    parser.add_argument("--output-dir", type=str, default="sample_test_results",
                        help="结果输出目录 (默认: sample_test_results)")

    parser.add_argument("--dataset", type=str,
                        help="测试指定数据集，可选: " + ", ".join([
                            "AffectNet_Classifier", "AffectNet_VA", "AffWild2_VA",
                            "AffWild2_EXPR", "EmotionTalk", "MELD", "MEMO_Bench",
                            "CH_SIMS", "CH_SIMS_V2", "CMU_MOSEI", "CMU_MOSI", "DFEW"
                        ]))

    parser.add_argument("--max-frames", type=int, default=8,
                        help="视频最大帧数 (默认: 8)")

    args = parser.parse_args()

    # 创建测试器
    tester = SampleTester(
        base_url=args.base_url,
        model_name=args.model_name,
        samples_per_dataset=args.samples_per_dataset,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        dataset_filter=getattr(args, 'dataset', None)
    )

    # 运行测试
    tester.run()


if __name__ == "__main__":
    main()
