#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_all_datasets.py
—— 完整的数据集评估脚本，用于评估Qwen3-Omni在所有情感数据集上的性能。

使用示例:
    # 评估所有数据集，每个数据集测试100个样本
    python evaluate_all_datasets.py --max-samples 100
    
    # 评估特定数据集
    python evaluate_all_datasets.py --dataset DFEW --max-samples 50
    
    # 使用不同的vLLM服务地址
    python evaluate_all_datasets.py --base-url http://192.168.1.100:8080/v1 --max-samples 20

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
from test_datasets import DatasetTester
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """完整的数据集评估器"""

    def __init__(self, base_url: str = "http://localhost:8080/v1",
                 model_name: str = "qwen3-omni",
                 max_samples: Optional[int] = None,
                 output_dir: str = "evaluation_results",
                 max_frames: int = 8):
        """
        初始化评估器

        Args:
            base_url: vLLM服务地址
            model_name: 模型名称
            max_samples: 每个数据集最大样本数
            output_dir: 结果输出目录
            max_frames: 视频最大帧数（考虑到32K上下文限制）
        """
        self.base_url = base_url
        self.model_name = model_name
        self.max_samples = max_samples
        self.output_dir = output_dir
        self.max_frames = max_frames

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化适配器（直接使用，避免冗余）
        self.adapter = Adapter(
            base_url=base_url,
            model_name=model_name,
            max_frames=max_frames,
            video_strategy="uniform"  # 均匀采样策略
        )

    def evaluate_single_dataset(self, dataset_name: str) -> Dict:
        """评估单个数据集"""
        logger.info(f"开始评估数据集: {dataset_name}")

        # 数据集映射
        dataset_map = {
            "AffectNet_Classifier": ("AffectNetClassifier", "affectnet.json"),
            "AffectNet_VA": ("AffectNetVA", "affectnet.json"),
            "AffWild2_VA": ("AffWild2VA", "affwild2_va.json"),
            "AffWild2_EXPR": ("AffWild2EXPR", "affwild2_expr.json"),
            "EmotionTalk": ("EmotionTalk", "emotiontalk.json"),
            "MELD": ("MELD", "meld.json"),
            "MEMO_Bench": ("MEMOBench", "memo_bench.json"),
            "CH_SIMS": ("CHSIMS", "ch_sims.json"),
            "CH_SIMS_V2": ("CHSIMSV2", "ch_sims_v2.json"),
            "CMU_MOSEI": ("CMUMOSEI", "cmu_mosei.json"),
            "CMU_MOSI": ("CMUMOSI", "cmu_mosi.json"),
            "DFEW": ("DFEW", "dfew.json"),
        }

        if dataset_name not in dataset_map:
            raise ValueError(f"未知数据集: {dataset_name}")

        # 动态导入数据集类
        import datasets
        dataset_class_name, config_file = dataset_map[dataset_name]
        dataset_class = getattr(datasets, dataset_class_name)

        # 加载配置
        config_path = os.path.join("configs", config_file)
        if not os.path.exists(config_path):
            return {
                "dataset_name": dataset_name,
                "status": "error",
                "message": f"配置文件不存在: {config_path}",
                "errors": [f"配置文件不存在: {config_path}"]
            }

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 设置最大样本数
        if self.max_samples is not None:
            config["max_samples"] = self.max_samples

        # 创建数据集实例
        try:
            dataset = dataset_class(config)
        except Exception as e:
            return {
                "dataset_name": dataset_name,
                "status": "error",
                "message": f"创建数据集失败: {e}",
                "errors": [f"创建数据集失败: {e}"]
            }

        # 评估数据集
        try:
            test_split = config.get(
                "test_split") or config.get("split") or "test"
            eval_result = dataset.evaluate(self.adapter, test_split)

            return {
                "dataset_name": dataset_name,
                "status": "success",
                "sample_count": eval_result.get("total_samples", 0),
                "evaluation_metrics": eval_result,
                "errors": []
            }
        except Exception as e:
            return {
                "dataset_name": dataset_name,
                "status": "partial_failure",
                "message": f"评估失败: {e}",
                "errors": [f"评估失败: {e}"],
                "sample_count": 0
            }

    def evaluate_all_datasets(self) -> Dict:
        """评估所有数据集"""
        logger.info("开始评估所有数据集...")

        # 所有数据集列表
        all_datasets = [
            "AffectNet_Classifier",
            "AffectNet_VA",
            "AffWild2_VA",
            "AffWild2_EXPR",
            "EmotionTalk",
            "MELD",
            "MEMO_Bench",
            "CH_SIMS",
            "CH_SIMS_V2",
            "CMU_MOSEI",
            "CMU_MOSI",
            "DFEW"
        ]

        results = {}

        for dataset_name in all_datasets:
            try:
                result = self.evaluate_single_dataset(dataset_name)
                results[dataset_name] = result

                # 打印简要结果
                status = result["status"]
                sample_count = result.get("sample_count", 0)
                error_count = len(result.get("errors", []))

                logger.info(
                    f"{dataset_name}: {status}, 样本数: {sample_count}, 错误数: {error_count}")

            except Exception as e:
                logger.error(f"评估 {dataset_name} 时发生异常: {e}")
                results[dataset_name] = {
                    "dataset_name": dataset_name,
                    "status": "error",
                    "message": str(e),
                    "errors": [str(e)]
                }

        return results

    def generate_comprehensive_report(self, results: Dict) -> str:
        """生成综合评估报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = []
        report.append("# Qwen3-Omni 多模态情感分析评估报告\n")
        report.append(
            f"**评估时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        report.append(f"**模型**: {self.model_name}\n")
        report.append(f"**服务地址**: {self.base_url}\n")
        report.append(
            f"**最大样本数**: {self.max_samples if self.max_samples else '无限制'}\n")

        # 总体统计
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

        # 数据集分类统计
        classification_datasets = [
            "AffectNet_Classifier", "AffWild2_EXPR", "EmotionTalk", "MELD", "MEMO_Bench", "DFEW"]
        regression_datasets = ["AffectNet_VA", "AffWild2_VA"]
        multimodal_datasets = ["CH_SIMS",
                               "CH_SIMS_V2", "CMU_MOSEI", "CMU_MOSI"]

        report.append("## 数据集类型统计\n")
        report.append(
            f"- **分类任务**: {len([d for d in classification_datasets if d in results])} 个数据集")
        report.append(
            f"- **回归任务**: {len([d for d in regression_datasets if d in results])} 个数据集")
        report.append(
            f"- **多模态任务**: {len([d for d in multimodal_datasets if d in results])} 个数据集\n")

        # 详细结果
        report.append("## 详细评估结果\n")

        for dataset_name, result in results.items():
            status = result.get("status", "unknown")
            status_emoji = {
                "success": "✅",
                "partial_failure": "⚠️",
                "error": "❌"
            }.get(status, "❓")

            report.append(f"### {status_emoji} {dataset_name}\n")
            report.append(f"**状态**: {status}\n")

            if status == "success":
                sample_count = result.get("sample_count", 0)
                report.append(f"**样本数**: {sample_count}\n")

                if "evaluation_metrics" in result:
                    metrics = result["evaluation_metrics"]
                    report.append("**评估指标**:\n")

                    # 按指标类型分组显示
                    acc_metrics = [
                        k for k in metrics.keys() if 'acc' in k.lower()]
                    f1_metrics = [
                        k for k in metrics.keys() if 'f1' in k.lower()]
                    ccc_metrics = [
                        k for k in metrics.keys() if 'ccc' in k.lower()]
                    mae_metrics = [
                        k for k in metrics.keys() if 'mae' in k.lower()]
                    corr_metrics = [
                        k for k in metrics.keys() if 'corr' in k.lower()]
                    other_metrics = [k for k in metrics.keys(
                    ) if k not in acc_metrics + f1_metrics + ccc_metrics + mae_metrics + corr_metrics]

                    if acc_metrics:
                        report.append(
                            f"- 准确率指标: {', '.join([f'{k}={metrics[k]:.4f}' for k in acc_metrics])}")
                    if f1_metrics:
                        report.append(
                            f"- F1指标: {', '.join([f'{k}={metrics[k]:.4f}' for k in f1_metrics])}")
                    if ccc_metrics:
                        report.append(
                            f"- CCC指标: {', '.join([f'{k}={metrics[k]:.4f}' for k in ccc_metrics])}")
                    if mae_metrics:
                        report.append(
                            f"- MAE指标: {', '.join([f'{k}={metrics[k]:.4f}' for k in mae_metrics])}")
                    if corr_metrics:
                        report.append(
                            f"- 相关系数: {', '.join([f'{k}={metrics[k]:.4f}' for k in corr_metrics])}")
                    if other_metrics:
                        report.append(
                            f"- 其他指标: {', '.join([f'{k}={metrics[k]}' for k in other_metrics])}")

                    report.append("")

            elif status == "partial_failure":
                sample_count = result.get("sample_count", 0)
                report.append(f"**样本数**: {sample_count}\n")
                report.append("**错误**:\n")
                for error in result.get("errors", []):
                    report.append(f"- {error}\n")

            else:  # error
                report.append(f"**错误**: {result.get('message', '未知错误')}\n")

            report.append("---\n")

        # 性能汇总
        report.append("## 性能汇总\n")

        # 分类任务性能
        classification_results = {}
        for dataset_name in classification_datasets:
            if dataset_name in results and results[dataset_name]["status"] == "success":
                metrics = results[dataset_name]["evaluation_metrics"]
                # 提取准确率相关指标
                acc_keys = [k for k in metrics.keys() if 'acc' in k.lower()]
                if acc_keys:
                    best_acc = max(metrics[k] for k in acc_keys)
                    classification_results[dataset_name] = best_acc

        if classification_results:
            report.append("### 分类任务准确率\n")
            report.append("| 数据集 | 最佳准确率 |")
            report.append("|--------|------------|")
            for dataset_name, acc in classification_results.items():
                report.append(f"| {dataset_name} | {acc:.4f} |")
            report.append("")

        # 回归任务性能
        regression_results = {}
        for dataset_name in regression_datasets:
            if dataset_name in results and results[dataset_name]["status"] == "success":
                metrics = results[dataset_name]["evaluation_metrics"]
                # 提取CCC指标
                ccc_keys = [k for k in metrics.keys() if 'ccc' in k.lower()]
                if ccc_keys:
                    avg_ccc = sum(metrics[k] for k in ccc_keys) / len(ccc_keys)
                    regression_results[dataset_name] = avg_ccc

        if regression_results:
            report.append("### 回归任务平均CCC\n")
            report.append("| 数据集 | 平均CCC |")
            report.append("|--------|---------|")
            for dataset_name, ccc in regression_results.items():
                report.append(f"| {dataset_name} | {ccc:.4f} |")
            report.append("")

        # 问题汇总
        report.append("## 问题与建议\n")
        all_errors = []
        for result in results.values():
            all_errors.extend(result.get("errors", []))

        if all_errors:
            error_summary = {}
            for error in all_errors:
                error_type = error.split(':')[0] if ':' in error else 'other'
                error_summary[error_type] = error_summary.get(
                    error_type, 0) + 1

            report.append("### 错误统计\n")
            for error_type, count in sorted(error_summary.items()):
                report.append(f"- {error_type}: {count} 次")
            report.append("")

        # 改进建议
        report.append("### 改进建议\n")
        report.append("1. **视频处理优化**: 考虑使用更智能的视频帧采样策略，如关键帧检测")
        report.append("2. **错误处理**: 改进类型错误处理，特别是字符串和整数的混合运算")
        report.append("3. **内存管理**: 对于大规模评估，考虑分批处理以避免内存溢出")
        report.append("4. **并行处理**: 可以考虑并行化多个数据集的评估过程")
        report.append("")

        return "\n".join(report)

    def save_results(self, results: Dict, report: str):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果JSON
        results_file = os.path.join(
            self.output_dir, f"evaluation_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"详细结果已保存到: {results_file}")

        # 保存报告Markdown
        report_file = os.path.join(
            self.output_dir, f"evaluation_report_{timestamp}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"评估报告已保存到: {report_file}")

        return results_file, report_file

    def run_evaluation(self, datasets: Optional[List[str]] = None):
        """运行完整评估"""
        logger.info("开始Qwen3-Omni多模态情感分析评估...")

        if datasets:
            # 评估指定数据集
            results = {}
            for dataset_name in datasets:
                result = self.evaluate_single_dataset(dataset_name)
                results[dataset_name] = result
        else:
            # 评估所有数据集
            results = self.evaluate_all_datasets()

        # 生成报告
        report = self.generate_comprehensive_report(results)

        # 保存结果
        results_file, report_file = self.save_results(results, report)

        # 打印摘要
        print("\n" + "="*60)
        print("Qwen3-Omni 多模态情感分析评估完成！")
        print("="*60)
        print(f"详细结果: {results_file}")
        print(f"评估报告: {report_file}")
        print("="*60)

        return results, report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Qwen3-Omni多模态情感分析完整评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                           # 评估所有数据集
  %(prog)s --dataset DFEW             # 只评估DFEW数据集
  %(prog)s --max-samples 100           # 每个数据集最多100个样本
  %(prog)s --base-url http://192.168.1.100:8080/v1  # 使用远程服务
        """
    )

    parser.add_argument("--dataset", type=str,
                        help="评估指定数据集，可选: " + ", ".join([
                            "AffectNet_Classifier", "AffectNet_VA", "AffWild2_VA",
                            "AffWild2_EXPR", "EmotionTalk", "MELD", "MEMO_Bench",
                            "CH_SIMS", "CH_SIMS_V2", "CMU_MOSEI", "CMU_MOSI", "DFEW"
                        ]))

    parser.add_argument("--base-url", type=str, default="http://localhost:8080/v1",
                        help="vLLM服务地址 (默认: http://localhost:8080/v1)")

    parser.add_argument("--model-name", type=str, default="qwen3-omni",
                        help="模型名称 (默认: qwen3-omni)")

    parser.add_argument("--max-samples", type=int, default=None,
                        help="每个数据集最大样本数 (默认: 使用配置文件设置)")

    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="结果输出目录 (默认: evaluation_results)")

    parser.add_argument("--dry-run", action="store_true",
                        help="只测试数据加载，不进行模型推理")

    args = parser.parse_args()

    # 创建评估器
    evaluator = ComprehensiveEvaluator(
        base_url=args.base_url,
        model_name=args.model_name,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )

    # 运行评估
    datasets = [args.dataset] if args.dataset else None
    evaluator.run_evaluation(datasets)


if __name__ == "__main__":
    main()
