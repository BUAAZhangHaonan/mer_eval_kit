# -*- coding: utf-8 -*-
"""
evaluate.py
—— 统一评测入口。根据 --dataset 选择数据读取器，按标准指标输出结果。
用法示例：
python evaluate.py --dataset affectnet_cls --split val --config configs/affectnet.json --adapter adapters/qwen3_omni_stub.py --output outputs/affectnet_cls_val.json
python evaluate.py --dataset dfew --split val --config configs/dfew.json --adapter adapters/qwen_adapter_http.py --output outputs/dfew.json
"""
import os
import json
import importlib.util
import argparse
from typing import Dict

from datasets import (
    AffectNetClassifier,
    AffectNetVA,
    AffWild2VA,
    AffWild2EXPR,
    CHSIMS,
    CMUMOSEI, DFEW, EmotionTalk, MELD, MEMOBench
)


def load_adapter(adapter_path: str, adapter_kwargs: Dict = None):
    """
    动态加载 adapter 模块，并实例化 Adapter 类。
    """
    adapter_kwargs = adapter_kwargs or {}
    spec = importlib.util.spec_from_file_location("user_adapter", adapter_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Adapter"):
        raise RuntimeError("适配器文件中未找到 Adapter 类")
    return mod.Adapter(**adapter_kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str,
                    choices=[
                        "affectnet_cls", "affectnet_va",
                        "affwild2_va", "affwild2_expr",
                        "chsims", "mosei",
                        "dfew", "emotiontalk", "meld", "memo_bench"
                    ])
    ap.add_argument("--split", default="test", type=str,
                    help="train/dev/val/test（视数据集而定）")
    ap.add_argument("--config", required=True, type=str, help="数据集配置 JSON 路径")
    ap.add_argument("--adapter", required=True,
                    type=str, help="模型适配器文件路径（.py）")
    ap.add_argument("--adapter-args", default="",
                    type=str, help="传给适配器的 JSON 字符串（可选）")
    ap.add_argument("--output", required=True, type=str, help="结果输出 JSON 路径")

    args = ap.parse_args()

    # 读取数据集配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 解析适配器参数
    adapter_kwargs = {}
    if args.adapter_args:
        try:
            adapter_kwargs = json.loads(args.adapter_args)
        except Exception as e:
            print("adapter-args JSON 解析失败：", e)

    # 实例化适配器
    adapter = load_adapter(args.adapter, adapter_kwargs)

    # 路由到对应数据集
    dataset = None
    if args.dataset == "affectnet_cls":
        dataset = AffectNetClassifier(cfg)
    elif args.dataset == "affectnet_va":
        dataset = AffectNetVA(cfg)
    elif args.dataset == "affwild2_va":
        dataset = AffWild2VA(cfg)
    elif args.dataset == "affwild2_expr":
        dataset = AffWild2EXPR(cfg)
    elif args.dataset == "chsims":
        dataset = CHSIMS(cfg)
    elif args.dataset == "mosei":
        dataset = CMUMOSEI(cfg)
    elif args.dataset == "dfew":
        dataset = DFEW(cfg)
    elif args.dataset == "emotiontalk":
        dataset = EmotionTalk(cfg)
    elif args.dataset == "meld":
        dataset = MELD(cfg)
    elif args.dataset == "memo_bench":
        dataset = MEMOBench(cfg)
    else:
        raise NotImplementedError(args.dataset)

    # 评测
    print(
        f"[INFO] Running evaluation: dataset={args.dataset}, split={args.split}")
    results = dataset.evaluate(
        adapter, split=args.split if hasattr(dataset, "evaluate") else "test")

    # 创建输出目录
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 写出 JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "split": args.split,
            "config": cfg,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    # 控制台打印
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k:20s}: {v:.6f}" if isinstance(
            v, (int, float)) else f"{k:20s}: {v}")


if __name__ == "__main__":
    main()
