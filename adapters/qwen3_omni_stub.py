# -*- coding: utf-8 -*-
"""
adapters/qwen3_omni_stub.py
—— 随机基线/预计算读取适配器示例（用于跑通评测流程）。
实际使用时，请将本文件改成你真实的 Qwen3-Omni 推理调用。
"""
from typing import Dict, List
import os
import json
import random

from adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 可选：读取预计算预测（JSONL，每行 {"id": ..., "pred": {...}}）
        self.precomputed = {}
        jsonl_path = self.kwargs.get("precomputed_jsonl", "")
        if jsonl_path and os.path.isfile(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.precomputed[rec["id"]] = rec["pred"]

        # 固定随机种子以保证可复现（仅演示用）
        seed = int(self.kwargs.get("seed", 2025))
        random.seed(seed)

    def _rand_label(self, label_space: List[str]) -> Dict:
        lab = random.choice(label_space)
        # 生成一个伪概率分布（仅演示）
        logits = [random.random() for _ in label_space]
        s = sum(logits)
        probs = {c: v/s for c, v in zip(label_space, logits)}
        return {"label": lab, "probs": probs}

    def predict(self, item: Dict, task: str) -> Dict:
        # 如果有预计算结果，优先返回
        item_id = item.get("id")
        if item_id in self.precomputed:
            return self.precomputed[item_id]

        if task in ("image_emotion_class", "video_emotion_class", "meld_dialog_emotion", "emotiontalk_dialog_emotion"):
            return self._rand_label(item["label_space"])

        if task in ("image_va_reg", "video_va_reg"):
            # 随机返回 [-1,1] 的 valence/arousal
            v = random.uniform(-1, 1)
            a = random.uniform(-1, 1)
            return {"valence": v, "arousal": a}

        if task in ("mosei_sentiment", "chsims_sentiment"):
            # MOSEI: [-3,3]；CH-SIMS: [-1,1] —— 这里统一返回 [-3,3]，数据集内部会 clip/缩放
            p = random.uniform(-3, 3)
            return {"polarity": p}

        # 未知任务
        return {}
