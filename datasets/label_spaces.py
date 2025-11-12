# -*- coding: utf-8 -*-
"""
label_spaces.py
—— 统一各数据集的情绪标签空间，便于模型输出与指标计算的一致化。
注意：不同数据集标签命名可能存在细微差异（anger/angry，happiness/happy 等）。
这里提供“标准名”与“别名映射”以做归一化。
"""
from typing import Dict, List

# AffectNet 8 类（含 contempt），官方顺序常见为：
AFFECTNET_8 = [
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "anger",
    "contempt",
]

# AffectNet 7 类（移除 contempt）
AFFECTNET_7 = [c for c in AFFECTNET_8 if c != "contempt"]

# Aff-Wild2 EXPR 8/7 类（包含 Other 类别）
AFFWILD2_EXPR_8 = [
    "neutral",
    "anger",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "other",
]

AFFWILD2_EXPR_7 = [c for c in AFFWILD2_EXPR_8 if c != "other"]

# DFEW 7 类（电影片段）
DFEW_7 = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# MELD 7 类（对话）
MELD_7 = [
    "neutral",
    "happy",
    "sad",
    "anger",
    "surprise",
    "fear",
    "disgust",
]

# EmotionTalk 与 MELD 使用相同的 7 类空间
EMOTIONTALK_7 = MELD_7

# MEMO-Bench 6 类（合成肖像，worry 归为 fear）
MEMO_6 = [
    "anger",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# 标签别名映射（输入 → 标准）
ALIASES: Dict[str, str] = {
    "happiness": "happy",
    "joy": "happy",
    "angry": "anger",
    "contemptuous": "contempt",
    "surprised": "surprise",
    "fearful": "fear",
    "disgusted": "disgust",
    "sadness": "sad",
    "happyness": "happy",
    "joy": "happy",
    "joyful": "happy",
    "smile": "happy",
    "joyous": "happy",
    "worry": "fear",
    "worried": "fear",
    "anxious": "fear",
    "anxiety": "fear",
    "mad": "anger",
    "madness": "anger",
    "neutrality": "neutral",
    "others": "other",
    "none": "neutral",
    "unknown": "other",
}


def normalize_label(label: str) -> str:
    if label is None:
        return ""
    l = str(label).strip().lower()
    return ALIASES.get(l, l)


def ensure_in_space(label: str, space: List[str]) -> str:
    """若标签不在空间内，尝试通过别名映射落入空间；否则原样返回（上游需保证一致）。"""
    n = normalize_label(label)
    return n if n in space else label
