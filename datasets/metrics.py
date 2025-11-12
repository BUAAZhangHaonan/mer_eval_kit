# -*- coding: utf-8 -*-
"""
metrics.py
—— 多模态情感评测常用指标的纯 Python 实现（无第三方依赖）。
包含：Accuracy、Precision/Recall/F1（Macro/Weighted）、UAR/WAR、Top-1、
MAE、Pearson 相关、CCC、一维情感的 Acc-2（两种定义）与 Acc-7。
"""
from typing import List, Dict, Tuple, Optional
import math
from collections import Counter, defaultdict


def _safe_div(n, d, default=0.0):
    return n / d if d != 0 else default


# ---------------------- 分类类指标 ----------------------
def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """总体准确率（WAR）。"""
    assert len(y_true) == len(y_pred)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return _safe_div(correct, len(y_true))


def confusion(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> Dict[Tuple[str, str], int]:
    """简单混淆矩阵计数：返回 (gold, pred) -> count"""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    mat = defaultdict(int)
    for g, p in zip(y_true, y_pred):
        mat[(g, p)] += 1
    return mat


def precision_recall_f1(y_true: List[str],
                        y_pred: List[str],
                        labels: Optional[List[str]] = None,
                        average: str = "macro") -> Dict[str, float]:
    """
    计算 Precision/Recall/F1；支持 average='macro' 或 'weighted'。
    - macro：各类别指标先算后平均（适合类不均衡）
    - weighted：按真实支持度加权平均（MELD等常用）
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    # 统计
    tp = {c: 0 for c in labels}
    fp = {c: 0 for c in labels}
    fn = {c: 0 for c in labels}
    support = Counter(y_true)
    for g, p in zip(y_true, y_pred):
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    # 每类P/R/F1
    per_class = {}
    for c in labels:
        prec = _safe_div(tp[c], tp[c] + fp[c])
        rec = _safe_div(tp[c], tp[c] + fn[c])
        f1 = _safe_div(2*prec*rec, prec+rec) if (prec+rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec,
                        "f1": f1, "support": support[c]}
    # 聚合
    if average == "macro":
        p = sum(per_class[c]["precision"] for c in labels) / len(labels)
        r = sum(per_class[c]["recall"] for c in labels) / len(labels)
        f = sum(per_class[c]["f1"] for c in labels) / len(labels)
    elif average == "weighted":
        total = sum(per_class[c]["support"] for c in labels)
        if total == 0:
            p = r = f = 0.0
        else:
            p = sum(per_class[c]["precision"]*per_class[c]
                    ["support"] for c in labels) / total
            r = sum(per_class[c]["recall"]*per_class[c]["support"]
                    for c in labels) / total
            f = sum(per_class[c]["f1"]*per_class[c]["support"]
                    for c in labels) / total
    else:
        raise ValueError("average must be 'macro' or 'weighted'")
    return {"precision": p, "recall": r, "f1": f}


def uar(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> float:
    """UAR = 宏平均召回（每类召回率先求后平均）。"""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    prf = precision_recall_f1(y_true, y_pred, labels=labels, average="macro")
    return prf["recall"]


def war(y_true: List[str], y_pred: List[str]) -> float:
    """WAR = 加权召回；在单标签多类下等同于整体 Accuracy。"""
    return accuracy(y_true, y_pred)


# ---------------------- 回归类指标 ----------------------
def mae(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    return sum(abs(a-b) for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: List[float], mu: Optional[float] = None) -> float:
    if not xs:
        return 0.0
    if mu is None:
        mu = _mean(xs)
    return sum((x-mu)**2 for x in xs) / len(xs)


def pearsonr(y_true: List[float], y_pred: List[float]) -> float:
    """皮尔逊相关；若方差为0则返回0（避免 NaN）。"""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    mu_t = _mean(y_true)
    mu_p = _mean(y_pred)
    num = sum((a-mu_t)*(b-mu_p) for a, b in zip(y_true, y_pred))
    den = math.sqrt(sum((a-mu_t)**2 for a in y_true)
                    * sum((b-mu_p)**2 for b in y_pred))
    return num/den if den != 0 else 0.0


def ccc(y_true: List[float], y_pred: List[float]) -> float:
    """
    Concordance Correlation Coefficient（一致性相关系数）
    CCC = 2 * r * σ_x * σ_y / (σ_x^2 + σ_y^2 + (μ_x - μ_y)^2)
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    mu_t, mu_p = _mean(y_true), _mean(y_pred)
    var_t, var_p = _var(y_true, mu_t), _var(y_pred, mu_p)
    r = pearsonr(y_true, y_pred)
    sigma_t = math.sqrt(var_t)
    sigma_p = math.sqrt(var_p)
    den = var_t + var_p + (mu_t - mu_p)**2
    return (2 * r * sigma_t * sigma_p / den) if den != 0 else 0.0


# ---------------------- 离散化评测（MOSI/MOSEI 习惯） ----------------------
def acc2_neg_nonneg(y_true: List[float], y_pred: List[float], zero: float = 0.0) -> float:
    """
    Acc-2（neg vs nonneg）：<0 为负类，其余为非负类。分母包含所有样本。
    """
    y_true_bin = [1 if v >= zero else 0 for v in y_true]  # 1: nonneg, 0: neg
    y_pred_bin = [1 if v >= zero else 0 for v in y_pred]
    return accuracy(y_true_bin, y_pred_bin)


def acc2_neg_pos(y_true: List[float], y_pred: List[float], zero: float = 0.0) -> float:
    """
    Acc-2（neg vs pos）：严格正 vs 严格负，忽略等于零的样本。
    """
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != zero]
    if not pairs:
        return 0.0
    y_t = [1 if t > zero else 0 for t, _ in pairs]  # 1: pos, 0: neg
    y_p = [1 if p > zero else 0 for _, p in pairs]
    return accuracy(y_t, y_p)


def acc7_from_continuous(y_true: List[float], y_pred: List[float]) -> float:
    """
    Acc-7（MOSEI/MOSI 常用）：将连续情感强度四舍五入到 {-3,-2,-1,0,1,2,3} 比较准确率。
    """
    def q7(x: float) -> int:
        x = max(-3.0, min(3.0, x))
        return int(round(x))
    y_t = [q7(v) for v in y_true]
    y_p = [q7(v) for v in y_pred]
    return accuracy(y_t, y_p)
