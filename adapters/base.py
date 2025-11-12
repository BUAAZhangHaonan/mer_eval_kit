# -*- coding: utf-8 -*-
"""
adapters/base.py
—— 模型适配器抽象接口。
你需要实现一个 `Adapter` 子类，并在 `predict` 中完成对单个样本的预测。
"""
from typing import Dict


class BaseAdapter:
    def __init__(self, **kwargs):
        """
        你可以在这里加载权重、创建客户端、初始化缓存等。
        kwargs 将透传 `--config` 或外部 JSON 里的 adapter 字段。
        """
        self.kwargs = kwargs

    def predict(self, item: Dict, task: str) -> Dict:
        """
        必须实现的统一接口。返回值格式因任务不同而不同：

        - image_emotion_class / video_emotion_class / meld_dialog_emotion / emotiontalk_dialog_emotion：
            return {"label": "<class_name>", "probs": {"happy": 0.7, "sad": 0.2, ...}}  # probs 可选
        - image_va_reg / video_va_reg：
            return {"valence": float in [-1,1], "arousal": float in [-1,1]}
        - mosei_sentiment / chsims_sentiment：
            return {"polarity": float}  # [-3,3] (MOSEI), [-1,1] (CH-SIMS)，若能给二分类概率也可返回 {"neg_prob": 0.3}
        """
        raise NotImplementedError
