# -*- coding: utf-8 -*-
"""
datasets/utils.py
—— 数据集通用工具：安全IO、视频抽帧、文本读取等。
尽量避免第三方依赖；如安装了 OpenCV，将优先使用以提升速度与兼容性。
"""
import os
import json
import csv
import glob
from typing import List, Dict

# 尝试导入 OpenCV，失败则退化到 imageio
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def read_csv(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def path_exists(p: str) -> bool:
    return p is not None and os.path.exists(p)


def sample_video_frames(video_path: str, num_frames: int = 8) -> List:
    """
    从视频中均匀抽取 num_frames 张帧（返回 BGR 或 RGB 的 numpy 数组，取决于后端）。
    若既没有 cv2 也没有 imageio，则返回空列表。
    """
    frames = []
    if _HAS_CV2 and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return frames
        idxs = [max(0, int(i)) for i in (total - 1) * (x/(num_frames-1)
                                                       if num_frames > 1 else 0 for x in range(num_frames))]
        idxs = sorted(set(idxs))
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)  # BGR
        cap.release()
        return frames
    elif _HAS_IMAGEIO and os.path.isfile(video_path):
        try:
            reader = imageio.get_reader(video_path)
            total = reader.get_length()
            if total <= 0:
                reader.close()
                return frames
            idxs = [max(0, int(i)) for i in (total - 1) * (x/(num_frames-1)
                                                           if num_frames > 1 else 0 for x in range(num_frames))]
            idxs = sorted(set(idxs))
            for idx in idxs:
                frame = reader.get_data(idx)  # RGB
                frames.append(frame)
            reader.close()
            return frames
        except Exception:
            return frames
    else:
        return frames


def glob_many(root_dir: str, patterns: List[str]) -> List[str]:
    """在 root_dir 下按多个通配符模式收集文件，返回排序后的完整路径列表。"""
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root_dir, pat), recursive=True))
    return sorted(files)
