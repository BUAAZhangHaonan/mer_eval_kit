#!/usr/bin/env python3
"""检查数据集路径是否正确"""

import os
import pandas as pd
from pathlib import Path


def check_path(path, description):
    """检查路径是否存在"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    if exists:
        if os.path.isfile(path):
            print(f"   文件大小: {os.path.getsize(path)} bytes")
        elif os.path.isdir(path):
            files = list(Path(path).rglob("*"))
            print(f"   目录内容: {len(files)} 个文件/目录")
    return exists


def main():
    print("=" * 60)
    print("数据集路径检查")
    print("=" * 60)

    # CH-SIMS
    print("\n【CH-SIMS】")
    ch_sims_video = "/home/remote1/lvshuyang/Datasets/CH-SIMS/Raw/video_0001/0001.mp4"
    ch_sims_label = "/home/remote1/lvshuyang/Datasets/CH-SIMS/label.csv"

    video_exists = check_path(ch_sims_video, "视频文件")
    label_exists = check_path(ch_sims_label, "标注文件")

    if label_exists:
        try:
            df = pd.read_csv(ch_sims_label)
            print(f"   标注数据: {len(df)} 行")
            print(f"   列名: {list(df.columns)}")
        except Exception as e:
            print(f"   读取标注失败: {e}")

    # CH-SIMS v2
    print("\n【CH-SIMS v2】")
    ch_sims_v2_video_s = "/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(s)/Raw/aqgy3_0001/00000.mp4"
    ch_sims_v2_video_u = "/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(u)/Raw/aqgy1_0001/00000.mp4"
    ch_sims_v2_meta_s = "/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(s)/meta.csv"
    ch_sims_v2_meta_u = "/home/remote1/lvshuyang/Datasets/CH-SIMS v2/CH-SIMS v2(u)/meta.csv"

    check_path(ch_sims_v2_video_s, "v2(s) 视频文件")
    check_path(ch_sims_v2_video_u, "v2(u) 视频文件")
    check_path(ch_sims_v2_meta_s, "v2(s) 标注文件")
    check_path(ch_sims_v2_meta_u, "v2(u) 标注文件")

    for meta_file in [ch_sims_v2_meta_s, ch_sims_v2_meta_u]:
        if os.path.exists(meta_file):
            try:
                df = pd.read_csv(meta_file)
                print(f"   {os.path.basename(meta_file)}: {len(df)} 行")
                print(f"   列名: {list(df.columns)}")
            except Exception as e:
                print(f"   读取 {os.path.basename(meta_file)} 失败: {e}")

    # DFEW
    print("\n【DFEW】")
    dfew_readme = "/home/remote1/lvshuyang/Datasets/DFEW/README.md"
    dfew_annotation = "/home/remote1/lvshuyang/Datasets/DFEW/Annotation/annotation.xlsx"
    dfew_clip_224 = "/home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224"
    dfew_clip_16f = "/home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224_16f"
    dfew_clip_avi = "/home/remote1/lvshuyang/Datasets/DFEW/Clip/clip_224x224_avi"
    dfew_clip_orig = "/home/remote1/lvshuyang/Datasets/DFEW/Clip/original"

    check_path(dfew_readme, "说明文件")
    check_path(dfew_annotation, "标注文件")
    check_path(dfew_clip_224, "Clip 224x224")
    check_path(dfew_clip_16f, "Clip 16f")
    check_path(dfew_clip_avi, "Clip AVI")
    check_path(dfew_clip_orig, "Clip Original")

    if os.path.exists(dfew_annotation):
        try:
            df = pd.read_excel(dfew_annotation)
            print(f"   标注数据: {len(df)} 行")
            print(f"   列名: {list(df.columns)}")
        except Exception as e:
            print(f"   读取标注失败: {e}")

    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
