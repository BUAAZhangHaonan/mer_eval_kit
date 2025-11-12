#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility script to cross-check dataset loaders, configs, and metric schemas.

The script verifies three things for every registered dataset key:

1. The JSON config can be loaded and instantiated with the corresponding
   ``DatasetConfig`` dataclass without raising errors.
2. The dataset exposes a working ``iter_samples`` path (patched with a
   synthetic iterator so we do not touch real assets) and ``evaluate`` can be
   executed end-to-end with a stub adapter.
3. The metrics returned by ``evaluate`` match the declared expectation for that
   dataset (no missing metrics, no unexpected valence/arousal metrics leaking
   into purely classification datasets, etc.).

Run ``python verify_dataset_configs.py`` to execute the full suite. Use the
``--datasets`` flag to limit the validation scope.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

from adapters.base import BaseAdapter
from adapters.qwen3_omni_stub import Adapter as StubAdapter
from datasets import (
    AffectNetClassifier,
    AffectNetVA,
    AffWild2EXPR,
    AffWild2VA,
    BaseDataset,
    CHSIMS,
    CHSIMSV2,
    CMUMOSEI,
    CMUMOSI,
    DFEW,
    EmotionTalk,
    MELD,
    MEMOBench,
)
from datasets.label_spaces import (
    AFFECTNET_8,
    AFFWILD2_EXPR_8,
    DFEW_7,
    EMOTIONTALK_7,
    MELD_7,
    MEMO_6,
)


@dataclass(frozen=True)
class DatasetSpec:
    """Wire up a dataset key with its config file and evaluation settings."""

    dataset_cls: type[BaseDataset]
    config_path: Path
    eval_kwargs: Mapping[str, object]
    samples: Sequence[MutableMapping[str, object]]
    required_metrics: Sequence[str]
    allowed_metrics: Sequence[str]


def _make_affectnet_cls_samples() -> List[MutableMapping[str, object]]:
    return [
        {
            "id": "synthetic_affectnet_cls_0",
            "image_id": "dummy",
            "image_path": "dummy_affectnet.jpg",
            "label": "happy",
            "label_space": AFFECTNET_8,
            "expression_idx": 1,
        }
    ]


def _make_affectnet_va_samples() -> List[MutableMapping[str, object]]:
    return [
        {
            "id": "synthetic_affectnet_va_0",
            "image_id": "dummy",
            "image_path": "dummy_affectnet.jpg",
            "valence": 0.2,
            "arousal": -0.1,
        }
    ]


def _make_affwild2_va_samples() -> List[MutableMapping[str, object]]:
    return [
        {
            "id": "synthetic_affwild2_va_0",
            "video_name": "dummy_video",
            "video_path": "dummy_affwild2.mp4",
            "valence_seq": [0.1, 0.0, -0.2],
            "arousal_seq": [0.0, 0.1, -0.1],
            "num_frames": 3,
        }
    ]


def _make_affwild2_expr_samples() -> List[MutableMapping[str, object]]:
    return [
        {
            "id": "synthetic_affwild2_expr_0",
            "video_name": "dummy_video",
            "video_path": "dummy_affwild2.mp4",
            "label": "neutral",
            "label_space": AFFWILD2_EXPR_8,
            "frame_labels": [0, 1, 2],
            "majority_label": "Neutral",
        }
    ]


def _make_chsims_samples() -> List[MutableMapping[str, object]]:
    return [
        {
            "id": "synthetic_chsims_0",
            "video_id": "dummy",
            "clip_id": "0001",
            "text": "你好，世界",
            "label": 0.1,
            "label_T": 0.1,
            "label_A": -0.2,
            "label_V": 0.0,
            "annotation": "dummy",
            "emotion": "neutral",
            "mode": "test",
            "video_path": "dummy_chsims.mp4",
            "features": {},
        }
    ]


def _make_chsims_v2_samples() -> List[MutableMapping[str, object]]:
    sample = _make_chsims_samples()[0].copy()
    sample["id"] = "synthetic_chsims_v2_0"
    sample["version"] = "s"
    return [sample]


def _make_mosei_samples() -> List[MutableMapping[str, object]]:
    return [
        {
            "id": "synthetic_cmu_mosei_0",
            "video_id": "YOUTUBE123",
            "clip_id": "0001",
            "text": "This is fine",
            "label": 2.0,
            "label_T": 2.0,
            "label_A": 1.5,
            "label_V": 2.5,
            "annotation": "sample",
            "mode": "test",
            "video_path": "dummy_mosei.mp4",
            "audio_path": "dummy_mosei.wav",
            "features": {},
        }
    ]


def _make_mosi_samples() -> List[MutableMapping[str, object]]:
    sample = _make_mosei_samples()[0].copy()
    sample["id"] = "synthetic_cmu_mosi_0"
    return [sample]


def _make_dfew_samples() -> List[MutableMapping[str, object]]:
    return [
        {
            "id": "synthetic_dfew_0",
            "clip_name": "clip_0001",
            "video_path": "dummy_dfew.mp4",
            "label": "anger",
            "label_space": DFEW_7,
            "annotation": 4,
            "fold": 1,
            "split": "test",
            "frames": [],
        }
    ]


def _make_emotiontalk_samples() -> List[MutableMapping[str, object]]:
    return [
        {
            "id": "synthetic_emotiontalk_0",
            "speaker_id": "spk1",
            "file_name": "utt_0001",
            "label": "neutral",
            "label_space": EMOTIONTALK_7,
            "text": "你好",
            "audio_path": "dummy_audio.wav",
            "video_path": "dummy_video.mp4",
            "text_label": "neutral",
            "audio_label": "neutral",
            "video_label": "neutral",
        }
    ]


def _make_meld_samples() -> List[MutableMapping[str, object]]:
    return [
        {
            "id": "synthetic_meld_0",
            "dialogue_id": "1",
            "utterance_id": "1-1",
            "text": "Hi there",
            "speaker": "Ross",
            "label": "neutral",
            "label_space": MELD_7,
            "video_path": "dummy_meld.mp4",
            "sentiment": "neutral",
            "start_time": "0",
            "end_time": "1",
            "season": "1",
            "episode": "1",
        }
    ]


def _make_memo_bench_samples() -> List[MutableMapping[str, object]]:
    return [
        {
            "id": "synthetic_memo_0",
            "image_name": "memo_0001",
            "image_path": "dummy_memo.png",
            "label": "happy",
            "label_space": MEMO_6,
            "emotion_score": "high",
            "distortions": "happy",
            "quality_score": 4.0,
        }
    ]


def _iter_from_samples(samples: Sequence[MutableMapping[str, object]]):
    """Create a synthetic ``iter_samples`` callable returning fresh copies."""

    def _iterator(split: str = "test", **_: object) -> Iterator[Dict[str, object]]:
        for sample in samples:
            yield dict(sample)

    return _iterator


class SyntheticAdapter(BaseAdapter):
    """Adapter that mirrors ground-truth fields to keep metrics deterministic."""

    def __init__(self) -> None:
        self._fallback = StubAdapter(seed=2025)

    def predict(self, item: Mapping[str, object], task: str) -> Dict[str, object]:
        if task in {"image_emotion_class", "video_emotion_class", "meld_dialog_emotion", "emotiontalk_dialog_emotion"}:
            label = item.get("label")
            if label is None:
                space = item.get("label_space")
                label = space[0] if isinstance(
                    space, Sequence) and space else "neutral"
            return {"label": label}

        if task in {"image_va_reg", "video_va_reg"}:
            if "valence_seq" in item and "arousal_seq" in item:
                return {
                    "valence": list(item.get("valence_seq", [])),
                    "arousal": list(item.get("arousal_seq", [])),
                }
            return {
                "valence": float(item.get("valence", 0.0)),
                "arousal": float(item.get("arousal", 0.0)),
            }

        if task in {"mosei_sentiment", "chsims_sentiment"}:
            return {"polarity": float(item.get("label", 0.0))}

        return self._fallback.predict(item, task)


def _load_config(path: Path, max_samples: int | None) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.setdefault("verify_files", False)
    cfg["strict_mode"] = False
    if max_samples is not None:
        cfg["max_samples"] = max_samples
    return cfg


def _build_registry() -> Dict[str, DatasetSpec]:
    root = Path("configs")

    return {
        "affectnet_cls": DatasetSpec(
            dataset_cls=AffectNetClassifier,
            config_path=(root / "affectnet.json").resolve(),
            eval_kwargs={"split": "val"},
            samples=_make_affectnet_cls_samples(),
            required_metrics=("acc", "macro_f1",
                              "weighted_f1", "total_samples"),
            allowed_metrics=("acc", "macro_f1",
                             "weighted_f1", "total_samples"),
        ),
        "affectnet_va": DatasetSpec(
            dataset_cls=AffectNetVA,
            config_path=(root / "affectnet.json").resolve(),
            eval_kwargs={"split": "val"},
            samples=_make_affectnet_va_samples(),
            required_metrics=(
                "valence_ccc",
                "arousal_ccc",
                "valence_mae",
                "arousal_mae",
                "valence_corr",
                "arousal_corr",
                "mean_ccc",
                "total_samples",
            ),
            allowed_metrics=(
                "valence_ccc",
                "arousal_ccc",
                "valence_mae",
                "arousal_mae",
                "valence_corr",
                "arousal_corr",
                "mean_ccc",
                "total_samples",
            ),
        ),
        "affwild2_va": DatasetSpec(
            dataset_cls=AffWild2VA,
            config_path=(root / "affwild2_va.json").resolve(),
            eval_kwargs={"split": "val"},
            samples=_make_affwild2_va_samples(),
            required_metrics=(
                "valence_ccc",
                "arousal_ccc",
                "valence_mae",
                "arousal_mae",
                "valence_corr",
                "arousal_corr",
                "mean_ccc",
                "total_frames",
            ),
            allowed_metrics=(
                "valence_ccc",
                "arousal_ccc",
                "valence_mae",
                "arousal_mae",
                "valence_corr",
                "arousal_corr",
                "mean_ccc",
                "total_frames",
            ),
        ),
        "affwild2_expr": DatasetSpec(
            dataset_cls=AffWild2EXPR,
            config_path=(root / "affwild2_expr.json").resolve(),
            eval_kwargs={"split": "val"},
            samples=_make_affwild2_expr_samples(),
            required_metrics=("acc", "macro_f1",
                              "weighted_f1", "total_samples"),
            allowed_metrics=("acc", "macro_f1",
                             "weighted_f1", "total_samples"),
        ),
        "chsims": DatasetSpec(
            dataset_cls=CHSIMS,
            config_path=(root / "ch_sims.json").resolve(),
            eval_kwargs={"split": "test"},
            samples=_make_chsims_samples(),
            required_metrics=(
                "mae",
                "corr",
                "acc2_neg_nonneg",
                "acc2_neg_pos",
                "f1_binary_macro",
                "acc_binary",
                "total_samples",
            ),
            allowed_metrics=(
                "mae",
                "corr",
                "acc2_neg_nonneg",
                "acc2_neg_pos",
                "f1_binary_macro",
                "acc_binary",
                "total_samples",
            ),
        ),
        "chsims_v2": DatasetSpec(
            dataset_cls=CHSIMSV2,
            config_path=(root / "ch_sims_v2.json").resolve(),
            eval_kwargs={"split": "test"},
            samples=_make_chsims_v2_samples(),
            required_metrics=(
                "mae",
                "corr",
                "acc2_neg_nonneg",
                "acc2_neg_pos",
                "f1_binary_macro",
                "acc_binary",
                "total_samples",
            ),
            allowed_metrics=(
                "mae",
                "corr",
                "acc2_neg_nonneg",
                "acc2_neg_pos",
                "f1_binary_macro",
                "acc_binary",
                "total_samples",
            ),
        ),
        "mosei": DatasetSpec(
            dataset_cls=CMUMOSEI,
            config_path=(root / "cmu_mosei.json").resolve(),
            eval_kwargs={"split": "test"},
            samples=_make_mosei_samples(),
            required_metrics=(
                "mae",
                "corr",
                "acc2_neg_nonneg",
                "acc2_neg_pos",
                "acc7",
                "f1_binary_macro",
                "acc_binary",
                "total_samples",
            ),
            allowed_metrics=(
                "mae",
                "corr",
                "acc2_neg_nonneg",
                "acc2_neg_pos",
                "acc7",
                "f1_binary_macro",
                "acc_binary",
                "total_samples",
            ),
        ),
        "mosi": DatasetSpec(
            dataset_cls=CMUMOSI,
            config_path=(root / "cmu_mosi.json").resolve(),
            eval_kwargs={"split": "test"},
            samples=_make_mosi_samples(),
            required_metrics=(
                "mae",
                "corr",
                "acc2_neg_nonneg",
                "acc2_neg_pos",
                "acc7",
                "f1_binary_macro",
                "acc_binary",
                "total_samples",
            ),
            allowed_metrics=(
                "mae",
                "corr",
                "acc2_neg_nonneg",
                "acc2_neg_pos",
                "acc7",
                "f1_binary_macro",
                "acc_binary",
                "total_samples",
            ),
        ),
        "dfew": DatasetSpec(
            dataset_cls=DFEW,
            config_path=(root / "dfew.json").resolve(),
            eval_kwargs={"split": "test", "fold": 1},
            samples=_make_dfew_samples(),
            required_metrics=(
                "war",
                "uar",
                "top1_acc",
                "macro_f1",
                "weighted_f1",
                "total_samples",
                "fold",
            ),
            allowed_metrics=(
                "war",
                "uar",
                "top1_acc",
                "macro_f1",
                "weighted_f1",
                "total_samples",
                "fold",
                "emotion_distribution",
                "prediction_distribution",
                "label_space",
            ),
        ),
        "emotiontalk": DatasetSpec(
            dataset_cls=EmotionTalk,
            config_path=(root / "emotiontalk.json").resolve(),
            eval_kwargs={"split": "test"},
            samples=_make_emotiontalk_samples(),
            required_metrics=("acc", "weighted_f1",
                              "macro_f1", "total_samples"),
            allowed_metrics=("acc", "weighted_f1",
                             "macro_f1", "total_samples"),
        ),
        "meld": DatasetSpec(
            dataset_cls=MELD,
            config_path=(root / "meld.json").resolve(),
            eval_kwargs={"split": "test"},
            samples=_make_meld_samples(),
            required_metrics=("acc", "weighted_f1",
                              "macro_f1", "total_samples"),
            allowed_metrics=(
                "acc",
                "weighted_f1",
                "macro_f1",
                "total_samples",
                "emotion_distribution",
                "prediction_distribution",
            ),
        ),
        "memo_bench": DatasetSpec(
            dataset_cls=MEMOBench,
            config_path=(root / "memo_bench.json").resolve(),
            eval_kwargs={"split": "test"},
            samples=_make_memo_bench_samples(),
            required_metrics=("acc", "macro_f1",
                              "weighted_f1", "total_samples"),
            allowed_metrics=(
                "acc",
                "macro_f1",
                "weighted_f1",
                "total_samples",
                "emotion_distribution",
                "prediction_distribution",
            ),
        ),
    }


def _verify_metrics(name: str, metrics: Mapping[str, object], spec: DatasetSpec) -> None:
    actual = set(metrics.keys())
    required = set(spec.required_metrics)
    allowed = set(spec.allowed_metrics)

    missing = required - actual
    unexpected = actual - allowed

    if missing:
        raise AssertionError(
            f"Dataset '{name}' missing required metrics: {sorted(missing)}"
        )
    if unexpected:
        raise AssertionError(
            f"Dataset '{name}' produced unexpected metrics: {sorted(unexpected)}"
        )


def _install_synthetic_iterator(dataset: BaseDataset, samples: Sequence[MutableMapping[str, object]]) -> None:
    dataset.iter_samples = _iter_from_samples(
        samples)  # type: ignore[assignment]


def _run_single_check(name: str, spec: DatasetSpec, max_samples: int) -> None:
    cfg = _load_config(spec.config_path, max_samples=max_samples)
    dataset = spec.dataset_cls(cfg)

    if getattr(dataset.config, "name", None) != cfg.get("name"):
        raise AssertionError(
            f"Dataset '{name}' name mismatch: config='{cfg.get('name')}', dataset='{dataset.config.name}'"
        )
    if getattr(dataset.config, "root", None) != cfg.get("root"):
        raise AssertionError(
            f"Dataset '{name}' root mismatch: config='{cfg.get('root')}', dataset='{dataset.config.root}'"
        )

    _install_synthetic_iterator(dataset, spec.samples)

    adapter = SyntheticAdapter()
    metrics = dataset.evaluate(adapter, **spec.eval_kwargs)
    _verify_metrics(name, metrics, spec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test dataset/config alignment and metric schemas.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["all"],
        help="Dataset keys to run (default: all).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Synthetic max_samples override (default: 5).",
    )
    args = parser.parse_args()

    registry = _build_registry()

    if not args.datasets or "all" in args.datasets:
        targets: Iterable[str] = registry.keys()
    else:
        targets = args.datasets

    failures: List[str] = []

    for name in targets:
        if name not in registry:
            print(f"[WARN] Unknown dataset key '{name}', skipping.")
            continue

        print(f"\n[INFO] Validating dataset '{name}'")
        try:
            _run_single_check(
                name, registry[name], max_samples=args.max_samples)
            print("  -> OK")
        except Exception as exc:  # pragma: no cover - debugging aid
            print(f"  -> FAILED: {exc}")
            failures.append(f"{name}: {exc}")

    if failures:
        print("\n[ERROR] Validation finished with failures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\n[INFO] All requested datasets passed validation.")


if __name__ == "__main__":
    main()
