#!/usr/bin/env python3
"""Quick sanity checks for dataset configs and loaders."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, Tuple

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

DatasetEntry = Tuple[str, type[BaseDataset], Dict[str, object]]


REGISTRY: Dict[str, DatasetEntry] = {
    "affectnet_cls": ("configs/affectnet.json", AffectNetClassifier, {"split": "val"}),
    "affectnet_va": ("configs/affectnet.json", AffectNetVA, {"split": "val"}),
    "affwild2_va": ("configs/affwild2_va.json", AffWild2VA, {"split": "val"}),
    "affwild2_expr": ("configs/affwild2_expr.json", AffWild2EXPR, {"split": "val"}),
    "chsims": ("configs/ch_sims.json", CHSIMS, {"split": "test"}),
    "chsims_v2": ("configs/ch_sims_v2.json", CHSIMSV2, {"split": "test"}),
    "mosei": ("configs/cmu_mosei.json", CMUMOSEI, {"split": "test"}),
    "mosi": ("configs/cmu_mosi.json", CMUMOSI, {"split": "test"}),
    "dfew": ("configs/dfew.json", DFEW, {"split": "test", "fold": 1}),
    "emotiontalk": ("configs/emotiontalk.json", EmotionTalk, {"split": "test"}),
    "meld": ("configs/meld.json", MELD, {"split": "test"}),
    "memo_bench": ("configs/memo_bench.json", MEMOBench, {"split": "test"}),
}


def load_config(config_path: Path, max_samples: int | None) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if max_samples is not None:
        cfg["max_samples"] = max_samples
    return cfg


def run_dataset_check(name: str, entry: DatasetEntry, max_samples: int, skip_eval: bool) -> Dict[str, object]:
    config_rel, dataset_cls, eval_kwargs = entry
    config_path = Path(config_rel).resolve()
    cfg = load_config(config_path, max_samples)
    dataset = dataset_cls(cfg)

    split = str(eval_kwargs.get("split", "test"))
    stats = dataset.get_stats(split=split)

    result: Dict[str, object] = {
        "dataset": name,
        "config": str(config_path),
        "split": split,
        "stats": stats,
    }

    if skip_eval:
        return result

    adapter = StubAdapter(seed=2025)
    metrics = dataset.evaluate(adapter, **eval_kwargs)
    result["metrics"] = metrics
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate dataset configs against loaders.")
    ap.add_argument("--datasets", nargs="*",
                    default=["all"], help="Dataset keys to validate (default: all)")
    ap.add_argument("--max-samples", type=int, default=5,
                    help="Limit samples per dataset during checks")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Skip calling dataset.evaluate and only iterate samples")
    args = ap.parse_args()

    if not args.datasets or "all" in args.datasets:
        targets = list(REGISTRY.keys())
    else:
        targets = args.datasets

    failures = []

    for name in targets:
        if name not in REGISTRY:
            print(f"[WARN] Unknown dataset key '{name}', skipping.")
            continue
        print(f"\n[INFO] Checking {name} ...")
        try:
            result = run_dataset_check(
                name, REGISTRY[name], args.max_samples, args.skip_eval)
            stats = result["stats"]
            print(f"  - config: {result['config']}")
            print(f"  - split: {result['split']}")
            print(
                "  - stats: total_samples={total_samples}, valid_samples={valid_samples}, errors={errors}".format(
                    **stats
                )
            )
            if "metrics" in result:
                print(f"  - metrics keys: {sorted(result['metrics'].keys())}")
        except Exception as exc:  # pragma: no cover - diagnostic output
            failures.append((name, exc))
            print(f"  ! FAILED: {exc}")
            traceback.print_exc()

    if failures:
        print("\n[ERROR] Some datasets failed validation:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        raise SystemExit(1)

    print("\n[INFO] All requested datasets passed the checks.")


if __name__ == "__main__":
    main()
