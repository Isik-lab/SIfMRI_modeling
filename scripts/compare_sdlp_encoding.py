#!/usr/bin/env python
"""Compare SDLP neural-encoding results: trained vs. untrained (baseline).

Reads the per-voxel parquet(s) written by video_neural_encoding.py and reports
mean held-out test_score overall and per region, plus the trained-minus-baseline
delta (the effect of Stage-1 training, since the architecture is identical).

Usage:
    python compare_sdlp_encoding.py \
        --trained  <data>/interim/VideoNeuralEncoding/model-sdlp.parquet \
        --baseline <data>/interim/VideoNeuralEncoding/model-sdlp-untrained.parquet
    # region column auto-detected; override with --region_col
"""
import argparse
import pandas as pd

REGION_CANDIDATES = ["roi", "region", "stream", "network", "parcel", "label", "roi_name"]
SCORE_COL = "test_score"


def _region_col(df, override=None):
    if override:
        return override if override in df.columns else None
    for c in REGION_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _summary(df, region_col):
    out = {"overall": df[SCORE_COL].mean(), "n_voxels": len(df)}
    if region_col:
        out["by_region"] = df.groupby(region_col)[SCORE_COL].mean()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trained", required=True, help="model-sdlp.parquet")
    p.add_argument("--baseline", default=None, help="model-sdlp-untrained.parquet")
    p.add_argument("--region_col", default=None)
    p.add_argument("--predictor_only", action="store_true",
                   help="restrict to voxels whose best layer is a predictor block")
    args = p.parse_args()

    trained = pd.read_parquet(args.trained)
    print(f"[trained]  {args.trained}  ({len(trained)} voxels)")
    print(f"  columns: {list(trained.columns)}")

    if args.predictor_only and "layer" in trained.columns:
        m = trained["layer"].astype(str).str.contains("predictor")
        trained = trained[m]
        print(f"  predictor-only voxels: {len(trained)}")

    region_col = _region_col(trained, args.region_col)
    print(f"  region column: {region_col or '(none found)'}")

    ts = _summary(trained, region_col)
    print(f"\nTRAINED  overall mean {SCORE_COL}: {ts['overall']:.4f}")

    if not args.baseline:
        if region_col:
            print("\nTrained by region:")
            print(ts["by_region"].to_string())
        return

    baseline = pd.read_parquet(args.baseline)
    if args.predictor_only and "layer" in baseline.columns:
        baseline = baseline[baseline["layer"].astype(str).str.contains("predictor")]
    bs = _summary(baseline, region_col)
    print(f"BASELINE overall mean {SCORE_COL}: {bs['overall']:.4f}")
    print(f"DELTA (trained - baseline):        {ts['overall'] - bs['overall']:+.4f}")

    if region_col:
        tbl = pd.DataFrame({
            "trained": ts["by_region"],
            "baseline": bs["by_region"],
        })
        tbl["delta"] = tbl["trained"] - tbl["baseline"]
        print("\nPer region (mean test_score):")
        print(tbl.sort_values("delta", ascending=False).to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
