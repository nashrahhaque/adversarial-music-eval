"""
One-command pipeline runner.

Executes the full adversarial music evaluation pipeline in order:
  1. fetch_artists.py    -  pull Last.fm data
  2. fingerprint.py      -  build 54-dim DNA vectors
  3. attack.py           -  run SD-MIAE on 5 pairs
  4. detect.py           -  statistical anomaly detection
  5. metrics.py          -  product metric computation
  6. eval_framework.py   -  A/B tests, fairness, ecosystem impact
  7. visualize.py        -  generate all figures

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-fetch    # if data/artists.json already exists
    python run_pipeline.py --only-viz      # only regenerate figures
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


PIPELINE_STEPS = [
    ("Fetching Last.fm artist data",        "fetch_artists.py",   "data/artists.json"),
    ("Building 54-dim fingerprints",        "fingerprint.py",     "data/fingerprints.json"),
    ("Running SD-MIAE adversarial attack",  "attack.py",          "data/results.json"),
    ("Statistical anomaly detection",       "detect.py",          None),
    ("Computing product metrics",           "metrics.py",         None),
    ("Running evaluation framework",        "eval_framework.py",  None),
    ("Generating figures",                  "visualize.py",       None),
]


def run_step(label: str, script: str, cache_path: str | None, force: bool) -> bool:
    if cache_path and Path(cache_path).exists() and not force:
        print(f"  [SKIP] {label} (cached: {cache_path})")
        return True

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ✗ FAILED: {script} (exit {result.returncode})")
        return False

    print(f"\n  ✓ Done in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run adversarial music eval pipeline")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetch_artists.py if data already exists")
    parser.add_argument("--only-viz", action="store_true",
                        help="Only run visualize.py")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if cached")
    args = parser.parse_args()

    print("=" * 60)
    print("  ADVERSARIAL MUSIC EVALUATION PIPELINE")
    print("  SD-MIAE x Last.fm x Artist Identity Evaluation")
    print("=" * 60)

    Path("data").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    if args.only_viz:
        steps = [PIPELINE_STEPS[-1]]
    else:
        steps = PIPELINE_STEPS
        if args.skip_fetch:
            steps = steps[1:]  # skip fetch step

    t_total = time.time()
    for label, script, cache in steps:
        force = args.force or (cache is None)
        ok = run_step(label, script, cache, force)
        if not ok:
            print(f"\nPipeline aborted at: {script}")
            sys.exit(1)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results  → data/results.json")
    print(f"  Figures  → figures/")
    print(f"{'='*60}")

    # Summary
    results_path = Path("data/results.json")
    if results_path.exists():
        import json
        with open(results_path) as f:
            results = json.load(f)

        pm = results.get("product_metrics", {})
        if pm:
            print(f"\n  Artist Health Score (AHS): {pm.get('afhs', 'N/A')}")
            print(f"  Adversarial Robustness:    {pm.get('ari', {}).get('ari', 'N/A')}")
            print(f"  Fairness Parity:           {pm.get('fps', {}).get('fps', 'N/A')}")
            print(f"  Style Transfer Quality:    {pm.get('mean_stq', 'N/A')}")

        det = results.get("detection", {}).get("metrics", {})
        if det:
            print(f"\n  Detection F1:  {det.get('f1', 'N/A')}")
            print(f"  Detection TPR: {det.get('recall', 'N/A')}")
            print(f"  Detection FPR: "
                  f"{det.get('fp', 0) / max(det.get('fp', 0) + det.get('tn', 0), 1):.4f}")


if __name__ == "__main__":
    main()
