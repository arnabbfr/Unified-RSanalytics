"""Consolidated Multi-Model Benchmark Aggregator for Remote Sensing Foundation Models.

Compares TerraMind, Prithvi-EO-2.0, SatMAE++, and GFM-Composition on the 6 standard flood metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


MODELS = [
    ("terramind", "TerraMind-1.0-base (IBM/ESA)"),
    ("prithvi", "Prithvi-EO-2.0-600M-TL (IBM/NASA)"),
    ("satmaepp", "SatMAE++ (Transformers)"),
    ("gfm", "GFM-Composition (Dual-Sensor)"),
]

METRIC_KEYS = [
    ("accuracy", "Overall Accuracy"),
    ("specificity", "Specificity (TNR)"),
    ("precision", "Flood Precision"),
    ("recall", "Flood Recall (TPR)"),
    ("dice", "Dice / F1-Score"),
    ("iou", "Flood IoU (Jaccard)"),
]


def load_model_metrics(results_dir: Path | str = "fine_tune/results") -> Dict[str, Dict[str, Any]]:
    base_dir = Path(results_dir)
    results = {}

    for model_key, model_display in MODELS:
        m_dir = base_dir / model_key
        test_json = m_dir / "metrics_test.json"
        valid_json = m_dir / "metrics_valid.json"

        target_json = test_json if test_json.is_file() else valid_json
        if target_json.is_file():
            try:
                with open(target_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                m = data.get("metrics", data)
                results[model_key] = {
                    "display_name": model_display,
                    "split": data.get("split", "test"),
                    "metrics": m,
                    "best_epoch": data.get("best_epoch", "N/A"),
                }
            except Exception as e:
                print(f"[Warning] Failed to parse metrics for {model_key}: {e}")

    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    if not results:
        print("[Benchmark Notice] No trained model results found under fine_tune/results/. Train models first.")
        return

    print("\n" + "=" * 92)
    print("🏆 CONSOLIDATED 4-FOUNDATION-MODEL BENCHMARK COMPARISON (SEN1FLOODS11)")
    print("=" * 92)

    # Header
    header = f"{'Metric':<25}"
    for model_key, _ in MODELS:
        if model_key in results:
            header += f" | {model_key.upper():<14}"
    print(header)
    print("-" * 92)

    for metric_key, metric_name in METRIC_KEYS:
        row = f"{metric_name:<25}"
        for model_key, _ in MODELS:
            if model_key in results:
                val = results[model_key]["metrics"].get(metric_key, 0.0)
                row += f" | {val * 100:>6.2f}%       "
        print(row)

    print("-" * 92)
    # Confusion row
    row_tp = f"{'True Positives (TP)':<25}"
    for model_key, _ in MODELS:
        if model_key in results:
            tp = results[model_key]["metrics"].get("tp", 0)
            row_tp += f" | {tp:>14,}"
    print(row_tp)

    row_fn = f"{'False Negatives (FN)':<25}"
    for model_key, _ in MODELS:
        if model_key in results:
            fn = results[model_key]["metrics"].get("fn", 0)
            row_fn += f" | {fn:>14,}"
    print(row_fn)

    print("=" * 92)


def plot_comparison_chart(results: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE or not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    models_found = [m for m, _ in MODELS if m in results]
    if not models_found:
        return

    metrics_labels = [label for _, label in METRIC_KEYS]
    x = np.arange(len(metrics_labels))
    width = 0.8 / len(models_found)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#1e1e24")
    ax.set_facecolor("#282a36")

    colors = ["#50fa7b", "#8be9fd", "#ff79c6", "#f1fa8c"]

    for i, m_key in enumerate(models_found):
        scores = [results[m_key]["metrics"].get(k, 0.0) * 100 for k, _ in METRIC_KEYS]
        offset = (i - len(models_found) / 2 + 0.5) * width
        rects = ax.bar(x + offset, scores, width, label=results[m_key]["display_name"], color=colors[i % len(colors)], alpha=0.9)

    ax.set_ylabel("Score (%)", color="white", fontsize=12)
    ax.set_title("Geospatial Foundation Model 6-Metric Benchmark on Sen1Floods11", color="white", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.3, color="gray", axis="y")
    ax.legend(facecolor="#1e1e24", edgecolor="none", labelcolor="white", loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Saved] Benchmark comparison chart written to: {output_path}")


def generate_benchmark_report(results_dir: str = "fine_tune/results") -> None:
    base = Path(results_dir)
    results = load_model_metrics(base)
    print_comparison_table(results)

    summary_json = base / "foundation_models_benchmark_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] Consolidated benchmark JSON: {summary_json}")

    chart_png = base / "foundation_models_benchmark_chart.png"
    plot_comparison_chart(results, chart_png)


def main():
    parser = argparse.ArgumentParser(description="Consolidated 4-model foundation benchmark report.")
    parser.add_argument("--results-dir", type=str, default="fine_tune/results", help="Directory containing model results.")
    args = parser.parse_args()
    generate_benchmark_report(args.results_dir)


if __name__ == "__main__":
    main()
