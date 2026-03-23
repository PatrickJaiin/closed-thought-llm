"""
Experiment 5D: Delta-norm recalibration.

The original exp2b used delta-norm thresholds [0.1-5.0], but actual norms
are ~190. This recalibrates with thresholds [10, 30, 50, 100, 150] on the
20 hand-crafted prompts.

Quick experiment (~30 min).
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, MAX_CONTINUOUS_STEPS, MAX_NEW_TOKENS,
)
from model_utils import load_model
from continuous_recurrence import continuous_recurrence
from gates_heuristic import DeltaNormHalt
from eval_prompts import get_prompts, check_answer


THRESHOLDS = [10, 30, 50, 100, 150]


def run_experiment():
    print("=" * 60)
    print("Experiment 5D: Delta-Norm Recalibration")
    print("=" * 60)

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    prompts = get_prompts()
    print(f"Running on {len(prompts)} eval prompts\n")

    results = {}

    for threshold in THRESHOLDS:
        print(f"\n--- Delta-norm threshold: {threshold} ---")
        t_start = time.time()

        halt = DeltaNormHalt(threshold=threshold, min_steps=2)
        correct = 0
        total_steps = 0
        details = []

        for p in prompts:
            result = continuous_recurrence(
                model, tokenizer,
                context_text=p["prompt"],
                query_text=p["prompt"],
                halt_fn=halt,
                max_steps=MAX_CONTINUOUS_STEPS,
                max_new_tokens=MAX_NEW_TOKENS,
                collect_diagnostics=True,
            )

            is_correct = check_answer(result["answer"], p["answer"])
            correct += int(is_correct)
            total_steps += result["n_steps_taken"]

            # Get final delta norm from diagnostics
            final_delta = None
            if result["diagnostics"]:
                final_delta = result["diagnostics"][-1].get("delta_norm")

            details.append({
                "id": p["id"],
                "category": p["category"],
                "correct": is_correct,
                "steps": result["n_steps_taken"],
                "halted": result["halted"],
                "answer": result["answer"][:100],
                "final_delta_norm": final_delta,
            })

            status = "OK" if is_correct else "WRONG"
            halt_str = (f"halted@{result['n_steps_taken']}"
                        if result["halted"]
                        else f"cap@{result['n_steps_taken']}")
            print(f"  [{status}] {p['id']}: {halt_str} "
                  f"(delta={final_delta:.1f})" if final_delta else
                  f"  [{status}] {p['id']}: {halt_str}")

        elapsed = time.time() - t_start
        accuracy = correct / len(prompts)
        avg_steps = total_steps / len(prompts)

        results[str(threshold)] = {
            "accuracy": accuracy,
            "avg_steps": avg_steps,
            "correct": correct,
            "total": len(prompts),
            "elapsed_s": elapsed,
            "details": details,
        }

        print(f"  Accuracy: {accuracy:.1%}, Avg steps: {avg_steps:.1f}, "
              f"Time: {elapsed:.0f}s")

    # Summary table
    print(f"\n{'='*55}")
    print("DELTA-NORM RECALIBRATION SUMMARY")
    print(f"{'='*55}")
    print(f"{'Threshold':>10} | {'Accuracy':>9} | {'Avg Steps':>10} | {'Time':>7}")
    print("-" * 55)

    for t in THRESHOLDS:
        r = results[str(t)]
        print(f"{t:>10} | {r['accuracy']:>8.1%} | "
              f"{r['avg_steps']:>9.1f} | {r['elapsed_s']:>6.0f}s")

    # Find best threshold (highest accuracy, prefer fewer steps on tie)
    best_t = max(THRESHOLDS, key=lambda t: (
        results[str(t)]["accuracy"],
        -results[str(t)]["avg_steps"],
    ))
    print(f"\n  Best threshold: {best_t} "
          f"({results[str(best_t)]['accuracy']:.1%} accuracy, "
          f"{results[str(best_t)]['avg_steps']:.1f} steps)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    accs = [results[str(t)]["accuracy"] * 100 for t in THRESHOLDS]
    steps = [results[str(t)]["avg_steps"] for t in THRESHOLDS]

    color_acc = "#1f77b4"
    color_steps = "#ff7f0e"

    bars = ax.bar([str(t) for t in THRESHOLDS], accs,
                  color=color_acc, alpha=0.7, label="Accuracy (%)")
    ax2 = ax.twinx()
    ax2.plot([str(t) for t in THRESHOLDS], steps, "o-",
             color=color_steps, linewidth=2, markersize=8, label="Avg Steps")

    ax.set_xlabel("Delta-Norm Threshold")
    ax.set_ylabel("Accuracy (%)", color=color_acc)
    ax2.set_ylabel("Avg Steps", color=color_steps)
    ax.set_title("Exp5D: Delta-Norm Recalibration")
    ax.grid(True, alpha=0.3, axis="y")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    save_path = RESULTS_DIR / "exp5d_delta_norm_recal.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.close(fig)

    # Save JSON
    results["metadata"] = {
        "experiment": "exp5d_delta_norm_recal",
        "thresholds": THRESHOLDS,
        "best_threshold": best_t,
        "timestamp": datetime.now().isoformat(),
    }
    output_path = RESULTS_DIR / "exp5d_delta_norm_recal.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run_experiment()
