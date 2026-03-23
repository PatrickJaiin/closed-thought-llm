"""
Experiment 5B: RL gate threshold sweep.

The RL-trained HaltGate outputs a probability, but _call_halt_fn thresholds
at 0.5. The optimal threshold for deterministic eval may differ.

Sweeps thresholds [0.2, 0.3, 0.4, 0.5] on:
- 20 hand-crafted prompts (eval_prompts.py)
- 50 GSM8K subset

Uses ThresholdedHaltGate wrapper to adjust the effective threshold.

Quick experiment (~20 min).
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
    RESULTS_DIR, DEVICE, HIDDEN_DIM,
    MAX_CONTINUOUS_STEPS, MAX_NEW_TOKENS,
    BENCHMARK_GSM8K_MAX_TOKENS,
)
from model_utils import load_model
from continuous_recurrence import continuous_recurrence
from gates import HaltGate
from eval_prompts import get_prompts, check_answer as check_eval_answer
from benchmarks import (
    load_gsm8k, extract_gsm8k_answer, check_gsm8k_answer,
    ThresholdedHaltGate,
)


THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


def run_eval_prompts(model, tokenizer, gate, threshold):
    """Run on 20 hand-crafted prompts."""
    wrapped = ThresholdedHaltGate(gate, threshold=threshold)
    prompts = get_prompts()
    correct = 0
    total_steps = 0

    for p in prompts:
        result = continuous_recurrence(
            model, tokenizer,
            context_text=p["prompt"],
            query_text=p["prompt"],
            halt_fn=wrapped,
            max_steps=MAX_CONTINUOUS_STEPS,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        if check_eval_answer(result["answer"], p["answer"]):
            correct += 1
        total_steps += result["n_steps_taken"]

    return {
        "accuracy": correct / len(prompts),
        "avg_steps": total_steps / len(prompts),
        "correct": correct,
        "total": len(prompts),
    }


def run_gsm8k_subset(model, tokenizer, gate, threshold, items):
    """Run on GSM8K subset."""
    wrapped = ThresholdedHaltGate(gate, threshold=threshold)
    correct = 0
    total_steps = 0

    for item in items:
        result = continuous_recurrence(
            model, tokenizer,
            context_text=item.prompt,
            query_text=item.prompt,
            halt_fn=wrapped,
            max_steps=MAX_CONTINUOUS_STEPS,
            max_new_tokens=BENCHMARK_GSM8K_MAX_TOKENS,
        )
        predicted = extract_gsm8k_answer(result["answer"])
        if check_gsm8k_answer(predicted, item.expected):
            correct += 1
        total_steps += result["n_steps_taken"]

    return {
        "accuracy": correct / len(items),
        "avg_steps": total_steps / len(items),
        "correct": correct,
        "total": len(items),
    }


def run_experiment():
    print("=" * 60)
    print("Experiment 5B: RL Gate Threshold Sweep")
    print("=" * 60)

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    print("Loading RL halt gate...")
    gate = HaltGate(hidden_dim=HIDDEN_DIM).to(DEVICE)
    gate.load_state_dict(torch.load(
        RESULTS_DIR / "halt_gate_rl.pt", map_location=DEVICE, weights_only=True
    ))
    gate.eval()
    print("Gate loaded.\n")

    print("Loading GSM8K subset (10 items for quick calibration)...")
    gsm8k_items = load_gsm8k(subset_n=10, seed=42)
    print(f"  Loaded {len(gsm8k_items)} items\n")

    results = {}

    for threshold in THRESHOLDS:
        print(f"\n--- Threshold: {threshold} ---")
        t_start = time.time()

        print("  Running on eval prompts...")
        eval_result = run_eval_prompts(model, tokenizer, gate, threshold)
        print(f"    Eval: {eval_result['accuracy']:.1%} acc, "
              f"{eval_result['avg_steps']:.1f} avg steps")

        print("  Running on GSM8K subset...")
        gsm8k_result = run_gsm8k_subset(model, tokenizer, gate, threshold, gsm8k_items)
        print(f"    GSM8K: {gsm8k_result['accuracy']:.1%} acc, "
              f"{gsm8k_result['avg_steps']:.1f} avg steps")

        elapsed = time.time() - t_start
        results[str(threshold)] = {
            "eval_prompts": eval_result,
            "gsm8k_10": gsm8k_result,
            "elapsed_s": elapsed,
        }
        print(f"  Done in {elapsed:.0f}s")

    # Summary table
    print(f"\n{'='*65}")
    print("THRESHOLD SWEEP SUMMARY")
    print(f"{'='*65}")
    print(f"{'Threshold':>10} | {'Eval Acc':>9} | {'Eval Steps':>11} | "
          f"{'GSM8K Acc':>10} | {'GSM8K Steps':>12}")
    print("-" * 65)

    for t in THRESHOLDS:
        r = results[str(t)]
        print(f"{t:>10.1f} | {r['eval_prompts']['accuracy']:>8.1%} | "
              f"{r['eval_prompts']['avg_steps']:>10.1f} | "
              f"{r['gsm8k_10']['accuracy']:>9.1%} | "
              f"{r['gsm8k_10']['avg_steps']:>11.1f}")

    # Find best
    best_t = max(THRESHOLDS, key=lambda t: (
        results[str(t)]["eval_prompts"]["accuracy"]
        + results[str(t)]["gsm8k_10"]["accuracy"]
    ) / 2)
    print(f"\n  Best threshold (avg of both): {best_t}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (dataset, key) in zip(axes, [("Eval Prompts", "eval_prompts"), ("GSM8K-50", "gsm8k_10")]):
        accs = [results[str(t)][key]["accuracy"] * 100 for t in THRESHOLDS]
        steps = [results[str(t)][key]["avg_steps"] for t in THRESHOLDS]

        color_acc = "#1f77b4"
        color_steps = "#ff7f0e"

        ax.bar([str(t) for t in THRESHOLDS], accs, color=color_acc, alpha=0.7, label="Accuracy (%)")
        ax2 = ax.twinx()
        ax2.plot([str(t) for t in THRESHOLDS], steps, "o-", color=color_steps, linewidth=2, label="Avg Steps")

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Accuracy (%)", color=color_acc)
        ax2.set_ylabel("Avg Steps", color=color_steps)
        ax.set_title(f"{dataset}")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Exp5B: RL Gate Threshold Sweep")
    plt.tight_layout()
    save_path = RESULTS_DIR / "exp5b_threshold_sweep.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.close(fig)

    # Save JSON
    results["metadata"] = {
        "experiment": "exp5b_threshold_sweep",
        "thresholds": THRESHOLDS,
        "best_threshold": best_t,
        "timestamp": datetime.now().isoformat(),
    }
    output_path = RESULTS_DIR / "exp5b_threshold_sweep.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run_experiment()
