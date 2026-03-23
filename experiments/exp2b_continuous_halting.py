"""
Experiment 2B: Continuous halting with heuristic gates.

Sweeps threshold parameters for each of the four heuristic gates:
1. Confidence gate — threshold sweep
2. Convergence gate — threshold sweep
3. Entropy gate — threshold sweep
4. Delta-norm gate — threshold sweep

Measures:
- Accuracy vs avg steps (adaptive compute efficiency)
- Per-problem step counts (do easy problems halt earlier?)
- Comparison to fixed N=32 baseline

Success criteria:
- At least one heuristic gate matches 85%+ accuracy with fewer avg steps than N=32
- Easy questions halt earlier than hard ones (adaptive behavior)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from config import (
    RESULTS_DIR, MID_LAYER_INDEX, MAX_CONTINUOUS_STEPS, MAX_NEW_TOKENS,
    HALT_CONFIDENCE_THRESHOLD, HALT_CONVERGENCE_THRESHOLD,
    HALT_ENTROPY_THRESHOLD, HALT_DELTA_NORM_THRESHOLD,
)
from config import WANDB_ENABLED, WANDB_PROJECT
from model_utils import load_model
from continuous_recurrence import continuous_recurrence
from gates_heuristic import ConfidenceHalt, ConvergenceHalt, EntropyHalt, DeltaNormHalt
from eval_prompts import get_prompts, check_answer

if WANDB_ENABLED:
    import wandb

# Threshold sweeps for each gate type
SWEEPS = {
    "confidence": {
        "thresholds": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "gate_class": ConfidenceHalt,
        "needs_model": True,
    },
    "convergence": {
        "thresholds": [0.90, 0.93, 0.95, 0.97, 0.98, 0.99],
        "gate_class": ConvergenceHalt,
        "needs_model": False,
    },
    "entropy": {
        "thresholds": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        "gate_class": EntropyHalt,
        "needs_model": True,
    },
    "delta_norm": {
        "thresholds": [0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
        "gate_class": DeltaNormHalt,
        "needs_model": False,
    },
}


def run_fixed_baseline(model, tokenizer, prompts):
    """Run fixed N=32 baseline for comparison."""
    print("\n--- Fixed N=32 baseline ---")
    correct = 0
    details = []

    for prompt_data in prompts:
        result = continuous_recurrence(
            model, tokenizer,
            context_text=prompt_data["prompt"],
            query_text=prompt_data["prompt"],
            n_steps=32,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        is_correct = check_answer(result["answer"], prompt_data["answer"])
        correct += int(is_correct)
        details.append({
            "id": prompt_data["id"],
            "category": prompt_data["category"],
            "correct": is_correct,
            "steps": 32,
            "answer": result["answer"][:100],
        })
        status = "OK" if is_correct else "WRONG"
        print(f"  [{status}] {prompt_data['id']}: {result['answer'][:60]}...")

    accuracy = correct / len(prompts)
    print(f"  Accuracy: {accuracy:.1%}")
    return {"accuracy": accuracy, "avg_steps": 32.0, "details": details}


def run_gate_sweep(model, tokenizer, prompts, gate_name, gate_config):
    """Sweep thresholds for a single gate type."""
    print(f"\n{'=' * 40}")
    print(f"Gate: {gate_name}")
    print(f"{'=' * 40}")

    gate_results = {}

    for threshold in gate_config["thresholds"]:
        print(f"\n  threshold={threshold}")

        # Create gate instance
        kwargs = {"threshold": threshold}
        if gate_config["needs_model"]:
            kwargs["model"] = model
        gate = gate_config["gate_class"](**kwargs)

        correct = 0
        total_steps = 0
        details = []

        for prompt_data in prompts:
            result = continuous_recurrence(
                model, tokenizer,
                context_text=prompt_data["prompt"],
                query_text=prompt_data["prompt"],
                halt_fn=gate,
                max_steps=MAX_CONTINUOUS_STEPS,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            is_correct = check_answer(result["answer"], prompt_data["answer"])
            correct += int(is_correct)
            total_steps += result["n_steps_taken"]

            details.append({
                "id": prompt_data["id"],
                "category": prompt_data["category"],
                "correct": is_correct,
                "steps": result["n_steps_taken"],
                "halted": result["halted"],
                "answer": result["answer"][:100],
            })

            status = "OK" if is_correct else "WRONG"
            halt_str = f"halted@{result['n_steps_taken']}" if result["halted"] else f"cap@{result['n_steps_taken']}"
            print(f"    [{status}] {prompt_data['id']}: {halt_str}")

        accuracy = correct / len(prompts)
        avg_steps = total_steps / len(prompts)

        gate_results[threshold] = {
            "accuracy": accuracy,
            "avg_steps": avg_steps,
            "correct": correct,
            "total": len(prompts),
            "details": details,
        }

        print(f"    Accuracy: {accuracy:.1%}, Avg steps: {avg_steps:.1f}")

    return gate_results


def plot_accuracy_vs_steps(all_results, baseline, save_path):
    """Plot accuracy vs average steps for all gates + baseline."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"confidence": "#1f77b4", "convergence": "#ff7f0e",
              "entropy": "#2ca02c", "delta_norm": "#d62728"}
    markers = {"confidence": "o", "convergence": "s", "entropy": "^", "delta_norm": "D"}

    for gate_name, gate_results in all_results.items():
        thresholds = sorted(gate_results.keys())
        accs = [gate_results[t]["accuracy"] for t in thresholds]
        steps = [gate_results[t]["avg_steps"] for t in thresholds]

        ax.plot(steps, accs,
                marker=markers.get(gate_name, "o"),
                color=colors.get(gate_name, "gray"),
                linewidth=2, markersize=8,
                label=gate_name)

        # Annotate thresholds
        for t, s, a in zip(thresholds, steps, accs):
            ax.annotate(f"t={t}", (s, a), textcoords="offset points",
                       xytext=(5, 5), fontsize=7, alpha=0.7)

    # Baseline
    ax.axhline(y=baseline["accuracy"], color="black", linestyle="--",
              alpha=0.5, label=f"Fixed N=32 ({baseline['accuracy']:.0%})")
    ax.axvline(x=32, color="black", linestyle=":", alpha=0.3)

    ax.set_xlabel("Average Steps to Halt")
    ax.set_ylabel("Accuracy")
    ax.set_title("Adaptive Halting: Accuracy vs Compute")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_steps_by_difficulty(all_results, save_path):
    """Plot steps-to-halt by problem category (easy vs hard)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (gate_name, gate_results) in zip(axes.flat, all_results.items()):
        # Use the median threshold
        thresholds = sorted(gate_results.keys())
        mid_threshold = thresholds[len(thresholds) // 2]
        details = gate_results[mid_threshold]["details"]

        categories = {}
        for d in details:
            cat = d["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(d["steps"])

        cats = sorted(categories.keys())
        positions = range(len(cats))
        means = [np.mean(categories[c]) for c in cats]
        stds = [np.std(categories[c]) for c in cats]

        ax.bar(positions, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(cats)
        ax.set_ylabel("Steps to halt")
        ax.set_title(f"{gate_name} (threshold={mid_threshold})")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Steps-to-Halt by Category (Adaptive Behavior Test)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def run_experiment():
    print("=" * 60)
    print("Experiment 2B: Continuous Halting — Heuristic Gate Sweep")
    print("=" * 60)

    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, name="exp2b_continuous_halting")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    prompts = get_prompts()

    # Run fixed baseline
    baseline = run_fixed_baseline(model, tokenizer, prompts)

    # Run all gate sweeps
    all_results = {}
    for gate_name, gate_config in SWEEPS.items():
        all_results[gate_name] = run_gate_sweep(
            model, tokenizer, prompts, gate_name, gate_config
        )

    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_vs_steps(
        all_results, baseline,
        save_path=RESULTS_DIR / "exp2b_accuracy_vs_steps.png"
    )
    plot_steps_by_difficulty(
        all_results,
        save_path=RESULTS_DIR / "exp2b_steps_by_difficulty.png"
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Best threshold per gate")
    print("=" * 70)
    print(f"{'Gate':>15} | {'Threshold':>10} | {'Accuracy':>10} | {'Avg Steps':>10} | {'vs N=32':>10}")
    print("-" * 65)

    for gate_name, gate_results in all_results.items():
        # Find best accuracy with fewer steps than N=32
        best = None
        for t, r in gate_results.items():
            if best is None or r["accuracy"] > best["accuracy"]:
                best = {"threshold": t, **r}
        if best:
            delta = f"{best['avg_steps'] - 32:.1f}"
            print(f"{gate_name:>15} | {best['threshold']:>10} | {best['accuracy']:>9.1%} | {best['avg_steps']:>9.1f} | {delta:>10}")

    print(f"\n  Fixed N=32 baseline: {baseline['accuracy']:.1%}")

    # Save results
    output_path = RESULTS_DIR / "exp2b_continuous_halting.json"
    serializable = {"baseline": baseline}
    for gate_name, gate_results in all_results.items():
        serializable[gate_name] = {}
        for t, r in gate_results.items():
            serializable[gate_name][str(t)] = {
                "accuracy": r["accuracy"],
                "avg_steps": r["avg_steps"],
                "correct": r["correct"],
                "total": r["total"],
                "details": r["details"],
            }

    serializable["metadata"] = {
        "experiment": "exp2b_continuous_halting",
        "timestamp": datetime.now().isoformat(),
        "max_continuous_steps": MAX_CONTINUOUS_STEPS,
        "mid_layer_index": MID_LAYER_INDEX,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if WANDB_ENABLED:
        wandb.finish()

    return all_results


if __name__ == "__main__":
    run_experiment()
