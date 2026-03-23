"""
Experiment 3B: REINFORCE refinement of learned HaltGate.

Starts from the supervised checkpoint (exp3a) and refines with RL:
- Reward: +1 for correct answer, -1 for incorrect, -0.01/step
- Shows that RL can improve over supervised-only training
- Compares: fixed N=32, heuristic, supervised, RL-refined

Success criteria:
- RL-refined gate >= supervised gate accuracy
- Gate shows per-problem adaptation (not just fixed N)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
from config import (
    DEVICE, RESULTS_DIR, MID_LAYER_INDEX, MAX_CONTINUOUS_STEPS,
    MAX_NEW_TOKENS, HIDDEN_DIM, GATE_HIDDEN_DIM,
    HALT_CONFIDENCE_THRESHOLD,
    GATE_RL_LR, GATE_RL_STEP_PENALTY,
    GATE_RL_CORRECT_REWARD, GATE_RL_INCORRECT_REWARD,
)
from config import WANDB_ENABLED, WANDB_PROJECT
from model_utils import load_model
from gates import HaltGate, count_parameters
from gates_heuristic import ConfidenceHalt
from gate_training import train_reinforce, save_gate, load_gate
from continuous_recurrence import continuous_recurrence
from eval_prompts import get_prompts, check_answer

if WANDB_ENABLED:
    import wandb


def evaluate_gate(model, tokenizer, prompts, halt_fn, gate_name, max_steps=MAX_CONTINUOUS_STEPS):
    """Run evaluation with a gate and return results."""
    correct = 0
    total_steps = 0
    details = []

    for prompt_data in prompts:
        result = continuous_recurrence(
            model, tokenizer,
            context_text=prompt_data["prompt"],
            query_text=prompt_data["prompt"],
            halt_fn=halt_fn,
            max_steps=max_steps,
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
        })

    accuracy = correct / len(prompts)
    avg_steps = total_steps / len(prompts)
    print(f"  {gate_name}: accuracy={accuracy:.1%}, avg_steps={avg_steps:.1f}")

    return {
        "accuracy": accuracy,
        "avg_steps": avg_steps,
        "correct": correct,
        "total": len(prompts),
        "details": details,
    }


def run_experiment():
    print("=" * 60)
    print("Experiment 3B: REINFORCE Gate Refinement")
    print("=" * 60)

    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, name="exp3b_rl_gate")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    prompts = get_prompts()

    # Load supervised checkpoint
    gate_path = RESULTS_DIR / "halt_gate_supervised.pt"
    if gate_path.exists():
        print("Loading supervised checkpoint...")
        halt_gate = load_gate(HaltGate, gate_path)
    else:
        print("No supervised checkpoint found. Training from scratch.")
        halt_gate = HaltGate(hidden_dim=HIDDEN_DIM, gate_dim=GATE_HIDDEN_DIM).to(DEVICE)

    print(f"  Parameters: {count_parameters(halt_gate):,}\n")

    # Evaluate pre-RL performance
    print("--- Pre-RL evaluation ---")
    pre_rl_results = evaluate_gate(
        model, tokenizer, prompts, halt_fn=halt_gate, gate_name="pre-RL"
    )

    # Run REINFORCE
    print("\n--- REINFORCE training ---")
    rl_history = train_reinforce(
        halt_gate,
        model, tokenizer,
        prompts,
        epochs=10,
        lr=GATE_RL_LR,
        step_penalty=GATE_RL_STEP_PENALTY,
        correct_reward=GATE_RL_CORRECT_REWARD,
        incorrect_reward=GATE_RL_INCORRECT_REWARD,
        max_steps=64,  # Shorter cap for RL efficiency
        mid_layer=MID_LAYER_INDEX,
        check_answer_fn=check_answer,
    )

    # Save RL-refined checkpoint
    rl_gate_path = RESULTS_DIR / "halt_gate_rl.pt"
    save_gate(halt_gate, rl_gate_path)

    # Evaluate post-RL
    print("\n--- Post-RL evaluation ---")
    post_rl_results = evaluate_gate(
        model, tokenizer, prompts, halt_fn=halt_gate, gate_name="post-RL"
    )

    # Baselines for comparison
    print("\n--- Baselines ---")
    print("  Fixed N=32:")
    fixed_results = evaluate_gate(
        model, tokenizer, prompts, halt_fn=None, gate_name="fixed_n32", max_steps=32
    )

    print("  Heuristic confidence:")
    heuristic_gate = ConfidenceHalt(model, threshold=HALT_CONFIDENCE_THRESHOLD)
    heuristic_results = evaluate_gate(
        model, tokenizer, prompts, halt_fn=heuristic_gate, gate_name="heuristic"
    )

    # Plot RL training curves
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(rl_history["reward"], "b-o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg Reward")
    ax1.set_title("RL Reward")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rl_history["accuracy"], "g-o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("RL Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    ax3.plot(rl_history["avg_steps"], "r-o", markersize=4)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Avg Steps")
    ax3.set_title("RL Steps-to-Halt")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3b_rl_curves.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {RESULTS_DIR / 'exp3b_rl_curves.png'}")
    plt.close(fig)

    # Plot comparison scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    configs = {
        "Fixed N=32": (fixed_results["avg_steps"], fixed_results["accuracy"], "#999999"),
        "Heuristic": (heuristic_results["avg_steps"], heuristic_results["accuracy"], "#ff7f0e"),
        "Supervised": (pre_rl_results["avg_steps"], pre_rl_results["accuracy"], "#2ca02c"),
        "RL-refined": (post_rl_results["avg_steps"], post_rl_results["accuracy"], "#1f77b4"),
    }

    for name, (s, a, c) in configs.items():
        ax.scatter(s, a, s=200, c=c, zorder=3, label=name)

    ax.set_xlabel("Average Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Gate Comparison: Fixed → Heuristic → Supervised → RL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3b_gate_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'exp3b_gate_comparison.png'}")
    plt.close(fig)

    # Per-problem analysis: steps by category
    print("\n--- Per-Problem Analysis (RL gate) ---")
    category_steps = {}
    for d in post_rl_results["details"]:
        cat = d["category"]
        if cat not in category_steps:
            category_steps[cat] = {"steps": [], "correct": []}
        category_steps[cat]["steps"].append(d["steps"])
        category_steps[cat]["correct"].append(d["correct"])

    for cat, data in sorted(category_steps.items()):
        avg_s = np.mean(data["steps"])
        acc = np.mean(data["correct"])
        print(f"  {cat}: avg_steps={avg_s:.1f}, accuracy={acc:.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Config':>15} | {'Accuracy':>10} | {'Avg Steps':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Fixed N=32':>15} | {fixed_results['accuracy']:>9.1%} | {'32.0':>10}")
    print(f"  {'Heuristic':>15} | {heuristic_results['accuracy']:>9.1%} | {heuristic_results['avg_steps']:>9.1f}")
    print(f"  {'Supervised':>15} | {pre_rl_results['accuracy']:>9.1%} | {pre_rl_results['avg_steps']:>9.1f}")
    print(f"  {'RL-refined':>15} | {post_rl_results['accuracy']:>9.1%} | {post_rl_results['avg_steps']:>9.1f}")

    # Save results
    output_path = RESULTS_DIR / "exp3b_rl_gate.json"
    serializable = {
        "fixed_n32": fixed_results,
        "heuristic": heuristic_results,
        "supervised": pre_rl_results,
        "rl_refined": post_rl_results,
        "rl_history": rl_history,
        "metadata": {
            "experiment": "exp3b_rl_gate",
            "timestamp": datetime.now().isoformat(),
            "gate_params": count_parameters(halt_gate),
            "rl_lr": GATE_RL_LR,
            "step_penalty": GATE_RL_STEP_PENALTY,
        },
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if WANDB_ENABLED:
        wandb.finish()

    return serializable


if __name__ == "__main__":
    run_experiment()
