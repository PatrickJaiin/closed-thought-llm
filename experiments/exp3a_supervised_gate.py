"""
Experiment 3A: Supervised bootstrap training for learned HaltGate.

Pipeline:
1. Run heuristic confidence gate on eval prompts to collect (h, halt_label) pairs
2. Train HaltGate MLP with BCE loss
3. Evaluate learned gate vs heuristic gate on same prompts
4. Compare accuracy and avg steps

Success criteria:
- Learned gate matches or exceeds heuristic gate accuracy
- Avg steps-to-halt decreases 20%+ vs fixed N=32
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
from config import (
    DEVICE, RESULTS_DIR, MID_LAYER_INDEX, MAX_CONTINUOUS_STEPS,
    MAX_NEW_TOKENS, HIDDEN_DIM, GATE_HIDDEN_DIM,
    HALT_CONFIDENCE_THRESHOLD,
)
from config import WANDB_ENABLED, WANDB_PROJECT
from model_utils import load_model
from gates import HaltGate, count_parameters
from gates_heuristic import ConfidenceHalt
from gate_training import collect_halt_labels, train_supervised, save_gate
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
            "correct": is_correct,
            "steps": result["n_steps_taken"],
            "halted": result["halted"],
        })

        status = "OK" if is_correct else "WRONG"
        print(f"  [{status}] {prompt_data['id']}: steps={result['n_steps_taken']}, "
              f"halted={result['halted']}")

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
    print("Experiment 3A: Supervised HaltGate Training")
    print("=" * 60)

    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, name="exp3a_supervised_gate")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    prompts = get_prompts()

    # Step 1: Collect labels from heuristic confidence gate
    print("Step 1: Collecting heuristic halt labels...")
    heuristic_gate = ConfidenceHalt(model, threshold=HALT_CONFIDENCE_THRESHOLD)
    dataset = collect_halt_labels(
        model, tokenizer, prompts,
        heuristic_halt_fn=heuristic_gate,
        max_steps=64,  # Don't need full 256 for label collection
        mid_layer=MID_LAYER_INDEX,
    )
    print(f"  Dataset: {len(dataset)} samples\n")

    # Step 2: Train HaltGate
    print("Step 2: Training HaltGate...")
    halt_gate = HaltGate(hidden_dim=HIDDEN_DIM, gate_dim=GATE_HIDDEN_DIM).to(DEVICE)
    print(f"  Parameters: {count_parameters(halt_gate):,}")

    train_history = train_supervised(
        halt_gate, dataset,
        epochs=20,
        batch_size=32,
        lr=1e-3,
    )

    # Save checkpoint
    gate_path = RESULTS_DIR / "halt_gate_supervised.pt"
    save_gate(halt_gate, gate_path)

    # Step 3: Evaluate both gates
    print("\nStep 3: Evaluating gates...")

    print("\n--- Heuristic confidence gate ---")
    heuristic_results = evaluate_gate(
        model, tokenizer, prompts,
        halt_fn=heuristic_gate,
        gate_name="heuristic",
    )

    print("\n--- Learned HaltGate ---")
    learned_results = evaluate_gate(
        model, tokenizer, prompts,
        halt_fn=halt_gate,
        gate_name="learned",
    )

    print("\n--- Fixed N=32 baseline ---")
    baseline_results = evaluate_gate(
        model, tokenizer, prompts,
        halt_fn=None,
        gate_name="fixed_n32",
        max_steps=32,
    )

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_history["loss"], "b-o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_history["accuracy"], "g-o", markersize=3, label="Accuracy")
    ax2.plot(train_history["precision"], "r-s", markersize=3, label="Precision")
    ax2.plot(train_history["recall"], "b-^", markersize=3, label="Recall")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Training Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3a_training_curves.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {RESULTS_DIR / 'exp3a_training_curves.png'}")
    plt.close(fig)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    names = ["Fixed N=32", "Heuristic", "Learned"]
    accs = [baseline_results["accuracy"], heuristic_results["accuracy"], learned_results["accuracy"]]
    steps = [baseline_results["avg_steps"], heuristic_results["avg_steps"], learned_results["avg_steps"]]

    colors = ["#999999", "#ff7f0e", "#1f77b4"]
    ax.scatter(steps, accs, s=200, c=colors, zorder=3)
    for i, name in enumerate(names):
        ax.annotate(name, (steps[i], accs[i]), textcoords="offset points",
                   xytext=(10, 5), fontsize=11)

    ax.set_xlabel("Average Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Gate Comparison: Accuracy vs Steps")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3a_gate_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'exp3a_gate_comparison.png'}")
    plt.close(fig)

    # Per-problem adaptation analysis
    print("\n--- Per-Problem Adaptation ---")
    for detail in learned_results["details"]:
        heur_detail = next(d for d in heuristic_results["details"] if d["id"] == detail["id"])
        print(f"  {detail['id']}: learned={detail['steps']} steps, "
              f"heuristic={heur_detail['steps']} steps")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Fixed N=32:  acc={baseline_results['accuracy']:.1%}, steps=32.0")
    print(f"  Heuristic:   acc={heuristic_results['accuracy']:.1%}, steps={heuristic_results['avg_steps']:.1f}")
    print(f"  Learned:     acc={learned_results['accuracy']:.1%}, steps={learned_results['avg_steps']:.1f}")

    # Save results
    output_path = RESULTS_DIR / "exp3a_supervised_gate.json"
    serializable = {
        "baseline": baseline_results,
        "heuristic": heuristic_results,
        "learned": learned_results,
        "training_history": train_history,
        "metadata": {
            "experiment": "exp3a_supervised_gate",
            "timestamp": datetime.now().isoformat(),
            "gate_params": count_parameters(halt_gate),
            "confidence_threshold": HALT_CONFIDENCE_THRESHOLD,
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
