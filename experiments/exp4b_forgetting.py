"""
Experiment 4B: Memory forgetting curves.

Tests how well different memory tiers retain information over time:
1. Store a set of facts as hidden states
2. Run recurrence steps (simulating time passing)
3. Periodically attempt to retrieve and check accuracy
4. Compare forgetting curves across memory tiers

Also tests:
- Rehearsal effect: re-accessed memories should survive longer
- Temporal decay impact

Success criteria:
- Forgetting curves show biologically-plausible decay
- Rehearsal works: re-accessed memories survive longer
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from config import (
    DEVICE, RESULTS_DIR, MID_LAYER_INDEX,
    HIDDEN_DIM, MEMORY_SLOTS, MEMORY_TEMPORAL_DECAY,
)
from config import WANDB_ENABLED, WANDB_PROJECT
from model_utils import load_model, full_forward, get_embeddings, partial_forward
from memory import create_memory

if WANDB_ENABLED:
    import wandb


# Facts to memorize (will be encoded as hidden states)
FACTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius.",
    "The speed of light is approximately 300,000 km per second.",
    "Shakespeare wrote Romeo and Juliet.",
    "Pi is approximately 3.14159.",
    "The Earth orbits the Sun.",
    "Gold has the chemical symbol Au.",
    "There are 7 continents on Earth.",
]


def encode_fact(model, tokenizer, text):
    """Encode a fact into a hidden state via full forward pass."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    embeds = get_embeddings(model, inputs.input_ids)
    hidden = full_forward(model, embeds, attention_mask=inputs.attention_mask)
    return hidden[:, -1:, :]  # (1, 1, hidden_dim)


def retrieval_accuracy(memory, query_states, stored_states):
    """
    Measure how well memory can retrieve stored states.

    For each query, retrieve from memory and compute cosine similarity
    to the original stored state. If cosine sim > 0.5, count as recalled.
    """
    stats = memory.stats()
    # NeuralMemory uses "num_slots", KV/Surprise use "size"
    mem_size = stats.get("size", stats.get("num_slots", 0))
    if mem_size == 0:
        return 0.0, [0.0] * len(query_states)

    recalled = 0
    sims = []

    for q_state, s_state in zip(query_states, stored_states):
        retrieved = memory.read(q_state)
        if retrieved is None:
            sims.append(0.0)
            continue

        sim = torch.nn.functional.cosine_similarity(
            retrieved.view(1, -1).float(), s_state.view(1, -1).float()
        ).item()
        sims.append(sim)
        if sim > 0.5:
            recalled += 1

    accuracy = recalled / len(query_states) if query_states else 0
    return accuracy, sims


def run_forgetting_curve(model, tokenizer, memory_tier, facts, checkpoints,
                         rehearsal_indices=None):
    """
    Run a forgetting curve experiment for one memory tier.

    1. Encode and store all facts
    2. Run "distractor" recurrence steps (noise / random processing)
    3. At each checkpoint, attempt retrieval and measure accuracy

    Args:
        memory_tier: "kv", "surprise", or "neural"
        facts: List of fact strings.
        checkpoints: List of step counts at which to measure retrieval.
        rehearsal_indices: Indices of facts to rehearse (re-read) periodically.
                         If None, no rehearsal.

    Returns:
        dict with forgetting curve data.
    """
    memory = create_memory(tier=memory_tier, device=DEVICE)

    # Step 1: Encode all facts
    fact_states = []
    with torch.no_grad():
        for fact in facts:
            h = encode_fact(model, tokenizer, fact)
            fact_states.append(h.clone())
            memory.write(h)

    print(f"  Stored {len(facts)} facts. Memory: {memory.stats()}")

    # Step 2: Run distractor steps and measure at checkpoints
    curve = {"steps": [], "accuracy": [], "avg_sim": []}

    # Measure initial retrieval
    acc, sims = retrieval_accuracy(memory, fact_states, fact_states)
    curve["steps"].append(0)
    curve["accuracy"].append(acc)
    curve["avg_sim"].append(float(np.mean(sims)))
    print(f"    Step 0: accuracy={acc:.2f}, avg_sim={np.mean(sims):.3f}")

    # Generate distractor hidden states (random recurrence)
    with torch.no_grad():
        distractor_text = "The quick brown fox jumps over the lazy dog."
        distractor_h = encode_fact(model, tokenizer, distractor_text)

        step = 0
        checkpoint_idx = 0

        while checkpoint_idx < len(checkpoints):
            target_step = checkpoints[checkpoint_idx]

            while step < target_step:
                # Run one distractor recurrence step
                position_ids = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
                distractor_h = partial_forward(
                    model, distractor_h, start_layer=MID_LAYER_INDEX,
                    position_ids=position_ids
                )

                # Write distractor to memory (simulates new info coming in)
                memory.write(distractor_h)

                # Apply decay for neural memory
                if hasattr(memory, "apply_decay"):
                    memory.apply_decay()

                # Rehearsal: periodically re-read some memories
                if rehearsal_indices and step % 10 == 0:
                    for ri in rehearsal_indices:
                        memory.read(fact_states[ri])

                step += 1

            # Measure retrieval at this checkpoint
            acc, sims = retrieval_accuracy(memory, fact_states, fact_states)
            curve["steps"].append(step)
            curve["accuracy"].append(acc)
            curve["avg_sim"].append(float(np.mean(sims)))
            print(f"    Step {step}: accuracy={acc:.2f}, avg_sim={np.mean(sims):.3f}")

            checkpoint_idx += 1

    return curve


def run_experiment():
    print("=" * 60)
    print("Experiment 4B: Memory Forgetting Curves")
    print("=" * 60)

    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, name="exp4b_forgetting")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    checkpoints = [5, 10, 20, 50, 100, 200, 500]
    tiers = ["kv", "surprise", "neural"]
    results = {}

    # Run forgetting curves without rehearsal
    for tier in tiers:
        print(f"\n--- {tier} memory (no rehearsal) ---")
        curve = run_forgetting_curve(
            model, tokenizer, tier, FACTS, checkpoints,
            rehearsal_indices=None,
        )
        results[f"{tier}_no_rehearsal"] = curve

    # Run forgetting curves with rehearsal (re-read facts 0, 1, 2 periodically)
    for tier in tiers:
        print(f"\n--- {tier} memory (with rehearsal on facts 0-2) ---")
        curve = run_forgetting_curve(
            model, tokenizer, tier, FACTS, checkpoints,
            rehearsal_indices=[0, 1, 2],
        )
        results[f"{tier}_rehearsal"] = curve

    # Plot forgetting curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"kv": "#1f77b4", "surprise": "#ff7f0e", "neural": "#2ca02c"}

    # No rehearsal
    for tier in tiers:
        key = f"{tier}_no_rehearsal"
        ax1.plot(results[key]["steps"], results[key]["accuracy"],
                marker="o", markersize=4, linewidth=2,
                color=colors[tier], label=tier)

    ax1.set_xlabel("Distractor Steps")
    ax1.set_ylabel("Retrieval Accuracy")
    ax1.set_title("Forgetting Curves (No Rehearsal)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # With rehearsal
    for tier in tiers:
        key = f"{tier}_rehearsal"
        ax2.plot(results[key]["steps"], results[key]["accuracy"],
                marker="s", markersize=4, linewidth=2,
                color=colors[tier], label=f"{tier} (rehearsed)")
        # Also plot no-rehearsal as dashed for comparison
        key_nr = f"{tier}_no_rehearsal"
        ax2.plot(results[key_nr]["steps"], results[key_nr]["accuracy"],
                linestyle="--", alpha=0.5, color=colors[tier])

    ax2.set_xlabel("Distractor Steps")
    ax2.set_ylabel("Retrieval Accuracy")
    ax2.set_title("Forgetting Curves (Solid = Rehearsal, Dashed = No Rehearsal)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4b_forgetting_curves.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {RESULTS_DIR / 'exp4b_forgetting_curves.png'}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Forgetting Curves")
    print("=" * 60)
    for key, curve in results.items():
        final_acc = curve["accuracy"][-1] if curve["accuracy"] else 0
        initial_acc = curve["accuracy"][0] if curve["accuracy"] else 0
        print(f"  {key:>25}: initial={initial_acc:.2f}, final={final_acc:.2f}, "
              f"decay={initial_acc - final_acc:.2f}")

    # Save
    output_path = RESULTS_DIR / "exp4b_forgetting.json"
    serializable = dict(results)
    serializable["metadata"] = {
        "experiment": "exp4b_forgetting",
        "timestamp": datetime.now().isoformat(),
        "n_facts": len(FACTS),
        "checkpoints": checkpoints,
        "memory_slots": MEMORY_SLOTS,
        "temporal_decay": MEMORY_TEMPORAL_DECAY,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if WANDB_ENABLED:
        wandb.finish()

    return results


if __name__ == "__main__":
    run_experiment()
