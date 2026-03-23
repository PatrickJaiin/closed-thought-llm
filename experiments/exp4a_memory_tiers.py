"""
Experiment 4A: Compare memory tiers on multi-query evaluation.

Tests all three memory tiers (KV, Surprise, Neural) against no-memory baseline.
Uses a multi-query setup: same context, multiple questions asked sequentially.
The memory retains information across queries — this is where memory should shine.

Memory tiers:
- Tier 1: KV Ring Buffer (no learnable params)
- Tier 2: Surprise-Based (Titans-inspired, no learnable params)
- Tier 3: Neural Memory (learned read/write, ~6.4M params)

Success criteria:
- Memory improves multi-turn accuracy by 10%+ over no-memory
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
    DEVICE, RESULTS_DIR, MID_LAYER_INDEX, MAX_CONTINUOUS_STEPS,
    MAX_NEW_TOKENS, HIDDEN_DIM, MEMORY_SLOTS,
    MEMORY_RESIDUAL_ALPHA,
)
from config import WANDB_ENABLED, WANDB_PROJECT
from model_utils import load_model
from continuous_recurrence import continuous_recurrence
from memory import create_memory
from eval_prompts import get_prompts, check_answer

if WANDB_ENABLED:
    import wandb


# Multi-query scenarios: same context concept, multiple questions
MULTI_QUERY_SCENARIOS = [
    {
        "id": "math_multi",
        "context": "Consider the number 144.",
        "queries": [
            {"q": "Question: What is the square root of 144?\nAnswer:", "a": "12"},
            {"q": "Question: What is 144 divided by 12?\nAnswer:", "a": "12"},
            {"q": "Question: What is 144 + 56?\nAnswer:", "a": "200"},
            {"q": "Question: Is 144 a perfect square? Answer yes or no.\nAnswer:", "a": "yes"},
        ],
    },
    {
        "id": "geo_multi",
        "context": "Paris is the capital of France. France is in Europe.",
        "queries": [
            {"q": "Question: What is the capital of France?\nAnswer:", "a": "Paris"},
            {"q": "Question: What continent is France on?\nAnswer:", "a": "Europe"},
            {"q": "Question: Is Paris in Europe? Answer yes or no.\nAnswer:", "a": "yes"},
        ],
    },
    {
        "id": "logic_multi",
        "context": "A is taller than B. B is taller than C. C is taller than D.",
        "queries": [
            {"q": "Question: Who is the tallest?\nAnswer:", "a": "A"},
            {"q": "Question: Who is the shortest?\nAnswer:", "a": "D"},
            {"q": "Question: Is B taller than D? Answer yes or no.\nAnswer:", "a": "yes"},
            {"q": "Question: Is C taller than A? Answer yes or no.\nAnswer:", "a": "no"},
        ],
    },
    {
        "id": "science_multi",
        "context": "Water boils at 100 degrees Celsius. Water freezes at 0 degrees Celsius.",
        "queries": [
            {"q": "Question: At what temperature does water boil in Celsius?\nAnswer:", "a": "100"},
            {"q": "Question: At what temperature does water freeze in Celsius?\nAnswer:", "a": "0"},
            {"q": "Question: What is the difference between the boiling and freezing points of water in Celsius?\nAnswer:", "a": "100"},
        ],
    },
]


def run_multi_query(model, tokenizer, scenarios, memory_tier, n_recurrence=32):
    """
    Run multi-query evaluation with a specific memory tier.

    For each scenario:
    1. Process context with recurrence
    2. Answer each query sequentially
    3. Memory persists across queries within a scenario

    Args:
        memory_tier: "none", "kv", "surprise", or "neural"
        n_recurrence: Number of recurrence steps per query.

    Returns:
        Results dict with per-scenario and aggregate accuracy.
    """
    all_details = []
    total_correct = 0
    total_queries = 0

    for scenario in scenarios:
        # Create fresh memory for each scenario
        if memory_tier != "none":
            memory = create_memory(tier=memory_tier, device=DEVICE)
        else:
            memory = None

        scenario_correct = 0

        for qi, query_data in enumerate(scenario["queries"]):
            context = scenario["context"]

            result = continuous_recurrence(
                model, tokenizer,
                context_text=context,
                query_text=query_data["q"],
                n_steps=n_recurrence,
                max_new_tokens=MAX_NEW_TOKENS,
                memory=memory,
                memory_gate=None,  # No gate — always read/write
                memory_alpha=MEMORY_RESIDUAL_ALPHA,
            )

            is_correct = check_answer(result["answer"], query_data["a"])
            scenario_correct += int(is_correct)
            total_correct += int(is_correct)
            total_queries += 1

            all_details.append({
                "scenario": scenario["id"],
                "query_idx": qi,
                "query": query_data["q"][:60],
                "expected": query_data["a"],
                "predicted": result["answer"][:80],
                "correct": is_correct,
                "steps": result["n_steps_taken"],
            })

            status = "OK" if is_correct else "WRONG"
            print(f"    [{status}] Q{qi}: {result['answer'][:60]}...")

        scenario_acc = scenario_correct / len(scenario["queries"])
        print(f"  {scenario['id']}: {scenario_correct}/{len(scenario['queries'])} ({scenario_acc:.0%})")

        # Print memory stats if applicable
        if memory is not None:
            stats = memory.stats()
            print(f"    Memory: {stats.get('size', 'N/A')} entries, "
                  f"avg_age={stats.get('avg_age', 0):.1f}")

    accuracy = total_correct / total_queries
    return {
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_queries,
        "details": all_details,
    }


def run_experiment():
    print("=" * 60)
    print("Experiment 4A: Memory Tier Comparison (Multi-Query)")
    print("=" * 60)

    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, name="exp4a_memory_tiers")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    tiers = ["none", "kv", "surprise", "neural"]
    results = {}

    for tier in tiers:
        print(f"\n{'=' * 40}")
        print(f"Memory tier: {tier}")
        print(f"{'=' * 40}")

        tier_results = run_multi_query(
            model, tokenizer, MULTI_QUERY_SCENARIOS,
            memory_tier=tier, n_recurrence=32,
        )
        results[tier] = tier_results

    # Also run single-query baseline on standard prompts
    print(f"\n{'=' * 40}")
    print("Single-query baselines (standard eval prompts)")
    print(f"{'=' * 40}")

    prompts = get_prompts()
    for tier in ["none", "kv"]:
        memory = create_memory(tier=tier, device=DEVICE) if tier != "none" else None
        correct = 0
        for p in prompts:
            result = continuous_recurrence(
                model, tokenizer,
                context_text=p["prompt"],
                query_text=p["prompt"],
                n_steps=32,
                max_new_tokens=MAX_NEW_TOKENS,
                memory=memory,
            )
            if check_answer(result["answer"], p["answer"]):
                correct += 1
        acc = correct / len(prompts)
        results[f"single_{tier}"] = {"accuracy": acc, "correct": correct, "total": len(prompts)}
        print(f"  {tier}: {acc:.1%} ({correct}/{len(prompts)})")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    tier_names = ["none", "kv", "surprise", "neural"]
    accs = [results[t]["accuracy"] for t in tier_names]
    colors = ["#999999", "#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(tier_names, accs, color=colors, alpha=0.8)
    ax.set_ylabel("Multi-Query Accuracy")
    ax.set_title("Memory Tier Comparison")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.0%}", ha="center", fontsize=11)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4a_memory_tiers.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {RESULTS_DIR / 'exp4a_memory_tiers.png'}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Multi-Query Accuracy by Memory Tier")
    print("=" * 60)
    for tier in tier_names:
        r = results[tier]
        print(f"  {tier:>10}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")

    improvement = results["kv"]["accuracy"] - results["none"]["accuracy"]
    print(f"\n  KV vs None improvement: {improvement:+.1%}")

    # Save
    output_path = RESULTS_DIR / "exp4a_memory_tiers.json"
    serializable = {}
    for k, v in results.items():
        serializable[k] = v

    serializable["metadata"] = {
        "experiment": "exp4a_memory_tiers",
        "timestamp": datetime.now().isoformat(),
        "n_scenarios": len(MULTI_QUERY_SCENARIOS),
        "memory_slots": MEMORY_SLOTS,
        "memory_alpha": MEMORY_RESIDUAL_ALPHA,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if WANDB_ENABLED:
        wandb.finish()

    return results


if __name__ == "__main__":
    run_experiment()
