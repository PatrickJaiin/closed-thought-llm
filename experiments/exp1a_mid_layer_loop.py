"""
Experiment 1A-mid: Mid-layer loop recurrence.
Feed final hidden state back to layer ~1/3, run through remaining layers, repeat N times.
Sweep N ∈ {0, 1, 2, 4, 8, 16, 32} across all eval prompts.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from config import RECURRENCE_STEPS, MAX_NEW_TOKENS, RESULTS_DIR, MID_LAYER_INDEX
from config import WANDB_ENABLED, WANDB_PROJECT
from model_utils import load_model
from recurrence import mid_layer_loop_recurrence
from eval_prompts import get_prompts, check_answer

if WANDB_ENABLED:
    import wandb


def run_experiment():
    print("=" * 60)
    print("Experiment 1A-mid: Mid-Layer Loop Recurrence")
    print(f"Looping back to layer {MID_LAYER_INDEX}")
    print("=" * 60)

    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, name="exp1a_mid_layer_loop")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    prompts = get_prompts()
    results = {}

    for n_steps in RECURRENCE_STEPS:
        print(f"\n--- N = {n_steps} recurrence steps (mid-layer) ---")
        step_results = []
        correct = 0

        for i, prompt_data in enumerate(prompts):
            context = prompt_data["prompt"]
            query = prompt_data["prompt"]

            result = mid_layer_loop_recurrence(
                model=model,
                tokenizer=tokenizer,
                context_text=context,
                query_text=query,
                n_steps=n_steps,
                mid_layer=MID_LAYER_INDEX,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            answer = result["answer"]
            is_correct = check_answer(answer, prompt_data["answer"])
            correct += int(is_correct)

            step_results.append({
                "id": prompt_data["id"],
                "category": prompt_data["category"],
                "question": prompt_data["question"],
                "expected": prompt_data["answer"],
                "predicted": answer,
                "correct": is_correct,
            })

            status = "OK" if is_correct else "WRONG"
            print(f"  [{status}] {prompt_data['id']}: {answer[:80]}...")

        accuracy = correct / len(prompts)
        results[n_steps] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(prompts),
            "details": step_results,
        }

        print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(prompts)})")

        if WANDB_ENABLED:
            wandb.log({"n_steps": n_steps, "accuracy": accuracy})

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Accuracy vs Recurrence Steps (Mid-Layer Loop)")
    print("=" * 60)
    print(f"{'N Steps':>10} | {'Accuracy':>10} | {'Correct':>10}")
    print("-" * 40)
    for n_steps in RECURRENCE_STEPS:
        r = results[n_steps]
        print(f"{n_steps:>10} | {r['accuracy']:>9.1%} | {r['correct']:>5}/{r['total']}")

    # Per-category breakdown
    for cat in ["math", "logic", "factual"]:
        print(f"\n  Category: {cat}")
        for n_steps in RECURRENCE_STEPS:
            cat_results = [d for d in results[n_steps]["details"] if d["category"] == cat]
            cat_correct = sum(1 for d in cat_results if d["correct"])
            print(f"    N={n_steps:>2}: {cat_correct}/{len(cat_results)}")

    # Save results
    output_path = RESULTS_DIR / "exp1a_mid_layer_loop.json"
    serializable = {}
    for k, v in results.items():
        serializable[str(k)] = {
            "accuracy": v["accuracy"],
            "correct": v["correct"],
            "total": v["total"],
            "details": v["details"],
        }
    serializable["metadata"] = {
        "experiment": "exp1a_mid_layer_loop",
        "timestamp": datetime.now().isoformat(),
        "recurrence_steps": RECURRENCE_STEPS,
        "mid_layer_index": MID_LAYER_INDEX,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if WANDB_ENABLED:
        wandb.finish()

    return results


if __name__ == "__main__":
    run_experiment()
