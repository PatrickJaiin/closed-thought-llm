"""
Experiment 1B: Text self-prompting baseline.
Let the model generate N tokens of "thinking" text, then answer the query.
Compare to latent recurrence experiments.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from config import MAX_NEW_TOKENS, RESULTS_DIR, WANDB_ENABLED, WANDB_PROJECT
from model_utils import load_model
from recurrence import text_baseline
from eval_prompts import get_prompts, check_answer

if WANDB_ENABLED:
    import wandb

# Map recurrence steps to approximate equivalent thinking tokens
# Each recurrence step processes the full model; we match by giving
# a comparable number of "thinking" tokens
THINKING_TOKEN_COUNTS = [0, 4, 8, 16, 32, 64, 128]


def run_experiment():
    print("=" * 60)
    print("Experiment 1B: Text Self-Prompting Baseline")
    print("=" * 60)

    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, name="exp1b_text_baseline")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    prompts = get_prompts()
    results = {}

    for n_tokens in THINKING_TOKEN_COUNTS:
        print(f"\n--- N = {n_tokens} thinking tokens ---")
        step_results = []
        correct = 0

        for i, prompt_data in enumerate(prompts):
            context = prompt_data["prompt"]
            query = prompt_data["prompt"]

            if n_tokens == 0:
                # No thinking — just direct generation
                inputs = tokenizer(query, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output = model.generate(
                        inputs.input_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                    )
                answer = tokenizer.decode(
                    output[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                thinking = ""
            else:
                result = text_baseline(
                    model=model,
                    tokenizer=tokenizer,
                    context_text=context,
                    query_text=query,
                    n_thinking_tokens=n_tokens,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                answer = result["answer"]
                thinking = result["thinking"]

            is_correct = check_answer(answer, prompt_data["answer"])
            correct += int(is_correct)

            step_results.append({
                "id": prompt_data["id"],
                "category": prompt_data["category"],
                "question": prompt_data["question"],
                "expected": prompt_data["answer"],
                "predicted": answer,
                "thinking": thinking[:200] if thinking else "",
                "correct": is_correct,
            })

            status = "OK" if is_correct else "WRONG"
            print(f"  [{status}] {prompt_data['id']}: {answer[:80]}...")

        accuracy = correct / len(prompts)
        results[n_tokens] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(prompts),
            "details": step_results,
        }

        print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(prompts)})")

        if WANDB_ENABLED:
            wandb.log({"n_thinking_tokens": n_tokens, "accuracy": accuracy})

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Accuracy vs Thinking Tokens (Text Baseline)")
    print("=" * 60)
    print(f"{'N Tokens':>10} | {'Accuracy':>10} | {'Correct':>10}")
    print("-" * 40)
    for n_tokens in THINKING_TOKEN_COUNTS:
        r = results[n_tokens]
        print(f"{n_tokens:>10} | {r['accuracy']:>9.1%} | {r['correct']:>5}/{r['total']}")

    # Save results
    output_path = RESULTS_DIR / "exp1b_text_baseline.json"
    serializable = {}
    for k, v in results.items():
        serializable[str(k)] = {
            "accuracy": v["accuracy"],
            "correct": v["correct"],
            "total": v["total"],
            "details": v["details"],
        }
    serializable["metadata"] = {
        "experiment": "exp1b_text_baseline",
        "timestamp": datetime.now().isoformat(),
        "thinking_token_counts": THINKING_TOKEN_COUNTS,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if WANDB_ENABLED:
        wandb.finish()

    return results


if __name__ == "__main__":
    run_experiment()
