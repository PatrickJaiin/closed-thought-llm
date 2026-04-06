"""
Quick interactive demo of KV recurrence on your local machine.
Usage: python try_model.py
"""

import sys
import torch
from model_utils import load_model
from kv_recurrence import kv_recurrence
from config import DEVICE, DTYPE

print(f"Device: {DEVICE} | Dtype: {DTYPE}")
print("Loading model...")
model, tokenizer = load_model()
print(f"Model loaded on {DEVICE}.\n")

print("=" * 60)
print("Interactive mode (type 'quit' to exit)")
print("=" * 60)
while True:
    try:
        context = input("\nContext: ").strip()
        if context.lower() == "quit":
            break
        query = input("Query:   ").strip()
        if query.lower() == "quit":
            break
        try:
            steps = int(input("Steps [0/4/8, default 4]: ").strip() or "4")
        except ValueError:
            steps = 4

        print("Thinking...")
        result = kv_recurrence(
            model,
            tokenizer,
            context_text=context,
            query_text=query,
            n_steps=steps,
            collect_diagnostics=True,
        )
        print(f"\nAnswer: {result['answer']}")
        print(f"Steps taken: {result['n_steps_taken']}, Halted: {result['halted']}")
        if result.get("diagnostics"):
            last = result["diagnostics"][-1]
            print(f"Final confidence: {last.get('max_prob', 'N/A')}")
    except KeyboardInterrupt:
        print("\nBye!")
        break
