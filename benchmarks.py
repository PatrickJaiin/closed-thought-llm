"""
Benchmark dataset loading and answer extraction for Phase 5 evaluation.

Supports:
- GSM8K: Grade school math (1319 test problems)
- ARC Challenge: AI2 Reasoning Challenge (1172 test problems)

Each loader returns a list of BenchmarkItem dataclasses with standardized
prompt format and expected answer.
"""

import re
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset

from config import HIDDEN_DIM, GATE_HIDDEN_DIM


# ── Data structures ───────────────────────────────────────────────────


@dataclass
class BenchmarkItem:
    """A single benchmark problem."""
    id: str
    prompt: str          # formatted prompt ready for model input
    expected: str        # ground-truth answer (number for GSM8K, letter for ARC)
    category: str        # "gsm8k" or "arc"
    raw_question: str    # original question text


# ── ThresholdedHaltGate wrapper ───────────────────────────────────────


class ThresholdedHaltGate(nn.Module):
    """
    Wraps a HaltGate (nn.Module) and maps p > custom_threshold to binary output.

    The continuous_recurrence loop's _call_halt_fn applies a hardcoded >0.5
    check for nn.Module gates. This wrapper pre-applies the custom threshold
    so that _call_halt_fn's >0.5 check works correctly.

    Usage:
        gate = HaltGate(...)
        gate.load_state_dict(torch.load("halt_gate_rl.pt"))
        wrapped = ThresholdedHaltGate(gate, threshold=0.3)
        # Pass wrapped as halt_fn to continuous_recurrence
    """

    def __init__(self, gate: nn.Module, threshold: float = 0.5):
        super().__init__()
        self.gate = gate
        self.threshold = threshold

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return 1.0 if gate(h) > threshold, else 0.0."""
        with torch.no_grad():
            p = self.gate(h)
            if isinstance(p, torch.Tensor):
                return (p > self.threshold).float()
            return torch.tensor(1.0 if p > self.threshold else 0.0)


# ── GSM8K ─────────────────────────────────────────────────────────────


def load_gsm8k(
    split: str = "test",
    subset_n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkItem]:
    """
    Load GSM8K dataset and format as BenchmarkItems.

    Prompt format: "Question: {question}\nAnswer:" (zero-shot)
    Answer: extract number after #### from dataset's answer field.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    if subset_n is not None and subset_n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(subset_n))

    items = []
    for i, row in enumerate(ds):
        question = row["question"].strip()
        # Extract answer: number after ####
        answer_text = row["answer"]
        expected = _extract_gsm8k_gold(answer_text)

        prompt = f"Question: {question}\nAnswer:"

        items.append(BenchmarkItem(
            id=f"gsm8k_{i:04d}",
            prompt=prompt,
            expected=expected,
            category="gsm8k",
            raw_question=question,
        ))

    return items


def _extract_gsm8k_gold(answer_text: str) -> str:
    """Extract the gold answer number from GSM8K's answer field (after ####)."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    # Fallback: last number in text
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return answer_text.strip()


def extract_gsm8k_answer(text: str) -> str:
    """
    Multi-tier answer extraction from model output.

    Priority:
    1. #### marker (if model mimics GSM8K format)
    2. "the answer is" pattern
    3. Last number in text
    """
    # Tier 1: #### marker
    match = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Tier 2: "the answer is" pattern
    match = re.search(r"the answer is\s*[:\s]*(-?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Tier 3: last number in text
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return text.strip()


def check_gsm8k_answer(predicted: str, expected: str) -> bool:
    """
    Check if predicted number matches expected.
    Normalizes by removing commas and comparing numeric values.
    """
    try:
        pred_num = float(predicted.replace(",", ""))
        exp_num = float(expected.replace(",", ""))
        return abs(pred_num - exp_num) < 1e-6
    except (ValueError, AttributeError):
        return predicted.strip() == expected.strip()


# ── ARC Challenge ─────────────────────────────────────────────────────


def load_arc(
    split: str = "test",
    subset_n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkItem]:
    """
    Load ARC Challenge dataset and format as BenchmarkItems.

    Prompt format: "Question: ...\nA) ...\nB) ...\nAnswer:"
    Answer: letter (A/B/C/D/E), handles both letter and numeric answerKey.
    """
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)

    if subset_n is not None and subset_n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(subset_n))

    items = []
    for i, row in enumerate(ds):
        question = row["question"].strip()
        choices = row["choices"]
        answer_key = row["answerKey"]

        # Format choices as multiple choice
        labels = choices["label"]
        texts = choices["text"]
        choice_lines = []
        for label, text in zip(labels, texts):
            choice_lines.append(f"{label}) {text}")
        choices_str = "\n".join(choice_lines)

        prompt = f"Question: {question}\n{choices_str}\nAnswer:"

        # Normalize answerKey: numeric (1,2,3,4) → letter (A,B,C,D)
        expected = _normalize_arc_answer(answer_key, labels)

        items.append(BenchmarkItem(
            id=f"arc_{i:04d}",
            prompt=prompt,
            expected=expected,
            category="arc",
            raw_question=question,
        ))

    return items


def _normalize_arc_answer(answer_key: str, labels: list) -> str:
    """Convert numeric answerKey to letter if needed."""
    # If already a letter
    if answer_key.upper() in ["A", "B", "C", "D", "E"]:
        return answer_key.upper()
    # Numeric: 1→A, 2→B, etc.
    try:
        idx = int(answer_key) - 1
        if 0 <= idx < len(labels):
            return labels[idx].upper()
    except ValueError:
        pass
    return answer_key.upper()


def extract_arc_answer(text: str) -> str:
    """
    Extract answer letter from model output.

    Priority:
    1. First character if it's A-E
    2. Pattern match "answer is [letter]"
    3. First standalone A-E letter
    """
    text = text.strip()

    # Tier 1: first character
    if text and text[0].upper() in "ABCDE":
        return text[0].upper()

    # Tier 2: "answer is" pattern
    match = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*([A-Ea-e])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Tier 3: first standalone A-E letter
    match = re.search(r"\b([A-Ea-e])\b", text)
    if match:
        return match.group(1).upper()

    return text[:1].upper() if text else ""


def check_arc_answer(predicted: str, expected: str) -> bool:
    """Check if predicted letter matches expected letter."""
    return predicted.strip().upper() == expected.strip().upper()


# ── Unified factory ───────────────────────────────────────────────────


def load_benchmark(
    name: str,
    split: str = "test",
    subset_n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkItem]:
    """
    Load a benchmark dataset by name.

    Args:
        name: "gsm8k" or "arc"
        split: Dataset split (default "test")
        subset_n: Number of items to load (None = full dataset)
        seed: Random seed for subset selection

    Returns:
        List of BenchmarkItem
    """
    loaders = {
        "gsm8k": load_gsm8k,
        "arc": load_arc,
    }
    if name not in loaders:
        raise ValueError(f"Unknown benchmark: {name}. Choose from: {list(loaders.keys())}")
    return loaders[name](split=split, subset_n=subset_n, seed=seed)


def extract_answer(text: str, benchmark: str) -> str:
    """Extract answer from model output based on benchmark type."""
    extractors = {
        "gsm8k": extract_gsm8k_answer,
        "arc": extract_arc_answer,
    }
    return extractors[benchmark](text)


def check_answer(predicted: str, expected: str, benchmark: str) -> bool:
    """Check if predicted answer is correct for the given benchmark."""
    checkers = {
        "gsm8k": check_gsm8k_answer,
        "arc": check_arc_answer,
    }
    return checkers[benchmark](predicted, expected)


# ── Self-test ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Testing GSM8K loader...")
    gsm_items = load_gsm8k(subset_n=5)
    print(f"  Loaded {len(gsm_items)} items")
    for item in gsm_items:
        print(f"  [{item.id}] expected={item.expected}")
        print(f"    prompt: {item.prompt[:80]}...")
    print()

    print("Testing ARC loader...")
    arc_items = load_arc(subset_n=5)
    print(f"  Loaded {len(arc_items)} items")
    for item in arc_items:
        print(f"  [{item.id}] expected={item.expected}")
        print(f"    prompt: {item.prompt[:80]}...")
    print()

    # Test answer extraction
    print("Testing GSM8K answer extraction:")
    tests = [
        ("#### 42", "42"),
        ("The answer is 42.", "42"),
        ("Let me compute. 3+4=7, so total is 42 dogs.", "42"),
        ("1,234", "1234"),
    ]
    for text, expected in tests:
        got = extract_gsm8k_answer(text)
        ok = "OK" if got == expected else "FAIL"
        print(f"  [{ok}] '{text}' -> '{got}' (expected '{expected}')")

    print("\nTesting ARC answer extraction:")
    tests = [
        ("A", "A"),
        ("B) gravity", "B"),
        ("The answer is C.", "C"),
        ("I think it's D because...", "D"),
    ]
    for text, expected in tests:
        got = extract_arc_answer(text)
        ok = "OK" if got == expected else "FAIL"
        print(f"  [{ok}] '{text}' -> '{got}' (expected '{expected}')")
