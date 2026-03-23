"""
Heuristic halting gates for the continuous recurrence loop (Phase 2).

Four strategies, all sharing the same interface:
    halt_fn(h: Tensor, step: int, diag: dict) -> bool

These serve as proof-of-concept gates and as training labels for learned gates (Phase 3).
"""

import torch
from model_utils import logit_lens, hidden_to_logits
from config import (
    HALT_CONFIDENCE_THRESHOLD,
    HALT_CONVERGENCE_THRESHOLD,
    HALT_ENTROPY_THRESHOLD,
    HALT_DELTA_NORM_THRESHOLD,
)


class ConfidenceHalt:
    """
    Halt when the model's top-token probability exceeds a threshold.

    Projects h through the LM head via logit_lens, checks if
    softmax(logits).max() > threshold. Requires at least min_steps.
    """

    def __init__(self, model, threshold=HALT_CONFIDENCE_THRESHOLD, min_steps=1):
        self.model = model
        self.threshold = threshold
        self.min_steps = min_steps

    def __call__(self, h: torch.Tensor, step: int, diag: dict) -> bool:
        if step < self.min_steps:
            return False
        with torch.no_grad():
            logits = self.model.lm_head(h)  # (1, 1, vocab_size)
            probs = torch.softmax(logits.float(), dim=-1)
            max_prob = probs.max().item()
        diag["confidence"] = max_prob
        return max_prob > self.threshold


class ConvergenceHalt:
    """
    Halt when consecutive hidden states become very similar.

    Checks cos_sim(h_t, h_{t-1}) > threshold, indicating the recurrence
    has converged to a near-fixed-point. Calibrated from Phase 1 data
    where cosine similarity plateaus around 0.85-0.95.
    """

    def __init__(self, threshold=HALT_CONVERGENCE_THRESHOLD, min_steps=2):
        self.threshold = threshold
        self.min_steps = min_steps

    def __call__(self, h: torch.Tensor, step: int, diag: dict) -> bool:
        if step < self.min_steps:
            return False
        cos_sim = diag.get("cos_sim")
        if cos_sim is None:
            return False
        return cos_sim > self.threshold


class EntropyHalt:
    """
    Halt when the logit distribution entropy drops below a threshold.

    Low entropy = the model is confident about one token = it has "decided."
    Entropy is measured in nats (natural log).
    """

    def __init__(self, model, threshold=HALT_ENTROPY_THRESHOLD, min_steps=1):
        self.model = model
        self.threshold = threshold
        self.min_steps = min_steps

    def __call__(self, h: torch.Tensor, step: int, diag: dict) -> bool:
        if step < self.min_steps:
            return False
        with torch.no_grad():
            logits = self.model.lm_head(h)  # (1, 1, vocab_size)
            probs = torch.softmax(logits.float(), dim=-1)
            log_probs = torch.log(probs.clamp(min=1e-10))
            entropy = -(probs * log_probs).sum().item()
        diag["entropy"] = entropy
        return entropy < self.threshold


class DeltaNormHalt:
    """
    Halt when the L2 distance between consecutive hidden states is small.

    This is the ACT-style criterion: stop when the update magnitude
    ||h_t - h_{t-1}||_2 drops below a threshold.
    """

    def __init__(self, threshold=HALT_DELTA_NORM_THRESHOLD, min_steps=2):
        self.threshold = threshold
        self.min_steps = min_steps

    def __call__(self, h: torch.Tensor, step: int, diag: dict) -> bool:
        if step < self.min_steps:
            return False
        delta_norm = diag.get("delta_norm")
        if delta_norm is None:
            return False
        return delta_norm < self.threshold


class CombinedHalt:
    """
    Halt when ANY of the provided halt functions returns True.
    Useful for combining multiple heuristics.
    """

    def __init__(self, *halt_fns):
        self.halt_fns = halt_fns

    def __call__(self, h: torch.Tensor, step: int, diag: dict) -> bool:
        for fn in self.halt_fns:
            if fn(h, step, diag):
                return True
        return False


def make_heuristic_halt(name: str, model=None, **kwargs):
    """
    Factory function to create a heuristic halt gate by name.

    Args:
        name: One of "confidence", "convergence", "entropy", "delta_norm", "combined"
        model: Required for "confidence" and "entropy" gates.
        **kwargs: Overrides for threshold, min_steps, etc.

    Returns:
        A halt function with the standard interface.
    """
    gates = {
        "confidence": lambda: ConfidenceHalt(model, **kwargs),
        "convergence": lambda: ConvergenceHalt(**kwargs),
        "entropy": lambda: EntropyHalt(model, **kwargs),
        "delta_norm": lambda: DeltaNormHalt(**kwargs),
    }

    if name == "combined":
        # Build all four and combine
        fns = [
            ConfidenceHalt(model, **{k: v for k, v in kwargs.items() if k in ("threshold", "min_steps")}),
            ConvergenceHalt(**{k: v for k, v in kwargs.items() if k in ("threshold", "min_steps")}),
            EntropyHalt(model, **{k: v for k, v in kwargs.items() if k in ("threshold", "min_steps")}),
            DeltaNormHalt(**{k: v for k, v in kwargs.items() if k in ("threshold", "min_steps")}),
        ]
        return CombinedHalt(*fns)

    if name not in gates:
        raise ValueError(f"Unknown gate: {name}. Choose from {list(gates.keys())}")

    return gates[name]()
