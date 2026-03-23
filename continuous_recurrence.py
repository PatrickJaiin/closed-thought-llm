"""
Continuous recurrence engine with pluggable halting and memory.

Refactored from recurrence.mid_layer_loop_recurrence() — replaces the fixed
`for step in range(n_steps)` loop with `while not halt_fn(h, step, diag)`.

Supports:
- Heuristic halt functions (Phase 2)
- Learned HaltGate modules (Phase 3)
- Memory read/write hooks (Phase 4)
- Fixed-N mode for backwards compatibility
"""

import torch
from typing import Optional, Callable, Union
from itertools import count

from model_utils import partial_forward, full_forward, get_embeddings, logit_lens
from config import (
    MID_LAYER_INDEX, DEVICE, MAX_CONTINUOUS_STEPS, MAX_NEW_TOKENS,
    MEMORY_RESIDUAL_ALPHA,
)


# Type alias for halt functions
# Args: (hidden_state: Tensor[1,1,H], step: int, diagnostics: dict) -> bool
HaltFn = Callable[[torch.Tensor, int, dict], bool]


def _default_halt_fn(h: torch.Tensor, step: int, diag: dict) -> bool:
    """Default: never halt (relies on max_steps safety cap)."""
    return False


def _fixed_n_halt(n_steps: int) -> HaltFn:
    """Return a halt function that stops after exactly n_steps."""
    def halt_fn(h: torch.Tensor, step: int, diag: dict) -> bool:
        return step >= n_steps
    return halt_fn


def continuous_recurrence(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    halt_fn: Optional[Union[HaltFn, "torch.nn.Module"]] = None,
    n_steps: Optional[int] = None,
    mid_layer: Optional[int] = None,
    max_steps: int = MAX_CONTINUOUS_STEPS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    collect_hidden: bool = False,
    collect_diagnostics: bool = False,
    memory: Optional[object] = None,
    memory_gate: Optional[object] = None,
    memory_alpha: float = MEMORY_RESIDUAL_ALPHA,
    halt_threshold: float = 0.5,
) -> dict:
    """
    Continuous mid-layer recurrence with pluggable halting and memory.

    The loop runs until halt_fn returns True or max_steps is reached.

    Args:
        model: Frozen causal LM.
        tokenizer: Tokenizer.
        context_text: Context/prompt to process.
        query_text: Query to answer after recurrence.
        halt_fn: Callable(h, step, diag) -> bool, or nn.Module with forward(h) -> p_halt.
                 If None and n_steps is None, runs until max_steps.
        n_steps: If provided, use fixed-N mode (backwards compatible).
        mid_layer: Layer index to inject recurrence (default: MID_LAYER_INDEX).
        max_steps: Safety cap on maximum iterations.
        max_new_tokens: Max tokens to generate in answer.
        collect_hidden: Whether to collect hidden state trajectory.
        collect_diagnostics: Whether to collect per-step diagnostics.
        memory: Memory object with read(h) and write(h) methods (Phase 4).
        memory_gate: Gate object with should_retrieve(h) and should_store(h) (Phase 4).
        memory_alpha: Blending weight for memory retrieval residual addition.
        halt_threshold: Threshold for nn.Module halt gates (default 0.5).

    Returns:
        dict with:
            - answer: Generated text answer.
            - n_steps_taken: Actual number of recurrence steps.
            - halted: Whether halt_fn triggered (vs max_steps cap).
            - hidden_states: List of hidden states if collect_hidden=True.
            - diagnostics: Per-step diagnostics if collect_diagnostics=True.
    """
    if mid_layer is None:
        mid_layer = MID_LAYER_INDEX

    # Resolve halt function
    if n_steps is not None:
        # Fixed-N mode for backwards compatibility
        effective_halt_fn = _fixed_n_halt(n_steps)
        max_steps = n_steps  # no safety cap needed
    elif halt_fn is not None:
        effective_halt_fn = halt_fn
    else:
        effective_halt_fn = _default_halt_fn

    hidden_trajectory = []
    step_diagnostics = []

    with torch.no_grad():
        # Step 1: Full forward pass on context
        context_inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        context_embeds = get_embeddings(model, context_inputs.input_ids)
        hidden = full_forward(
            model, context_embeds, attention_mask=context_inputs.attention_mask
        )
        h = hidden[:, -1:, :]  # (1, 1, hidden_dim)

        if collect_hidden:
            hidden_trajectory.append(h.clone().cpu())

        # Step 2: Continuous recurrence loop
        h_prev = None
        halted = False
        steps_taken = 0

        for step in count():
            if step >= max_steps:
                break

            # Build per-step diagnostics dict (available to halt_fn)
            diag = {}
            if h_prev is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    h.view(1, -1), h_prev.view(1, -1)
                ).item()
                delta_norm = (h - h_prev).norm().item()
                diag["cos_sim"] = cos_sim
                diag["delta_norm"] = delta_norm
            diag["h_norm"] = h.norm().item()
            diag["step"] = step

            # Check halt condition
            if _call_halt_fn(effective_halt_fn, h, step, diag, threshold=halt_threshold):
                halted = True
                break

            h_prev = h.clone()

            # Memory retrieval (Phase 4) — BEFORE forward pass
            if memory is not None and memory_gate is not None:
                if memory_gate.should_retrieve(h):
                    mem = memory.read(h)
                    if mem is not None:
                        h = h + memory_alpha * mem.to(h.dtype)  # residual addition
            elif memory is not None:
                # No gate — always try to retrieve
                mem = memory.read(h)
                if mem is not None:
                    h = h + memory_alpha * mem.to(h.dtype)

            # Core recurrence step: partial forward through upper layers
            position_ids = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
            h = partial_forward(
                model, h, start_layer=mid_layer,
                position_ids=position_ids
            )

            # Memory storage (Phase 4) — AFTER forward pass
            if memory is not None and memory_gate is not None:
                if memory_gate.should_store(h):
                    memory.write(h)
            elif memory is not None:
                # No gate — always store
                memory.write(h)

            steps_taken = step + 1

            if collect_hidden:
                hidden_trajectory.append(h.clone().cpu())

            if collect_diagnostics:
                # Add logit lens info
                lens = logit_lens(model, h)
                diag["max_prob"] = lens["max_prob"].item()
                diag["entropy"] = lens["entropy"].item()
                step_diagnostics.append(diag)

        # Step 3: Generate answer with recurrence state prepended to query
        answer = _generate_with_prefix_state(
            model, tokenizer, h, query_text, max_new_tokens
        )

    return {
        "answer": answer,
        "n_steps_taken": steps_taken,
        "halted": halted,
        "hidden_states": hidden_trajectory if collect_hidden else None,
        "diagnostics": step_diagnostics if collect_diagnostics else None,
    }


def _call_halt_fn(halt_fn, h, step, diag, threshold=0.5):
    """
    Call halt_fn, handling both plain callables and nn.Module gates.

    For nn.Module gates (Phase 3), calls forward(h) and thresholds at given threshold.
    For plain callables (Phase 2 heuristics), calls halt_fn(h, step, diag).
    """
    if isinstance(halt_fn, torch.nn.Module):
        with torch.no_grad():
            p_halt = halt_fn(h)
            if isinstance(p_halt, torch.Tensor):
                return p_halt.item() > threshold
            return p_halt > threshold
    else:
        return halt_fn(h, step, diag)


def _generate_with_prefix_state(model, tokenizer, hidden_state, query_text, max_new_tokens):
    """
    Generate answer by prepending the recurrence hidden state (as a pseudo-embedding)
    to the query's token embeddings, then using model.generate() with inputs_embeds.

    Reused from recurrence.py — identical logic.
    """
    query_inputs = tokenizer(query_text, return_tensors="pt").to(DEVICE)
    query_embeds = get_embeddings(model, query_inputs.input_ids)

    h = hidden_state.to(dtype=query_embeds.dtype)
    combined_embeds = torch.cat([h, query_embeds], dim=1)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


def continuous_recurrence_trajectory(
    model,
    tokenizer,
    context_text: str,
    max_steps: int = MAX_CONTINUOUS_STEPS,
    mid_layer: Optional[int] = None,
) -> dict:
    """
    Run continuous recurrence without a query — just collect the trajectory.
    Used for stability analysis (exp2a).

    Returns:
        dict with:
            - trajectory: list of hidden state numpy arrays
            - norms: list of L2 norms
            - cosine_sims: list of cosine similarities between consecutive steps
            - nan_detected: bool
    """
    import numpy as np

    if mid_layer is None:
        mid_layer = MID_LAYER_INDEX

    trajectory = []
    norms = []
    cosine_sims = []

    with torch.no_grad():
        context_inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        context_embeds = get_embeddings(model, context_inputs.input_ids)
        hidden = full_forward(
            model, context_embeds, attention_mask=context_inputs.attention_mask
        )
        h = hidden[:, -1:, :]

        h_np = h.squeeze().cpu().float().numpy()
        trajectory.append(h_np)
        norms.append(float(np.linalg.norm(h_np)))

        nan_detected = False

        for step in range(max_steps):
            position_ids = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
            h = partial_forward(
                model, h, start_layer=mid_layer,
                position_ids=position_ids
            )

            h_np = h.squeeze().cpu().float().numpy()

            if np.any(np.isnan(h_np)) or np.any(np.isinf(h_np)):
                nan_detected = True
                print(f"  NaN/Inf detected at step {step + 1}")
                break

            norm = float(np.linalg.norm(h_np))
            norms.append(norm)

            # Cosine similarity with previous
            h_prev = trajectory[-1]
            cos = float(
                np.dot(h_prev, h_np)
                / (np.linalg.norm(h_prev) * np.linalg.norm(h_np) + 1e-8)
            )
            cosine_sims.append(cos)

            trajectory.append(h_np)

    return {
        "trajectory": trajectory,
        "norms": norms,
        "cosine_sims": cosine_sims,
        "nan_detected": nan_detected,
        "steps_completed": len(trajectory) - 1,
    }
