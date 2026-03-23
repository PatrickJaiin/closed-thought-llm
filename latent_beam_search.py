"""
Latent Beam Search: Tree-of-Thoughts in hidden space on a frozen model.

Instead of generating text to explore reasoning paths, we branch in latent
space using the model's own logit lens predictions. Each branch represents
an alternative "next thought direction" the model was considering.

Algorithm:
1. Forward pass on prompt → hidden state h₀
2. Logit lens on h₀ → top-k candidate tokens (the model's uncertainty)
3. For each candidate, inject its embedding into h₀ and run partial forward
   through layers 12-35 → produces a new hidden state per branch
4. Score each branch via logit lens confidence
5. Keep top beam_width branches, prune the rest
6. Repeat from step 2 on each surviving branch
7. When confidence exceeds threshold → generate answer from that branch

This is Tree-of-Thoughts but entirely in latent space:
- No text generation until the final answer
- Branching is driven by the model's own uncertainty (not random noise)
- Orders of magnitude cheaper than text-space tree search
- Works on a frozen model with zero training
"""

import math
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from model_utils import (
    partial_forward, full_forward, get_embeddings,
    logit_lens, encode_and_forward,
)
from continuous_recurrence import _generate_with_prefix_state
from config import (
    MID_LAYER_INDEX, DEVICE, MAX_NEW_TOKENS,
    HIDDEN_DIM,
)


# ── Configuration defaults ────────────────────────────────────────────

BEAM_WIDTH = 3           # number of beams to keep after pruning
BRANCH_FACTOR = 5        # top-k candidates from logit lens at each step
BEAM_MAX_DEPTH = 8       # maximum branching depth
BEAM_CONFIDENCE_THRESHOLD = 0.9  # halt when max_prob exceeds this
BEAM_INJECTION_ALPHA = 1.0       # scaling for token embedding addition


# ── Data structures ───────────────────────────────────────────────────


@dataclass
class Beam:
    """A single beam (reasoning path) in the latent search tree."""
    hidden_state: torch.Tensor    # [1, 1, hidden_dim]
    score: float                  # cumulative log-probability
    depth: int                    # number of branching steps taken
    confidence: float             # current max_prob from logit lens
    token_path: list = field(default_factory=list)        # token IDs chosen at each branch
    confidence_history: list = field(default_factory=list) # max_prob at each step


# ── Core beam search ──────────────────────────────────────────────────


def latent_beam_search(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    beam_width: int = BEAM_WIDTH,
    branch_factor: int = BRANCH_FACTOR,
    max_depth: int = BEAM_MAX_DEPTH,
    confidence_threshold: float = BEAM_CONFIDENCE_THRESHOLD,
    injection_alpha: float = BEAM_INJECTION_ALPHA,
    mid_layer: int = MID_LAYER_INDEX,
    max_new_tokens: int = MAX_NEW_TOKENS,
    collect_diagnostics: bool = False,
) -> dict:
    """
    Run latent beam search on a prompt.

    Args:
        model: Frozen causal LM.
        tokenizer: Tokenizer.
        context_text: The prompt/context to reason about.
        query_text: The query for final answer generation.
        beam_width: Number of beams to keep after each pruning step.
        branch_factor: Number of top-k candidates to branch from each beam.
        max_depth: Maximum tree depth (branching iterations).
        confidence_threshold: Halt a beam when logit lens confidence exceeds this.
        injection_alpha: Scaling factor for token embedding injection.
        mid_layer: Layer index for partial forward injection.
        max_new_tokens: Max tokens for final answer generation.
        collect_diagnostics: Whether to collect per-step diagnostic info.

    Returns:
        dict with:
            - answer: Generated text answer from the best beam.
            - best_beam: The winning Beam object's metadata.
            - depth_reached: Actual tree depth explored.
            - total_forward_calls: Total partial_forward calls made (compute proxy).
            - halted: Whether a beam hit the confidence threshold.
            - all_beams: List of all final beams (if collect_diagnostics).
            - diagnostics: Per-depth diagnostic info (if collect_diagnostics).
    """
    diagnostics = [] if collect_diagnostics else None
    total_forward_calls = 0

    with torch.no_grad():
        # Step 1: Full forward pass on context → initial hidden state
        hidden, _, _ = encode_and_forward(model, tokenizer, context_text)
        h0 = hidden[:, -1:, :]  # [1, 1, hidden_dim]

        # Score the initial state
        lens0 = logit_lens(model, h0)
        initial_confidence = lens0["max_prob"].item()

        # Check if model is already confident enough (trivial problem)
        if initial_confidence >= confidence_threshold:
            answer = _generate_with_prefix_state(
                model, tokenizer, h0, query_text, max_new_tokens
            )
            return {
                "answer": answer,
                "best_beam": {
                    "score": 0.0,
                    "depth": 0,
                    "confidence": initial_confidence,
                    "token_path": [],
                    "confidence_history": [initial_confidence],
                },
                "depth_reached": 0,
                "total_forward_calls": 0,
                "halted": True,
                "all_beams": None,
                "diagnostics": diagnostics,
            }

        # Step 2: Create initial beams by branching from h0
        beams = _branch_and_score(
            model, h0, branch_factor, injection_alpha, mid_layer,
            parent_score=0.0, parent_depth=0, parent_path=[],
            parent_conf_history=[initial_confidence],
        )
        total_forward_calls += len(beams)

        # Prune to beam_width
        beams = _select_top_beams(beams, beam_width)

        if collect_diagnostics:
            diagnostics.append({
                "depth": 1,
                "num_candidates": branch_factor,
                "num_surviving": len(beams),
                "best_confidence": beams[0].confidence if beams else 0.0,
                "beam_scores": [b.score for b in beams],
                "beam_confidences": [b.confidence for b in beams],
            })

        # Step 3: Iterative deepening
        halted = False
        depth_reached = 1

        for depth in range(2, max_depth + 1):
            # Check if any beam has halted
            best_beam = max(beams, key=lambda b: b.confidence)
            if best_beam.confidence >= confidence_threshold:
                halted = True
                break

            # Branch from each surviving beam
            new_beams = []
            for beam in beams:
                children = _branch_and_score(
                    model, beam.hidden_state, branch_factor,
                    injection_alpha, mid_layer,
                    parent_score=beam.score, parent_depth=beam.depth,
                    parent_path=beam.token_path,
                    parent_conf_history=beam.confidence_history,
                )
                total_forward_calls += len(children)
                new_beams.extend(children)

            # Prune all children to beam_width
            beams = _select_top_beams(new_beams, beam_width)
            depth_reached = depth

            if collect_diagnostics:
                diagnostics.append({
                    "depth": depth,
                    "num_candidates": len(new_beams),
                    "num_surviving": len(beams),
                    "best_confidence": beams[0].confidence if beams else 0.0,
                    "beam_scores": [b.score for b in beams],
                    "beam_confidences": [b.confidence for b in beams],
                })

        # Step 4: Generate answer from the best beam
        best_beam = max(beams, key=lambda b: b.confidence)
        answer = _generate_with_prefix_state(
            model, tokenizer, best_beam.hidden_state, query_text, max_new_tokens
        )

    result = {
        "answer": answer,
        "best_beam": {
            "score": best_beam.score,
            "depth": best_beam.depth,
            "confidence": best_beam.confidence,
            "token_path": best_beam.token_path,
            "confidence_history": best_beam.confidence_history,
        },
        "depth_reached": depth_reached,
        "total_forward_calls": total_forward_calls,
        "halted": halted,
        "all_beams": [
            {
                "score": b.score,
                "depth": b.depth,
                "confidence": b.confidence,
                "token_path": b.token_path,
            }
            for b in beams
        ] if collect_diagnostics else None,
        "diagnostics": diagnostics,
    }

    return result


# ── Internal helpers ──────────────────────────────────────────────────


def _branch_and_score(
    model,
    h: torch.Tensor,
    branch_factor: int,
    injection_alpha: float,
    mid_layer: int,
    parent_score: float,
    parent_depth: int,
    parent_path: list,
    parent_conf_history: list,
) -> List[Beam]:
    """
    Branch from a hidden state: get top-k candidates from logit lens,
    inject each candidate's embedding, run partial forward, score.

    Returns a list of Beam objects (one per candidate).
    """
    # Get top-k candidates from current hidden state
    lens = logit_lens(model, h)
    top_k_ids = lens["top_k_ids"][0, 0, :branch_factor]  # [branch_factor]
    top_k_probs = lens["top_k_probs"][0, 0, :branch_factor]  # [branch_factor]

    beams = []
    position_ids = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)

    for i in range(min(branch_factor, len(top_k_ids))):
        token_id = top_k_ids[i].item()
        token_prob = top_k_probs[i].item()

        # Get token embedding and inject into hidden state
        # Scale embedding to match hidden state norm so it actually influences the result
        token_tensor = torch.tensor([[token_id]], device=DEVICE)
        emb = get_embeddings(model, token_tensor)  # [1, 1, hidden_dim]

        # Normalize embedding to hidden state scale, then apply alpha
        h_norm = h.norm()
        emb_norm = emb.norm().clamp(min=1e-8)
        scaled_emb = emb * (h_norm / emb_norm)

        # Combine: hidden state + scaled token embedding direction
        h_branch = h + injection_alpha * scaled_emb

        # Run through upper layers
        h_new = partial_forward(
            model, h_branch, start_layer=mid_layer,
            position_ids=position_ids,
        )

        # Score the result via logit lens
        lens_new = logit_lens(model, h_new)
        new_confidence = lens_new["max_prob"].item()

        # Cumulative log-prob score
        new_score = parent_score + math.log(max(token_prob, 1e-10))

        beams.append(Beam(
            hidden_state=h_new,
            score=new_score,
            depth=parent_depth + 1,
            confidence=new_confidence,
            token_path=parent_path + [token_id],
            confidence_history=parent_conf_history + [new_confidence],
        ))

    return beams


def _select_top_beams(beams: List[Beam], beam_width: int) -> List[Beam]:
    """Select top beam_width beams by confidence (primary) and score (tiebreak)."""
    beams.sort(key=lambda b: (b.confidence, b.score), reverse=True)
    return beams[:beam_width]


# ── Convenience wrapper for benchmarks ────────────────────────────────


def run_beam_search_on_item(
    model, tokenizer, item, max_tokens,
    beam_width=BEAM_WIDTH,
    branch_factor=BRANCH_FACTOR,
    max_depth=BEAM_MAX_DEPTH,
    confidence_threshold=BEAM_CONFIDENCE_THRESHOLD,
    injection_alpha=BEAM_INJECTION_ALPHA,
):
    """
    Run latent beam search on a BenchmarkItem.
    Returns dict compatible with the ablation experiment format.
    """
    result = latent_beam_search(
        model, tokenizer,
        context_text=item.prompt,
        query_text=item.prompt,
        beam_width=beam_width,
        branch_factor=branch_factor,
        max_depth=max_depth,
        confidence_threshold=confidence_threshold,
        injection_alpha=injection_alpha,
        max_new_tokens=max_tokens,
    )

    return {
        "answer": result["answer"],
        "n_steps_taken": result["total_forward_calls"],
        "depth_reached": result["depth_reached"],
        "best_confidence": result["best_beam"]["confidence"],
        "halted": result["halted"],
    }
