"""
KV-Cache Recurrence: Latent reasoning with full prompt attention.

Unlike the original single-vector recurrence (which processes an isolated hidden
state with no attention to context), this module maintains the prompt's KV cache
during recurrence. Each thought token can attend to:
  1. All original prompt tokens (via the frozen KV cache)
  2. All prior thought tokens (via accumulated KV entries)

This is COCONUT's mechanism on a frozen model — latent thought tokens occupy
real positions in the attention sequence, enabling the model to "re-read" the
question while thinking.

Two recurrence modes:
  - Partial (exp7a): layers 12-35 only, KV cache grows in those layers only.
    Generation discards the cache (single-vector bottleneck).
  - Full (exp7b): ALL 36 layers per step, uniform KV cache growth.
    Generation uses the full cache — output tokens attend to prompt + thoughts.
"""

import torch
import copy
from typing import Optional, Callable, Union
from transformers.cache_utils import DynamicCache

from model_utils import get_embeddings, logit_lens
from config import MID_LAYER_INDEX, DEVICE, MAX_CONTINUOUS_STEPS, MAX_NEW_TOKENS


def kv_recurrence(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    n_steps: int = 32,
    mid_layer: int = MID_LAYER_INDEX,
    max_new_tokens: int = MAX_NEW_TOKENS,
    confidence_threshold: Optional[float] = None,
    max_steps: int = MAX_CONTINUOUS_STEPS,
    collect_diagnostics: bool = False,
) -> dict:
    """
    Run latent recurrence with KV cache persistence.

    Each recurrence step:
    1. Takes the current hidden state h
    2. Runs it through layers mid_layer..35 with full KV cache attention
    3. The thought token's KV entries are appended to the cache
    4. The model can attend to prompt + all prior thoughts

    Args:
        model: Frozen causal LM.
        tokenizer: Tokenizer.
        context_text: The prompt/context to reason about.
        query_text: The query for final answer generation.
        n_steps: Number of recurrence steps (if no confidence threshold).
        mid_layer: Layer index for partial forward injection (default: 12).
        max_new_tokens: Max tokens for final answer generation.
        confidence_threshold: If set, halt when logit lens confidence exceeds this.
        max_steps: Safety cap when using confidence_threshold.
        collect_diagnostics: Whether to collect per-step diagnostic info.

    Returns:
        dict with:
            - answer: Generated text answer.
            - n_steps_taken: Number of recurrence steps executed.
            - halted: Whether confidence threshold was hit.
            - diagnostics: Per-step info (if collect_diagnostics).
    """
    effective_max = n_steps if confidence_threshold is None else max_steps
    diagnostics = [] if collect_diagnostics else None

    with torch.no_grad():
        # Step 1: Full forward on prompt with KV cache
        inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        outputs = model.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
        )
        h = outputs.last_hidden_state[:, -1:, :]  # [1, 1, 4096]
        kv_cache = outputs.past_key_values  # DynamicCache with all 36 layers
        prompt_len = inputs.input_ids.shape[1]

        # Step 2: Recurrence loop with KV cache
        halted = False
        steps_taken = 0

        for step in range(effective_max):
            # Position ID for this thought token
            thought_pos = prompt_len + step
            pos_ids = torch.tensor([[thought_pos]], device=DEVICE)
            cache_position = pos_ids.squeeze(0)  # [1]

            # Compute RoPE for this position
            pos_emb = model.model.rotary_emb(h, pos_ids)

            # Run through layers mid_layer..35 with KV cache
            for layer_idx in range(mid_layer, len(model.model.layers)):
                layer = model.model.layers[layer_idx]
                layer_out = layer(
                    h,
                    past_key_values=kv_cache,
                    use_cache=True,
                    position_embeddings=pos_emb,
                    cache_position=cache_position,
                )
                h = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            # Apply final norm
            h = model.model.norm(h)

            steps_taken = step + 1

            # Diagnostics
            if collect_diagnostics or confidence_threshold is not None:
                lens = logit_lens(model, h)
                max_prob = lens["max_prob"].item()

                if collect_diagnostics:
                    diagnostics.append({
                        "step": step,
                        "max_prob": max_prob,
                        "entropy": lens["entropy"].item(),
                        "h_norm": h.norm().item(),
                        "cache_seq_len": kv_cache.get_seq_length(mid_layer),
                    })

                # Check confidence halt
                if confidence_threshold is not None and max_prob >= confidence_threshold:
                    halted = True
                    break

        # Step 3: Generate answer
        # Prepend the final thought state to query embeddings
        answer = _generate_with_kv_state(
            model, tokenizer, h, query_text, max_new_tokens
        )

    return {
        "answer": answer,
        "n_steps_taken": steps_taken,
        "halted": halted,
        "diagnostics": diagnostics,
    }


def kv_beam_search(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    beam_width: int = 3,
    branch_factor: int = 5,
    max_depth: int = 4,
    mid_layer: int = MID_LAYER_INDEX,
    max_new_tokens: int = MAX_NEW_TOKENS,
    injection_alpha: float = 0.1,
    collect_diagnostics: bool = False,
) -> dict:
    """
    Latent beam search WITH KV cache attention to prompt.

    Each beam maintains its own copy of the KV cache, so different branches
    have different thought histories. All branches share the same prompt KV
    (layers 0-11 are identical across beams).

    Args:
        model: Frozen causal LM.
        tokenizer: Tokenizer.
        context_text: The prompt/context.
        query_text: The query for answer generation.
        beam_width: Number of beams to keep after pruning.
        branch_factor: Top-k candidates from logit lens at each step.
        max_depth: Maximum branching depth.
        mid_layer: Layer injection point (default: 12).
        max_new_tokens: Max tokens for final answer.
        injection_alpha: Scale for token embedding injection.
        collect_diagnostics: Whether to collect diagnostic info.

    Returns:
        dict with answer, best beam info, compute stats.
    """
    import math
    diagnostics = [] if collect_diagnostics else None
    total_forward_calls = 0

    with torch.no_grad():
        # Step 1: Full forward on prompt with KV cache
        inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        outputs = model.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
        )
        h0 = outputs.last_hidden_state[:, -1:, :]  # [1, 1, 4096]
        base_cache = outputs.past_key_values
        prompt_len = inputs.input_ids.shape[1]

        # Score initial state
        lens0 = logit_lens(model, h0)
        initial_conf = lens0["max_prob"].item()

        # Initialize beams: branch from h0
        beams = _kv_branch_and_score(
            model, h0, base_cache, prompt_len, 0,
            branch_factor, injection_alpha, mid_layer,
            parent_score=0.0, parent_path=[], parent_conf_history=[initial_conf],
        )
        total_forward_calls += len(beams)
        beams.sort(key=lambda b: b["confidence"], reverse=True)
        beams = beams[:beam_width]

        if collect_diagnostics:
            diagnostics.append({
                "depth": 1,
                "num_candidates": len(beams),
                "best_confidence": beams[0]["confidence"] if beams else 0,
                "beam_confidences": [b["confidence"] for b in beams],
            })

        # Step 2: Iterative deepening
        depth_reached = 1
        for depth in range(2, max_depth + 1):
            new_beams = []
            for beam in beams:
                children = _kv_branch_and_score(
                    model, beam["hidden_state"], beam["kv_cache"],
                    prompt_len, beam["depth"],
                    branch_factor, injection_alpha, mid_layer,
                    parent_score=beam["score"],
                    parent_path=beam["token_path"],
                    parent_conf_history=beam["confidence_history"],
                )
                total_forward_calls += len(children)
                new_beams.extend(children)

            new_beams.sort(key=lambda b: b["confidence"], reverse=True)
            beams = new_beams[:beam_width]
            depth_reached = depth

            if collect_diagnostics:
                diagnostics.append({
                    "depth": depth,
                    "num_candidates": len(new_beams),
                    "best_confidence": beams[0]["confidence"] if beams else 0,
                    "beam_confidences": [b["confidence"] for b in beams],
                })

        # Step 3: Generate from best beam
        best = max(beams, key=lambda b: b["confidence"])
        answer = _generate_with_kv_state(
            model, tokenizer, best["hidden_state"], query_text, max_new_tokens
        )

    return {
        "answer": answer,
        "best_beam": {
            "score": best["score"],
            "depth": best["depth"],
            "confidence": best["confidence"],
            "token_path": best["token_path"],
            "confidence_history": best["confidence_history"],
        },
        "depth_reached": depth_reached,
        "total_forward_calls": total_forward_calls,
        "diagnostics": diagnostics,
    }


def _kv_branch_and_score(
    model, h, kv_cache, prompt_len, current_depth,
    branch_factor, injection_alpha, mid_layer,
    parent_score, parent_path, parent_conf_history,
):
    """Branch from a hidden state, each branch gets its own KV cache copy."""
    import math

    lens = logit_lens(model, h)
    top_ids = lens["top_k_ids"][0, 0, :branch_factor]
    top_probs = lens["top_k_probs"][0, 0, :branch_factor]

    beams = []
    for i in range(min(branch_factor, len(top_ids))):
        token_id = top_ids[i].item()
        token_prob = top_probs[i].item()

        # Deep copy KV cache for this branch
        branch_cache = _clone_cache(kv_cache)

        # Get token embedding, scale to hidden state norm
        token_tensor = torch.tensor([[token_id]], device=DEVICE)
        emb = get_embeddings(model, token_tensor)
        h_norm = h.norm()
        emb_norm = emb.norm().clamp(min=1e-8)
        scaled_emb = emb * (h_norm / emb_norm)

        # Inject: h + scaled embedding direction
        h_branch = h + injection_alpha * scaled_emb

        # Position for this thought token
        thought_pos = prompt_len + current_depth
        pos_ids = torch.tensor([[thought_pos]], device=DEVICE)
        cache_position = pos_ids.squeeze(0)
        pos_emb = model.model.rotary_emb(h_branch, pos_ids)

        # Run through layers mid_layer..35 with KV cache
        h_new = h_branch
        for layer_idx in range(mid_layer, len(model.model.layers)):
            layer = model.model.layers[layer_idx]
            layer_out = layer(
                h_new,
                past_key_values=branch_cache,
                use_cache=True,
                position_embeddings=pos_emb,
                cache_position=cache_position,
            )
            h_new = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

        h_new = model.model.norm(h_new)

        # Score
        lens_new = logit_lens(model, h_new)
        new_conf = lens_new["max_prob"].item()
        new_score = parent_score + math.log(max(token_prob, 1e-10))

        beams.append({
            "hidden_state": h_new,
            "kv_cache": branch_cache,
            "score": new_score,
            "depth": current_depth + 1,
            "confidence": new_conf,
            "token_path": parent_path + [token_id],
            "confidence_history": parent_conf_history + [new_conf],
        })

    return beams


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    """Deep copy a DynamicCache so branches don't share state."""
    return copy.deepcopy(cache)


def kv_recurrence_full(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    n_steps: int = 32,
    mid_layer: int = MID_LAYER_INDEX,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_steps: int = MAX_CONTINUOUS_STEPS,
    decay_window: int = 3,
    decay_threshold: float = 0.0,
    adaptive_halt: bool = False,
    norm_mode: str = "final_only",
    generation_mode: str = "prefix",
    collect_diagnostics: bool = False,
) -> dict:
    """
    KV recurrence with consolidation and KV-aware generation.

    Recurrence: partial layers (mid_layer..35) with KV cache, same as exp7a.
    Generation: instead of discarding the cache, runs a consolidation pass
    through ALL layers to create a uniform cache, then generates autoregressively
    with the cache — output tokens attend to prompt + thought summary.

    Args:
        model: Frozen causal LM.
        tokenizer: Tokenizer.
        context_text: The prompt/context to reason about.
        query_text: Unused (kept for API compat).
        n_steps: Max recurrence steps (exact if adaptive_halt=False).
        mid_layer: Layer index for partial forward (default: 12).
        max_new_tokens: Max tokens for answer generation.
        max_steps: Safety cap when using adaptive halting.
        decay_window: Steps to average confidence change over.
        decay_threshold: Halt when avg confidence change < this.
        adaptive_halt: Whether to use adaptive halting.
        norm_mode: "every_step" (exp7a compat) or "final_only" (no drift).
        collect_diagnostics: Whether to collect per-step diagnostic info.

    Returns:
        dict with answer, n_steps_taken, halted, diagnostics.
    """
    effective_max = max_steps if adaptive_halt else n_steps
    diagnostics = [] if collect_diagnostics else None
    confidence_history = []

    with torch.no_grad():
        # Step 1: Full forward on prompt with KV cache
        inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        outputs = model.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
        )
        h = outputs.last_hidden_state[:, -1:, :]  # [1, 1, 4096] post-norm
        kv_cache = outputs.past_key_values
        prompt_len = inputs.input_ids.shape[1]

        # Step 2: Recurrence loop — partial layers (mid_layer..35)
        halted = False
        steps_taken = 0

        for step in range(effective_max):
            thought_pos = prompt_len + step
            pos_ids = torch.tensor([[thought_pos]], device=DEVICE)
            cache_position = pos_ids.squeeze(0)

            pos_emb = model.model.rotary_emb(h, pos_ids)

            # Run through layers mid_layer..35 with KV cache
            for layer_idx in range(mid_layer, len(model.model.layers)):
                layer = model.model.layers[layer_idx]
                layer_out = layer(
                    h,
                    past_key_values=kv_cache,
                    use_cache=True,
                    position_embeddings=pos_emb,
                    cache_position=cache_position,
                )
                h = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            # Norm handling
            if norm_mode == "every_step":
                h = model.model.norm(h)
            # else final_only: no norm during loop

            steps_taken = step + 1

            # Diagnostics / adaptive halting
            if collect_diagnostics or adaptive_halt:
                h_for_lens = h if norm_mode == "every_step" else model.model.norm(h)
                lens = logit_lens(model, h_for_lens)
                max_prob = lens["max_prob"].item()
                confidence_history.append(max_prob)

                if collect_diagnostics:
                    diagnostics.append({
                        "step": step,
                        "max_prob": max_prob,
                        "entropy": lens["entropy"].item(),
                        "h_norm": h.norm().item(),
                        "cache_seq_len": kv_cache.get_seq_length(mid_layer),
                    })

                if adaptive_halt and len(confidence_history) >= decay_window + 1:
                    recent_changes = [
                        confidence_history[i] - confidence_history[i - 1]
                        for i in range(-decay_window, 0)
                    ]
                    avg_change = sum(recent_changes) / len(recent_changes)
                    if avg_change < decay_threshold:
                        halted = True
                        break

        # Step 3: Generate answer
        # Apply norm for generation if not already applied
        h_gen = h if norm_mode == "every_step" else model.model.norm(h)

        if generation_mode == "split":
            answer = _generate_with_split_layers(
                model, tokenizer, h, kv_cache,
                prompt_len, steps_taken, mid_layer, max_new_tokens,
            )
        elif generation_mode == "consolidation":
            answer = _generate_with_split_layers(
                model, tokenizer, h, kv_cache,
                prompt_len, steps_taken, mid_layer, max_new_tokens,
            )
        else:
            # prefix mode: prepend h to query embeddings (same as exp7a)
            answer = _generate_with_kv_state(
                model, tokenizer, h_gen, query_text, max_new_tokens,
            )

    return {
        "answer": answer,
        "n_steps_taken": steps_taken,
        "halted": halted,
        "diagnostics": diagnostics,
    }


def _generate_with_split_layers(model, tokenizer, h, kv_cache, prompt_len,
                                 n_steps, mid_layer, max_new_tokens,
                                 baseline_logits=None, prompt_weight=0.0):
    """
    Split-layer generation: layers 0-11 clean, layers 12-35 with thought attention.

    After partial-layer recurrence (layers 12-35), the KV cache is:
      - Layers 0-11: prompt_len entries (prompt only)
      - Layers 12-35: prompt_len + n_steps entries (prompt + thought tokens)

    Instead of trying to fix the mismatch, we EMBRACE it:
      - Layers 0-11: process generated tokens normally (attend to prompt only)
      - Layers 12-35: process with full recurrence cache (attend to prompt + thoughts)

    Each generated token gets a clean forward through lower layers, then
    thought-enriched attention at upper layers. Position IDs account for
    thought tokens so RoPE is consistent with the recurrence phase.

    If baseline_logits and prompt_weight > 0 are provided, the first token is
    chosen from a weighted blend: prompt_weight * baseline + (1-prompt_weight) * thought.
    This anchors the output format to what the prompt expects while injecting reasoning.
    """
    with torch.no_grad():
        # Get first token from final thought state
        h_normed = model.model.norm(h) if h is not h else model.model.norm(h)
        thought_logits = model.lm_head(h_normed)

        # Blend with baseline logits for first token if provided
        if baseline_logits is not None and prompt_weight > 0:
            blended = prompt_weight * baseline_logits[:, -1:, :] + (1 - prompt_weight) * thought_logits[:, -1:, :]
            first_token_id = blended.argmax(dim=-1)  # [1, 1]
        else:
            first_token_id = thought_logits[:, -1:, :].argmax(dim=-1)  # [1, 1]

        # Position for generated tokens starts after prompt + thoughts
        gen_start_pos = prompt_len + n_steps

        generated_ids = [first_token_id.item()]

        # Process first token through split layers
        h_gen = model.model.embed_tokens(first_token_id)
        pos = torch.tensor([[gen_start_pos]], device=DEVICE)
        cache_pos = torch.tensor([gen_start_pos], device=DEVICE)
        pos_emb = model.model.rotary_emb(h_gen, pos)

        # Layers 0 to mid_layer-1: clean forward (attend to prompt only)
        for layer_idx in range(mid_layer):
            layer_out = model.model.layers[layer_idx](
                h_gen,
                past_key_values=kv_cache,
                use_cache=True,
                position_embeddings=pos_emb,
                cache_position=cache_pos,
            )
            h_gen = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

        # Layers mid_layer to 35: thought-enriched (attend to prompt + thoughts)
        for layer_idx in range(mid_layer, len(model.model.layers)):
            layer_out = model.model.layers[layer_idx](
                h_gen,
                past_key_values=kv_cache,
                use_cache=True,
                position_embeddings=pos_emb,
                cache_position=cache_pos,
            )
            h_gen = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

        h_gen = model.model.norm(h_gen)

        # Continue generating
        for gen_step in range(max_new_tokens - 1):
            gen_logits = model.lm_head(h_gen)
            next_token = gen_logits[:, -1:, :].argmax(dim=-1)
            token_id = next_token.item()

            if token_id == tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)

            next_embeds = model.model.embed_tokens(next_token)
            next_pos_val = gen_start_pos + gen_step + 1
            next_pos = torch.tensor([[next_pos_val]], device=DEVICE)
            next_cache_pos = torch.tensor([next_pos_val], device=DEVICE)
            next_pos_emb = model.model.rotary_emb(next_embeds, next_pos)

            h_gen = next_embeds

            # Layers 0 to mid_layer-1: clean
            for layer_idx in range(mid_layer):
                layer_out = model.model.layers[layer_idx](
                    h_gen,
                    past_key_values=kv_cache,
                    use_cache=True,
                    position_embeddings=next_pos_emb,
                    cache_position=next_cache_pos,
                )
                h_gen = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            # Layers mid_layer to 35: thought-enriched
            for layer_idx in range(mid_layer, len(model.model.layers)):
                layer_out = model.model.layers[layer_idx](
                    h_gen,
                    past_key_values=kv_cache,
                    use_cache=True,
                    position_embeddings=next_pos_emb,
                    cache_position=next_cache_pos,
                )
                h_gen = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            h_gen = model.model.norm(h_gen)

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer


def kv_recurrence_gated(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    n_steps: int = 4,
    mid_layer: int = MID_LAYER_INDEX,
    max_new_tokens: int = MAX_NEW_TOKENS,
    confidence_threshold: float = 0.9,
    norm_mode: str = "every_step",
    generation_mode: str = "split",
    prompt_weight: float = 0.0,
) -> dict:
    """
    Confidence-gated KV recurrence: only apply recurrence when the model is uncertain.

    1. Run baseline forward pass, get first-token logit distribution.
    2. If top-1 softmax confidence > threshold → use baseline generation (no recurrence).
    3. If confidence <= threshold → run full recurrence + split-layer generation.

    If prompt_weight > 0, the first generated token blends baseline logits (format-
    preserving) with thought logits (reasoning-enriched):
        first_logits = prompt_weight * baseline + (1 - prompt_weight) * thought
    This anchors the output format to the prompt's expectations.

    This avoids corrupting easy tasks (ARC-like) while boosting hard tasks (GSM8K-like).
    """
    with torch.no_grad():
        inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        input_ids = inputs.input_ids

        # Baseline forward pass to get confidence
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
        )
        h_baseline = outputs.last_hidden_state[:, -1:, :]
        kv_cache = outputs.past_key_values
        prompt_len = input_ids.shape[1]

        # Check first-token confidence and save baseline logits for blending
        h_normed = model.model.norm(h_baseline)
        baseline_logits = model.lm_head(h_normed)  # save for prompt_weight blending
        probs = torch.softmax(baseline_logits[:, -1, :], dim=-1)
        top_prob = probs.max().item()

        if top_prob >= confidence_threshold:
            # High confidence → baseline generation, skip recurrence
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            answer = tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            return {
                "answer": answer,
                "n_steps_taken": 0,
                "halted": False,
                "routed": "baseline",
                "baseline_confidence": top_prob,
                "diagnostics": None,
            }

        # Low confidence → run recurrence on the existing KV cache
        h = h_baseline
        steps_taken = 0

        for step in range(n_steps):
            thought_pos = prompt_len + step
            pos_ids = torch.tensor([[thought_pos]], device=DEVICE)
            cache_position = pos_ids.squeeze(0)
            pos_emb = model.model.rotary_emb(h, pos_ids)

            for layer_idx in range(mid_layer, len(model.model.layers)):
                layer = model.model.layers[layer_idx]
                layer_out = layer(
                    h,
                    past_key_values=kv_cache,
                    use_cache=True,
                    position_embeddings=pos_emb,
                    cache_position=cache_position,
                )
                h = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            if norm_mode == "every_step":
                h = model.model.norm(h)

            steps_taken = step + 1

        # Generate with split-layer or prefix
        if generation_mode == "split":
            answer = _generate_with_split_layers(
                model, tokenizer, h, kv_cache,
                prompt_len, steps_taken, mid_layer, max_new_tokens,
                baseline_logits=baseline_logits if prompt_weight > 0 else None,
                prompt_weight=prompt_weight,
            )
        else:
            h_gen = h if norm_mode == "every_step" else model.model.norm(h)
            answer = _generate_with_kv_state(
                model, tokenizer, h_gen, query_text, max_new_tokens,
            )

    return {
        "answer": answer,
        "n_steps_taken": steps_taken,
        "halted": False,
        "routed": "recurrence",
        "baseline_confidence": top_prob,
        "diagnostics": None,
    }


def kv_recurrence_first_token_override(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    n_steps: int = 4,
    mid_layer: int = MID_LAYER_INDEX,
    max_new_tokens: int = MAX_NEW_TOKENS,
    norm_mode: str = "every_step",
) -> dict:
    """
    Approach 1: First-token override.

    Always run recurrence + split-layer generation, but force the first generated
    token to come from baseline logits (prompt_weight=1.0 on token 1 only).
    Subsequent tokens use split-layer gen normally.

    This preserves output format (A/B/C/D for ARC) while letting multi-token
    answers (GSM8K) benefit from thought-enriched generation.
    """
    with torch.no_grad():
        inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        input_ids = inputs.input_ids

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
        )
        h = outputs.last_hidden_state[:, -1:, :]
        kv_cache = outputs.past_key_values
        prompt_len = input_ids.shape[1]

        # Save baseline logits for first-token override
        h_normed = model.model.norm(h)
        baseline_logits = model.lm_head(h_normed)

        # Recurrence loop
        for step in range(n_steps):
            thought_pos = prompt_len + step
            pos_ids = torch.tensor([[thought_pos]], device=DEVICE)
            cache_position = pos_ids.squeeze(0)
            pos_emb = model.model.rotary_emb(h, pos_ids)

            for layer_idx in range(mid_layer, len(model.model.layers)):
                layer = model.model.layers[layer_idx]
                layer_out = layer(
                    h, past_key_values=kv_cache, use_cache=True,
                    position_embeddings=pos_emb, cache_position=cache_position,
                )
                h = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            if norm_mode == "every_step":
                h = model.model.norm(h)

        # Generate with split layers, first token forced to baseline
        answer = _generate_with_split_layers(
            model, tokenizer, h, kv_cache,
            prompt_len, n_steps, mid_layer, max_new_tokens,
            baseline_logits=baseline_logits,
            prompt_weight=1.0,  # 100% baseline for first token only
        )

    return {
        "answer": answer,
        "n_steps_taken": n_steps,
        "halted": False,
        "routed": "first_token_override",
        "baseline_confidence": None,
        "diagnostics": None,
    }


def kv_recurrence_kl_gated(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    n_steps: int = 4,
    mid_layer: int = MID_LAYER_INDEX,
    max_new_tokens: int = MAX_NEW_TOKENS,
    kl_threshold: float = 1.0,
    norm_mode: str = "every_step",
    prompt_weight: float = 0.7,
) -> dict:
    """
    Approach 2: KL-divergence gating.

    Run 1 recurrence step, compare logits to baseline via KL-divergence.
    If KL > threshold → recurrence is changing the model's mind → continue to n_steps.
    If KL <= threshold → recurrence isn't adding signal → use baseline.
    """
    with torch.no_grad():
        inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        input_ids = inputs.input_ids

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
        )
        h = outputs.last_hidden_state[:, -1:, :]
        kv_cache = outputs.past_key_values
        prompt_len = input_ids.shape[1]

        # Baseline logits
        h_normed = model.model.norm(h)
        baseline_logits = model.lm_head(h_normed)
        baseline_log_probs = torch.log_softmax(baseline_logits[:, -1, :], dim=-1)

        # Run 1 recurrence step
        pos_ids = torch.tensor([[prompt_len]], device=DEVICE)
        cache_position = pos_ids.squeeze(0)
        pos_emb = model.model.rotary_emb(h, pos_ids)

        h_probe = h
        for layer_idx in range(mid_layer, len(model.model.layers)):
            layer = model.model.layers[layer_idx]
            layer_out = layer(
                h_probe, past_key_values=kv_cache, use_cache=True,
                position_embeddings=pos_emb, cache_position=cache_position,
            )
            h_probe = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

        if norm_mode == "every_step":
            h_probe = model.model.norm(h_probe)

        # Compare logits after 1 step
        h_probe_normed = h_probe if norm_mode == "every_step" else model.model.norm(h_probe)
        step1_logits = model.lm_head(h_probe_normed)
        step1_log_probs = torch.log_softmax(step1_logits[:, -1, :], dim=-1)

        # KL(step1 || baseline)
        step1_probs = torch.exp(step1_log_probs)
        kl_div = (step1_probs * (step1_log_probs - baseline_log_probs)).sum().item()

        if kl_div <= kl_threshold:
            # Low divergence → recurrence isn't doing much → baseline
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            answer = tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            return {
                "answer": answer,
                "n_steps_taken": 0,
                "halted": False,
                "routed": "baseline",
                "baseline_confidence": kl_div,
                "diagnostics": None,
            }

        # High divergence → continue recurrence from step 1 state
        h = h_probe
        for step in range(1, n_steps):
            thought_pos = prompt_len + step
            pos_ids = torch.tensor([[thought_pos]], device=DEVICE)
            cache_position = pos_ids.squeeze(0)
            pos_emb = model.model.rotary_emb(h, pos_ids)

            for layer_idx in range(mid_layer, len(model.model.layers)):
                layer = model.model.layers[layer_idx]
                layer_out = layer(
                    h, past_key_values=kv_cache, use_cache=True,
                    position_embeddings=pos_emb, cache_position=cache_position,
                )
                h = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            if norm_mode == "every_step":
                h = model.model.norm(h)

        answer = _generate_with_split_layers(
            model, tokenizer, h, kv_cache,
            prompt_len, n_steps, mid_layer, max_new_tokens,
            baseline_logits=baseline_logits if prompt_weight > 0 else None,
            prompt_weight=prompt_weight,
        )

    return {
        "answer": answer,
        "n_steps_taken": n_steps,
        "halted": False,
        "routed": "recurrence",
        "baseline_confidence": kl_div,
        "diagnostics": None,
    }


def kv_recurrence_answer_mass_gated(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    n_steps: int = 4,
    mid_layer: int = MID_LAYER_INDEX,
    max_new_tokens: int = MAX_NEW_TOKENS,
    mass_threshold: float = 0.5,
    norm_mode: str = "every_step",
    prompt_weight: float = 0.7,
) -> dict:
    """
    Approach 3: Answer-token probability mass gating.

    Check if "answer-format" tokens (A-E, digits 0-9) dominate the baseline
    distribution. If combined mass > threshold → simple answer task → baseline.
    If mass is low → complex continuation needed → recurrence.
    """
    with torch.no_grad():
        inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        input_ids = inputs.input_ids

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
        )
        h = outputs.last_hidden_state[:, -1:, :]
        kv_cache = outputs.past_key_values
        prompt_len = input_ids.shape[1]

        h_normed = model.model.norm(h)
        baseline_logits = model.lm_head(h_normed)
        probs = torch.softmax(baseline_logits[:, -1, :], dim=-1)

        # Gather probability mass on answer-format tokens
        answer_tokens = []
        for tok_str in ["A", "B", "C", "D", "E",
                        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                        " A", " B", " C", " D", " E",
                        " 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]:
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            answer_tokens.extend(ids)
        answer_tokens = list(set(answer_tokens))

        answer_mass = probs[0, answer_tokens].sum().item()

        if answer_mass >= mass_threshold:
            # High mass on answer tokens → simple task → baseline
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            answer = tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            return {
                "answer": answer,
                "n_steps_taken": 0,
                "halted": False,
                "routed": "baseline",
                "baseline_confidence": answer_mass,
                "diagnostics": None,
            }

        # Low mass → complex task → recurrence
        for step in range(n_steps):
            thought_pos = prompt_len + step
            pos_ids = torch.tensor([[thought_pos]], device=DEVICE)
            cache_position = pos_ids.squeeze(0)
            pos_emb = model.model.rotary_emb(h, pos_ids)

            for layer_idx in range(mid_layer, len(model.model.layers)):
                layer = model.model.layers[layer_idx]
                layer_out = layer(
                    h, past_key_values=kv_cache, use_cache=True,
                    position_embeddings=pos_emb, cache_position=cache_position,
                )
                h = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            if norm_mode == "every_step":
                h = model.model.norm(h)

        answer = _generate_with_split_layers(
            model, tokenizer, h, kv_cache,
            prompt_len, n_steps, mid_layer, max_new_tokens,
            baseline_logits=baseline_logits if prompt_weight > 0 else None,
            prompt_weight=prompt_weight,
        )

    return {
        "answer": answer,
        "n_steps_taken": n_steps,
        "halted": False,
        "routed": "recurrence",
        "baseline_confidence": answer_mass,
        "diagnostics": None,
    }


def _generate_with_kv_state(model, tokenizer, hidden_state, query_text, max_new_tokens):
    """Generate answer by prepending recurrence state to query embeddings (legacy)."""
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
