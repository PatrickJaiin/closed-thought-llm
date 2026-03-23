"""
Core recurrence loops for closed-thought LLM experiments.

Two modes:
1. Full-loop: final hidden state → input embedding layer → full forward pass (repeat N times)
2. Mid-layer-loop: final hidden state → layer L (~1/3 depth) → forward to end (repeat N times)
"""

import torch
from typing import Optional
from model_utils import (
    get_embeddings,
    full_forward,
    partial_forward,
    hidden_to_logits,
)
from config import MID_LAYER_INDEX, DEVICE


def full_loop_recurrence(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    n_steps: int,
    max_new_tokens: int = 64,
    collect_hidden: bool = False,
) -> dict:
    """
    Full-loop recurrence: feed final hidden state back as input embeddings.

    Process:
    1. Forward pass on context → get final-layer hidden state of last token
    2. Use that hidden state as the sole input embedding, run full forward again
    3. Repeat step 2 for n_steps iterations
    4. Prepend the recurrence hidden state to query embeddings and generate
    """
    hidden_trajectory = []

    with torch.no_grad():
        # Step 1: Initial forward pass on context
        context_inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        context_embeds = get_embeddings(model, context_inputs.input_ids)
        hidden = full_forward(
            model, context_embeds, attention_mask=context_inputs.attention_mask
        )
        # Extract last token's hidden state: (1, 1, hidden_dim)
        h = hidden[:, -1:, :]

        if collect_hidden:
            hidden_trajectory.append(h.clone().cpu())

        # Step 2-3: Recurrence loop
        for step in range(n_steps):
            ones_mask = torch.ones(1, 1, device=DEVICE, dtype=torch.long)
            position_ids = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
            hidden = full_forward(
                model, h, attention_mask=ones_mask, position_ids=position_ids
            )
            h = hidden[:, -1:, :]

            if collect_hidden:
                hidden_trajectory.append(h.clone().cpu())

        # Step 4: Generate with recurrence state prepended to query
        answer = _generate_with_prefix_state(
            model, tokenizer, h, query_text, max_new_tokens
        )

    return {
        "answer": answer,
        "hidden_states": hidden_trajectory if collect_hidden else None,
        "n_steps": n_steps,
    }


def mid_layer_loop_recurrence(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    n_steps: int,
    mid_layer: Optional[int] = None,
    max_new_tokens: int = 64,
    collect_hidden: bool = False,
) -> dict:
    """
    Mid-layer loop: feed final hidden state back to layer L (~1/3 depth).

    Process:
    1. Full forward pass on context → get final hidden state of last token
    2. Feed that hidden state into layer L, run through layers[L:] → new hidden state
    3. Repeat step 2 for n_steps iterations
    4. Prepend the recurrence hidden state to query embeddings and generate
    """
    if mid_layer is None:
        mid_layer = MID_LAYER_INDEX

    hidden_trajectory = []

    with torch.no_grad():
        # Step 1: Full forward pass on context
        context_inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)
        context_embeds = get_embeddings(model, context_inputs.input_ids)
        hidden = full_forward(
            model, context_embeds, attention_mask=context_inputs.attention_mask
        )
        h = hidden[:, -1:, :]

        if collect_hidden:
            hidden_trajectory.append(h.clone().cpu())

        # Step 2-3: Mid-layer recurrence loop
        for step in range(n_steps):
            position_ids = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
            h = partial_forward(
                model, h, start_layer=mid_layer,
                position_ids=position_ids
            )

            if collect_hidden:
                hidden_trajectory.append(h.clone().cpu())

        # Step 4: Generate with recurrence state
        answer = _generate_with_prefix_state(
            model, tokenizer, h, query_text, max_new_tokens
        )

    return {
        "answer": answer,
        "hidden_states": hidden_trajectory if collect_hidden else None,
        "n_steps": n_steps,
    }


def text_baseline(
    model,
    tokenizer,
    context_text: str,
    query_text: str,
    n_thinking_tokens: int,
    max_new_tokens: int = 64,
) -> dict:
    """
    Text self-prompting baseline: let the model generate N tokens of
    "thinking" text, then answer the query.
    """
    thinking_prompt = context_text + "\nLet me think step by step.\n"

    inputs = tokenizer(thinking_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        thinking_output = model.generate(
            inputs.input_ids,
            max_new_tokens=n_thinking_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    thinking_text = tokenizer.decode(
        thinking_output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    # Now append the query and generate the answer
    full_prompt = thinking_prompt + thinking_text + "\n" + query_text
    full_inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        answer_output = model.generate(
            full_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    answer = tokenizer.decode(
        answer_output[0][full_inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return {
        "answer": answer,
        "thinking": thinking_text,
        "n_thinking_tokens": n_thinking_tokens,
    }


def _generate_with_prefix_state(model, tokenizer, hidden_state, query_text, max_new_tokens):
    """
    Generate answer by prepending the recurrence hidden state (as a pseudo-embedding)
    to the query's token embeddings, then using model.generate() with inputs_embeds.

    The hidden state acts as a "latent context token" that encodes the recurrence
    processing. The model sees [recurrence_state, query_tokens] as its input.
    """
    query_inputs = tokenizer(query_text, return_tensors="pt").to(DEVICE)
    query_embeds = get_embeddings(model, query_inputs.input_ids)

    # Cast hidden_state to match query_embeds dtype
    h = hidden_state.to(dtype=query_embeds.dtype)

    # Concatenate: [recurrence_hidden_state(1 token), query_embeddings]
    combined_embeds = torch.cat([h, query_embeds], dim=1)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # model.generate with inputs_embeds returns only the generated token IDs
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer
