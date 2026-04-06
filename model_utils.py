"""
Model loading, hidden state extraction, and partial forward pass utilities.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import config as _cfg
from config import MODEL_NAME, DTYPE, DEVICE


def load_model(use_flash_attn=True, compile_model=False):
    """Load Qwen3-8B with tokenizer. Returns (model, tokenizer).

    Args:
        use_flash_attn: Use Flash Attention 2 if available (CUDA only, ~1.5-2x speedup).
        compile_model: Apply torch.compile for faster repeated forward passes.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Flash Attention 2 only works on CUDA with Ampere+ GPUs
    attn_impl = "flash_attention_2" if (use_flash_attn and DEVICE == "cuda") else None
    extra_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}

    if _cfg.LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            **extra_kwargs,
        )
    elif DEVICE == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=DTYPE,
            trust_remote_code=True,
        ).to(DEVICE)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True,
            **extra_kwargs,
        )

    model.eval()

    if compile_model and DEVICE == "cuda":
        model = torch.compile(model, mode="reduce-overhead")

    return model, tokenizer


def get_embeddings(model, input_ids):
    """Get token embeddings from input_ids. Shape: (batch, seq_len, hidden_dim)."""
    return model.model.embed_tokens(input_ids)


def full_forward(model, inputs_embeds, attention_mask=None, position_ids=None):
    """
    Run a full forward pass from embeddings through all layers.
    Returns the final hidden states (before LM head). Shape: (batch, seq_len, hidden_dim).
    """
    with torch.no_grad():
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    return outputs.last_hidden_state


def partial_forward(model, hidden_states, start_layer, attention_mask=None, position_ids=None):
    """
    Run forward pass from a specific layer index to the final layer.
    This calls model.model.layers[start_layer:] sequentially.

    Args:
        model: The loaded causal LM.
        hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
        start_layer: Index of the first layer to run (inclusive).
        attention_mask: Optional attention mask.
        position_ids: Optional position IDs.

    Returns:
        Final hidden states after running through layers[start_layer:] and final norm.
    """
    with torch.no_grad():
        # Compute RoPE position embeddings (required by Qwen3 layers)
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        for layer in model.model.layers[start_layer:]:
            # Note: attention_mask omitted — for single-token recurrence (seq_len=1)
            # no causal masking is needed, and for short sequences the model handles
            # it via the default (no mask = attend to everything).
            layer_outputs = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )
            # Qwen3DecoderLayer.forward returns a tensor directly (not a tuple)
            hidden_states = layer_outputs if isinstance(layer_outputs, torch.Tensor) else layer_outputs[0]

        hidden_states = model.model.norm(hidden_states)

    return hidden_states


def hidden_to_logits(model, hidden_states):
    """Apply LM head to hidden states → logits. Shape: (batch, seq_len, vocab_size)."""
    with torch.no_grad():
        return model.lm_head(hidden_states)


def logit_lens(model, hidden_states, tokenizer=None, top_k=5):
    """
    Project hidden states through the LM head and return top-k token info.

    This is the "logit lens" technique: peek at what the model is "thinking"
    at intermediate recurrence steps by projecting hidden states to vocabulary space.

    Args:
        model: The loaded causal LM.
        hidden_states: Shape (batch, seq_len, hidden_dim) or (1, 1, hidden_dim).
        tokenizer: Optional tokenizer for decoding token IDs to strings.
        top_k: Number of top tokens to return.

    Returns:
        dict with:
            - top_k_probs: Tensor of shape (batch, seq_len, top_k) — probabilities
            - top_k_ids: Tensor of shape (batch, seq_len, top_k) — token IDs
            - top_k_tokens: list of token strings (only if tokenizer provided)
            - entropy: Tensor of shape (batch, seq_len) — distribution entropy in nats
            - max_prob: Tensor of shape (batch, seq_len) — max probability (confidence)
    """
    with torch.no_grad():
        logits = model.lm_head(hidden_states)  # (batch, seq_len, vocab_size)
        probs = torch.softmax(logits.float(), dim=-1)

        top_k_probs, top_k_ids = torch.topk(probs, k=top_k, dim=-1)

        # Entropy: -sum(p * log(p)), clamped to avoid log(0)
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=-1)

        max_prob = probs.max(dim=-1).values

        result = {
            "top_k_probs": top_k_probs,
            "top_k_ids": top_k_ids,
            "entropy": entropy,
            "max_prob": max_prob,
        }

        if tokenizer is not None:
            tokens = []
            for batch_idx in range(top_k_ids.shape[0]):
                for seq_idx in range(top_k_ids.shape[1]):
                    step_tokens = [
                        tokenizer.decode([tid.item()])
                        for tid in top_k_ids[batch_idx, seq_idx]
                    ]
                    tokens.append(step_tokens)
            result["top_k_tokens"] = tokens

        return result


def generate_from_hidden(model, tokenizer, hidden_states, attention_mask=None, max_new_tokens=256):
    """
    Given hidden states (already processed through all layers + norm),
    generate text by iteratively sampling from the LM head.
    Uses greedy decoding (argmax).

    Args:
        model: The loaded causal LM.
        tokenizer: The tokenizer.
        hidden_states: Shape (1, seq_len, hidden_dim) — final hidden states.
        attention_mask: Shape (1, seq_len).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text string.
    """
    generated_ids = []

    with torch.no_grad():
        # Get first token from last position's logits
        logits = model.lm_head(hidden_states[:, -1:, :])
        next_token = logits.argmax(dim=-1)  # (1, 1)
        generated_ids.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Continue generating autoregressively using the model's normal generate
        # by feeding the token through embeddings
        for _ in range(max_new_tokens - 1):
            next_embed = model.model.embed_tokens(next_token)
            # Run full forward on just this token (no KV cache for simplicity)
            # For efficiency in production, we'd use KV cache, but for experiments this is fine
            hidden = full_forward(model, next_embed)
            logits = model.lm_head(hidden[:, -1:, :])
            next_token = logits.argmax(dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token.item())

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def encode_and_forward(model, tokenizer, text):
    """
    Convenience: tokenize text, get embeddings, run full forward.
    Returns (hidden_states, input_ids, attention_mask).
    """
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    embeds = get_embeddings(model, inputs.input_ids)
    hidden = full_forward(model, embeds, attention_mask=inputs.attention_mask)
    return hidden, inputs.input_ids, inputs.attention_mask


def _make_causal_mask(attention_mask, seq_len, dtype, device):
    """
    Build a 4D causal attention mask from a 2D padding mask.
    Shape: (batch, 1, seq_len, seq_len).
    """
    batch_size = attention_mask.shape[0]
    # Causal: lower triangular
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    # Expand and combine with padding mask
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    # Apply padding mask: positions where attention_mask == 0 should be masked
    pad_mask = attention_mask[:, None, None, :].to(dtype)  # (batch, 1, 1, seq_len)
    combined = causal * pad_mask
    # Convert to additive mask: 0 → -inf, 1 → 0
    combined = (1.0 - combined) * torch.finfo(dtype).min
    return combined
