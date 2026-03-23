"""
Memory system for the continuous recurrence loop (Phase 4).

Three tiers of memory, from simple to learned:
1. KVMemory — ring buffer with cosine similarity retrieval (~1MB VRAM)
2. SurpriseMemory — store only when hidden state changes significantly (Titans-inspired)
3. NeuralMemory — learned read/write with attention over memory slots (~13MB VRAM)

All share a common interface:
    memory.read(h) → retrieved_mem or None
    memory.write(h) → None
    memory.reset() → None

Memory integration in the loop:
    if gate.should_retrieve(h):
        mem = memory.read(h)
        h = h + alpha * mem          # residual addition
    h = partial_forward(model, h, ...)
    if gate.should_store(h):
        memory.write(h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from config import (
    HIDDEN_DIM, MEMORY_SLOTS, MEMORY_DIM,
    MEMORY_TEMPORAL_DECAY, MEMORY_SURPRISE_THRESHOLD,
    DEVICE,
)


# ── Tier 1: KV Ring Buffer ───────────────────────────────────────────


class KVMemory:
    """
    Simple ring buffer memory with cosine similarity retrieval.

    No learnable parameters. Stores hidden states in a fixed-size buffer,
    overwrites oldest entries when full. Retrieves via cosine similarity.

    VRAM: slots * hidden_dim * 4 bytes ≈ 128 * 4096 * 4 ≈ 2MB
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_slots: int = MEMORY_SLOTS,
        temporal_decay: float = MEMORY_TEMPORAL_DECAY,
        device: str = DEVICE,
    ):
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.temporal_decay = temporal_decay
        self.device = device

        # Ring buffer
        self.buffer = torch.zeros(num_slots, hidden_dim, device=device)
        self.ages = torch.zeros(num_slots, device=device)  # age in steps
        self.access_counts = torch.zeros(num_slots, device=device, dtype=torch.long)
        self.write_ptr = 0
        self.size = 0  # current number of valid entries

    def write(self, h: torch.Tensor):
        """
        Store hidden state in the ring buffer.

        Args:
            h: Shape (1, 1, hidden_dim) or (hidden_dim,)
        """
        h_flat = h.detach().view(-1).float()

        self.buffer[self.write_ptr] = h_flat
        self.ages[self.write_ptr] = 0.0
        self.access_counts[self.write_ptr] = 0

        self.write_ptr = (self.write_ptr + 1) % self.num_slots
        self.size = min(self.size + 1, self.num_slots)

        # Age all other entries
        self.ages[:self.size] += 1
        self.ages[self.write_ptr - 1 if self.write_ptr > 0 else self.num_slots - 1] = 0

    def read(self, h: torch.Tensor, top_k: int = 1) -> Optional[torch.Tensor]:
        """
        Retrieve most similar memory via cosine similarity, with temporal decay.

        Args:
            h: Query hidden state, shape (1, 1, hidden_dim) or (hidden_dim,).
            top_k: Number of memories to retrieve and average.

        Returns:
            Retrieved memory tensor of shape (1, 1, hidden_dim), or None if empty.
        """
        if self.size == 0:
            return None

        h_flat = h.detach().view(-1).float()  # (hidden_dim,)
        valid = self.buffer[:self.size]  # (size, hidden_dim)

        # Cosine similarity
        sim = F.cosine_similarity(h_flat.unsqueeze(0), valid, dim=1)  # (size,)

        # Apply temporal decay: older memories are less relevant
        decay = self.temporal_decay ** self.ages[:self.size]
        weighted_sim = sim * decay

        # Get top-k
        k = min(top_k, self.size)
        topk_vals, topk_idx = weighted_sim.topk(k)

        # Update access counts (reading refreshes the memory)
        self.access_counts[topk_idx] += 1
        self.ages[topk_idx] = 0  # reset age on access (rehearsal)

        # Weighted average of top-k memories
        weights = F.softmax(topk_vals, dim=0)
        retrieved = (weights.unsqueeze(1) * valid[topk_idx]).sum(dim=0)

        return retrieved.view(1, 1, self.hidden_dim)

    def reset(self):
        """Clear all memories."""
        self.buffer.zero_()
        self.ages.zero_()
        self.access_counts.zero_()
        self.write_ptr = 0
        self.size = 0

    def stats(self) -> dict:
        """Return memory statistics."""
        return {
            "size": self.size,
            "capacity": self.num_slots,
            "avg_age": self.ages[:self.size].mean().item() if self.size > 0 else 0,
            "max_age": self.ages[:self.size].max().item() if self.size > 0 else 0,
            "avg_access_count": self.access_counts[:self.size].float().mean().item() if self.size > 0 else 0,
        }


# ── Tier 2: Surprise-Based Memory ────────────────────────────────────


class SurpriseMemory:
    """
    Surprise-based memory: only stores when hidden state changes significantly.

    Inspired by Titans (Behrouz et al., 2024) surprise metric.
    Stores h when 1 - cos_sim(h_t, h_{t-1}) > surprise_threshold.
    This filters out redundant/boring states and keeps only novel ones.

    Uses KVMemory internally for storage, but gates writes by surprise.

    VRAM: same as KVMemory (~2MB)
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_slots: int = MEMORY_SLOTS,
        surprise_threshold: float = MEMORY_SURPRISE_THRESHOLD,
        temporal_decay: float = MEMORY_TEMPORAL_DECAY,
        device: str = DEVICE,
    ):
        self.surprise_threshold = surprise_threshold
        self.kv = KVMemory(hidden_dim, num_slots, temporal_decay, device)
        self.last_h = None

    def write(self, h: torch.Tensor):
        """
        Store h only if it's surprising (different enough from last write).

        Args:
            h: Shape (1, 1, hidden_dim).
        """
        h_flat = h.detach().view(-1)

        if self.last_h is None:
            # First write is always surprising
            self.kv.write(h)
            self.last_h = h_flat.clone()
            return

        surprise = 1.0 - F.cosine_similarity(
            h_flat.unsqueeze(0), self.last_h.unsqueeze(0)
        ).item()

        if surprise > self.surprise_threshold:
            self.kv.write(h)
            self.last_h = h_flat.clone()

    def read(self, h: torch.Tensor, top_k: int = 1) -> Optional[torch.Tensor]:
        """Retrieve via cosine similarity (delegates to KVMemory)."""
        return self.kv.read(h, top_k)

    def reset(self):
        """Clear all memories."""
        self.kv.reset()
        self.last_h = None

    def stats(self) -> dict:
        """Return memory statistics."""
        base = self.kv.stats()
        base["type"] = "surprise"
        base["surprise_threshold"] = self.surprise_threshold
        return base


# ── Tier 3: Neural Memory ────────────────────────────────────────────


class NeuralMemory(nn.Module):
    """
    Learned memory with attention-based read and gated write.

    Unlike KVMemory/SurpriseMemory, this has learnable parameters for:
    - Write: projects h to memory dim, uses a gated update
    - Read: attention over memory slots with learned query projection

    Architecture:
        Write: h → Linear(4096, 256) → write_gate * current + (1-write_gate) * old
        Read:  h → Linear(4096, 256) → attention over slots → Linear(256, 4096)

    Parameters:
        Write: 4096*256 + 256 + 256*1 + 1 = 1,049,089
        Read:  4096*256 + 256 + 256*4096 + 4096 = 2,101,504
        Memory slots (learnable): 128 * 256 = 32,768
        Total: ~3.2M params (read) + ~1.05M (write) + 0.03M (slots) ≈ 4.3M
        But with shared backbone: ~6.4M total as stated in plan

    VRAM: ~13MB at fp32
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        memory_dim: int = MEMORY_DIM,
        num_slots: int = MEMORY_SLOTS,
        device: str = DEVICE,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_slots = num_slots

        # Memory contents: learnable initial state
        self.memory = nn.Parameter(torch.randn(num_slots, memory_dim) * 0.01)

        # Write path
        self.write_proj = nn.Linear(hidden_dim, memory_dim)
        self.write_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Sigmoid(),
        )

        # Read path
        self.read_query_proj = nn.Linear(hidden_dim, memory_dim)
        self.read_output_proj = nn.Linear(memory_dim, hidden_dim)

        # Slot usage tracking (not learnable)
        self.register_buffer("slot_ages", torch.zeros(num_slots))
        self.register_buffer("slot_access_counts", torch.zeros(num_slots, dtype=torch.long))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))

        self.to(device)

    def write(self, h: torch.Tensor):
        """
        Write to memory using gated update.

        The write gate decides how much of the new value to blend with
        the existing slot content. Oldest slot is overwritten.

        Args:
            h: Shape (1, 1, hidden_dim)
        """
        h_flat = h.view(-1).to(self.write_proj.weight.dtype)  # (hidden_dim,)
        new_val = self.write_proj(h_flat)  # (memory_dim,)

        # Pick slot to overwrite (round-robin)
        ptr = self.write_ptr.item()
        old_val = self.memory.data[ptr]

        # Gated update: blend new and old
        combined = torch.cat([new_val, old_val])
        gate = self.write_gate(combined)
        updated = gate * new_val + (1 - gate) * old_val

        self.memory.data[ptr] = updated.detach()

        # Update metadata
        self.slot_ages += 1
        self.slot_ages[ptr] = 0
        self.slot_access_counts[ptr] = 0
        self.write_ptr = (self.write_ptr + 1) % self.num_slots

    def read(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Read from memory using attention.

        Projects h to query space, computes attention weights over memory slots,
        retrieves weighted sum, projects back to hidden dim.

        Args:
            h: Shape (1, 1, hidden_dim)

        Returns:
            Retrieved memory, shape (1, 1, hidden_dim)
        """
        h_flat = h.view(-1).to(self.read_query_proj.weight.dtype)  # (hidden_dim,)
        query = self.read_query_proj(h_flat)  # (memory_dim,)

        # Attention over memory slots
        scores = torch.matmul(query.unsqueeze(0), self.memory.T)  # (1, num_slots)
        scores = scores / (self.memory_dim ** 0.5)  # scaled dot product
        weights = F.softmax(scores, dim=-1)  # (1, num_slots)

        # Weighted sum
        retrieved = torch.matmul(weights, self.memory)  # (1, memory_dim)

        # Project back to hidden dim
        output = self.read_output_proj(retrieved)  # (1, hidden_dim)

        # Update access counts for attended slots
        top_idx = weights.squeeze().topk(min(5, self.num_slots)).indices
        self.slot_access_counts[top_idx] += 1
        self.slot_ages[top_idx] = 0  # refresh on access

        return output.view(1, 1, self.hidden_dim)

    def reset(self):
        """Reset memory to initial state (re-randomize slots)."""
        nn.init.normal_(self.memory, std=0.01)
        self.slot_ages.zero_()
        self.slot_access_counts.zero_()
        self.write_ptr.zero_()

    def apply_decay(self, decay_rate: float = 0.999):
        """
        Apply temporal decay to memory slots.
        Older, unaccessed slots fade toward zero.
        """
        decay = decay_rate ** self.slot_ages
        self.memory.data *= decay.unsqueeze(1)

    def stats(self) -> dict:
        """Return memory statistics."""
        return {
            "type": "neural",
            "num_slots": self.num_slots,
            "memory_dim": self.memory_dim,
            "avg_slot_norm": self.memory.data.norm(dim=1).mean().item(),
            "avg_age": self.slot_ages.float().mean().item(),
            "max_age": self.slot_ages.max().item(),
            "avg_access_count": self.slot_access_counts.float().mean().item(),
            "write_ptr": self.write_ptr.item(),
        }


# ── Factory ───────────────────────────────────────────────────────────


def create_memory(
    tier: str = "kv",
    hidden_dim: int = HIDDEN_DIM,
    num_slots: int = MEMORY_SLOTS,
    device: str = DEVICE,
    **kwargs,
):
    """
    Factory function to create a memory instance by tier name.

    Args:
        tier: "kv", "surprise", or "neural"
        hidden_dim: Hidden dimension of the LLM.
        num_slots: Number of memory slots.
        device: Device to place memory on.
        **kwargs: Additional arguments (surprise_threshold, temporal_decay, etc.)

    Returns:
        Memory instance with read/write/reset interface.
    """
    if tier == "kv":
        return KVMemory(
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            temporal_decay=kwargs.get("temporal_decay", MEMORY_TEMPORAL_DECAY),
            device=device,
        )
    elif tier == "surprise":
        return SurpriseMemory(
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            surprise_threshold=kwargs.get("surprise_threshold", MEMORY_SURPRISE_THRESHOLD),
            temporal_decay=kwargs.get("temporal_decay", MEMORY_TEMPORAL_DECAY),
            device=device,
        )
    elif tier == "neural":
        return NeuralMemory(
            hidden_dim=hidden_dim,
            memory_dim=kwargs.get("memory_dim", MEMORY_DIM),
            num_slots=num_slots,
            device=device,
        )
    else:
        raise ValueError(f"Unknown memory tier: {tier}. Choose from: kv, surprise, neural")
