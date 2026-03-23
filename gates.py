"""
Learned gate modules for the continuous recurrence loop (Phase 3).

Three gate types, all small nn.Modules trained while keeping the LLM frozen:
- HaltGate: decides when to stop the recurrence loop (~1.05M params)
- InjectGate: decides when to inject a new query into the loop (~2.1M params)
- MemoryGate: decides when to store/retrieve from memory (~1.1M params)

Total: ~4.2M params, ~8MB VRAM — negligible compared to the 5GB frozen model.
"""

import torch
import torch.nn as nn
from config import HIDDEN_DIM, GATE_HIDDEN_DIM


class HaltGate(nn.Module):
    """
    Learned halting gate: h → p_halt ∈ [0, 1].

    Architecture: Linear(4096, 256) → GELU → Linear(256, 1) → Sigmoid
    Parameters: 4096*256 + 256 + 256*1 + 1 = 1,049,089 (~1.05M)

    Compatible with the continuous_recurrence loop — can be passed as halt_fn.
    When used as halt_fn, thresholds at 0.5 (via _call_halt_fn in continuous_recurrence.py).
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, gate_dim=GATE_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, gate_dim),
            nn.GELU(),
            nn.Linear(gate_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h):
        """
        Args:
            h: Hidden state tensor of shape (batch, seq_len, hidden_dim)
               or (batch, 1, hidden_dim) for single-token recurrence.

        Returns:
            p_halt: Probability of halting, shape (batch, seq_len, 1) or scalar.
        """
        h = h.to(self.net[0].weight.dtype)
        return self.net(h).squeeze(-1)

    def get_halt_prob(self, h):
        """Return halt probability as a scalar float."""
        with torch.no_grad():
            p = self.forward(h)
            return p.item() if p.numel() == 1 else p.mean().item()


class InjectGate(nn.Module):
    """
    Learned injection gate: [h_loop, h_query] → p_inject ∈ [0, 1].

    Decides when the loop should accept a new query for processing.
    Takes both the current loop state and the query embedding as input.

    Architecture:
        h_loop  → Linear(4096, 256)  ─┐
                                       ├→ concat(512) → Linear(512, 256) → GELU → Linear(256, 1) → Sigmoid
        h_query → Linear(4096, 256)  ─┘

    Parameters: 2 * (4096*256 + 256) + 512*256 + 256 + 256*1 + 1 = 2,229,505 (~2.1M)
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, gate_dim=GATE_HIDDEN_DIM):
        super().__init__()
        self.proj_loop = nn.Linear(hidden_dim, gate_dim)
        self.proj_query = nn.Linear(hidden_dim, gate_dim)
        self.decision = nn.Sequential(
            nn.Linear(gate_dim * 2, gate_dim),
            nn.GELU(),
            nn.Linear(gate_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_loop, h_query):
        """
        Args:
            h_loop: Current loop hidden state (batch, 1, hidden_dim).
            h_query: Query hidden state (batch, 1, hidden_dim).

        Returns:
            p_inject: Probability of injecting query, shape (batch, 1).
        """
        dtype = self.proj_loop.weight.dtype
        z_loop = self.proj_loop(h_loop.to(dtype))
        z_query = self.proj_query(h_query.to(dtype))
        combined = torch.cat([z_loop, z_query], dim=-1)
        return self.decision(combined).squeeze(-1)

    def should_inject(self, h_loop, h_query, threshold=0.5):
        """Binary decision: should we inject the query now?"""
        with torch.no_grad():
            p = self.forward(h_loop, h_query)
            return p.item() > threshold


class MemoryGate(nn.Module):
    """
    Learned memory gate: h → (p_store, p_retrieve).

    Two-headed output: one sigmoid for store decision, one for retrieve decision.
    Operates inside the recurrence loop to control memory read/write.

    Architecture:
        h → Linear(4096, 256) → GELU → [store_head(256→1→Sigmoid), retrieve_head(256→1→Sigmoid)]

    Parameters: 4096*256 + 256 + 256*1 + 1 + 256*1 + 1 = 1,049,346 (~1.1M)
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, gate_dim=GATE_HIDDEN_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, gate_dim),
            nn.GELU(),
        )
        self.store_head = nn.Sequential(
            nn.Linear(gate_dim, 1),
            nn.Sigmoid(),
        )
        self.retrieve_head = nn.Sequential(
            nn.Linear(gate_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h):
        """
        Args:
            h: Hidden state (batch, 1, hidden_dim).

        Returns:
            (p_store, p_retrieve): Both shape (batch, 1).
        """
        z = self.backbone(h.to(self.backbone[0].weight.dtype))
        p_store = self.store_head(z).squeeze(-1)
        p_retrieve = self.retrieve_head(z).squeeze(-1)
        return p_store, p_retrieve

    def should_store(self, h, threshold=0.5):
        """Binary decision: should we write to memory?"""
        with torch.no_grad():
            p_store, _ = self.forward(h)
            return p_store.item() > threshold

    def should_retrieve(self, h, threshold=0.5):
        """Binary decision: should we read from memory?"""
        with torch.no_grad():
            _, p_retrieve = self.forward(h)
            return p_retrieve.item() > threshold


def count_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def create_all_gates(hidden_dim=HIDDEN_DIM, gate_dim=GATE_HIDDEN_DIM, device="cuda"):
    """
    Create all three gates and move to device.

    Returns:
        dict with "halt", "inject", "memory" gate modules.
        Also prints parameter counts.
    """
    gates = {
        "halt": HaltGate(hidden_dim, gate_dim).to(device),
        "inject": InjectGate(hidden_dim, gate_dim).to(device),
        "memory": MemoryGate(hidden_dim, gate_dim).to(device),
    }

    total = 0
    for name, gate in gates.items():
        n = count_parameters(gate)
        total += n
        print(f"  {name}: {n:,} params ({n * 4 / 1e6:.1f} MB fp32)")
    print(f"  Total: {total:,} params ({total * 4 / 1e6:.1f} MB fp32)")

    return gates
