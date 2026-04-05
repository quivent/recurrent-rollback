"""
Split-Recurrence Rollback for Speculative Decoding
====================================================

Core technique: decouple batched matmuls from sequential recurrence in hybrid
attention/recurrent layers. Process matmuls at T=N (all draft tokens), but split
the recurrence into T=1 steps, capturing intermediate state references for
zero-cost rollback on rejection.

This module is architecture-agnostic. It defines the rollback protocol and
orchestrates the split forward pass. Architecture-specific recurrence
implementations (DeltaNet, Mamba, etc.) plug in via the RecurrentLayer protocol.

Key invariant: MLX arrays are immutable. Saving a reference to an intermediate
state tensor is zero-copy — the array is never mutated in place. This makes
the entire rollback mechanism allocation-free beyond the list of references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, Tuple

import mlx.core as mx


# ---------------------------------------------------------------------------
# Protocols — what architecture-specific layers must implement
# ---------------------------------------------------------------------------

class RecurrentLayer(Protocol):
    """Protocol for a recurrent layer that supports split-step execution.

    Implementors must provide:
      - input_proj:  batched linear projections (T-agnostic)
      - recurrence_step: single-token recurrence update
      - output_proj: batched output projection (T-agnostic)
      - get_state / set_state: state accessors for rollback
    """

    def input_proj(self, x: mx.array) -> mx.array:
        """Project input hidden states. Called once with shape (B, T, D).
        Returns projected tensor of shape (B, T, D_proj)."""
        ...

    def recurrence_step(
        self,
        projected: mx.array,       # (B, 1, D_proj) — single token slice
        state: Any,                 # current recurrence state
    ) -> Tuple[mx.array, Any]:
        """Execute one step of the recurrence.

        Args:
            projected: Single-token projected input, shape (B, 1, D_proj).
            state: Current recurrence state (architecture-specific).

        Returns:
            (output, new_state): Single-token recurrence output and updated state.
            CRITICAL: new_state must be a NEW array, not an in-place mutation.
        """
        ...

    def output_proj(self, recurrence_out: mx.array) -> mx.array:
        """Project recurrence output back to hidden dim. Shape (B, T, D)."""
        ...

    def get_state(self) -> Any:
        """Return current recurrence state (reference, not copy)."""
        ...

    def set_state(self, state: Any) -> None:
        """Restore recurrence state from a saved reference."""
        ...


class AttentionLayer(Protocol):
    """Protocol for an attention layer with trimmable KV cache."""

    def forward(self, x: mx.array, cache: Any) -> mx.array:
        """Standard attention forward pass."""
        ...


# ---------------------------------------------------------------------------
# Rollback point — captured at each token position
# ---------------------------------------------------------------------------

@dataclass
class RollbackPoint:
    """Snapshot of model state after processing tokens 0..position.

    All state references are zero-copy (immutable arrays). Restoring a
    rollback point is O(num_recurrent_layers) assignment operations.

    Attributes:
        position: Number of tokens processed (0-indexed, inclusive).
        recurrent_states: Dict mapping layer_index -> state ref for each
            recurrent layer. These are the states AFTER processing the
            token at this position.
        kv_cache_lengths: Dict mapping layer_index -> KV cache length for
            each attention layer. Used to trim KV caches on rollback.
    """
    position: int
    recurrent_states: dict[int, Any] = field(default_factory=dict)
    kv_cache_lengths: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core forward pass with split recurrence
# ---------------------------------------------------------------------------

@dataclass
class SplitForwardOutput:
    """Output of split_recurrence_forward.

    Attributes:
        hidden_states: Final hidden states, shape (B, T, D).
        logits: Output logits, shape (B, T, V).
        rollback_points: List of RollbackPoint, one per input token.
            rollback_points[i] captures state after processing token i.
    """
    hidden_states: mx.array
    logits: mx.array
    rollback_points: List[RollbackPoint]


def split_recurrence_forward(
    layers: List[Tuple[str, Any]],
    head: Callable[[mx.array], mx.array],
    x: mx.array,
    cache: Optional[Any] = None,
    *,
    embed_fn: Optional[Callable[[mx.array], mx.array]] = None,
) -> SplitForwardOutput:
    """Execute a forward pass with split recurrence for rollback support.

    This is the core of the technique. For each layer in the model:
      - If attention: run normally (KV cache is trimmable)
      - If recurrent: batch the matmuls, split the recurrence into T=1 steps,
        capture state after each step

    Args:
        layers: List of (layer_type, layer_module) pairs.
            layer_type is "attention" or "recurrent".
        head: Language model head (hidden -> logits).
        x: Input token IDs, shape (B, T) if embed_fn provided,
           or embedded hidden states, shape (B, T, D).
        cache: Model cache object (KV caches, etc.).
        embed_fn: Optional embedding function (token IDs -> hidden states).

    Returns:
        SplitForwardOutput with hidden_states, logits, and rollback_points.

    Complexity:
        Same FLOPs as a standard forward pass. Additional overhead is
        N_tokens * N_recurrent_layers kernel dispatches for the split
        recurrence (~0.02ms each on M4 Max).
    """
    if embed_fn is not None:
        h = embed_fn(x)
    else:
        h = x

    B, T, D = h.shape
    num_tokens = T

    # Initialize rollback points — one per token position
    rollback_points: List[RollbackPoint] = [
        RollbackPoint(position=t) for t in range(num_tokens)
    ]

    for layer_idx, (layer_type, layer) in enumerate(layers):
        if layer_type == "attention":
            # Attention layers process all tokens at once.
            # KV cache is append-only and trimmable.
            layer_cache = cache[layer_idx] if cache is not None else None
            h = layer.forward(h, layer_cache)

            # Record KV cache length at each token position.
            # After processing T tokens, cache length increases by T.
            if layer_cache is not None:
                # offset is the cache length AFTER processing all T tokens.
                # Subtract T to get the cache length BEFORE this forward pass,
                # so base_len + t + 1 gives the cache length after token t.
                base_len = getattr(layer_cache, 'offset', 0) - T
                for t in range(num_tokens):
                    rollback_points[t].kv_cache_lengths[layer_idx] = base_len + t + 1

        elif layer_type == "recurrent":
            # --- THE SPLIT ---
            # Step 1: Batch the input projection across all T tokens.
            projected = layer.input_proj(h)  # (B, T, D_proj)

            # Step 2: Run recurrence one token at a time, saving state refs.
            state = layer.get_state()
            recurrence_outputs = []

            for t in range(num_tokens):
                token_proj = projected[:, t : t + 1, :]  # (B, 1, D_proj)
                token_out, state = layer.recurrence_step(token_proj, state)
                recurrence_outputs.append(token_out)

                # Capture state reference. This is zero-copy because `state`
                # is a new array (immutable semantics). The recurrence_step
                # created it; the old state is untouched.
                rollback_points[t].recurrent_states[layer_idx] = state

            # Commit the final state to the layer.
            layer.set_state(state)

            # Step 3: Concatenate recurrence outputs and batch the output projection.
            rec_out = mx.concatenate(recurrence_outputs, axis=1)  # (B, T, D_rec)
            h = layer.output_proj(rec_out)  # (B, T, D)

        else:
            raise ValueError(f"Unknown layer type: {layer_type!r}")

    logits = head(h)

    return SplitForwardOutput(
        hidden_states=h,
        logits=logits,
        rollback_points=rollback_points,
    )


def rollback_to(
    point: RollbackPoint,
    layers: List[Tuple[str, Any]],
    cache: Optional[Any] = None,
) -> None:
    """Restore model state to a rollback point. No recomputation.

    This is the payoff: on rejection at position `i`, call
    `rollback_to(rollback_points[i-1], ...)` to restore state to after
    processing tokens 0..i-1. Cost: O(num_layers) reference assignments.

    Args:
        point: RollbackPoint to restore to.
        layers: Same layer list passed to split_recurrence_forward.
        cache: Model cache object.
    """
    for layer_idx, (layer_type, layer) in enumerate(layers):
        if layer_type == "recurrent":
            if layer_idx in point.recurrent_states:
                layer.set_state(point.recurrent_states[layer_idx])

        elif layer_type == "attention":
            if cache is not None and layer_idx in point.kv_cache_lengths:
                layer_cache = cache[layer_idx]
                target_len = point.kv_cache_lengths[layer_idx]
                _trim_kv_cache(layer_cache, target_len)


def _trim_kv_cache(cache: Any, target_length: int) -> None:
    """Trim a KV cache to target_length entries.

    For MLX KV caches, this means slicing the key and value tensors
    along the sequence dimension and updating the offset.
    """
    if hasattr(cache, 'offset') and hasattr(cache, 'keys') and hasattr(cache, 'values'):
        current_len = cache.offset
        if target_length < current_len:
            cache.keys = cache.keys[:, :target_length, :]
            cache.values = cache.values[:, :target_length, :]
            cache.offset = target_length
    elif hasattr(cache, 'trim'):
        cache.trim(target_length)
