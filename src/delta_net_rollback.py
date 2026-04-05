"""
DeltaNet-Specific Split-Recurrence Rollback
=============================================

Reference implementation of the split-recurrence rollback technique for
DeltaNet layers as found in Qwen3.5-27B. DeltaNet uses a gated delta
network (GDN) recurrence with the state update:

    S_t = gate * S_{t-1} + key_t * (value_t - key_t^T @ S_{t-1}) * beta_t

where S is a (d_k, d_v) state matrix, gate is a scalar decay, and beta
is a learned scaling factor. This is a nonlinear recurrence — S_t depends
on S_{t-1} both multiplicatively (gate * S) and through the delta rule
(key^T @ S). The state cannot be decomposed or trimmed.

This module wraps a DeltaNet layer to implement the RecurrentLayer protocol,
splitting the fused conv1d + GDN recurrence into per-token steps while
keeping the surrounding linear projections batched.

Architecture (Qwen3.5-27B):
    48 DeltaNet layers + 16 attention layers = 64 total
    d_model = 3584, d_k = d_v = 256, n_heads = 14
    GDN state per layer: 14 heads * 256 * 256 * 2 bytes = 1.75 MB
    Total recurrent state: 48 * 1.75 MB = 84 MB
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DeltaNetState:
    """DeltaNet recurrent state.

    Attributes:
        conv_state: Short convolution state, shape (B, n_heads, d_conv-1, d_k).
            Used by the causal conv1d before the GDN. Also non-trimmable
            (sliding window of recent inputs).
        rnn_state: GDN state matrix, shape (B, n_heads, d_k, d_v).
            The core non-trimmable state updated via the delta rule.
    """
    conv_state: mx.array
    rnn_state: mx.array


class DeltaNetRollbackLayer:
    """Wrapper around a DeltaNet layer enabling split-recurrence rollback.

    Decomposes the standard DeltaNet forward pass:

        Standard:  x -> in_proj -> conv1d+GDN(T=N) -> out_proj -> y
                                    ↑ fused, only final state

        Split:     x -> in_proj(T=N) -> [conv1d+GDN(T=1)]xN -> out_proj(T=N) -> y
                                          ↑ per-step state saved

    The in_proj and out_proj are standard linear layers that process all T
    tokens in a single matmul. Only the conv1d+GDN recurrence is split.

    Usage:
        layer = DeltaNetRollbackLayer(model.layers[i].recurrence)
        # Then pass to split_recurrence_forward as a RecurrentLayer
    """

    def __init__(self, delta_net_module: Any):
        """Wrap a DeltaNet module.

        Args:
            delta_net_module: The recurrence sub-module of a hybrid layer.
                Expected to have: in_proj, out_proj weights, and GDN parameters
                (beta, gate/alpha weights, q/k/v projections).
        """
        self.module = delta_net_module
        self._state: Optional[DeltaNetState] = None

    def input_proj(self, x: mx.array) -> mx.array:
        """Batched input projection: x @ W_in.

        Produces the concatenated [query, key, value, gate, beta] projections
        for all T tokens simultaneously. Shape: (B, T, proj_dim).

        This is a single large matmul — the most expensive operation in the
        layer, and it runs at full T=N efficiency.
        """
        return self.module.in_proj(x)

    def recurrence_step(
        self,
        projected: mx.array,
        state: DeltaNetState,
    ) -> Tuple[mx.array, DeltaNetState]:
        """Execute one step of the DeltaNet recurrence (conv1d + GDN).

        This is the sequential part that we split from the batched matmuls.
        For a single token, the operations are:

        1. Short convolution (causal conv1d with kernel size 4):
           - Shift conv_state left, append new input
           - Output = conv_state @ conv_weight + bias
           - Apply SiLU activation (fused_conv1d_silu in optimized path)

        2. GDN recurrence:
           - Split projected into q, k, v, gate, beta
           - beta = sigmoid(beta)
           - gate = sigmoid(gate)  [or exp(-softplus(alpha))]
           - delta = v - k^T @ S
           - S_new = gate * S + k * delta * beta
           - output = q^T @ S_new

        Args:
            projected: Single-token projection, shape (B, 1, proj_dim).
            state: Current DeltaNetState (conv_state + rnn_state).

        Returns:
            (output, new_state): Recurrence output (B, 1, d_out) and new state.
            new_state contains NEW arrays — old state refs remain valid.
        """
        B = projected.shape[0]
        m = self.module

        # --- Unpack projections ---
        # The in_proj concatenates q, k, v, gate, beta along the last dim.
        # Exact split depends on model config; Qwen3.5-27B uses:
        #   proj_dim = n_heads * (3*d_k + 2)  [q,k,v each d_k, gate+beta each 1]
        n_heads = m.n_heads if hasattr(m, 'n_heads') else m.num_heads
        d_k = m.d_k if hasattr(m, 'd_k') else m.head_dim
        d_v = m.d_v if hasattr(m, 'd_v') else d_k

        proj = projected.reshape(B, 1, n_heads, -1)
        q = proj[..., :d_k]
        k = proj[..., d_k:2*d_k]
        v = proj[..., 2*d_k:3*d_k]
        gate_logit = proj[..., 3*d_k:3*d_k+1]
        beta_logit = proj[..., 3*d_k+1:3*d_k+2]

        # --- Short convolution (causal conv1d) ---
        # Shift conv_state: drop oldest, append current key
        # conv_state shape: (B, n_heads, d_conv-1, d_k)
        conv_state = state.conv_state
        k_for_conv = k.reshape(B, n_heads, 1, d_k)
        new_conv_state = mx.concatenate(
            [conv_state[:, :, 1:, :], k_for_conv], axis=2
        )

        # Apply conv weights + SiLU
        if hasattr(m, 'conv_weight'):
            conv_weight = m.conv_weight  # (n_heads, d_conv, 1) or similar
            conv_out = (new_conv_state * conv_weight).sum(axis=2, keepdims=True)
            if hasattr(m, 'conv_bias') and m.conv_bias is not None:
                conv_out = conv_out + m.conv_bias
        else:
            conv_out = k_for_conv

        k_activated = mx.sigmoid(conv_out) * conv_out  # SiLU = x * sigmoid(x)

        # --- GDN recurrence ---
        beta = mx.sigmoid(beta_logit)  # (B, 1, n_heads, 1)
        gate = mx.sigmoid(gate_logit)  # decay gate

        rnn_state = state.rnn_state  # (B, n_heads, d_k, d_v)

        k_step = k_activated.reshape(B, n_heads, d_k, 1)
        v_step = v.reshape(B, n_heads, 1, d_v)
        q_step = q.reshape(B, n_heads, 1, d_k)
        beta_step = beta.reshape(B, n_heads, 1, 1)
        gate_step = gate.reshape(B, n_heads, 1, 1)

        # Delta rule: delta = v - k^T @ S, then S += k * delta * beta
        k_row = k_step.reshape(B, n_heads, 1, d_k)
        retrieval = k_row @ rnn_state  # (B, n_heads, 1, d_v)
        delta = v_step - retrieval

        # State update: S_new = gate * S + k * delta * beta
        update = k_step @ (delta * beta_step)  # (B, n_heads, d_k, d_v)
        new_rnn_state = gate_step * rnn_state + update

        # Output: q^T @ S_new
        output = q_step @ new_rnn_state  # (B, n_heads, 1, d_v)
        output = output.reshape(B, 1, n_heads * d_v)

        new_state = DeltaNetState(
            conv_state=new_conv_state,
            rnn_state=new_rnn_state,
        )

        return output, new_state

    def output_proj(self, recurrence_out: mx.array) -> mx.array:
        """Batched output projection: rec_out @ W_out.

        Processes all T tokens in a single matmul. Same cost regardless
        of whether recurrence was split or fused.
        """
        return self.module.out_proj(recurrence_out)

    def get_state(self) -> DeltaNetState:
        """Return current state reference (zero-copy)."""
        if self._state is None:
            raise RuntimeError(
                "State not initialized. Call set_state() with initial state "
                "before running split_recurrence_forward."
            )
        return self._state

    def set_state(self, state: DeltaNetState) -> None:
        """Set state from a reference. Used for both init and rollback."""
        self._state = state

    @staticmethod
    def wrap_model_layers(model: Any) -> list:
        """Wrap a Qwen3.5-27B model's layers for split-recurrence rollback.

        Inspects each layer to determine if it's attention or recurrent,
        and wraps recurrent layers in DeltaNetRollbackLayer.

        Args:
            model: A Qwen3.5 model instance with model.layers.

        Returns:
            List of (layer_type, wrapped_layer) tuples suitable for
            split_recurrence_forward.
        """
        wrapped = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'self_attn') and not hasattr(layer, 'recurrence'):
                wrapped.append(("attention", layer))
            elif hasattr(layer, 'recurrence') or hasattr(layer, 'delta_net'):
                recurrence = getattr(layer, 'recurrence', None) or layer.delta_net
                rollback_layer = DeltaNetRollbackLayer(recurrence)
                # Initialize state from the model's cache if available
                if hasattr(recurrence, 'state'):
                    rollback_layer.set_state(DeltaNetState(
                        conv_state=recurrence.state.conv_state,
                        rnn_state=recurrence.state.rnn_state,
                    ))
                wrapped.append(("recurrent", rollback_layer))
            else:
                # Fallback: treat as attention (e.g., MLP-only layers)
                wrapped.append(("attention", layer))
        return wrapped
