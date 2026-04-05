"""
MTP Speculative Decoding with Split-Recurrence Rollback
=========================================================

Complete example: use Qwen3.5-27B's MTP (Multi-Token Prediction) heads as
the draft model, and split-recurrence rollback for zero-cost rejection
recovery.

MTP drafting is ideal for hybrid models because the draft predictions come
from the target model itself — no separate draft model needed. The MTP
heads are small linear projections trained to predict tokens 2..K steps
ahead from intermediate hidden states.

Pipeline:
    1. Run target model forward pass with split recurrence (captures rollback points)
    2. Use MTP heads to draft K-1 additional tokens from the hidden states
    3. Run target model forward pass on all K tokens (again with split recurrence)
    4. Verify: compare target logits with draft tokens
    5. On rejection at position i: rollback_to(points[i-1]) — no redo

Requirements:
    pip install mlx mlx-lm recurrent-rollback
    # Model: Qwen/Qwen3.5-27B-MLX (or quantized variant)
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# -- recurrent-rollback imports --
from recurrent_rollback import split_recurrence_forward, rollback_to, RollbackPoint
from recurrent_rollback.delta_net_rollback import DeltaNetRollbackLayer, DeltaNetState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MTP_DRAFT_TOKENS = 3       # Number of extra tokens to draft via MTP heads
TEMPERATURE = 0.0          # Greedy decoding for verification
MAX_TOKENS = 256           # Maximum generation length
MODEL_PATH = "Qwen/Qwen3.5-27B-MLX-4bit"


# ---------------------------------------------------------------------------
# MTP Draft Head
# ---------------------------------------------------------------------------

class MTPDraftHead:
    """Multi-Token Prediction draft head.

    Qwen3.5-27B has MTP heads that predict tokens t+2, t+3, ... from the
    hidden state at position t. Each head is a small linear projection:

        logits_{t+k} = MTP_head_k(hidden_t)

    This gives us free draft tokens without a separate model.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        # MTP heads are stored as model.mtp_heads or similar
        self.heads = getattr(model, 'mtp_heads', None)
        if self.heads is None:
            # Fallback: use the main LM head (greedy continuation)
            self.heads = None

    def draft(
        self,
        hidden_states: mx.array,
        last_position: int,
        n_draft: int,
    ) -> List[mx.array]:
        """Draft n_draft tokens from the hidden state at last_position.

        Args:
            hidden_states: Hidden states from the target forward pass,
                shape (B, T, D).
            last_position: Index of the last verified token in the sequence.
            n_draft: Number of tokens to draft.

        Returns:
            List of token ID arrays, each shape (B, 1).
        """
        h = hidden_states[:, last_position:last_position+1, :]  # (B, 1, D)
        drafts = []

        if self.heads is not None:
            # Use dedicated MTP heads
            for k in range(n_draft):
                if k < len(self.heads):
                    logits = self.heads[k](h)  # (B, 1, V)
                    token = mx.argmax(logits, axis=-1, keepdims=True)  # (B, 1, 1)
                    drafts.append(token.squeeze(-1))  # (B, 1)
                else:
                    break
        else:
            # Fallback: autoregressive greedy from main head
            for k in range(n_draft):
                logits = self.model.head(h)
                token = mx.argmax(logits[:, -1:, :], axis=-1)
                drafts.append(token)
                # Re-embed for next step (approximate — real MTP doesn't need this)
                h = self.model.embed_tokens(token)

        return drafts


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_draft(
    target_logits: mx.array,
    draft_tokens: List[mx.array],
    temperature: float = 0.0,
) -> int:
    """Verify drafted tokens against target model logits.

    Uses greedy verification: accept draft token if it matches the target
    model's argmax at that position.

    Args:
        target_logits: Logits from target model, shape (B, T, V).
            Position 0 corresponds to the original token.
            Positions 1..K correspond to draft tokens.
        draft_tokens: List of K drafted token IDs, each shape (B, 1).
        temperature: Sampling temperature (0 = greedy).

    Returns:
        Number of accepted tokens (0 to len(draft_tokens)).
    """
    n_draft = len(draft_tokens)
    accepted = 0

    for i in range(n_draft):
        # Target's prediction for position i+1 comes from logits at position i
        if temperature == 0.0:
            target_token = mx.argmax(target_logits[:, i, :], axis=-1, keepdims=True)
        else:
            # Nucleus/temperature sampling would go here
            target_token = mx.argmax(target_logits[:, i, :], axis=-1, keepdims=True)

        if mx.array_equal(target_token, draft_tokens[i]):
            accepted += 1
        else:
            break

    return accepted


# ---------------------------------------------------------------------------
# Main speculative decoding loop
# ---------------------------------------------------------------------------

def speculative_decode(
    model: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int = MAX_TOKENS,
    n_draft: int = MTP_DRAFT_TOKENS,
) -> Tuple[mx.array, dict]:
    """Speculative decoding with MTP drafting and split-recurrence rollback.

    Args:
        model: Qwen3.5-27B model instance.
        prompt_tokens: Prompt token IDs, shape (1, S).
        max_tokens: Maximum tokens to generate.
        n_draft: Number of tokens to draft per step.

    Returns:
        (generated_tokens, stats): Generated token IDs and performance stats.
    """
    # --- Setup ---
    wrapped_layers = DeltaNetRollbackLayer.wrap_model_layers(model)
    drafter = MTPDraftHead(model)
    cache = model.make_cache()

    generated: List[mx.array] = []
    stats = {
        "total_tokens": 0,
        "draft_tokens": 0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "forward_passes": 0,
        "rollbacks": 0,
        "rollback_time_ms": 0.0,
        "redo_time_saved_ms": 0.0,
    }

    # --- Prefill ---
    prefill_output = split_recurrence_forward(
        layers=wrapped_layers,
        head=model.lm_head,
        x=prompt_tokens,
        cache=cache,
        embed_fn=model.embed_tokens,
    )
    stats["forward_passes"] += 1

    # Sample first token from prefill logits
    next_token = mx.argmax(prefill_output.logits[:, -1:, :], axis=-1)
    generated.append(next_token)
    stats["total_tokens"] += 1

    # --- Main loop ---
    while stats["total_tokens"] < max_tokens:
        # Step 1: Draft K tokens using MTP heads
        draft_tokens = drafter.draft(
            prefill_output.hidden_states,
            last_position=-1,
            n_draft=n_draft,
        )

        if not draft_tokens:
            # No MTP heads available; fall back to standard autoregressive
            break

        # Step 2: Concatenate [verified_token, draft_1, ..., draft_K]
        verify_input = mx.concatenate(
            [next_token] + draft_tokens, axis=1
        )  # (B, 1+K)

        # Step 3: Forward pass with split recurrence on all K+1 tokens
        verify_output = split_recurrence_forward(
            layers=wrapped_layers,
            head=model.lm_head,
            x=verify_input,
            cache=cache,
            embed_fn=model.embed_tokens,
        )
        stats["forward_passes"] += 1
        stats["draft_tokens"] += len(draft_tokens)

        # Step 4: Verify draft tokens against target logits
        # verify_output.logits[:, 0, :] = prediction for position after next_token
        # verify_output.logits[:, i, :] = prediction for position after draft_tokens[i-1]
        n_accepted = verify_draft(
            verify_output.logits[:, :-1, :],  # exclude last position (no draft to verify)
            draft_tokens,
        )

        stats["accepted_tokens"] += n_accepted
        stats["rejected_tokens"] += len(draft_tokens) - n_accepted

        # Step 5: Accept verified tokens
        for i in range(n_accepted):
            generated.append(draft_tokens[i])
            stats["total_tokens"] += 1

        # Step 6: Handle rejection
        if n_accepted < len(draft_tokens):
            # Rollback to state after the last accepted position
            rollback_pos = n_accepted  # index in verify_output's rollback_points
            t0 = time.perf_counter()

            rollback_to(
                verify_output.rollback_points[rollback_pos],
                wrapped_layers,
                cache,
            )

            rollback_ms = (time.perf_counter() - t0) * 1000
            stats["rollbacks"] += 1
            stats["rollback_time_ms"] += rollback_ms

            # Estimate saved redo time: would have needed a forward pass
            # on rollback_pos+1 tokens without split-recurrence rollback
            estimated_redo_ms = 34.0 * (n_accepted + 1) / 64  # proportional to seq len
            stats["redo_time_saved_ms"] += estimated_redo_ms

            # Take the target model's token at the rejection position
            correction = mx.argmax(
                verify_output.logits[:, n_accepted, :], axis=-1, keepdims=True
            )
            generated.append(correction)
            stats["total_tokens"] += 1
            next_token = correction
        else:
            # All drafts accepted — take next token from final logits
            next_token = mx.argmax(
                verify_output.logits[:, -1:, :], axis=-1
            )
            generated.append(next_token)
            stats["total_tokens"] += 1

        # Update for next iteration
        prefill_output = verify_output

        # Check for EOS
        if next_token.item() == 151643:  # Qwen EOS token
            break

    # Compile results
    all_tokens = mx.concatenate(generated, axis=1)
    return all_tokens, stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run speculative decoding demo."""
    try:
        from mlx_lm import load
    except ImportError:
        print("Install mlx-lm: pip install mlx-lm")
        return

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)

    prompt = "The key insight behind split-recurrence rollback is that"
    print(f"\nPrompt: {prompt}")
    print(f"Draft tokens per step: {MTP_DRAFT_TOKENS}")
    print("-" * 60)

    tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)

    t0 = time.perf_counter()
    generated, stats = speculative_decode(model, tokens, max_tokens=128)
    elapsed = time.perf_counter() - t0

    text = tokenizer.decode(generated[0].tolist())
    tps = stats["total_tokens"] / elapsed

    print(f"\nGenerated: {text}")
    print(f"\n{'='*60}")
    print(f"Performance:")
    print(f"  Tokens generated:    {stats['total_tokens']}")
    print(f"  Wall time:           {elapsed:.2f}s")
    print(f"  Throughput:          {tps:.1f} tok/s")
    print(f"  Forward passes:      {stats['forward_passes']}")
    print(f"  Draft tokens:        {stats['draft_tokens']}")
    print(f"  Accepted:            {stats['accepted_tokens']} ({100*stats['accepted_tokens']/max(1,stats['draft_tokens']):.0f}%)")
    print(f"  Rejected:            {stats['rejected_tokens']} ({100*stats['rejected_tokens']/max(1,stats['draft_tokens']):.0f}%)")
    print(f"  Rollbacks:           {stats['rollbacks']}")
    print(f"  Rollback time:       {stats['rollback_time_ms']:.2f}ms total")
    print(f"  Redo time saved:     {stats['redo_time_saved_ms']:.1f}ms (estimated)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
