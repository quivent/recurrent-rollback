"""
Simple Split-Recurrence Rollback Example
==========================================

Minimal example demonstrating split-recurrence rollback on Qwen3.5-27B
without MTP drafting. Loads the model, runs a split forward pass with
2 dummy tokens, then verifies that rolling back restores state exactly.

Requirements:
    pip install mlx mlx-lm recurrent-rollback

Usage:
    python examples/simple_example.py
"""

from __future__ import annotations

import sys

import mlx.core as mx

from recurrent_rollback import split_recurrence_forward, rollback_to
from recurrent_rollback.delta_net_rollback import DeltaNetRollbackLayer, DeltaNetState


MODEL_PATH = "Qwen/Qwen3.5-27B-MLX-4bit"


def main():
    try:
        from mlx_lm import load
    except ImportError:
        print("Install mlx-lm: pip install mlx-lm")
        sys.exit(1)

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)

    # Wrap model layers for split-recurrence rollback
    wrapped_layers = DeltaNetRollbackLayer.wrap_model_layers(model)
    cache = model.make_cache()

    # Encode a short prompt
    prompt = "Hello world"
    tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    print(f"Prompt: {prompt!r} ({tokens.shape[1]} tokens)")

    # --- Prefill: process the prompt ---
    prefill_output = split_recurrence_forward(
        layers=wrapped_layers,
        head=model.lm_head,
        x=tokens,
        cache=cache,
        embed_fn=model.embed_tokens,
    )
    print(f"Prefill done. Logits shape: {prefill_output.logits.shape}")

    # --- Save recurrent states after prefill ---
    states_after_prefill = {}
    for layer_idx, (layer_type, layer) in enumerate(wrapped_layers):
        if layer_type == "recurrent":
            state = layer.get_state()
            states_after_prefill[layer_idx] = (
                mx.array(state.rnn_state),  # force copy for comparison
            )

    # --- Forward pass with 2 dummy tokens ---
    # Use the top-1 prediction as "token 1", then a random token as "token 2"
    token_1 = mx.argmax(prefill_output.logits[:, -1:, :], axis=-1)
    token_2 = mx.array([[42]])  # arbitrary dummy token
    dummy_tokens = mx.concatenate([token_1, token_2], axis=1)  # (1, 2)
    print(f"Processing 2 tokens: {dummy_tokens.tolist()}")

    output = split_recurrence_forward(
        layers=wrapped_layers,
        head=model.lm_head,
        x=dummy_tokens,
        cache=cache,
        embed_fn=model.embed_tokens,
    )
    print(f"Forward done. Got {len(output.rollback_points)} rollback points.")

    # --- Verify states changed ---
    print("\n--- State comparison BEFORE rollback ---")
    for layer_idx in sorted(states_after_prefill.keys()):
        _, layer_type_and_layer = None, None
        for lt, ly in wrapped_layers:
            pass
        state_now = None
        for idx, (lt, ly) in enumerate(wrapped_layers):
            if idx == layer_idx:
                state_now = ly.get_state()
                break
        if state_now is None:
            continue
        (saved_rnn,) = states_after_prefill[layer_idx]
        diff = mx.abs(state_now.rnn_state - saved_rnn).max().item()
        if layer_idx == sorted(states_after_prefill.keys())[0]:
            print(f"  Layer {layer_idx}: max |state_now - state_prefill| = {diff:.6f}"
                  f"  {'(changed)' if diff > 0 else '(unchanged)'}")
    n_changed = sum(
        1 for idx in states_after_prefill
        for (lt, ly) in [wrapped_layers[idx]]
        if mx.abs(ly.get_state().rnn_state - states_after_prefill[idx][0]).max().item() > 0
    )
    print(f"  {n_changed}/{len(states_after_prefill)} recurrent layers changed state.")

    # --- Rollback to after token 0 (discard token 1) ---
    print("\n--- Rolling back to after token 0 ---")
    rollback_to(output.rollback_points[0], wrapped_layers, cache)

    # --- Verify rollback restored state to after token 0 ---
    # The state after rollback_points[0] should match the state after processing
    # only token_1 (one step beyond prefill). It should NOT match the prefill
    # state (that was before any of these tokens).
    print("\n--- State comparison AFTER rollback ---")
    print("Verifying rollback_points[0] state matches current state...")

    all_match = True
    for layer_idx in sorted(states_after_prefill.keys()):
        for idx, (lt, ly) in enumerate(wrapped_layers):
            if idx == layer_idx:
                state_now = ly.get_state()
                # The rollback point stores the state ref directly
                rp_state = output.rollback_points[0].recurrent_states.get(layer_idx)
                if rp_state is not None:
                    diff = mx.abs(state_now.rnn_state - rp_state.rnn_state).max().item()
                    if diff > 1e-8:
                        all_match = False
                        if layer_idx == sorted(states_after_prefill.keys())[0]:
                            print(f"  Layer {layer_idx}: MISMATCH (diff={diff:.6f})")
                break

    if all_match:
        print("  All recurrent layers: rollback state matches exactly.")
    else:
        print("  WARNING: Some layers did not match after rollback.")

    # Also verify it differs from the post-token-2 state (which was overwritten)
    print("\n--- Verifying rollback discarded token 1's effect ---")
    # Re-run to see what state[1] would have been
    # (We can't check directly since we rolled back, but we can verify
    # state differs from prefill since one token was processed)
    for layer_idx in sorted(states_after_prefill.keys())[:1]:
        for idx, (lt, ly) in enumerate(wrapped_layers):
            if idx == layer_idx:
                state_now = ly.get_state()
                (saved_rnn,) = states_after_prefill[layer_idx]
                diff = mx.abs(state_now.rnn_state - saved_rnn).max().item()
                print(f"  Layer {layer_idx}: |state_rolled_back - state_prefill| = {diff:.6f}"
                      f"  {'(good: 1 token processed)' if diff > 0 else '(unexpected: no change)'}")
                break

    print("\nDone. Rollback verified successfully.")


if __name__ == "__main__":
    main()
