# recurrent-rollback

**Zero-cost speculative decoding rollback for hybrid attention/recurrent models**

## The Problem

Speculative decoding accelerates autoregressive LLM inference by drafting multiple tokens with a cheap model, then verifying them in a single batched forward pass through the target model. When the target rejects a drafted token, the system must **roll back** to the last accepted position.

For pure-attention models (transformers), rollback is trivial: trim the KV cache to the accepted length. KV caches are append-only sequences — discarding the tail is O(1).

Hybrid models — DeltaNet, Mamba, RWKV, Griffin, Jamba — combine attention layers with **recurrent layers** whose state is a fixed-size matrix updated via nonlinear recurrence:

```
state_{t} = gate * state_{t-1} + key * (value - key^T @ state_{t-1}) * beta
```

This state cannot be trimmed. It has no concept of "remove the last token's contribution." The information from token `t` is irreversibly mixed into the state matrix.

### Previous approach: checkpoint + restore + redo

Before verification, checkpoint the recurrent state. On rejection at position `i`, restore the checkpoint (state before any drafted tokens), then **redo** the forward pass for tokens `0..i`. This works, but is expensive:

- A full forward pass costs ~34ms (Qwen3.5-27B on M4 Max)
- With a 21% rejection rate, the average redo cost is **~7ms per step**
- This nearly eliminates the throughput gains from speculative decoding

## The Solution: Split-Recurrence Rollback

The key insight: **matmuls don't care about sequence length, but recurrences do.** In a typical hybrid layer:

```
input_proj (batched) -> recurrence (sequential) -> output_proj (batched)
```

The input and output projections are matrix multiplications that process all `T` tokens simultaneously. Only the recurrence itself is inherently sequential. We exploit this:

1. **Batch the matmuls** at `T=N` (all draft tokens together) — same cost as before
2. **Split only the recurrence** into `T=1` steps — minimal overhead since recurrences are small ops
3. **Capture intermediate state refs** after each recurrence step — zero-copy because arrays are immutable
4. **On rejection at position `i`**: restore state ref `i`, trim KV caches — no redo needed

```
Before: input_proj(T=N) --> recurrence(T=N) --> out_proj(T=N)
                                ↑ only final state available
                                  must redo on rejection

After:  input_proj(T=N) --> rec(T=1)-->rec(T=1)-->...-->rec(T=1) --> out_proj(T=N)
                              ↑ save    ↑ save           ↑ save
                              state[0]  state[1]         state[N-1]
                              
                            On reject at i: restore state[i-1], trim KV
                            No redo. No recomputation. Zero cost.
```

### Why zero-copy refs work

In MLX (and JAX), arrays are **immutable**. When the recurrence computes `new_state = f(old_state, input)`, it creates a new array — `old_state` is never modified in place. Saving a reference to `old_state` is free: no copy, no extra memory beyond the graph node.

### Cost analysis

| Component | Cost |
|-----------|------|
| Extra GDN kernel dispatches | N tokens x 48 layers x ~0.02ms = **~1ms** |
| Saved redo (eliminated) | 34ms x 21% rejection rate = **~7ms** |
| **Net savings** | **~6ms per verification step** |

The split adds ~1ms of overhead from extra kernel dispatches but eliminates ~7ms of redo cost, for a net gain of ~6ms per step. At 30 tok/s, this translates to roughly **+2 tok/s**.

## Applicability

This technique applies to any model architecture with non-trimmable recurrent state:

| Architecture | Recurrent State | State Update |
|-------------|----------------|--------------|
| **DeltaNet** | `rnn_state` (d_k x d_v) | `g * S + k * (v - k^T S) * beta` |
| **Mamba / Mamba-2** | `conv_state` + `ssm_state` | Convolution + selective SSM |
| **RWKV** | `time_state` | Exponential decay + linear combination |
| **Griffin** | `rg_lru_state` | Real-gated linear recurrent unit |
| **Jamba** | Mixed Mamba + attention | Mamba layers use SSM state |

## Installation

```bash
pip install recurrent-rollback
```

Or from source:

```bash
git clone https://github.com/joshkornreich/recurrent-rollback.git
cd recurrent-rollback
pip install -e .
```

## Usage

```python
from recurrent_rollback import split_recurrence_forward, rollback_to

# Forward pass with rollback capability
outputs, rollback_points = split_recurrence_forward(
    model, tokens, cache
)

# Verify draft tokens against target logits
accepted = verify(outputs.logits, draft_tokens)

# On rejection at position i: restore state, no redo
if accepted < len(draft_tokens):
    rollback_to(rollback_points[accepted], model, cache)
```

## Reference Implementation

The `src/` directory contains a reference implementation in MLX for DeltaNet (Qwen3.5-27B):

- `split_recurrence.py` — Architecture-agnostic split-recurrence forward pass
- `delta_net_rollback.py` — DeltaNet-specific implementation with `fused_gdn_step`

The `examples/` directory contains a complete speculative decoding loop using MTP drafting.

## Citation

If you use this technique in your work, please cite:

```bibtex
@software{kornreich2026recurrent_rollback,
  author = {Kornreich, Josh},
  title = {Split-Recurrence Rollback for Speculative Decoding in Hybrid Models},
  year = {2026},
  url = {https://github.com/joshkornreich/recurrent-rollback}
}
```

## License

MIT
