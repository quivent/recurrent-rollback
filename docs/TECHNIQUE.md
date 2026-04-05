# Split-Recurrence Rollback: Technical Deep Dive

## 1. Mathematical Formulation

### The Delta Rule Recurrence

DeltaNet layers maintain a state matrix `S in R^{d_k x d_v}` updated at each timestep `t` by the **gated delta rule**:

```
S_{t+1} = g_t * S_t + beta_t * k_t * (v_t - k_t^T @ S_t)
```

where:
- `S_t in R^{d_k x d_v}` is the recurrent state matrix
- `g_t in R` is a scalar decay gate (sigmoid of a learned projection)
- `beta_t in R` is a learned scaling factor (sigmoid of a learned projection)
- `k_t in R^{d_k}` is the key vector (after conv1d + SiLU activation)
- `v_t in R^{d_v}` is the value vector
- `k_t^T @ S_t in R^{d_v}` is the retrieval: what the state "remembers" for key `k_t`

The term `(v_t - k_t^T @ S_t)` is the **delta** -- the difference between the desired value `v_t` and what the state currently retrieves for key `k_t`. This is analogous to the delta rule in classical Hebbian learning: the update is proportional to the prediction error.

Expanding:

```
S_{t+1} = g_t * S_t + beta_t * k_t * v_t^T - beta_t * k_t * (k_t^T @ S_t)
        = g_t * S_t + beta_t * k_t @ v_t^T - beta_t * (k_t @ k_t^T) @ S_t
        = (g_t * I - beta_t * k_t @ k_t^T) @ S_t + beta_t * k_t @ v_t^T
```

This is an **affine** recurrence: `S_{t+1} = A_t @ S_t + B_t` where `A_t = g_t * I - beta_t * k_t @ k_t^T` and `B_t = beta_t * k_t @ v_t^T`. The matrix `A_t` is **state-independent** but **input-dependent** -- it changes every timestep based on the input token's projections.

### Why the Delta Rule Is Non-Invertible

To "undo" one step (recover `S_t` from `S_{t+1}`), we would need:

```
S_t = A_t^{-1} @ (S_{t+1} - B_t)
```

This requires:
1. **Storing `A_t` and `B_t`** for every timestep (or equivalently `g_t`, `beta_t`, `k_t`, `v_t`)
2. **`A_t` must be invertible**: `A_t = g_t * I - beta_t * k_t @ k_t^T` is a rank-1 perturbation of a scaled identity. By the matrix determinant lemma, `det(A_t) = g_t^{d_k} * (1 - beta_t * ||k_t||^2 / g_t)`. When `beta_t * ||k_t||^2 = g_t`, the matrix is singular and inversion fails entirely.

Even when `A_t` is invertible, computing `A_t^{-1}` requires a `d_k x d_k` matrix inverse per step per head per layer -- 48 layers x 14 heads x 256x256 inverse per token. This is far more expensive than simply saving the intermediate state.

The `k_t^T @ S_t` term is what makes the recurrence **state-dependent**: the update to `S` depends on what is currently stored in `S`. This distinguishes it from a pure linear recurrence `S_{t+1} = g * S_t + k * v^T`, which is invertible as long as `g != 0` (just compute `S_t = (S_{t+1} - k * v^T) / g`).

## 2. The Split-Recurrence Technique

### Formal Definition

Given a recurrent layer with forward pass:

```
y = out_proj(recurrence(in_proj(x)))
```

where `x in R^{B x T x D}`, `in_proj: R^D -> R^{D_proj}` and `out_proj: R^{D_out} -> R^D` are linear projections, and `recurrence` is the sequential state update, the **split-recurrence** decomposes this into:

```
p = in_proj(x)                    # Batched: (B, T, D) -> (B, T, D_proj)
for t in 0..T-1:
    r_t, s_t = step(p[:, t], s_{t-1})   # Sequential: one token at a time
    save_ref(s_t)                         # Zero-copy state capture
r = concat(r_0, ..., r_{T-1})     # Reassemble: (B, T, D_out)
y = out_proj(r)                   # Batched: (B, T, D_out) -> (B, T, D)
```

### Proof: Splitting Preserves Correctness

The correctness of splitting depends on one property: **matmuls distribute over concatenation along the sequence dimension**.

**Claim**: For `X = concat(x_0, x_1, ..., x_{T-1})` along axis 1, and weight matrix `W`:

```
matmul(concat(x_0, x_1, ..., x_{T-1}), W) = concat(matmul(x_0, W), matmul(x_1, W), ..., matmul(x_{T-1}, W))
```

**Proof**: Let `X in R^{B x T x D_in}` and `W in R^{D_in x D_out}`. The matmul computes:

```
Y[b, t, j] = sum_k X[b, t, k] * W[k, j]
```

Each `(b, t)` slice is computed independently. Therefore:

```
Y[:, t, :] = X[:, t, :] @ W = x_t @ W
```

And:

```
Y = concat(x_0 @ W, x_1 @ W, ..., x_{T-1} @ W)
```

This means `in_proj` and `out_proj` produce identical results whether applied to the full `(B, T, D)` tensor or to individual `(B, 1, D)` slices concatenated afterward. The recurrence is inherently sequential regardless, so splitting it changes nothing about the computation -- only the granularity of state capture.

### Why We Split but Still Batch the Matmuls

We batch `in_proj` and `out_proj` at `T=N` because GPU matmuls achieve higher utilization with larger batch dimensions. Splitting those would add overhead for no benefit. The recurrence is already sequential -- running it as N steps of T=1 versus 1 step of T=N adds only kernel dispatch overhead, not extra FLOPs.

## 3. Cost Model

### Definitions

- `C_batched_matmul`: Cost of `in_proj(T=N)` + `out_proj(T=N)` -- identical in all approaches
- `C_recurrence_fused`: Cost of fused recurrence over T=N tokens (single kernel dispatch)
- `C_recurrence_dispatch`: Cost of one T=1 recurrence kernel dispatch
- `N`: Number of tokens in the verification batch
- `C_forward`: Cost of a full model forward pass
- `P_reject`: Probability of rejecting at least one draft token

### Split-Recurrence Cost

```
C_split = C_batched_matmul + N * C_recurrence_dispatch
```

The split pays `N` individual dispatches instead of one fused dispatch.

### Original (Checkpoint + Redo) Cost

```
C_original = C_batched_matmul + C_recurrence_fused + C_redo
C_redo = C_forward * P_reject * E[accepted / N]
```

where `E[accepted / N]` is the expected fraction of tokens that must be re-processed on rejection.

### Break-Even Analysis

The split wins when its overhead is less than the expected redo cost:

```
N * C_recurrence_dispatch - C_recurrence_fused < C_forward * P_reject * E[accepted / N]
```

The left side is the extra dispatch overhead (split cost minus fused cost). The right side is the eliminated redo cost.

For Qwen3.5-27B on M4 Max:
- `C_recurrence_dispatch` ~ 0.02ms (single GDN step)
- `C_recurrence_fused` ~ 0.01ms (fused T=4 GDN, marginal over T=1)
- `N` = 4 (1 verified + 3 draft)
- `C_forward` ~ 34ms
- `P_reject` ~ 0.51 (with K=3 draft tokens, 21% per-token rejection)
- `E[accepted / N]` ~ 0.5

Extra dispatch overhead per layer: `4 * 0.02 - 0.01 = 0.07ms`
Over 48 recurrent layers: `48 * 0.07 = 3.4ms`

Eliminated redo: `34 * 0.51 * 0.5 = 8.7ms`

**Net savings: 8.7 - 3.4 = 5.3ms per verification step.**

The split wins when:

```
N * C_dispatch < C_forward * P_reject
```

For this model: `4 * 0.02 * 48 = 3.84ms < 34 * 0.51 = 17.3ms` -- comfortably above break-even.

## 4. Why Recurrent State Cannot Be Trimmed

### The Attention Case (Trimmable)

In a standard transformer, the KV cache is an append-only sequence:

```
KV_cache = [kv_0, kv_1, ..., kv_t]
```

To roll back to position `i`, simply discard entries `i+1..t`:

```
KV_cache = KV_cache[:i+1]
```

This works because each token's contribution to the cache is an independent row. Token `t`'s key/value vectors do not modify token `t-1`'s. The operation is O(1) -- a pointer/length update.

### The Recurrent Case (Non-Trimmable)

As shown in Section 1, the DeltaNet state update `S_{t+1} = g_t * S_t + beta_t * k_t * (v_t - k_t^T @ S_t)` creates a nonlinear dependency on the previous state. The retrieval term `k_t^T @ S_t` means the update depends on what was *read* from `S_t`, not just what was written. You cannot algebraically invert this to recover `S_t` from `S_{t+1}` without storing the intermediate values (and even then, the matrix may be singular).

Even the simpler linear recurrence `S_t = g * S_{t-1} + k * v^T` would require storing all `(g, k, v)` tuples to invert -- and the delta rule makes it strictly harder.

### Other Architectures

| Architecture | State Update | Why Non-Invertible |
|-------------|-------------|-------------------|
| **DeltaNet** | `g*S + k*(v - k^T@S)*beta` | Nonlinear (retrieval in delta) |
| **Mamba** | `A*S + B*x` (discretized SSM) | `A` is input-dependent (selective) |
| **Mamba-2** | Same + `conv_state` | Convolution is a sliding window |
| **RWKV** | `exp(-w)*S + k*v` | Exponential decay mixes all history |
| **Griffin** | `a*S + b*(1-a)*x` | Real-gated LRU, same mixing issue |
| **Jamba** | Mamba layers interleaved with attention | Mamba layers have SSM state |

In all cases, the state at time `t` is a lossy compression of the entire history `0..t`. There is no "undo last token" operation.

#### Mamba Generalization

Mamba has two recurrent states. The **conv_state** is a FIFO buffer (sliding window of the last `d_conv - 1` inputs). It *could* be trimmed by popping the most recent entry and restoring the oldest, but this requires saving the evicted entry -- it is simpler to use the split-recurrence approach and save the full conv_state ref. The **ssm_state** uses selective scan: `S_{t+1} = A_bar_t * S_t + B_bar_t * x_t` where `A_bar` and `B_bar` are input-dependent (the "selective" mechanism). This is a linear recurrence with time-varying coefficients -- non-invertible because `A_bar_t` depends on the input at time `t`, which is not stored. The split-recurrence technique applies identically: batch the input/output projections, split conv + SSM into per-token steps, save state refs.

#### RWKV Generalization

RWKV's time-mixing uses: `state_{t+1} = exp(-w) * state_t + k_t * v_t^T` where `w` is a learned channel-wise decay. The exponential decay accumulates a weighted sum of all history -- each token's contribution is scaled by `exp(-w * (T - t))`. The state is non-invertible because undoing one step requires knowing the exact `k_t * v_t^T` that was added, which is not retained after the state update. The split-recurrence technique applies directly with no modifications to the protocol.

## 2. Why Checkpoint + Redo Is Expensive

### The Standard Approach

Before running the verification forward pass on draft tokens, save the recurrent state:

```python
checkpoint = copy_state(model)        # Deep copy all recurrent states
logits = model.forward(draft_tokens)  # Updates recurrent state
accepted = verify(logits, drafts)

if accepted < len(drafts):
    restore_state(model, checkpoint)  # Restore to before all drafts
    model.forward(draft_tokens[:accepted])  # REDO accepted tokens
```

### Cost Breakdown

For Qwen3.5-27B on M4 Max (546 GB/s memory bandwidth):

| Operation | Cost |
|-----------|------|
| Full forward pass (64 layers) | ~34ms |
| State checkpoint (84 MB copy) | ~0.3ms |
| State restore | ~0.3ms |
| **Redo forward pass** | **~34ms x (accepted/total)** |

With K=3 draft tokens and 21% rejection rate:
- Rejection probability per step: ~1 - 0.79^3 = ~51%
- Average redo when rejecting: ~34ms * 1.5/3 = ~17ms (average half the tokens)
- Expected redo cost per step: 0.51 * 17ms = **~8.7ms**

At 30 tok/s baseline (33ms/tok), the redo penalty consumes **26%** of the time budget. This makes speculative decoding barely break even — the throughput gain from accepting multiple tokens is nearly offset by the redo cost on rejection.

### The Deep Copy Problem

The checkpoint itself is also expensive to do correctly:

```python
# Naive: Python reference copy (WRONG — state gets mutated)
checkpoint = model.state  

# Correct: deep copy of all arrays
checkpoint = mx.array(model.rnn_state)  # Forces a copy
```

With 48 DeltaNet layers, each with a `(14, 256, 256)` state matrix in float16:
- Per-layer state: 14 * 256 * 256 * 2 bytes = 1.75 MB
- Total: 48 * 1.75 MB = 84 MB
- Copy cost: 84 MB / 546 GB/s = ~0.15ms (negligible)

The copy cost is small, but the **redo cost dominates**.

## 3. Why Splitting Only the Recurrence Works

### Key Observation: Matmuls Are T-Agnostic

A linear projection `Y = X @ W` where `X` has shape `(B, T, D_in)` and `W` has shape `(D_in, D_out)` produces `Y` with shape `(B, T, D_out)`. The computation is:

```
Y[b, t, :] = X[b, t, :] @ W    for all t independently
```

Each token's projection is **independent** of every other token's. Computing `X @ W` for T=4 costs the same as computing it 4 times for T=1 — actually less, because the batched matmul has better GPU utilization.

This means we can process the matmuls at full batch size `T=N` and split *only* the recurrence.

### The Split Architecture

Standard DeltaNet layer forward pass:
```
h = in_proj(x)          # (B, T, D) -> (B, T, D_proj)     [matmul, T-agnostic]
h = conv1d(h)           # (B, T, D_proj) -> (B, T, D_proj) [sequential]
h = gdn(h, state)       # (B, T, D_proj) -> (B, T, D_out)  [sequential]
y = out_proj(h)          # (B, T, D_out) -> (B, T, D)       [matmul, T-agnostic]
```

Split forward pass:
```
h = in_proj(x)           # Batched at T=N — same cost

for t in range(N):       # Split recurrence into T=1 steps
    h_t = conv1d(h[:, t:t+1, :], conv_state)
    h_t, rnn_state = gdn_step(h_t, rnn_state)
    save_ref(t, conv_state, rnn_state)  # Zero-copy ref
    outputs.append(h_t)

h = concat(outputs)      # Reassemble for output projection
y = out_proj(h)           # Batched at T=N — same cost
```

The matmuls (`in_proj`, `out_proj`) run at full T=N efficiency. Only the recurrence is split, and it was already sequential by nature — running it as N steps of T=1 instead of 1 step of T=N adds only kernel dispatch overhead.

### What the Split Costs

For Qwen3.5-27B with N=3 draft tokens + 1 verified token (T=4):

| Component | Standard | Split | Delta |
|-----------|----------|-------|-------|
| in_proj matmul | 1 dispatch | 1 dispatch | 0 |
| conv1d + GDN | 1 dispatch (fused T=4) | 4 dispatches (T=1 each) | +3 dispatches |
| out_proj matmul | 1 dispatch | 1 dispatch | 0 |

Per recurrent layer: +3 extra dispatches at ~0.02ms each = +0.06ms.
For 48 DeltaNet layers: 48 * 0.06ms = **~2.9ms total overhead**.

But this is per verification step, which processes K+1 tokens. Per output token:
~2.9ms / (1 + 3*0.79) = ~0.85ms per token.

### What the Split Saves

On rejection (probability ~51% per step with K=3):
- No redo forward pass needed
- Rollback cost: O(48) reference assignments = **~0.001ms**
- Savings: ~8.7ms expected redo cost eliminated

**Net: +0.85ms overhead, -8.7ms redo cost = ~7.85ms saved per step.**

## 4. Why Zero-Copy References Work

### Array Immutability in MLX

MLX arrays follow **functional semantics**: operations produce new arrays rather than mutating existing ones. When the GDN recurrence computes:

```python
new_state = gate * old_state + key @ delta.T * beta
```

This creates a **new** `mx.array` for `new_state`. The `old_state` array is untouched. Both arrays exist in GPU memory simultaneously until one is garbage collected.

Saving a reference is therefore free:

```python
saved_states = []
for t in range(N):
    output, state = gdn_step(input_t, state)
    saved_states.append(state)  # Reference to immutable array — no copy
```

After the loop, `saved_states[0]` still points to the state after token 0, even though `state` has been updated N times. Each update created a new array; the old one is preserved by the reference in `saved_states`.

### Memory Overhead

Each saved state reference adds one pointer (8 bytes) to the Python list. The actual state arrays already exist in GPU memory — they would have been computed and discarded in the standard forward pass. The split retains N intermediate arrays instead of just the final one.

For Qwen3.5-27B:
- State per layer per step: 1.75 MB
- Additional retained states: (N-1) * 48 layers * 1.75 MB
- With N=4: 3 * 48 * 1.75 MB = **252 MB**

This is modest compared to the model's 15.3 GB weight footprint and the KV cache.

### JAX Compatibility

The same technique works in JAX, which shares MLX's immutable array semantics. In PyTorch, explicit `.clone()` would be needed since PyTorch tensors are mutable — but the cost is still small (just the state matrices, not the full hidden states).

## 5. Generalization to Other Architectures

### Mamba / Mamba-2

Mamba has two recurrent states:

**conv_state** — short causal convolution (kernel size 4):
```python
# Standard: conv1d over T tokens
# Split: shift-and-append per token
conv_state = concat(conv_state[:, 1:], new_input)
output = conv_state @ conv_weight
```

**ssm_state** — selective state space model:
```python
# Discretized: S_t = A_bar * S_{t-1} + B_bar * x_t
# A_bar, B_bar are input-dependent (selective mechanism)
A_bar = exp(delta * A)  # delta is input-dependent
B_bar = delta * B       # B is input-dependent
ssm_state = A_bar * ssm_state + B_bar * x
output = C @ ssm_state
```

Both are non-trimmable. The split technique applies identically: batch the input/output projections, split the conv + SSM into per-token steps, save state refs.

### RWKV

RWKV's time-mixing recurrence:
```python
# Exponential decay + linear combination
state = exp(-w) * state + k * v
output = state @ r  # receptance-gated readout
```

The exponential decay `exp(-w)` is channel-wise and learned. State captures a weighted average of all history. Split applies directly.

### Griffin (Real-Gated Linear Recurrent Unit)

Griffin uses an RG-LRU:
```python
# Real-valued gated linear recurrence
a = sigmoid(W_a @ x)     # input-dependent gate
state = a * state + (1 - a) * (W_x @ x)
output = state * sigmoid(W_o @ x)
```

The input-dependent gate `a` makes this non-invertible. Split works the same way.

### Implementation Pattern

For any architecture, the adaptation follows the same pattern:

```python
class MyArchRollbackLayer:
    def input_proj(self, x):
        """Batch all linear projections at T=N."""
        return self.module.in_proj(x)
    
    def recurrence_step(self, projected, state):
        """Single-token recurrence update.
        Must return (output, NEW_state) — never mutate state in place.
        """
        # Architecture-specific recurrence
        new_state = f(projected, state)
        return output, new_state
    
    def output_proj(self, rec_out):
        """Batch output projection at T=N."""
        return self.module.out_proj(rec_out)
    
    def get_state(self): return self._state
    def set_state(self, s): self._state = s
```

## 6. Dispatch Barrier Analysis

A critical discovery during optimization: the gap between theoretical bandwidth time (25.1ms) and actual GPU time (36.1ms) is dominated by **dispatch barriers between kernels**, not kernel execution overhead.

```
Matmul chain only:        22.4 ms  (weight reads, pipelined, no barriers)
Matmul + norm chain:      31.0 ms  (+8.6ms from norm dispatch barriers)
Full model:               36.1 ms  (+5.1ms from GDN, activations, reshapes)
```

Each dispatch barrier is a GPU pipeline stall: L2 cache coherency sync between the output of one kernel and the input of the next. Individually ~5-15us, but 500+ barriers per forward pass aggregate to ~14ms.

**Key insight**: reducing dispatch count within a small subsystem (e.g., fusing 2 GDN T=1 calls into 1 T=2 call) shows 0ms improvement because those barriers are hidden behind adjacent matmul work. But reducing barriers **between matmuls** (fusing rms_norm into the matmul kernel) saves real time because those barriers are in the critical path.

Measured: fusing rms_norm into quantized_matmul saves **2.0ms** (256 barrier eliminations) via custom kernel. Full MLX integration expected to save more.

## 7. Limitations and Future Work

### When Split-Recurrence Rollback Doesn't Help

1. **Pure attention models**: KV cache trimming is already O(1). No recurrent state to worry about.

2. **Very long draft sequences (K >> 10)**: Memory overhead from retained intermediate states grows linearly with K. At K=32 with a large model, this could be significant.

3. **Very low rejection rates**: If the draft model is highly accurate (< 5% rejection), the redo cost is already negligible. The split overhead may exceed the savings.

4. **Architectures with fused recurrence + matmul kernels**: If the recurrence and surrounding matmuls are fused into a single kernel (e.g., FlashLinearAttention), splitting requires breaking the fusion. The overhead may be larger than the 0.02ms per dispatch assumed here.

### Measured Performance Gaps

With Qwen3.5-27B on M4 Max (79% acceptance, split-recurrence rollback):

```
Achieved: 42.7 tok/s (1.45x over 29.5 baseline)
Ceiling:  58.8 tok/s (2.0x, with 100% acceptance and zero overhead)

Gap breakdown (0.55x lost):
  - MTP head + split dispatch overhead (4ms/step):     ~15%
  - 21% rejection (1 token instead of 2):              ~12%
  - eval sync + Python loop per step:                   ~5%
```

The ceiling is NOT 2x — it's O(1) per token. With N draft tokens and 100% acceptance, one T=N+1 forward reads weights once and produces N+1 tokens. DeltaNet recurrence adds only N × ~0.02ms per layer. At N=8: 34ms + 8ms = 42ms for 9 tokens = 4.7ms/tok = 213 tok/s. The practical limit is draft accuracy at depth.

### Future Directions

1. **Reduce step overhead**: Compile the MTP head into the main model's computation graph (eliminate 3ms separate dispatch). Fuse the GDN split into existing Metal kernels (eliminate 1ms). Target: 34.5ms per step → 52 tok/s.

2. **Eliminate Python loop overhead**: Move accept/reject decision to GPU (argmax + compare as lazy ops). Batch multiple steps into one eval. Target: save 1-2ms per step.

3. **Train better MTP heads**: Current Qwen3.5 MTP is a single transformer layer, giving 79% acceptance. Fine-tuning on the main model's own hidden states (distillation, no labeled data needed) should push to 85-90%. Multi-layer MTP heads would sustain accuracy at greater draft depth, enabling N>1 drafts. Training cost is minimal — freeze main model, train only 15 tensors (~800M params).

4. **Adaptive splitting**: Track acceptance rates at runtime. Switch between standard T=2 and split modes dynamically. Below 26.5% acceptance, standard T=2 with checkpoint/redo is cheaper.

5. **Compiler-level integration**: Teach `mx.compile` to automatically insert state capture points when it detects a split-recurrence pattern. This would eliminate the manual protocol and let the compiler optimize dispatch scheduling.

6. **Speculative state prediction**: Instead of rolling back and continuing from a checkpoint, predict what the state *would have been* if only the accepted tokens were processed. Likely intractable for nonlinear recurrences but could work for linear ones (Mamba with fixed A).

7. **Cross-architecture validation**: The technique is theoretically applicable to Mamba, RWKV, Griffin, and Jamba. Reference implementations and benchmarks on these architectures would validate the generalization claims.
