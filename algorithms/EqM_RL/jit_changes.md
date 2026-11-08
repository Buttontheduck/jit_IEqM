# torch.compile ("jit") in Aware: what changes in the code

This document compares the `jit` branch with `diag_offline` (commit `96e9333`), which is the same code without the compile option.

All numbers were measured on this machine on 2026-11-07: RTX 4090, PyTorch 2.12.1+cu130, the `hawk` env, HalfCheetah sizes (state 17, action 6), and Aware's 4 × 256 networks. Every Otter measurement uses the config's `num_actions: 1`.

---

## 1. Summary

| | |
|---|---|
| **Switch** | `compile: true` in `Aware/config.yaml`, or `compile=true` on the command line or in a sweep. The default is `false`. |
| **What gets compiled** | Only the EqM sampler's K-step loop, `Actor._descend`. `torch.compile(mode="reduce-overhead")` traces it once into one graph, fuses its small operations into Triton kernels, and replays the whole 20-step descent as one CUDA graph. |
| **Effect per sampler call** | The CPU starts 1 launch instead of 401. |
| **Second change, always on** | The 12 `.item()` calls per training step are gone. They made the CPU wait for the GPU 12 times per step. Statistics and losses are now logged as GPU tensors and averaged when the logger writes. |
| **Math** | Unchanged. Actions agree to about 1e-7. Training and scores agree: 50.1 ± 0.2 on halfcheetah-medium-v2 with and without compile. |
| **Speed** | Sampler: 8.6× faster at the training batch, 16× at the eval batch. One update: about 2×. A full 120k-step run: 54.2 → 24.3 min. |

---

## 2. Why Aware is slow without it

In normal ("eager") PyTorch, every tensor operation in Python does three things:
1. Python calls into PyTorch's C++ dispatcher.
2. The dispatcher picks a GPU kernel.
3. The CPU launches that kernel on the GPU, which costs about 5–20 µs of CPU time.

The GPU runs kernels in order while the CPU queues the next ones. For large networks, the GPU is always the slower side. Aware's networks are small: a 256 × 256 matrix multiply over 256 rows takes the GPU only a few microseconds. So **the GPU waits for the CPU**, and the time goes into launching kernels, not computing them.

The EqM sampler is the worst case. It is a loop of 20 small network calls:

```python
m = torch.zeros_like(x)
for _ in range(20):                                   # num_step = K = 20
    x_lookahead = x - self.step_size * m * self.moment  # 3 kernels: mul, mul, sub
    grad = self.model(x_lookahead, state)              # 15 kernels: cat, 5 Linear (2 cuBLAS kernels each), 4 Mish
    m = grad                                           # no kernel
    x = x - self.step_size * m                         # 2 kernels: mul, sub
```

That is 20 kernels per step, × 20 steps, + 1 for `zeros_like` = **401 kernel launches per sampler call**. The profiler confirms it: 200 cuBLAS matrix-multiply kernels, 80 Mish, 20 concatenations and 101 small arithmetic kernels. One call takes 5.8 ms, almost all of it launch time.

One Aware training step (awac critic) calls the sampler twice:
1. for the next action a′ in the critic target;
2. for the action that gives V(s) in the advantage weights.

That is 11.6 ms of the 20.7 ms an update takes.

**JAX and PyTorch, side by side:**

| JAX | PyTorch | Used here |
|---|---|---|
| `jax.jit(f)` | `torch.compile(f)` | yes |
| `lax.scan` over a fixed number of steps | a plain Python `for` loop over `range(K)`, which `torch.compile` unrolls | yes, the loop is unrolled 20 times |
| XLA fuses operations | Inductor fuses operations into Triton kernels | yes |
| one compiled program per call | CUDA graphs (`mode="reduce-overhead"`): the recorded kernel sequence is replayed with one launch | yes |

---

## 3. Where the sampler runs

All callers reach the compiled loop through `Actor._implicit`, which calls `self._descend_fn`:

```
main.py training loop
 └─ Otter.update(batch)
     ├─ _critic_loss  ─► self._actor(next_states) ─► Actor.forward ─► _sample_actions_for_training ─► _implicit   B = 256      (critic: awac)
     └─ _update_actor ─► _actor_loss ─► _offline_weights
                          └─ self._actor(states, num_actions) ─► ... ─► _implicit                               B = 256 × num_actions (critic: awac)
main.py eval_actor    ─► actor.sample(states)  ─► _implicit                                                    B = unfinished episodes (≤ 10)
main.py diagnostics   ─► actor.diagnostics(...) ─► _implicit                                                   B = 256 states × 8 = 2048
```

- With `agent.critic: iql`, training never samples from the actor, because IQL's critic is in-sample. Compiling then speeds up only evals and diagnostics, so the gain is much smaller.
- Not compiled:
  - The other samplers: `_implicit_stop` (used when `actor.early_stop` is set), `_implicit_OOD*`, `_implicit_langevin`, `_explicit`, and `sample2`.
  - The losses, backward passes, Adam steps and soft target updates.
  - The other scripts, `fit_actor.py`, `train_critic.py` and `finetune_actor.py`. They build their own `Actor` and never call `compile_sampler()`.

---

## 4. The code changes, file by file

### 4.1 `Aware/actor_models.py`: the sampler loop becomes compilable

**Before.** The loop sat inside `_implicit`, mixed with switching the network between eval and train mode:

```python
    def _implicit(self, x , state):

        is_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            if self.opt_type == "gd":
                for _ in range(self.num_step):
                    grad = self.model(x,state)
                    x = x - self.step_size * grad

            elif self.opt_type == "nag":
                m = torch.zeros_like(x)
                for _ in range(self.num_step):
                    x_lookahead = x - self.step_size * m * self.moment
                    grad = self.model(x_lookahead,state)
                    m = grad
                    x = x - self.step_size * m
            else:
                raise ValueError(f"\n Action Gradient optimizer must be 'gd' or 'nag', got '{self.opt_type}' \n ")
        
        if is_training:
            self.model.train()
        return x
```

**After, part 1: two new attributes at the end of `Actor.__init__`, and a new method.**

```python
        # The K-step descent of _implicit. compile_sampler() swaps in a torch.compile'd version.
        self._descend_fn = self._descend
        self._compiled = False

    def compile_sampler(self, mode: str = "reduce-overhead"):
        """Compile the K-step descent of _implicit with torch.compile, PyTorch's counterpart of
        jax.jit over a lax.scan loop: the fixed-K loop is unrolled and its ops fused, and mode
        'reduce-overhead' replays the whole descent as one CUDA graph. Only _implicit uses it (the
        training actions, the eval sampler and diagnostics()); the other samplers stay eager.
        The first calls compile (about a minute); each new batch size records one more graph."""
        self._descend_fn = torch.compile(self._descend, mode=mode)
        self._compiled = True
        return self
```

| Line | What it does |
|---|---|
| `self._descend_fn = self._descend` | By default, the function `_implicit` calls is the plain Python method, so nothing changes. |
| `self._compiled = False` | Remembers whether outputs need copying (see `.clone()` below). |
| `torch.compile(self._descend, mode=mode)` | Returns a wrapper. **Nothing is compiled at this point.** Compiling happens on the first call, when the input shapes are known. |
| `return self` | Allows `actor.compile_sampler()` inside an expression, as in `check_compile.py`. |

**After, part 2: `_implicit` keeps the mode switching and calls the loop through `_descend_fn`.**

```python
    def _implicit(self, x , state):

        is_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            x = self._descend_fn(x, state)
            if self._compiled:
                x = x.clone()   # a CUDA-graph output is overwritten by the graph's next replay

        if is_training:
            self.model.train()
        return x
```

| Line | Why |
|---|---|
| `self.model.eval()` / `.train()` stay here, outside the compiled part | These switches change the module's state. Inside a compiled function they would add a guard and complicate the graph. Outside, the compiled function always sees an eval-mode network. Aware's MLP has no dropout or batch norm, so the mode doesn't change the math either way. |
| `with torch.no_grad():` stays here | The compiler records that it is called without gradients, so it builds an inference-only graph with no backward pass. |
| `x = self._descend_fn(x, state)` | The only call site of the loop: eager or compiled. |
| `x = x.clone()` when compiled | In CUDA-graph mode, the graph's output lives in memory the graph owns, and the next replay writes into the same memory. Aware calls the sampler twice per update, so without the copy the first result would be overwritten by the second call. PyTorch would raise an error rather than silently return wrong numbers, but the copy avoids the problem. It costs one tiny copy kernel (256 × 6 floats). |

**After, part 3: the loop, moved unchanged into its own method.**

```python
    def _descend(self, x, state):
        """The K steps of _implicit, without the train/eval switching (so torch.compile sees a
        pure function). Called under no_grad."""
        if self.opt_type == "gd":
            for _ in range(self.num_step):
                grad = self.model(x,state)
                x = x - self.step_size * grad

        elif self.opt_type == "nag":
            m = torch.zeros_like(x)
            for _ in range(self.num_step):
                x_lookahead = x - self.step_size * m * self.moment
                grad = self.model(x_lookahead,state)
                m = grad
                x = x - self.step_size * m
        else:
            raise ValueError(f"\n Action Gradient optimizer must be 'gd' or 'nag', got '{self.opt_type}' \n ")
        return x
```

The lines are the same as in the old `_implicit`. A **pure function** here means tensors in, tensors out, with no side effects such as mode switching, logging or `.item()`. That is what the compiler can trace into one graph without breaks.

### 4.2 `Aware/agents.py`: no more GPU→CPU waits during a step

**Before, in `Otter._offline_weights`:**

```python
                weights = torch.clamp_max(input=torch.exp(adv/self._temperature), max=self._exp_adv_max) 
                gl.log("weighting/q_mean",            q.mean().item())
                gl.log("weighting/v_mean",            v.mean().item())
                gl.log("weighting/adv_mean",          adv.mean().item())
                gl.log("weighting/adv_std",           adv.std(unbiased=False).item())
                gl.log("weighting/weights_mean",      weights.mean().item())
                gl.log("weighting/weights_max",       weights.max().item())
                gl.log("weighting/weights_min",       weights.min().item())
                gl.log("weighting/weights_var",       weights.var(unbiased=False).item())
                gl.log("weighting/weights_median",    weights.median().item())
                gl.log("weighting/weights_clip_frac", (weights >= self._exp_adv_max).float().mean().item())
```

**After:**

```python
                weights = torch.clamp_max(input=torch.exp(adv/self._temperature), max=self._exp_adv_max)
                # GPU tensors, no .item(): each .item() makes the CPU wait for the GPU. The logger
                # turns them into numbers once per log_frequency steps (logger.py, dump).
                gl.log("weighting/q_mean",            q.mean())
                gl.log("weighting/v_mean",            v.mean())
                gl.log("weighting/adv_mean",          adv.mean())
                gl.log("weighting/adv_std",           adv.std(unbiased=False))
                gl.log("weighting/weights_mean",      weights.mean())
                gl.log("weighting/weights_max",       weights.max())
                gl.log("weighting/weights_min",       weights.min())
                gl.log("weighting/weights_var",       weights.var(unbiased=False))
                gl.log("weighting/weights_median",    weights.median())
                gl.log("weighting/weights_clip_frac", (weights >= self._exp_adv_max).float().mean())
```

**Before, at the end of `_update_critic`, `_update_value` and `_update_actor`:**

```python
        return loss.item()
```

**After:**

```python
        return loss.detach()   # no .item() sync; logged as a tensor
```

- The same statistics are computed. Only *when* they become Python numbers changes.
- `.detach()` drops the autograd history, so the logger keeps no graph alive. It does not copy the data.
- `Otter.update` still returns `{"critic_loss": ..., "actor_loss": ...}`; the values are now 0-dim GPU tensors. `main.py` passes them straight to `gl.log`. No other script calls `Otter.update`, `_update_*` or `_offline_weights` (checked with grep), so no other script is affected.

Why this matters is explained in section 7.

### 4.3 `Aware/logger.py`: the logger accepts GPU tensors

**Before:**

```python
import wandb
from collections import defaultdict
import numpy as np
...
        aggregated_metrics = {
            key: np.mean(values) for key, values in self.buffer.items()
        }
```

**After:**

```python
import wandb
from collections import defaultdict
import numpy as np
import torch


def _mean(values) -> float:
    """Mean of the buffered values of one key: Python/NumPy numbers, or 0-dim GPU tensors (logged
    without .item() to avoid a GPU sync per value; here they are read back once, in float64 as
    np.mean would)."""
    if isinstance(values[0], torch.Tensor):
        return torch.stack([v.detach() for v in values]).double().mean().item()
    return float(np.mean(values))
...
        aggregated_metrics = {
            key: _mean(values) for key, values in self.buffer.items()
        }
```

- `np.mean` cannot average CUDA tensors, so `_mean` handles both kinds of value.
- Tensors (training statistics): `torch.stack` makes one 1-D tensor of the up to 1000 buffered values, `.double()` averages in float64 as `np.mean` did, and `.item()` reads the result back once. That is **one GPU→CPU wait per metric per `log_frequency` steps**, instead of one per metric per step.
- Plain numbers (the eval scores, which are NumPy floats) still go through `np.mean`.
- Checked in `check_compile.py`, check 4: for the same 1000 values, both paths give the same mean to 12 digits.

### 4.4 `Aware/config.yaml`: two new keys

```yaml
device: cuda # or 'mps' (Mac) / 'cpu'
compile: false                  # true = torch.compile the EqM sampler's K-step loop (Actor.compile_sampler)
compile_mode: reduce-overhead   # 'reduce-overhead' = also replay it as one CUDA graph; 'default' = fusion only
```

- `compile_mode: default` fuses the operations but launches each fused kernel separately. That is about 300 launches per call instead of 401, so most of the launch cost stays.
- `reduce-overhead` adds CUDA graphs, which brings it down to 1 launch. Use `reduce-overhead`.

### 4.5 `Aware/main.py`: turn it on after building the actor

**Before:**

```python
        early_stop    = cfg.actor.early_stop
        )
    
    # IQL critic option (agent.critic: iql): a V(s) network, same size as the critics
```

**After:**

```python
        early_stop    = cfg.actor.early_stop
        )
    if cfg.get("compile", False):
        actor.compile_sampler(mode=cfg.get("compile_mode", "reduce-overhead"))

    # IQL critic option (agent.critic: iql): a V(s) network, same size as the critics
```

`cfg.get(..., default)` means an old config without these keys still runs, uncompiled. The same `actor` object is used by `Otter`, by `eval_actor` and by `diagnostics`, so all three get the compiled loop.

### 4.6 New files

| File | What it is |
|---|---|
| `Aware/check_compile.py` | Four checks on random data or a trained checkpoint: (1) sampler eager vs compiled at B = 256, 10, 3 and 2048; (2) 200 training steps from identical agents; (3) speed; (4) logger means. Run `python check_compile.py [--checkpoint <file>]`. **Fix in this commit:** it now builds `Otter` with `num_actions=1`, as `config.yaml` does. Before, it used `Otter`'s default of 10, so its update timings drew V(s) from 2,560 samples per step instead of 256. Its correctness results held either way. |
| `Aware/rollout_check.py` | Plays real episodes with a trained policy. (A) It follows the compiled actions and compares them with the eager action at every step. (B) It counts failed and flipped episodes for each sampler. |
| `sweeps/jit_check.yaml` | `compile` false vs true × seeds 1–3, halfcheetah-medium-v2, 120k steps, no checkpoints. |

---

## 5. What happens at run time with `compile: true`

**At start-up**, `compile_sampler()` only wraps `_descend`. Nothing is compiled yet.

**At the first update step (B = 256)**, the critic loss calls `_implicit`, which calls `_descend_fn(x, state)` for the first time. Five stages run:

1. **TorchDynamo** (graph capture). It runs the Python bytecode of `_descend` symbolically and records every tensor operation into one graph (an FX graph).
   - `if self.opt_type == "gd"` / `"nag"` is decided while tracing, so only the NAG branch ends up in the graph.
   - `for _ in range(self.num_step)` with `num_step = 20` is **unrolled**: the graph holds 20 copies of the step.
   - `self.step_size = 0.01` and `self.moment = 0.3` are baked in as constants.
   - `self.model` is traced through: `torch.concat` and the 5 `Linear` and 4 `Mish` layers become graph operations, and the weights become graph inputs, so optimizer updates are always picked up.
   - Measured: **1 graph, 0 graph breaks, 301 operations** (20 steps × 15 operations + `zeros_like`).
   - It also stores **guards**: conditions under which this graph may be reused. They cover the attributes it used (opt_type, num_step, step_size, moment), gradient mode (no_grad), and the input shapes, dtypes and devices (x of shape [256, 6], float32, cuda).
2. **AOTAutograd.** Because the call runs under `no_grad`, it builds only the forward graph. There is no backward pass to prepare.
3. **TorchInductor** (code generation).
   - Matrix multiplies stay with cuBLAS. The generated code calls `extern_kernels.mm`, 100 times, which is 5 per step.
   - Everything elementwise around them is fused into **Triton** kernels: bias add + Mish, and the NAG update + lookahead + concat.
   - The generated Python module and its kernels are cached on disk in `/tmp/torchinductor_root/`. The first compile on a fresh cache took 26 s; later runs took 7–8 s.
4. **CUDA graph recording** (`reduce-overhead`).
   - After a warm-up call, PyTorch records the whole kernel sequence of one call as a CUDA graph: 309 kernels.
   - From then on, each call copies the new `x` and `state` into the graph's fixed input buffers (1 kernel) and replays the graph (1 launch). The output is in graph-owned memory, which is why `_implicit` clones it.
5. **The `.clone()`** copies the result into normal memory (1 kernel).

**At the first eval (B = 10)**, the shape guard fails and Dynamo recompiles, this time with the batch size as a variable. The log line is:

```
Recompiling function _descend in .../Aware/actor_models.py:292
    triggered by the following guard failure(s):
    - 0/0: tensor 'x' size mismatch at index 0. expected 256, actual 10
```

After that, every batch size reuses this second graph: 10 and fewer at eval, 2048 in diagnostics, 256 in training. No more recompiles happen; this was checked with `TORCH_LOGS=recompiles` over update → eval (10) → diagnostics (2048) → eval (3) → more updates. CUDA graphs keep one recording per batch size they have seen. **So a run compiles twice, then only replays.**

**Recompiles and fallback:**
- Changing `num_step`, `step_size`, `moment` or `opt_type` on a compiled actor during a run breaks a guard and triggers one recompile. `fit_actor.py`'s `evaluate_at` does this, but it doesn't compile.
- After 8 recompiles of one function, Dynamo stops compiling it and runs it eagerly, with a warning.

---

## 6. What the compiler produced

This is the code Inductor generated for `_descend` at B = 256, logged with `TORCH_LOGS=output_code` (variable names as generated; comments mine).

**Kernels per sampler step:**

| | Eager | Compiled |
|---|---|---|
| `torch.concat([x_lookahead, state])` | 1 kernel | fused into the update kernel of the step before |
| 5 × `Linear` | 5 × 2 cuBLAS kernels (`addmm`, bias included) | 5 × 2 cuBLAS kernels (`mm`, no bias) |
| bias add + `Mish`, 4 hidden layers | 4 Mish kernels (bias is inside `addmm`) | 4 fused Triton kernels (bias + Mish, in place) |
| NAG arithmetic: lookahead (mul, mul, sub) + update (mul, sub), head bias | 5 kernels | 1 fused Triton kernel (head bias + update + lookahead + concat) |
| **per step** | **20 kernels, 20 CPU launches** | **about 15 kernels, all inside one CUDA graph** |
| **per call (20 steps)** | **401 kernels, 401 CPU launches** | **309 kernels, 1 CPU launch** (200 cuBLAS + 107 Triton + 1 input copy + 1 clone) |

**The generated step sequence** (start of `partition_0`; this whole function is what the CUDA graph records):

```python
# step 1 input: cat(x, state) -> buf2 (256 x 23). m = 0 at step 1, so the lookahead is x itself:
# the compiler folded `x - 0.01 * 0 * 0.3` away; this kernel only copies x and state side by side.
triton_for_fused_0.run(arg0_1, arg1_1, buf0, buf1, ...)                # arg0_1 = x, arg1_1 = state
extern_kernels.mm(buf2, <W1 transposed>, out=buf3)                       # layer 1: cuBLAS, no bias
triton_poi_fused_addmm_mish_1.run(buf4, arg3_1, 65536, ...)             # + b1, Mish, in place (256*256 values)
extern_kernels.mm(buf4, <W2 transposed>, out=buf5)                       # layer 2
triton_poi_fused_addmm_mish_1.run(buf6, arg5_1, 65536, ...)             # + b2, Mish
extern_kernels.mm(buf6, <W3 transposed>, out=buf7)                       # layer 3
triton_poi_fused_addmm_mish_1.run(buf8, arg7_1, 65536, ...)             # + b3, Mish
extern_kernels.mm(buf8, <W4 transposed>, out=buf9)                       # layer 4
triton_poi_fused_addmm_mish_1.run(buf10, arg9_1, 65536, ...)            # + b4, Mish
extern_kernels.mm(buf10, <W_head transposed>, out=buf11)                 # head: 256 x 6, bias not added yet
# one kernel: m1 = buf11 + b_head; x1 = x - 0.01*m1; lookahead = x1 - 0.01*m1*0.3; write cat(lookahead, state)
triton_poi_fused_addmm_cat_mul_sub_2.run(arg0_1, arg11_1, buf11, arg1_1, buf12, 5888, ...)
extern_kernels.mm(buf12, <W1 transposed>, out=buf13)                     # step 2, layer 1 ... and so on
```

`buf4 = buf3; del buf3  # reuse` and similar lines in the real file show that Inductor also plans memory: each Mish writes in place into the matrix multiply's output, and buffers are reused across layers.

**Generated kernel 1: bias add + Mish, fused** (`triton_poi_fused_addmm_mish_1`, one GPU thread block per chunk of the 256 × 256 activations):

```python
def triton_poi_fused_addmm_mish_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536                                   # 256 rows x 256 units
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = (xindex % 256)                              # which unit (column): picks the bias
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')   # bias b[j]
    tmp1 = tl.load(in_out_ptr0 + (x2), None)                             # (h @ W^T)[i, j] from cuBLAS
    tmp2 = tmp0 + tmp1                                                    # z = h W^T + b
    tmp3 = tl.full([1], 20.0, tl.float32)
    tmp4 = tmp2 > tmp3
    tmp5 = libdevice.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)                # softplus(z) = log(1 + e^z), or z itself above 20 (as PyTorch)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp2 * tmp8                               # Mish(z) = z * tanh(softplus(z))
    tl.store(in_out_ptr0 + (x2), tmp9, None)         # written back in place
```

In eager mode, the 65,536 values are written by `addmm`, read again by the Mish kernel, and written again. Here they are read once and written once, by one kernel.

**Generated kernel 2: the NAG update, lookahead and concat, fused** (`triton_poi_fused_addmm_cat_mul_sub_3`; it builds the network input for step 3):

```python
def triton_poi_fused_addmm_cat_mul_sub_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5888                                    # 256 rows x 23 columns (6 action + 17 state)
    ...
    x0 = (xindex % 23)                               # column
    x1 = xindex // 23                                # row
    tmp4 = tmp0 < tmp3                               # column < 6: an action column
    tmp5 = tl.load(in_ptr0 + (6*x1 + (x0)), ...)     # x0: the sampler's starting noise
    tmp6 = tl.load(in_ptr1 + (x0), ...)              # head bias
    tmp7 = tl.load(in_ptr2 + (6*x1 + (x0)), ...)     # head output of step 1 (from cuBLAS)
    tmp8 = tmp6 + tmp7                               # m1 = field value at step 1
    tmp9 = tl.full([1], 0.01, tl.float32)            # step_size, baked in
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 - tmp10                             # x1 = x0 - 0.01 m1
    tmp12 = tl.load(in_ptr3 + (6*x1 + (x0)), ...)    # head output of step 2
    tmp13 = tmp6 + tmp12                             # m2
    tmp14 = tmp13 * tmp9
    tmp15 = tmp11 - tmp14                            # x2 = x1 - 0.01 m2
    tmp16 = tl.full([1], 0.3, tl.float32)            # moment, baked in
    tmp17 = tmp14 * tmp16
    tmp18 = tmp15 - tmp17                            # lookahead = x2 - 0.01 * m2 * 0.3
    ...
    tmp24 = tl.load(in_ptr4 + (17*x1 + ((-6) + x0)), ...)   # state, for columns 6..22
    tmp25 = tl.where(tmp4, tmp20, tmp24)             # concat: action columns from the lookahead, then the state
    tl.store(out_ptr0 + (x2), tmp25, xmask)          # the 256 x 23 input of step 3's first layer
```

This one kernel replaces 6 eager kernels: the update's multiply and subtract, the lookahead's two multiplies and subtract, and `torch.concat`. It also adds the head bias, which eager mode adds inside cuBLAS's `addmm`. Note also that it never stored x1 or x2. It keeps the small 256 × 6 head outputs and **recomputes** the running x from them. Recomputing is cheaper than writing and reading x back between kernels. This changes the order of floating-point operations slightly, which is one reason the compiled actions differ from eager ones by about 1e-7.

---

## 7. The `.item()` change: what a GPU→CPU wait costs

The CPU and GPU work asynchronously: the CPU puts kernels in a queue and moves on. `.item()` needs the *value*, so the CPU must stop until the GPU has finished everything queued so far, then copy one number back. During that time the CPU queues nothing, and right after it the GPU has nothing queued:

```
with .item() (before)            CPU: [queue A][queue B]...wait......[queue C][queue D]...wait......
                                 GPU:   [A][B]                     idle [C][D]                   idle
without .item() (after)          CPU: [queue A][queue B][queue C][queue D][queue E] ...
                                 GPU:   [A][B][C][D][E] ...                  (never waits for the CPU)
```

Measured over one `Otter.update`:

| Code | GPU→CPU waits per step |
|---|---|
| `diag_offline` | 12 (10 weighting statistics + 2 losses) |
| `jit`, either setting | **0** |

`aten::_local_scalar_dense` still shows up 60 times per step. Those are Adam reading its step counters, which live on the CPU (`capturable=False`), so the CPU never waits for the GPU there. They are the same in both versions.

**Alone, removing the waits barely changes the eager speed** (20.7 ms per update before, about 20.4 ms after). The eager CPU is the bottleneck anyway, so the GPU is idle at those points regardless. It matters once the sampler is compiled: then the CPU can run ahead, and each wait would throw that lead away.

---

## 8. Measurements

### 8.1 Kernels, launches and waits

One sampler call (B = 256) and one `Otter.update` (awac critic, `num_actions = 1`):

| | `diag_offline` | `jit`, `compile: false` | `jit`, `compile: true` |
|---|---|---|---|
| sampler: GPU kernels | 401 | 401 | 309 |
| sampler: launches from the CPU | 401 | 401 | **1** (a CUDA-graph launch) |
| update: GPU kernels | 1,161 | 1,161 | 975 |
| update: launches from the CPU | 1,158 | 1,158 | **358** + 2 CUDA-graph launches |
| update: GPU→CPU waits | 12 | **0** | **0** |

The remaining 358 launches per update are the parts that are not compiled: the critic and actor losses, three backward passes, three Adam steps, and the soft target update.

### 8.2 Speed

| What | Eager | Compiled | Speedup |
|---|---|---|---|
| one sampler call, B = 256 (training) | 5.84 ms | 0.68 ms | 8.6× |
| one sampler call, B = 10 (eval) | 5.37 ms | 0.33 ms | 16× |
| one `Otter.update` | 20.7 ms | 8–12.5 ms * | about 2× |
| `main.py`, 3k steps alone on the GPU | 51 steps/s | 115 steps/s | 2.25× |
| `jit_check` sweep, 120k steps, 6 runs sharing the GPU | 54.2 min per run | 24.3 min per run | 2.2× |

\* The machine is shared, with a load average of 20–30 from other users of the host. What remains after compiling is mostly CPU work, so repeated measurements of the compiled update range from 8 to 12.5 ms.

### 8.3 Correctness

**`check_compile.py`, sampler, same noise and states** (largest |compiled − eager| over all actions):

| Batch size | Fresh networks | Trained (Aware 1M T = 4 seed 1, medium-expert) |
|---|---|---|
| 256 | 2.4e-7 | 2.4e-7 |
| 10 | 0 | 1.2e-7 |
| 3 | 3.7e-9 | 1.2e-7 |
| 2048 | 1.2e-7 | 3.6e-7 |

**`check_compile.py`, 200 identical training steps** (relative difference):

| | Fresh | Trained |
|---|---|---|
| losses (worst step) | 1.9e-7 | 1.3e-4 |
| EqM field weights | 8.8e-8 | 5.5e-6 |
| critic 1 / critic 2 weights | 8.2e-8 / 8.1e-8 | 9.2e-8 / 7.8e-8 |

With the trained networks, the difference grows during training, because the advantage weights `exp(A/T)` turn a 1e-7 difference in V(s) into larger differences in the loss. It stays far below the 1e-3 tolerance. Over thousands of steps, the two runs of one seed drift apart the way any two runs with slightly different rounding do, so they are compared by scores, not step by step.

**`jit_check` sweep** (halfcheetah-medium-v2, 120k steps; `sweep_runs/jit_check_20261107-214731`; mean of the evals in the last 20k steps):

| Seed | Score, eager / compiled | Q at dataset actions | Critic loss | Share of weights at the cap |
|---|---|---|---|---|
| 1 | 50.2 / 50.2 | 553.6 / 553.5 | 26.0 / 26.1 | 0.0018 / 0.0018 |
| 2 | 50.2 / 50.3 | 556.2 / 556.6 | 27.0 / 27.0 | 0.0018 / 0.0018 |
| 3 | 49.8 / 49.8 | 545.9 / 545.3 | 27.6 / 27.3 | 0.0018 / 0.0017 |
| mean | 50.1 ± 0.2 / 50.1 ± 0.2 | | | |

Compiled seed 3 had three single failed eval episodes, at 29k, 35k and 84k: one of 10 episodes ended with a return of 167, 1113 and 973. Over the run, eager had 3 and compiled 7 such evals out of 333 each (3,330 episodes per setting). **`rollout_check.py` shows they come from the policy, not from compile:**

| Test | Medium-expert, 100k steps | Medium-replay, 150k steps |
|---|---|---|
| A. largest action difference at any step of 50 × 1000 steps | 7.8e-7 | 4.2e-7 |
| B. failed episodes of 200, eager / compiled | 22 / 22 | 0 / 0 |
| B. flipped episodes of 200, eager / compiled | 62 / 63 | 1 / 0 |

---

## 9. What did not change

- **The math.**
  - The EqM loss, the advantage weights, the critic's TD target and the sampler's formula are the same lines of code.
  - Everything stays float32. TF32 matrix multiplies are *not* enabled. PyTorch's compiler suggests them, but they would change the numbers.
- **Random numbers.** The sampler's starting noise is drawn *before* the compiled function is called (`torch.randn` in `_sample_actions_for_training`, `sample` and `diagnostics`). The compiled code draws no random numbers, so a run's random stream is the same with and without compile.
- **`compile: false`** runs exactly the old code path, except that statistics are logged as tensors. Their logged values are identical.
- **The other scripts** (`fit_actor.py`, `train_critic.py`, `finetune_actor.py`, `make_weights.py`, `diagnostics/*`) and **`Aware_Minari/`** are unchanged. They don't call `compile_sampler()`.

---

## 10. Rules and pitfalls

1. **Don't keep a compiled output without copying it.** `_implicit` already clones. If you call `_descend_fn` directly, copy its result before the next call.
2. **Keep shapes stable during training.** Each new batch size records another CUDA graph. That is fine for a handful of sizes, as in Aware: 256, eval sizes up to 10, and 2048.
3. **Don't change sampler settings in the middle of a run** without expecting a recompile of a few seconds. Examples: `actor.num_step = 10`, or a different `step_size`.
4. **The first steps of a run are slower**, because of compiling (7–30 s, two compiles per run). Short smoke runs gain less.
5. **No `.item()`, `print(tensor)` or `if tensor > ...:` inside `_descend`.** Each would split the graph or force a GPU wait. Logging belongs outside the compiled function.
6. **Early stopping (`actor.early_stop`) uses `_implicit_stop`, which is not compiled.** Its `if not active_mask.any(): break` depends on the data and would split the graph.
7. **With `agent.critic: iql`** training doesn't use the sampler, so expect only faster evals.

---

## 11. What is left

After compiling, an update still takes 8–12 ms, spent on the 358 remaining launches. Options, from easiest to hardest:

| Option | Expected effect |
|---|---|
| `torch.optim.Adam(..., fused=True)` | Each Adam step becomes one kernel instead of several. |
| Soft target update with `torch._foreach_lerp_` | 2 kernels instead of about 80 (20 parameters × 4 operations). `train_critic.py` already does this. |
| `torch.compile` the critic and actor loss functions | Fuses the forward passes and the backward passes that autograd generates. |
| `torch.compile` a whole training step, including the optimizer steps | Largest gain, but the most work to get without graph breaks. |

None of these is implemented yet.

---

## 12. Files and commands

```bash
git checkout jit                                     # this code; diag_offline = without it
git diff diag_offline jit -- algorithms/EqM_RL/Aware # the code changes above

cd algorithms/EqM_RL/Aware
python check_compile.py                              # equivalence + speed on fresh networks
python check_compile.py --checkpoint <run>/checkpoint_N.pt
python rollout_check.py <run_dir> checkpoint_N.pt    # real episodes (MuJoCo on LD_LIBRARY_PATH)
python main.py compile=true                          # train with the compiled sampler

cd .. && python run_sweep.py sweeps/jit_check.yaml   # eager vs compiled, 3 seeds

TORCH_LOGS=recompiles python main.py compile=true    # print every recompile and its reason
TORCH_LOGS=output_code python main.py compile=true   # print the generated code (section 6)
```
