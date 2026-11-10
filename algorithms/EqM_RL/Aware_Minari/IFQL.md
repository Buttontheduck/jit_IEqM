# IFQL: how it trains, how its critic works, and how the critic is used

Written 2026-11-09, and corrected the same day after a second check against the sources below. IFQL ("implicit flow
Q-learning") is one of the baselines in the FQL paper (Park, Li, Levine 2025). It is the best-performing method in that paper on
our main task, `humanoidmaze-medium-navigate-singletask-v0` (69 % success on task 1), which is why it is written up here.

**Sources** (everything below was checked against these; anything that could not be checked is marked as such):
- FQL paper: `papers/FQL.pdf` (arXiv 2502.02538v2). Main text Section 4.1 (rejection sampling, Equation 11) and Section 5.1;
  Appendix B (implementation details), C (ablations), E.1 (environments and rewards), E.2 (how each baseline was run);
  Tables 2, 3, 4, 5, 6 and 7.
- Official code of the FQL paper, <https://github.com/seohongpark/fql> (MIT license): `agents/ifql.py`, `agents/iql.py`,
  `utils/networks.py`, `utils/flax_utils.py`, `utils/datasets.py`, `envs/env_utils.py`, `utils/evaluation.py`, `main.py`, and the
  exact IFQL command for humanoidmaze in `README.md`.
- IDQL paper (Hansen-Estruch et al. 2023, arXiv 2304.10573), Sections 4.1, 4.2 and 6.3. IFQL is "the flow counterpart of IDQL"
  (FQL paper, Section 5.1); the IDQL paper defines the critic-based weights described in Section 9.3.
- OGBench 1.2.2 (in the `ogbench` conda env): `ogbench/utils.py`, `ogbench/relabel_utils.py`, `ogbench/locomaze/maze.py`, and
  the humanoidmaze-medium dataset itself. Used for how the dataset's rewards and masks are set (Section 2).
- Our own runs (Sections 9.4 and 11): `sweeps/exp13_ogbench_aware.yaml`, `sweeps/exp14_ogbench_*.yaml`,
  `sweeps/exp15_ogbench_bestofn.yaml`, `sweeps/exp18_ogbench_aware_v.yaml`, the summaries in `sweep_runs/`, and `Aware/agents.py`.

## 1. Summary

1. IFQL trains three things **separately**, all from the first step, on the same batches of dataset rows:
   a value network V, a critic Q (two copies), and a flow network that copies the dataset's actions (behaviour cloning).
2. The critic is IQL's critic, unchanged. It is trained only at the dataset's own actions: during training, it never scores
   an action that a policy invented. It scores the flow network's proposals only when an action is chosen (point 4).
3. The flow network never sees Q or V. Every dataset row has weight 1.
4. The critic and the flow network are combined only when an action is chosen: the flow network proposes 32 actions,
   and the critic picks the one with the highest score.
5. So IFQL builds **no weights** from the critic during training. At decision time, the "weight" is 1 for the best candidate
   and 0 for the other 31.

## 2. Words used in this file

| word | exact meaning |
|---|---|
| state s | the numbers describing the situation at one time step; in humanoidmaze, 69 numbers (robot body and position) |
| action a | the command sent to the robot at one step; in humanoidmaze, 21 numbers (one per motor), each between −1 and 1 |
| reward r | always −1 or 0 (FQL appendix E.1: "rewards are always −1 or 0"). **In the dataset:** 0 on the rows whose state s is already inside the goal circle, −1 on all other rows, including the row whose next state s′ first enters the goal circle. **In the evaluation environment:** 0 on the step that reaches the goal, −1 on every step before it. See the note below this table |
| next state s′ | the state one step after s |
| mask | **in the dataset:** 0 on the rows whose state s is inside the goal circle (the same rows that have reward 0), 1 on all other rows. It stands in for the end of the episode: in the evaluation environment, the episode ends on the step that reaches the goal (appendix E.1) |
| dataset | 2,000,000 rows (s, a, r, s′, mask) recorded in advance (`humanoidmaze-medium-navigate-v0`, the training split). IFQL collects no new data in offline training |
| batch | 256 rows drawn uniformly at random from the dataset (with replacement) |
| discount γ | how much a later reward counts: a reward k steps later counts γᵏ times as much. Humanoidmaze: γ = 0.995 |
| return | the sum of all future rewards, each multiplied by its discount |
| floor of Q | the return of never reaching the goal: −1 / (1 − γ) = −200 for γ = 0.995. Every true Q value lies between −200 and 0 |
| Q(s, a), the critic | a network that takes a state and an action and outputs an estimate of the return after taking that action |
| Q1, Q2 | two critic networks with the same shape and different random starting weights; "min Q" = the smaller of their outputs |
| V(s), the value network | a network that takes only a state and outputs one number: how good that state is (defined exactly in Section 4, step 2) |
| target copy | a second set of weights that follows the trained weights slowly (Section 4, step 6). In IFQL, only Q1 and Q2 have one, and only V's training target is computed with it, so that this target does not jump around. Q's own target uses V, which has no copy |
| flow network v(s, x, t) | a network that takes a state s, a point x in action space (21 numbers) and a time t between 0 and 1, and outputs a direction (21 numbers) saying how to move x |
| behaviour cloning (BC) | training a network to reproduce the dataset's actions, with no notion of good or bad |
| expectile loss with τ = 0.9 | a squared error that counts with weight 0.9 when the prediction is too low and 0.1 when it is too high |
| GELU | a smooth curve applied to each number after a linear layer (like ReLU, without the sharp corner) |
| LayerNorm | rescales the 512 numbers of a layer to mean 0 and spread 1, then applies a learned scale and shift |
| IQL | Implicit Q-Learning (Kostrikov et al. 2022), the method IFQL takes its V and Q training from |
| IDQL | Implicit Diffusion Q-Learning (Hansen-Estruch et al. 2023): IQL's critic plus a diffusion BC model; IFQL is the same idea with a flow BC model |

**Note on the goal and the reward timing.** The goal circle is the set of x-y positions within 0.5 of the task's goal position
(task 1: (20, 20)). The dataset and the evaluation environment give reward 0 one step apart:
- OGBench sets the dataset's rewards and masks from the position in each row's state s, not in s′ (`relabel_dataset` in
  `ogbench/relabel_utils.py` reads `qpos`, and `ogbench/utils.py` aligns `qpos` with s).
- The evaluation environment checks for the goal after each step.
- Checked on the task 1 data: all 3,921 of the 2,000,000 rows with reward 0 have s inside the goal circle. All 209 rows that
  enter the circle (s outside, s′ inside) have reward −1 and mask 1.
- Effect: for every state s outside the goal circle, Q with the dataset's rewards = γ · (Q with the evaluation rewards) − 1.
  So the values learned from the data are lower by between 0 and 1: by 1 right before the goal, by almost 0 far from it.
  Both rank actions the same way.

## 3. The networks

| network | input | layers | output | copies |
|---|---|---|---|---|
| **Q1, Q2** (critic) | state and action joined: 69 + 21 = 90 numbers | 4 hidden layers of 512; each hidden layer is linear → GELU → LayerNorm; then linear → 1 | 1 number | 2 networks, different random starting weights |
| **target Q1, Q2** | same | same | same | 1 slow copy of Q1 and Q2; starts equal to them |
| **V** (value network) | state only: 69 numbers | same as Q: 4 × 512, GELU, LayerNorm | 1 number | 1 network, **no target copy** |
| **flow network** (actor) | state, point x (21 numbers), time t (one plain number, no sinusoidal embedding) joined | 4 hidden layers of 512, GELU, **no LayerNorm**; then linear → 21 | 21 numbers (a direction) | 1 network |

- Starting weights: Glorot-uniform (`variance_scaling(1.0, 'fan_avg', 'uniform')`), biases 0.
- States go into the networks as raw numbers: the code does not normalize states and does not scale rewards.
- Dataset actions are clipped to [−1 + 0.00001, 1 − 0.00001] when the data is loaded (`envs/env_utils.py`, `action_clip_eps=1e-5`).
- One Adam optimizer for all networks: learning rate 0.0003, optax defaults (β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸), no gradient
  clipping, no weight decay, no learning-rate schedule.

## 4. Training, step by step

One training step is described below. It is repeated 1,000,000 times. All networks are trained from the first step;
there is no separate pre-training phase.

**Step 1. Draw a batch** of 256 rows (s, a, r, s′, mask).

**Step 2. Train V with the expectile loss.**
- For each row: q = min(Q1_target(s, a), Q2_target(s, a)), the target copies' score of the dataset's own action a.
- d = q − V(s).
- Loss = 0.9 · d² if d ≥ 0 (V is too low), and 0.1 · d² if d < 0 (V is too high). Averaged over the batch.
- Effect: a prediction that is too low costs 9 times more than one that is too high, so V(s) does not settle at the average
  Q of the actions taken at similar states. It settles between that average and the highest of them. Where exactly depends on
  how the Q values are spread, as the two examples show.
- Example 1: if the dataset's actions at similar states have Q = −120, −100 and −80, a plain squared loss puts V at −100 (the
  average); the 0.9 expectile puts V at about −85.5 (solve 0.9 · (−80 − V) = 0.1 · ((V + 120) + (V + 100))). Here V is
  closer to the highest Q.
- Example 2: if 99 actions have Q = 0 and 1 action has Q = 100, the average is 1 and the 0.9 expectile is about 8.3 (solve
  0.9 · (100 − V) = 0.1 · 99 · V). Here V is much closer to the average than to the highest Q.
- In plain words: V(s) is "how good s is if you act like the better actions in the data".

**Step 3. Train Q1 and Q2.**
- Target for each row: y = r + γ · mask · V(s′), with γ = 0.995 and V from the value network as it is now (not a copy).
- Loss = (Q1(s, a) − y)² + (Q2(s, a) − y)², averaged over the batch.
- V(s′) is used as a fixed number here: this loss does not change V.
- The target needs no action from any policy, only V(s′). Together with step 2 (V fitted at dataset actions only), this means
  Q is never evaluated during training at an action that is not in the dataset. This is what "implicit" refers to.

**Step 4. Train the flow network by flow matching (plain BC).**
- Draw x₀: 21 random numbers from a standard normal distribution. Let x₁ = the dataset action a.
- Draw a time t uniformly between 0 and 1.
- Take the point on the straight line from the noise to the action: xₜ = (1 − t) · x₀ + t · x₁.
- The correct direction at that point is x₁ − x₀.
- Loss = (v(s, xₜ, t) − (x₁ − x₀))², averaged over the 21 numbers and the batch.
- Every row counts equally. The flow network never sees Q or V.

**Step 5. One Adam step on the sum of the three losses** (V loss + Q loss + flow loss). Each loss depends only on its own
network (V's loss uses the target copies, which get no gradient; Q's loss uses V as a fixed number; the flow loss uses only the
flow network). So this is the same as three separate updates on the same batch, all computed from the weights before the step.

**Step 6. Move the target copies:** target = 0.995 × target + 0.005 × Q, for Q1 and Q2. (Exact code detail: the target moves
toward the Q weights from before this step's update.) V has no target copy.

**What never happens in training:** no step uses an action produced by the flow network. The critic and the flow network meet
only when an action is chosen (Section 5).

## 5. Choosing an action ("rejection sampling", FQL paper Equation 11)

At a state s, in every evaluation step:
1. Draw N = 32 noise vectors, each 21 numbers from a standard normal distribution.
2. Move each one 10 times: x ← x + v(s, x, t) / 10, with t = 0, 0.1, …, 0.9 (the Euler method with 10 steps). Each noise vector
   becomes one action proposed by the flow network.
3. Clip each proposed action to [−1, 1]. These are the 32 candidates.
4. Score each candidate with min(Q1, Q2)(s, candidate), using the trained critic (not the target copies).
5. Execute the candidate with the highest score.

Cost per environment step: 32 × 10 = 320 flow-network evaluations and 32 × 2 = 64 critic evaluations.

## 6. The critic in one place

| part | what it is |
|---|---|
| networks | Q1, Q2: 90 → 4 × 512 (linear → GELU → LayerNorm) → 1; V: 69 → same → 1; target copies of Q1, Q2 only |
| V's target | min(Q1_target, Q2_target)(s, a) at the dataset action |
| V's loss | expectile, τ = 0.9: 0.9 · d² if V too low, 0.1 · d² if V too high |
| Q's target | r + 0.995 · mask · V(s′); no policy action, no min, no max |
| Q's loss | (Q1 − y)² + (Q2 − y)² |
| where min(Q1, Q2) is used | only in V's target (with the target copies) and when choosing the action (with the trained Q) |
| target copies | Q only; updated every step with rate 0.005 |
| range of Q | true values lie in [−200, 0]; nothing in the code enforces this (no clipping of the target) |
| what the critic is used for | only to pick 1 of 32 candidates at decision time; never to weight or steer the actor in training |

## 7. Hyperparameters for humanoidmaze-medium

The exact command from the official README (everything not in it is a code default):

```
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent=agents/ifql.py --agent.discount=0.995 --agent.num_samples=32
```

| setting | value | source |
|---|---|---|
| discount γ | 0.995 | README command; paper Table 5 ("0.995 (antmaze-giant, humanoidmaze, antsoccer)") |
| expectile τ | 0.9 | `agents/ifql.py` default; appendix E.2 says IFQL uses IQL's defaults, and IQL uses 0.9 "for all environments" |
| V and Q networks | 4 × 512, GELU, LayerNorm (`layer_norm=True`) | code default; appendix B |
| flow network | 4 × 512, GELU, no LayerNorm (`actor_layer_norm=False`) | code default |
| number of Q networks | 2 | code |
| target update rate | 0.005 per step (Q only) | code; Table 5 |
| optimizer | Adam, learning rate 0.0003 | code; Table 5 |
| batch size | 256 | code; Table 5 |
| training steps | 1,000,000 | `main.py` (`offline_steps`); appendix B |
| candidates N | 32 (tried 32, 64 and 128 on task 1 and used for all five tasks) | README; Table 6; appendix E.2 |
| Euler steps of the flow | 10 | code; Table 5 |
| flow time distribution | uniform on [0, 1] | code; Table 5 |
| evaluation | every 100k steps, 50 episodes; score = mean of the evals at 800k, 900k and 1M | `main.py`; appendix B |
| seeds | 8 | paper Section 5.1 |
| **not used** | gradient clipping, reward scaling, state normalization, clipping of the TD target, weight decay, dropout, learning-rate schedule | checked in the code |

## 8. The tricks, and what each one does

1. **Expectile τ = 0.9 for V.** Makes V track the better dataset actions without ever taking a maximum over actions.
2. **Targets built only from dataset actions** (Q's target uses V(s′); V is fitted at dataset actions). Avoids the usual offline-RL
   failure where Q gives high values to actions that the data never tried.
3. **Two Q networks; min(Q1, Q2) in exactly two places:** V's target and the choice of action. Q's own target takes no min,
   because it comes from V.
4. **A slow target copy only for Q.** V has none, and Q's target uses the current V.
5. **LayerNorm in V and Q** (not in the flow network). Appendix B: "to further stabilize training".
6. **Large networks (4 × 512).** Appendix B: "using a large enough network is especially important in navigation environments".
7. **The mask.** Stops the target from adding value after the goal step.
8. **Dataset actions clipped to ±(1 − 0.00001)** when the data is loaded; **candidates clipped to [−1, 1]** before being scored.

**Two places in the FQL appendix that are easy to misread:**
- Appendix B ("Value learning") says the critic target uses the mean of Q1 and Q2 by default, and their min for adroit and
  antmaze-large/giant. This describes **FQL's own critic, not IFQL's**. IFQL's Q target is r + γ · mask · V(s′); its min is in V's
  target.
- Appendix E.2 writes "FIQL"; it is a typo for IFQL.

## 9. Weights from the critic

### 9.1 IFQL during training: none
The flow network copies every dataset action with weight 1. It never reads Q or V.

### 9.2 IFQL when acting: weight 1 for the best candidate, 0 for the rest
Section 5: score the 32 candidates with min(Q1, Q2) and execute the highest.

### 9.3 Original IDQL also defines softer weights (IFQL does not use them)
IDQL paper, Algorithm 2 and Equation 9. For each candidate aᵢ:
1. wᵢ = |τ − 1(Q(s, aᵢ) < V(s))|, which for τ = 0.9 is **0.9 if Q(s, aᵢ) ≥ V(s), and 0.1 if Q(s, aᵢ) < V(s)**.
2. Normalize: pᵢ = wᵢ / (w₁ + … + w_N).
3. Draw the executed action at random with probabilities pᵢ.

Example: if 8 of 32 candidates score above V, each of them gets probability 0.9 / (8 · 0.9 + 24 · 0.1) = 0.094 and each of the
other 24 gets 0.010; an above-V action is executed 75 % of the time.

The IDQL paper reports that taking the highest-scoring candidate works better at evaluation: "the expectile objective
performs the strongest with greedy extraction" (Section 6.3); taking the maximum "corresponds to selecting w(s, a) to be a
one-hot" (Section 4.2). That is why IFQL takes the maximum.

### 9.4 For comparison: methods that train the actor with weights from a critic

Results on humanoidmaze-medium task 1 (FQL paper Table 3, 8 seeds; our runs 3 seeds, except 2 for Exp 13's IQL critic).

| method | weight of each dataset row during actor training | success |
|---|---|---|
| IQL (Gaussian actor; same code base, same critic as IFQL) | min(exp(10 · (min(Q1, Q2)(s, a) − V(s))), 100); α = 10 (Table 6); uses the trained Q, not the target copies | 32 ± 7 % |
| FAWAC (flow actor with AWR weights) | exp(α · (Q(s, a) − V(s))) with α = 3 (Equation 10, Table 6). Its code is not public, so its V and any cap could not be checked | 6 ± 2 % |
| **IFQL** | **none (weight 1 for every row); best of 32 at decision time** | **69 ± 19 %** |
| our Aware with its AWAC critic (Experiment 13, AWAC runs; Experiment 18) | min(exp((Q − V)/T), 100) with T = 1/3, the same form as α = 3; Q = min(Q1, Q2)(s, a) from the trained critic; V = mean min Q over the actor's own samples (1 sample in Exp 13, 16 in Exp 18), not an expectile network | 0–2 % (stopped early; last eval at 100–200k in Exp 13, 300k in Exp 18) |
| our Aware with an IQL critic (Experiment 13, IQL runs) | the same formula with T = 1/3, but Q = min(Q1, Q2)(s, a) from the target copies, and V from a value network trained with the 0.9 expectile loss, as in IFQL | 0–2 % (2 seeds, the third run was never started; stopped early, last eval at 200k) |

## 10. Results on humanoidmaze-medium (FQL paper)

Success in %, 8 seeds, mean of the evals at 800k, 900k and 1M (Table 3; Table 2 for the average over the five tasks).
Exception: IDQL was trained for 3M steps, and its score is the mean of the evals at 2.5M, 2.75M and 3M (appendix E.2).

| task | BC | IQL | IDQL | CAC | FAWAC | FBRAC | **IFQL** | FQL |
|---|---|---|---|---|---|---|---|---|
| task 1 (default) | 1 ± 0 | 32 ± 7 | 1 ± 1 | 38 ± 19 | 6 ± 2 | 25 ± 8 | **69 ± 19** | 19 ± 12 |
| task 2 | 1 ± 0 | 41 ± 9 | 1 ± 1 | 47 ± 35 | 40 ± 2 | 76 ± 10 | **85 ± 11** | 94 ± 3 |
| task 3 | 6 ± 2 | 25 ± 5 | 0 ± 1 | 83 ± 18 | 19 ± 2 | 27 ± 11 | **49 ± 49** | 74 ± 18 |
| task 4 | 0 ± 0 | 0 ± 1 | 1 ± 1 | 5 ± 4 | 1 ± 1 | 1 ± 2 | **1 ± 1** | 3 ± 4 |
| task 5 | 2 ± 1 | 66 ± 4 | 1 ± 1 | 91 ± 5 | 31 ± 7 | 63 ± 9 | **98 ± 2** | 97 ± 2 |
| average of the 5 tasks | 2 ± 1 | 33 ± 2 | 1 ± 0 | 53 ± 8 | 19 ± 1 | 38 ± 5 | **60 ± 14** | 58 ± 5 |

- **Offline-to-online** (Table 4, task 1, success at 1M offline steps → after 1M more online steps): IFQL 56 ± 35 → 82 ± 20;
  FQL 12 ± 7 → 22 ± 12; IQL 21 ± 13 → 16 ± 8. The offline number (56) differs from Table 3 (69). One reason: Table 4 reports
  only the eval at 1M (appendix B), while Table 3 reports the mean of the evals at 800k, 900k and 1M. The paper does not say
  whether Table 4 uses the same runs as Table 3. For FQL it cannot: FQL used α = 30 in Table 6 and α = 100 in Table 7. IFQL
  used N = 32 in both tables.
- **The same idea with a different implementation scored 1 %:** IDQL (the official diffusion code, run by the FQL authors,
  appendix E.2) uses the same kind of critic (IQL, expectile loss) and also chooses among N = 32 samples (Table 6), but scored
  1 ± 1 % on task 1. The differences the appendix lists: value networks of 2 × 512 (4 layers were "unstable" in that code base), expectile 0.7 for locomotion
  (IFQL: 0.9), a diffusion model with residual layers instead of a flow MLP, and 3M training steps (1.5M for the value
  networks). The paper does not test which of these differences matters.

## 11. Comparison with our experiments on the same task

| | IFQL | our Experiment 15 (best of N) | our Aware with its AWAC critic (Exp 13 AWAC runs, Exp 18) | our Aware with an IQL critic (Exp 13 IQL runs) |
|---|---|---|---|---|
| actor proposing actions | flow BC, 10 Euler steps, 1M steps, never sees the critic | EqM-BC (Exp 14), K = 20 sampler steps, 500k steps, frozen | EqM actor trained with AWAC weights from the critic | EqM actor trained with AWAC weights from the critic |
| critic target | r + γ · V(s′); V by expectile 0.9 at dataset actions | QIPO softmax: average of Q over the dataset's next action and 16 EqM-BC samples at K = 10, weighted by softmax(3 · Q) | min Q at one next action from the current EqM actor | r + γ · V(s′); V by expectile 0.9 at dataset actions, as in IFQL |
| critic sees policy-made actions in training | no | yes (the 16 EqM-BC samples) | yes (the actor's next action) | no |
| critic network | 4 × 512, LayerNorm, γ = 0.995, no target clipping | 4 × 512, LayerNorm, γ = 0.995, target clipped to [−200, 0] | Exp 13: 4 × 256, γ = 0.995; Exp 18: 4 × 512, LayerNorm, γ = 0.999 | Q and V: 4 × 256, no LayerNorm, γ = 0.995 |
| how the critic is used | best of 32 at decision time | best of 16 / 32 / 64 at decision time | weights in actor training | weights in actor training |
| seeds | 8 | 3 | 3 | 2 (the third run was never started) |
| success, task 1 | 69 ± 19 % | 33 / 39 / 42 % (N = 16 / 32 / 64) | 0–2 % (stopped early; last eval at 100k–300k) | 0–2 % (stopped early; last eval at 200k) |

Experiment 15 and IFQL use the critic in the same way (choose one of N samples from a BC actor that the critic never changes).
They differ in how the critic is trained (IQL expectile V on dataset actions vs QIPO's softmax target over BC samples), in the
actor (EqM at K = 20 vs flow), in the actor's width (EqM-BC 4 × 256 vs flow 4 × 512), in BC training length (500k vs 1M
steps), in the evaluation (Exp 15: 100 episodes at the final checkpoints; FQL: the mean of 3 evals of 50 episodes each, at
800k, 900k and 1M), and in the number of seeds.

**Not tested yet (possible next experiment):** the IFQL recipe with our actor, i.e. train an IQL critic exactly as in Sections 4
and 7 (expectile 0.9, 4 × 512 with LayerNorm, γ = 0.995, 1M steps) and choose the best of 32 EqM-BC samples at K = 20. This would
separate the effect of the critic training from the effect of the actor type. Experiment 13 already trained an IQL critic
(expectile 0.9), but a different one, used differently: smaller networks (4 × 256, no LayerNorm), used only for the actor's
training weights and never to choose among samples, and stopped early (last eval at 200k, 0–2 %).
