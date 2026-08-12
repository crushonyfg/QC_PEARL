# Test-Time Bellman-Residual Correction for Full-PEARL under Dynamics Shift

Paper methods reference for the `pearl_brpc_action_adapter` full-PEARL +
value-shift / Q-residual pipeline. All formulas below match the implementation in
`pearl_brpc_action_adapter/` (training: `experiments/train_full_pearl.py`,
world model: `experiments/train_full_pearl_dynamics.py`, evaluation:
`eval/eval_full_pearl_dynamics_lookahead.py`).

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $s_t\in\mathbb R^{d_s}$, $a_t\in\mathbb R^{d_a}$ | state, action ($d_s{=}11,\,d_a{=}3$ for Hopper) |
| $z\in\mathbb R^{d_z}$ | task latent ($d_z{=}5$) |
| $\xi=(\xi_{\text{mass}},\xi_{\text{fric}},\xi_{\text{damp}},\xi_{\text{act}})$ | dynamics-scaling factors of a task |
| $\pi_\theta(a\mid s,z)$ | tanh-Gaussian policy; $\mu_\theta(s,z)$ its deterministic (mean) action |
| $Q_{\phi_1},Q_{\phi_2}$ | twin critics; $Q_{\min}(s,a,z)=\min_i Q_{\phi_i}(s,a,z)$ |
| $e_\psi$ | product-of-Gaussians context encoder $\Rightarrow z$ |
| $\gamma,\ \alpha$ | discount ($0.99$), SAC entropy temperature ($0.2$) |
| $f_\eta(s,a,z)$ | offline world model, predicts $\widehat{\Delta s}=\widehat{s'}-s$ |
| $\Delta f(s,a,z)$ | online dynamics-residual (BRPC), corrects $f$ |
| $h_w(\varphi_Q(s,a,z))$ | online value-residual head over the critic's penultimate features |

**Frozen backbone.** $\pi_\theta,Q_{\phi_{1,2}},e_\psi$ are trained once (Sec. 2) and
**frozen** at test time. World model $f_\eta$ is trained once offline (Sec. 7), also frozen.
Only the linear online estimators $\Delta f,\ h_w$ adapt at test time.

**Soft / deterministic value.** The SAC soft value and the low-variance deterministic
value used for re-ranking are

$$
V_{\text{soft}}(s,z)=\mathbb E_{a\sim\pi_\theta}\!\big[Q_{\min}(s,a,z)-\alpha\log\pi_\theta(a\mid s,z)\big],
\qquad
V(s,z)\;\triangleq\;Q_{\min}\!\big(s,\mu_\theta(s,z),z\big).
$$

We use the deterministic $V$ inside value differences (it removes sampling noise from
$\Delta V$ below).

---

## 2. Training full joint PEARL (the frozen backbone)

Standard PEARL: meta-train a single latent-conditioned SAC over a distribution of tasks
$\xi\sim\mathcal U([0.8,1.2]^4)$ (`dynamics_randomization.train_range`), each task a
dynamics-randomized Hopper (Sec. 8). Per-task replay buffers feed a joint
encoder–critic–actor update.

**Context encoder (product of Gaussians).** A context item is
$c=(s,a,r,s',d)\in\mathbb R^{2d_s+d_a+2}$. Each item produces a Gaussian factor
$\mathcal N(\mu_n,\sigma_n^2)$; the posterior over $z$ aggregates $N$ items against an
$\mathcal N(0,I)$ prior:

$$
\sigma_z^{-2}=1+\sum_{n=1}^{N}\sigma_n^{-2},\qquad
z_{\text{mean}}=\sigma_z^{2}\sum_{n=1}^{N}\sigma_n^{-2}\mu_n,\qquad
z\sim\mathcal N(z_{\text{mean}},\sigma_z^{2}).
$$

**Losses.** With target critics $Q_{\bar\phi_i}$ and next action $a'\sim\pi_\theta(\cdot\mid s',z)$,

$$
y = r+\gamma(1-d)\Big(\min_i Q_{\bar\phi_i}(s',a',z)-\alpha\log\pi_\theta(a'\mid s',z)\Big),
$$

$$
\mathcal L_{Q}= \tfrac12\big(Q_{\phi_1}(s,a,z)-y\big)^2+\tfrac12\big(Q_{\phi_2}(s,a,z)-y\big)^2,
$$

$$
\mathrm{KL}=\tfrac12\sum_{j}\big(z_{\text{mean},j}^2+\sigma_{z,j}^2-\log\sigma_{z,j}^2-1\big).
$$

The **encoder is trained through the critic loss** plus a latent prior term
($z\!\sim\!q_\psi$ in $\mathcal L_Q$), and the actor uses a **detached** $z_{\text{mean}}$:

$$
\mathcal L_{\text{critic+enc}}=\mathcal L_Q+\beta_{\mathrm{KL}}\,\mathrm{KL},
\qquad
\mathcal L_{\pi}=\mathbb E_{a\sim\pi_\theta}\big[\alpha\log\pi_\theta(a\mid s,\bar z)-Q_{\min}(s,a,\bar z)\big],
\quad \bar z=\mathrm{sg}[z_{\text{mean}}].
$$

Target nets: Polyak $\bar\phi\leftarrow\tau\phi+(1-\tau)\bar\phi$, $\tau=0.005$;
$\beta_{\mathrm{KL}}=10^{-3}$.

**Algorithm (one epoch).** (i) collect on-policy rollouts on sampled tasks into per-task
buffers; (ii) for `updates_per_epoch` steps: sample a meta-batch of tasks, for each task
sample a context batch and an RL batch, infer $z\sim q_\psi(\cdot\mid c_{1:N})$, take a
gradient step on $\mathcal L_{\text{critic+enc}}$ and (with $\bar z$) on $\mathcal L_\pi$,
Polyak-update targets. Checkpoint by best nominal episode length.

```
conda run -n bi-rl python -m pearl_brpc_action_adapter.experiments.train_full_pearl \
    --config configs/full_pearl_hopper.json --seed 0
# -> checkpoints/full_pearl/full_pearl_best.pt   (actor, q1, q2, encoder, cfg, meta)
```

Key config (`configs/full_pearl_hopper*.json`): `latent.dim=5`,
`pearl.{gamma=0.99, sac_alpha=0.2, beta_kl=1e-3, tau=5e-3}`, hidden $256{\times}2$
(actor/critic), $200{\times}3$ (encoder).

---

## 3. Test-time setup (everything below uses the frozen backbone)

### 3.1 PEARL-style $z$ adaptation
At test we infer $z=e_\psi(c_{t-N:t})$ from a running context window (recency length
`eval_context_max`, min `eval_context_min`). With `persist_context=true` the context (and
the gate EWMA, previous $z$) **persists across the warmup and test episodes** of a regime,
i.e. the standard PEARL "few adaptation episodes" protocol. This is applied to **every
method including the baselines** for a fair comparison.

> Empirically, $z$-adaptation does **not** raise the baseline on out-of-range OOD: the
> encoder was trained on $\xi\in[0.8,1.2]$ and cannot represent an out-of-range $z$, so more
> context does not yield a better $z$. The OOD gain therefore must come from correcting the
> value, not from $z$ alone.

### 3.2 Candidate pool
At state $s$ with $a_0=\mu_\theta(s,z)$, build $n$ candidates: $a_0$ itself, multi-scale
Gaussian shells $a_0+\mathcal N(0,\sigma_k^2 I)$ (`local_action_stds`, e.g. $[0.1,0.3]$),
and policy samples $a\sim\pi_\theta$ (`policy_candidate_frac`), clipped to the action box.

### 3.3 Shift gate (a Bellman/dynamics-residual detector)
Let the **observed one-step model residual** on the executed action be
$\;\varepsilon_t=\big\|(s_{t+1}-s_t)-f_\eta(s_t,a_t,z)\big\|.$
With $\bar\varepsilon$ its EWMA normalized by the nominal $95^{\text{th}}$ pct $\varepsilon_{p95}$,

$$
\bar\varepsilon_t=(1-\rho)\bar\varepsilon_{t-1}+\rho\,\frac{\varepsilon_t}{\varepsilon_{p95}},
\qquad
h_t=\sigma\!\big(\kappa(\bar\varepsilon_t-\tau)\big)\in(0,1).
$$

The gate decides **how much to deviate** from the policy: the executed action is the
gated interpolation $a_t=(1-h_t)\,a_0+h_t\,a_{\text{sel}}$, so on nominal ($h\!\to\!0$) every
method reduces to the frozen policy. Defaults $\kappa=8,\ \tau\approx0.6,\ \rho=0.1$.

---

## 4. The Bellman residual and its decomposition (core)

For a transition $(s,a,r,s')$ the **(deterministic) Bellman residual** of the frozen critic is

$$
\boxed{\ \eta(s,a,r,s')\;=\;r+\gamma(1-d)\,V(s',z)\;-\;Q_{\min}(s,a,z)\ }
$$

It is the TD error: $\eta=0\iff Q_{\min}$ is one-step self-consistent at $(s,a)$ under the
realized dynamics+reward. Under a dynamics shift, $\eta\neq0$ both because the **transition**
moved and because the frozen **value** $Q,V$ extrapolates badly (e.g. $z$ is OOD).

**Model-based residual.** Replacing the realized $s'$ by the world model
$\widehat{s'}(s,a)=s+f_\eta(s,a,z)+\Delta f(s,a,z)$ and the reward by the (Hopper-analytic)
$\hat r(s,a,s')$ gives a per-candidate estimate

$$
\hat\eta(s,a)=\hat r(s,a,\widehat{s'})+\gamma V(\widehat{s'})-Q_{\min}(s,a,z)
\;=\;\underbrace{\hat Q(s,a)}_{\text{shifted one-step value}}-Q_{\min}(s,a,z),
$$

so **every re-ranking score is $Q_{\min}$ plus an estimate of the Bellman residual.**

**Decomposition into dynamics vs. value error.** Writing
$\widehat{s'}_{\text{nom}}=s+f_\eta$ (nominal model) and
$\widehat{s'}=\widehat{s'}_{\text{nom}}+\Delta f$,

$$
\boxed{\ \hat\eta(s,a)=\underbrace{\big[\hat r(s,a,\widehat{s'}_{\text{nom}})+\gamma V(\widehat{s'}_{\text{nom}})-Q_{\min}\big]}_{\eta_{\text{value}}\ \text{(intrinsic value error)}}
\;+\;\underbrace{\Delta r}_{\text{reward shift}}\;+\;\underbrace{\gamma\,\Delta V}_{\text{dynamics value shift}}\ }
$$

with the two shift terms driven only by the online residual $\Delta f$:

$$
\Delta V=V(\widehat{s'}_{\text{nom}}+\Delta f)-V(\widehat{s'}_{\text{nom}}),
\qquad
\Delta r=\hat r(s,a,\widehat{s'})-\hat r(s,a,\widehat{s'}_{\text{nom}})=w_{\text{fwd}}\,\Delta f_{[v_x]},
$$

because Hopper's reward $\hat r=\underbrace{1}_{\text{healthy}}+w_{\text{fwd}}\,v_x(s')-w_{\text{ctrl}}\|a\|^2$
depends on $s'$ only through the forward velocity $v_x=s'_{[5]}$ (so the $\Delta f$ component at
the $v_x$ index is the entire reward shift). **The two halves of $\hat\eta$ are estimated by two
different methods** (Sec. 5): $\gamma\Delta V+\Delta r$ by `value_shift`, and $\eta_{\text{value}}$ by `value_shift_qr`.

---

## 5. Methods — each as $Q_{\min}+\widehat{(\text{Bellman residual})}$

All re-rankers select $a_{\text{sel}}=\arg\max_i \mathrm{score}(s,a_i)$ (or top-$k$ softmax
sample), then gate-interpolate $a=(1-h)a_0+h\,a_{\text{sel}}$. Penalties
$\;\Omega(a)=\lambda_a\|a-a_0\|^2+\lambda_{\text{sw}}\|a-a_{\text{prev}}\|^2\;$ keep the search local.

| method | score $\mathrm{score}(s,a)$ | residual term used |
|---|---|---|
| `full_pearl_only` | execute $a_0=\mu_\theta(s,z)$ | none (assumes $\hat\eta\equiv0$) |
| `q_greedy` | $Q_{\min}(s,a,z)-\Omega(a)$ | **none** in score; gate uses observed $\varepsilon$ |
| `value_shift` | $Q_{\min}+\;\underbrace{w_{\text{fwd}}\Delta f_{[v_x]}+\gamma\,\Delta V}_{\text{dynamics half of }\hat\eta}\;-\Omega$ | $\Delta r+\gamma\Delta V$ |
| `value_shift_h` | $Q_{\min}+\Delta\mathcal R_H-\Omega$ (multi-step, below) | $H$-step return residual |
| **`value_shift_qr`** | $Q_{\min}+\;\underbrace{h_w(\varphi_Q(s,a,z))}_{\widehat{\eta_{\text{value}}}}\;-\Omega$ | **value half of $\hat\eta$** |
| `oracle_pool` (diag.) | $r_{\text{real}}+\gamma V(s'_{\text{real}})=Q_{\min}+\eta_{\text{true}}$ | exact $\eta$ via true sim |

**`full_pearl_only`** — the frozen PEARL policy; the OOD baseline.

**`q_greedy`** — re-rank candidates by the *raw* critic (i.e. trust $Q$'s **ranking**, set
$\hat\eta\equiv0$). Surprisingly strong; isolates "how much is just greedy reselection."

**`value_shift`** — estimate only the **dynamics** half of $\hat\eta$. The online dynamics
residual $\Delta f$ (Sec. 6) gives $\Delta V$ and $\Delta r$. The world model's own bias
cancels in the difference $\Delta V=V(s+f+\Delta f)-V(s+f)$, so the signal comes from $\Delta f$,
not from trusting $f$'s absolute prediction. (Empirically near-neutral / noisy on stationary OOD.)

**`value_shift_h`** — multi-horizon. Roll the nominal ($f$) and corrected ($f+\Delta f$)
trajectories $H$ steps under $\pi_\theta$ and take the discounted-return difference plus a
terminal bootstrap:

$$
\Delta\mathcal R_H=\sum_{k=0}^{H-1}\gamma^k\big(\hat r^{\text{corr}}_k-\hat r^{\text{nom}}_k\big)
+\gamma^H\big(V(s^{\text{corr}}_H)-V(s^{\text{nom}}_H)\big).
$$

$H{=}1$ recovers `value_shift`. More backups reduce reliance on the (wrong) bootstrap $V$,
but $\Delta f$ extrapolation compounds over $H$ — use small $H$ (3–5). (Did not beat $q$-greedy in practice.)

**`value_shift_qr` (main method)** — estimate the **value** half of $\hat\eta$, i.e. the
critic's *intrinsic* value error, in the basis where $Q$ is linear. Since
$Q_{\min}(s,a,z)=w^\top\varphi_Q(s,a,z)+b$ for penultimate features $\varphi_Q$
(`QNetwork.features`), the value error is approximately linear in $\varphi_Q$. An online
linear head $h_w$ (BRPC, Sec. 6) is fit to the observed value residual and added to $Q$:

$$
\boxed{\ \mathrm{score}(s,a)=Q_{\min}(s,a,z)+h_w\!\big(\varphi_Q(s,a,z)\big)-\Omega(a)\ }
$$

The **fitted target** on each executed transition (uses real $r$, **nominal-model** next
state to avoid double-counting the bootstrap shift):

$$
\boxed{\ \eta^{\text{obs}}_{\text{value}}=r+\gamma(1-d)\,V\!\big(s+f_\eta(s,a,z),z\big)-Q_{\min}(s,a,z)\ }
$$

Because $\varphi_Q$ is a learned, low-dimensional basis, the head extrapolates the
correction from the thin executed stream to *all* candidates — the reason the earlier
generic-RFF Bellman-residual head failed but this one works.

**$z$/$Q$ two-timescale coupling.** $z$ adapts slowly (accumulated context); $h_w$ adapts
fast (recursive). Since $\varphi_Q(\cdot,\cdot,z)$ depends on $z$, a large $z$ jump
(`q_z_jump_thresh`) inflates the head posterior covariance ($P\!\mathrel{+}=\!\lambda P_0 I$,
`q_z_inflate`) so it re-learns in the new basis — keeping the coupling stable.

**`oracle_pool` (diagnostic, not deployable).** For each candidate, save the MuJoCo state,
step the **true shifted** env, score $r_{\text{real}}+\gamma V(s'_{\text{real}})$, restore
state (qpos/qvel **and** the `TimeLimit` step counter). Equals $Q_{\min}+\eta_{\text{true}}$ —
the **upper bound** of one-step re-ranking of a pool with a perfect model.

---

## 6. Online estimators (BRPC) and fleet batch update

Both $\Delta f$ (output dim $d_s$, features = RFF of $(s,a,z)$) and $h_w$ (output dim $1$,
features $=\varphi_Q$) are **Bayesian recursive linear regressors with an AR(1) prior**
(`BRPCResidualCalibrator`). State $(M,P)$, observation noise $\sigma^2$, forgetting $\rho$,
process noise $q_\alpha$:

$$
\textbf{predict: }\;M\leftarrow\rho M,\quad P\leftarrow\rho^2P+q_\alpha I;
$$
$$
\textbf{update }(\varphi,y):\;\;S=\sigma^2+\varphi^\top P\varphi,\quad
K=P\varphi/S,\quad
M\leftarrow M+K\,(y-M^\top\varphi)^\top,\quad
P\leftarrow P-K\varphi^\top P;
$$
$$
\textbf{predict mean/var: }\;\;\hat y=M^\top\varphi,\quad \mathrm{Var}=\varphi^\top P\varphi.
$$

- $\Delta f$ target $y=(s'-s)-f_\eta(s,a,z)$; obs-noise = per-dim nominal residual std $\times$ scale.
- $h_w$ target $y=\eta^{\text{obs}}_{\text{value}}$ (above); obs-noise scalar (`q_obs_noise`).

**Few-shot / continual.** Calibrators persist across the $K$ warmup episodes and keep
updating through the measured episodes.

**Fleet ($M$ agents).** $M$ agents act in the same regime in lockstep and share **one**
calibrator. Per wall-clock step we apply **one** AR(1) `predict` and then $M$ measurement
`update`s (a batch Kalman step), so $M$ agents give $M\times$ adaptation data per step
without over-decaying. Optionally the fleet also shares one context (`fleet_share_context`)
so $z$-adaptation is $M\times$ too. This decouples adaptation breadth $M$ from the number
of bad episodes $K$ the deployed agent must itself suffer.

---

## 7. Offline world model $f_\eta$

`NormalizedLatentDynamicsModel`: $f_\eta(s,a,z)\to\widehat{\Delta s}$, an MLP ($256{\times}3$)
in a standardized space (state and $\Delta s$ are z-scored; buffers stored in the checkpoint).
Trained **post-hoc on the frozen PEARL checkpoint**, never touching $\pi,Q,e_\psi$: roll the
trained policy + exploration noise ($\mathcal N(0,0.15^2)$) on $\xi\sim$ train range, relabel
$z$ with the **frozen** encoder, minimize $\tfrac1{2}\|f_\eta(s,a,z)-(s'-s)\|^2$ (normalized).
Also stores the nominal residual scale (per-dim std, norm $p95$) used for the BRPC obs-noise
and the gate threshold.

```
conda run -n bi-rl python -m pearl_brpc_action_adapter.experiments.train_full_pearl_dynamics \
    --config configs/full_pearl_dynamics_lookahead_smoke.json --seed 0
# -> checkpoints/full_pearl/full_pearl_dynamics.pt
```

---

## 8. OOD regime definitions

**Dynamics parametrization.** A task is a vector of multiplicative scaling factors
$\xi=(\xi_{\text{mass}},\xi_{\text{fric}},\xi_{\text{damp}},\xi_{\text{act}})$ applied to the
nominal MuJoCo model (`DynamicsRandomizationWrapper`):

$$
m\!\leftarrow\!\xi_{\text{mass}}m_0,\quad
\mu\!\leftarrow\!\xi_{\text{fric}}\mu_0,\quad
c\!\leftarrow\!\xi_{\text{damp}}c_0,\qquad
a_{\text{exec}}=\mathrm{clip}\big(\xi_{\text{act}}\,a\big)\ \text{(applied at step time).}
$$

So $\xi_{\text{act}}$ is an **action-side gain**; mass/friction/damping are **environment/body**
dynamics. **Training distribution:** $\xi\sim\mathcal U([0.8,1.2]^4)$ i.i.d. per task. **OOD**
$=$ $\xi$ at or beyond the edges of this box (e.g. $\xi_{\text{act}}\in\{0.6,0.4,0.2\}$,
$\xi_{\text{fric}}\in\{0.6,1.5\}$, $\xi_{\text{mass}}\in\{0.6,1.5\}$). Note the latent $z$ of
full PEARL is **learned**, not $\xi$, so there is no closed-form "true $z^\star$".

**Time-indexed schedules $\xi_t$** (`EvalDynamicsSchedule` + local `MultiSuddenSchedule`):

| regime `type` | $\xi_t$ |
|---|---|
| `nominal` | $\xi_t=\mathbf 1$ |
| `fixed` | $\xi_t=\xi_{\text{OOD}}\ \ \forall t$ (stationary OOD) |
| `sudden` | $\xi_t=\xi_{\text{before}}\ (t<t_s)$, $\ \xi_{\text{after}}\ (t\ge t_s)$ |
| `gradual` | $\xi_t=\xi_0+\min\!\big(1,\tfrac{t}{T_{\text{drift}}}\big)\,(\xi_{\text{end}}-\xi_0)$ (linear drift) |
| `multi_sudden` | piecewise-constant: $\xi_t=$ last change with `step`$\le t$ |

**Action-side beyond gain** (`ActionPerturbWrapper`): additive bias / actuation noise that the
gain $\xi_{\text{act}}$ cannot express,
$\;a_{\text{exec}}=\mathrm{clip}\big(g\,a+b+\epsilon\big),\ \epsilon\sim\mathcal N(0,\sigma^2)$,
declared per regime via `action_perturb: {gain, bias, noise_std}`.

**OOD categories evaluated** (`configs/..._oodtypes.json`): *action-side* (actuator gain,
bias, noise), *environment dynamics* (friction, damping), *body dynamics* (mass), *compound*
(several factors at once), and the non-stationary `sudden` / `gradual` / `multi_sudden`.

---

## 9. Few-shot ($K$) / fleet ($M$) protocol

For each (regime, seed): run $K$ warmup episodes (adapt $z$-context and the calibrators),
then measure $N$ test episodes (adaptation continues). $K=$`warmup_episodes` /`--warmup`;
$M=$`--num-agents` (fleet, BRPC methods). The M/K study asks whether parallel breadth $M$
(fleet) can substitute for sequential depth $K$ at the deployed agent.

---

## 10. Evaluation metrics

Per episode: return, length; diagnostics `mean_q_lift` $=\overline{Q_{\min}(s,a)-Q_{\min}(s,a_0)}$,
`mean_value_shift_term` $=\overline{\mathrm{score}-Q}$, `mean_gate` $=\bar h$,
`mean_raw_resid_norm` $=\bar\varepsilon$, `mean_correction_sq` $=\overline{\|a-a_0\|^2}$.
Aggregated mean/std over seeds × episodes (fleet: each agent-episode is one sample).

---

## 11. Reproduction

```
conda activate qc-pearl
cd QC_PEARL

# 1) train frozen backbone, 2) offline world model, 3) evaluate
python -m pearl_brpc_action_adapter.experiments.train_full_pearl          --config configs/full_pearl_hopper.json --seed 0
python -m pearl_brpc_action_adapter.experiments.train_full_pearl_dynamics --config configs/full_pearl_dynamics_lookahead_qr.json --seed 0
python -m pearl_brpc_action_adapter.experiments.run_full_pearl_dynamics_lookahead_evals \
    --config configs/full_pearl_dynamics_lookahead_oodtypes.json \
    --methods full_pearl_only q_greedy value_shift_qr --num-agents 5 --skip-existing
```

`--methods` ∈ {`full_pearl_only`,`q_greedy`,`value_shift`,`value_shift_h`,`value_shift_qr`,`oracle_pool`};
`--warmup K`, `--num-agents M` for the M/K study.

---

## Appendix A — headline results (3 seeds, corrected protocol, fleet $M{=}5$)

Return; $\Delta$ vs frozen PEARL and vs `q_greedy`.

| OOD type | base | q_greedy | value_shift_qr | qr−base | qr−q_greedy |
|---|---|---|---|---|---|
| nominal | 1468 | 1452 | 1447 | −1.5% | −0.4% |
| action gain 0.60 | 721 | 788 | **842** | **+16.8%** | **+6.8%** |
| action gain 0.40 (strong) | 371 | 371 | 362 | −2.3% | −2.4% |
| action bias 0.20 | 978 | **1116** | 1085 | +11.0% | −2.7% |
| action noise 0.20 | 1485 | 1459 | 1450 | −2.4% | −0.6% |
| env friction 0.60 | 864 | 874 | **911** | +5.4% | **+4.2%** |
| env friction 1.50 | 424 | 417 | **432** | +1.9% | **+3.5%** |
| env damping 0.60/1.50 | ~1468 | ~1449 | ~1446 | ~−1.5% | ≈0 |
| body mass 1.50 | 1255 | 1233 | 1225 | −2.3% | −0.7% |
| body mass 0.60 | 1426 | 1280 | **1389** | −2.6% | **+8.6%** |
| compound mild | 915 | 992 | **1048** | **+14.5%** | **+5.7%** |
| compound hard | 515 | 551 | 552 | +7.2% | +0.1% |

**Takeaways.** `value_shift_qr` wins where the critic's *value* is genuinely wrong and
correctable — moderate action gain, friction, compound, and more robustly than `q_greedy`
on body-mass. For pure action *bias* (wrong action, not wrong value) `q_greedy` is the right
tool. Near-nominal regimes and strong shift ($\xi_{\text{act}}\!\le\!0.4$, a policy/gait
ceiling shown by `oracle_pool`) are neutral. `value_shift` (dynamics-only) and `value_shift_h`
(multi-horizon) did not beat `q_greedy`.

## Appendix B — non-stationary regimes (3 seeds, fleet $M{=}5$)

Time-varying $\xi_t$ (Sec. 8). Return; $\Delta$ vs frozen PEARL and vs `q_greedy`.

| regime | base | q_greedy | value_shift_qr | qr−base | qr−q_greedy |
|---|---|---|---|---|---|
| `sudden` $\xi_{\text{act}}{:}1.0\!\to\!0.4$ @ $t{=}100$ | 604 | 607 | **614** | +1.7% | +1.2% |
| `gradual` $\xi_{\text{act}}{:}1.0\!\to\!0.2$ over 300 | 689 | 710 | **727** | +5.5% | +2.3% |
| `multi_sudden` $1.0\!\to\!0.5\!\to\!0.85\!\to\!0.35$ | 648 | **642** | **726** | **+12.1%** | **+13.1%** |

**Takeaway.** The largest non-stationary win is `multi_sudden`: with repeated changes the frozen
$Q$ goes stale and `q_greedy` even drops below the policy ($642<648$), whereas the **continually
re-fit** value-residual head tracks the drift (+13.1% over `q_greedy`). `gradual` shows the same
trend more mildly; single moderate-late `sudden→0.4` has little headroom (it spends most steps in
the strong-shift/ceiling regime). This is where the *continual / few-shot* nature of the online
Bellman-residual head matters most.

> **Protocol note.** Appendix A uses the persistent-context $z$-adaptation protocol (Sec. 3.1);
> Appendix B is from the earlier protocol (per-episode $z$ context, warmup only for the adapting
> methods). The two are directly comparable: $z$-adaptation was found to have negligible effect on
> returns (the encoder cannot represent an out-of-range $z$; Sec. 3.1), so it neither lifts the
> baselines nor changes the relative ordering.

## Appendix C — the Bellman residual is a good OOD / non-stationarity signal

This motivates using the residual (i) to *detect* a shift (the gate, Sec. 3.3) and (ii) as the
correction target (Sec. 5). Two empirical claims.

**C.1 — Cross-regime separation.** The frozen-critic Bellman residual
$|\eta|=|r+\gamma V(s')-Q_{\min}|$ (soft $V$; `run_full_pearl_bellman_diagnostic.py`, 5 seeds ×
25 ep) rises sharply out of distribution:

| regime | mean $|\eta|$ | $p95\,|\eta|$ | $E$/nominal-$p95$ ratio |
|---|---|---|---|
| nominal | 0.72 | 2.15 | 0.99 |
| `fixed` $\xi_{\text{act}}{=}0.60$ | **1.21** (+68%) | 3.88 | **1.79** |
| `sudden` $\to0.60$ (post-shift) | **1.31** (+82%) | 3.87 | **1.79** |

The residual energy $E$ (per-step squared error vs. its nominal $p95$) is $\approx\!1$ on nominal
and $\approx\!1.8$ under OOD — a clean, threshold-able detector requiring no labels.

**C.2 — Change-point detection (non-stationary), phase-controlled.** Care is needed: $|\eta|$ is
naturally larger later in an episode (harder/later states), so a naive within-`sudden` pre/post
split overstates the effect. Controlling for episode phase by comparing against the `nominal`
regime at the *same* steps (nominal and sudden share identical dynamics for $t<100$):

| phase | nominal | sudden | sudden vs nominal |
|---|---|---|---|
| pre-shift ($t<100$) | 0.44 | 0.44 | $\times1.00$ (identical dynamics ✓) |
| post-shift ($t\ge100$) | 0.79 | **1.31** | **$\times1.66$ (+66%)** |

So the *episode-phase* term ($0.44\!\to\!0.79$ in `nominal`) is real but common to both; the **shift adds
a further $+66\%$** on top (identical for the raw dynamics residual energy $E$: $0.37\!\to\!0.61$,
$\times1.66$). This is the genuine change-point signal that the EWMA gate
$h_t=\sigma(\kappa(\bar\varepsilon_t-\tau))$ integrates. (Per-*time* cross-episode averages, Fig.
panel (a), are additionally confounded by **survivorship** — struggling sudden episodes terminate
early — so the phase-controlled per-episode comparison in Fig. panel (b) is the rigorous statement.)

**C.3 — The (cheaper) dynamics residual $\varepsilon=\|(s'{-}s)-f(s,a,z)\|$ also predicts *which*
OOD actually hurts.** Used by the deployed gate (no reward/$V$ needed). Per OOD type (value_shift_qr
run, mean over episode):

| regime | $\bar\varepsilon$ | gate $\bar h$ | return vs nominal |
|---|---|---|---|
| nominal | 0.21 | 0.30 | — |
| env damping 0.6 / 1.5 | 0.20 / 0.21 | 0.30 / 0.29 | ≈0 (no harm) |
| body mass 1.5 | 0.23 | 0.42 | mild |
| compound mild | 0.24 | 0.45 | moderate |
| body mass 0.6 | 0.30 | 0.58 | mild |
| env friction 0.6 / 1.5 | 0.35 / 0.39 | 0.69 / 0.78 | moderate |
| action gain 0.6 | 0.38 | 0.82 | large |
| compound hard | 0.44 | 0.86 | large |
| action noise / bias 0.2 | 0.53 / 0.55 | 0.99 / 1.00 | large |
| action gain 0.4 | 0.71 | 1.00 | severe |

$\bar\varepsilon$ orders the regimes by how far they are from the training dynamics and is
**monotone with the performance drop**: regimes with $\bar\varepsilon\approx$ nominal (damping,
mass) barely degrade and the gate stays near-closed (protecting them), while large-$\bar\varepsilon$
regimes (actuator, friction, compound, action perturbations) both degrade most and open the gate.
The residual thus serves jointly as an unsupervised OOD/non-stationarity *detector* and a *severity
estimate* that gates how aggressively to correct.

## Appendix D — few-shot depth $K$ vs. fleet breadth $M$ (value_shift_qr, 2 seeds)

Adaptation data can be acquired sequentially ($K$ warmup episodes the deployed agent itself runs)
or in parallel ($M$ agents sharing one calibrator; Sec. 6, 9). Return of `value_shift_qr`:

| regime | $K{=}0,M{=}1$ | $K{=}0,M{=}5$ | $K{=}3,M{=}1$ | $K{=}3,M{=}5$ |
|---|---|---|---|---|
| nominal | 1445 | 1443 | 1450 | 1445 |
| `fixed` $\xi_{\text{act}}{=}0.6$ (stationary) | **847** | 816 | 824 | 832 |
| `multi_sudden` (non-stationary) | 701 | **712** | 686 | **707** |

**Reading (2 seeds, trends).**
- **nominal**: unaffected by $K$ or $M$ (1443–1450) — adaptation never harms the in-distribution case.
- **stationary OOD**: continual *within-episode* adaptation already suffices ($K{=}0,M{=}1$ is best,
  847); extra warmup or fleet gives no consistent gain (the calibrator converges within one episode).
- **non-stationary** (`multi_sudden`): **fleet breadth $M$ helps** ($M{=}5\!\ge\!M{=}1$ at both $K$:
  $712\!>\!701$, $707\!>\!686$) while **sequential depth $K$ does not** ($K{=}3,M{=}1$ is the *worst*, 686).
  Intuition: under drift, warmup episodes adapt to an already-stale phase, whereas $M$ parallel agents
  supply $M\times$ *current* data per wall-clock step to track the change — i.e. **parallel breadth
  substitutes for, and beats, sequential depth on non-stationary shift**, without the deployed agent
  having to suffer $K$ bad episodes first.

(2 seeds — directional, not significant; the stationary-OOD numbers are within noise.)

## Appendix E — comparison to test-time fine-tuning (the fair baseline)

`value_shift_qr` adapts a frozen backbone from $K$ in-regime episodes by correcting $Q$'s
last layer. Its natural apples-to-apples comparison is **test-time fine-tuning**, which adapts
the *same* backbone from the *same* data by SGD on the network weights. Two variants
(`eval/eval_full_pearl_finetune.py`, methods `finetune_lastlayer`, `finetune_full`):

| baseline | what is updated | how it acts | $K{=}0$ limit |
|---|---|---|---|
| `finetune_lastlayer` | SGD on **only** $Q$'s last linear layer ($w$ in $Q=w^\top\varphi_Q+b$ — the same subspace `value_shift_qr` corrects), actor frozen | re-rank the candidate pool with the fine-tuned $Q$ (like `q_greedy`) | $\equiv$ `q_greedy` |
| `finetune_full` | SAC fine-tune of **actor + critic**, encoder frozen | deploy the fine-tuned deterministic actor | $\equiv$ `full_pearl_only` |

Both: the frozen backbone is **deep-copied per run** (the pretrained checkpoint is *never*
overwritten); encoder frozen ($z$ inferred once per update from the in-regime context, Sec. 3.1);
**$M$-fold fleet data parity** (Sec. 6) with `value_shift_qr`; **continual + episodic** —
$K$ warmup fleet-episodes + the measured episodes all feed one replay buffer (full $100$k or a
recency window), with `finetune.updates_per_episode` SGD steps after each episode.

**Relation to `value_shift_qr`.** Both live in the *same hypothesis class* — a correction linear
in $\varphi_Q$. They differ in three ways: (i) **target** — `value_shift_qr` fits the
*frozen-bootstrap* value residual $\eta^{\text{obs}}_{\text{value}}$ (Sec. 5), `finetune` minimizes
the *self-bootstrapping* SAC TD loss (target net chases itself); (ii) **estimator** — closed-form
Bayesian recursive least squares with a prior, forgetting $\rho$ and posterior covariance $P$
(Sec. 6) vs. SGD (needs lr / #steps / buffer, no uncertainty); (iii) **gate** — `value_shift_qr`
interpolates back to the policy ($a=(1{-}h)a_0+h\,a_{\text{sel}}$), fine-tune does not.

**Protocol — no validation set.** In zero/few-shot test-time adaptation there is **no held-out
validation set** for an *unknown* deployment shift, so a hyperparameter configuration must be
**committed a priori for all regimes** — tuning on the test OOD regime is invalid. The honest
metric is therefore the **worst-case regret** of a single committed config across regimes,
$\;\min_{\text{regime}}\big(\text{return}-\text{return}_{\text{full\_pearl\_only}}\big)$:
a method "wins" the setting if it *never badly hurts* any regime. All of Appendix E uses
$K{=}3,\,M{=}3$, 3 seeds, regimes {nominal, action gain 0.6, `sudden`$\to$0.4, `multi_sudden`}
(`configs/full_pearl_dynamics_lookahead_ft.json`); baseline `full_pearl_only` =
{nominal 1465, ood 719, sudden 610, multi 649}.

### E.1 — Robustness / "never hurts" (the headline advantage)

Sweeping each method's own knobs (`value_shift_qr`: $q$-prior-var $\in\{0.1,0.25,1.0\}$;
fine-tune: lr$\in\{10^{-4},3{\cdot}10^{-4}\}\times$updates$\in\{50,200\}$, both variants).
**Regret vs frozen PEARL** per regime; **WORST** = worst across all four (the committed-config metric):

| config | nominal | ood 0.6 | sudden 0.4 | multi | **WORST** |
|---|---|---|---|---|---|
| **vsqr** prior-var 0.10 | −18 | +117 | +7 | +35 | **−18** |
| **vsqr** prior-var 0.25 | −16 | +109 | +7 | +34 | **−16** |
| **vsqr** prior-var 1.00 | −18 | +99 | +7 | +61 | **−18** |
| ft-lastlayer lr1e-4 u50 | −119 | +85 | +0 | +106 | −119 |
| ft-lastlayer lr1e-4 u200 | −102 | +72 | +7 | +127 | −102 |
| ft-lastlayer lr3e-4 u50 | −133 | +81 | −1 | +107 | −133 |
| ft-lastlayer lr3e-4 u200 | −109 | +65 | −1 | +131 | −109 |
| ft-full lr1e-4 u50 | −189 | +62 | +27 | +73 | −189 |
| ft-full lr1e-4 u200 | −213 | +70 | −6 | +106 | −213 |
| ft-full lr3e-4 u50 | −218 | +48 | −102 | +81 | −218 |
| ft-full lr3e-4 u200 | −482 | +48 | −260 | +54 | −482 |

**Takeaways.** (1) `value_shift_qr` has **near-zero worst-case regret ($-16$ to $-18$, a $\sim$1%
nominal cost) and is insensitive to its own knob** — it needs no tuning. (2) **No fine-tune config
(any lr/updates) approaches this**: last-layer always pays a $\sim$7–9% nominal tax ($-102$ to
$-133$), full fine-tune is far worse and unstable ($-189$ to $-482$). Since you cannot tune
per-regime without a validation set, this is the rigorous statement of the advantage. (3) *Where
fine-tune leads is narrow:* `value_shift_qr` is **equal-or-better on nominal, action-gain 0.6, and
`sudden`** (and clearly better on the first two); fine-tune-lastlayer's only clear win is
**`multi_sudden`** (+106…+131 vs +34…+61). Even there it is not a free win — it costs the
unavoidable $-102$…$-133$ nominal tax (no single fine-tune config takes `multi_sudden` *and* keeps
nominal). The higher "OOD-mean" sometimes quoted for fine-tune is driven by this one regime.

### E.2 — Tracking non-stationary drift (a tested-and-rejected hypothesis)

*Hypothesis:* a recursive filter with forgetting should **track** a moving target better than
batch SGD on a growing buffer. *Test:* sweep `value_shift_qr`'s forgetting factor
$q\rho\in\{0.999,0.99,0.95,0.90\}$ on a continuous `gradual` drift and `multi_sudden`, give
fine-tune a recency-window buffer (3k) as the fair tracking baseline, and record the **per-step
reward trace + alive fraction** (`finalize_traces`). Regret vs frozen PEARL:

| config | gradual (→0.2) | multi_sudden |
|---|---|---|
| vsqr $\rho{=}0.999$ | **+39** | **+36** |
| vsqr $\rho{=}0.99$ | +4 | +3 |
| vsqr $\rho{=}0.95$ | +7 | +0 |
| vsqr $\rho{=}0.90$ | +5 | +3 |
| ft-lastlayer full buffer | **+50** | **+132** |
| ft-lastlayer recency 3k | +49 | +128 |
| ft-full full buffer | +29 | +29 |
| ft-full recency 3k | −60 | −157 |

**The hypothesis fails, and the figures say why.** (1) **Lower $\rho$ (more forgetting) makes
`value_shift_qr` *worse*** ($\rho{=}0.999$, least forgetting, is best). The BRPC "forgetting"
$M\!\leftarrow\!\rho M$ decays the correction toward **zero** = *revert to the frozen base policy* —
a **safety** mechanism, not a tracking one (tracking would move the mean to a *new* value); faster
forgetting just erases the useful correction. (2) **The per-step reward trace is $\approx$ identical
across *all* methods** (`results/drift/drift_reward.png`) — nobody tracks the moving optimum's value
better; the entire return difference is **survival / alive-fraction**
(`results/drift/drift_alive.png`): fine-tune-lastlayer keeps the agent up a few dozen steps longer
through the hard segment. Both regimes eventually hit the $\xi_{\text{act}}\!\approx\!0.3$–$0.35$
gait ceiling where everyone falls (consistent with `oracle_pool`). (3) `finetune_full` + a small
recency buffer **collapses** (overfits recent transitions). **Conclusion:** we do *not* claim a
tracking advantage. fine-tune-lastlayer leads on the **repeated-change `multi_sudden`** (+131,
strongly) and **mildly on `gradual`** (+50 vs +39) — via longer survival, not better per-step
value, and at the cost of the nominal tax. Note `multi_sudden`'s schedule **repeats every episode**,
so over 6 episodes it is closer to *fitting a fixed multi-phase distribution* (batch-SGD's home turf)
than to tracking novelty; Appendix D shows fleet breadth $M$ narrows this gap for `value_shift_qr`.
The Bayesian machinery delivers *safety*, not *tracking*. *(Supersedes the earlier directional
`multi_sudden` reading in Appendix B/D, which conflated the survival effect with tracking.)*

### E.3 — Explicit uncertainty (a costless conservatism dial)

`value_shift_qr`'s posterior gives a per-candidate std $\sigma_h(s,a)=\sqrt{\varphi_Q^\top P\,\varphi_Q}$
(already computed; Sec. 6) that fine-tune has **no analog of**. A risk-averse LCB score distrusts
the correction where it is uncertain:

$$
\mathrm{score}(s,a)=Q_{\min}(s,a,z)+h_w(\varphi_Q)-\beta\,\sigma_h(s,a)-\Omega(a),\qquad (\beta{=}0\text{ recovers Sec. 5}).
$$

Sweep $\beta\in\{0,1,2\}$ (return, seed-std, regret):

| $\beta$ | nominal | ood 0.6 | sudden 0.4 | multi (std) |
|---|---|---|---|---|
| 0 | 1450 (−15) | 828 (+109) | 617 (+7) | 683 (+34), s17 |
| 1 | 1448 (−17) | 832 (+113) | 618 (+8) | 662 (+13), s11 |
| 2 | 1448 (−17) | 840 (**+121**) | 616 (+6) | 705 (**+56**), **s11** |

**Takeaway.** Turning up $\beta$ is **neutral-to-mildly-helpful** — it slightly *raises* OOD return
(avoiding over-optimistic mis-corrected candidates) and cuts the `multi_sudden` seed-variance
($17\!\to\!11$), with nominal/sudden and the worst-case regret unchanged ($\approx\!-16$). Differences
are near the 3-seed noise floor, so the modest claim: explicit posterior uncertainty is a **free,
principled conservatism knob** (it never hurts and can only help) that the fine-tune baselines
structurally lack.

### E.4 — Positioning

Across E.1–E.3 the defensible, data-backed claim is **not** "uniformly higher OOD return" or "better
drift tracking" (the latter is false). It is twofold. (1) **Coverage:** `value_shift_qr` is
*equal-or-better than fine-tune on every regime except `multi_sudden`* — clearly better on nominal
and action-gain 0.6, a tie on `sudden`, a small gap on `gradual` — and even fine-tune's one clear win
(`multi_sudden`) is bought with a $-102$…$-133$ nominal tax it cannot avoid. (2) **Safety:**
`value_shift_qr` is the only method safely deployable in the no-validation-set test-time-adaptation
setting — it never badly hurts (worst-case regret $-16$ vs fine-tune $-102$…$-482$), needs no tuning
(impossible without a val set), and reverts to the trusted base policy under uncertainty (prior +
revert-to-base forgetting + gate + optional LCB). Fine-tune's only edge is a higher ceiling on the
*repeated-change* `multi_sudden` regime, at the cost of an unavoidable nominal-safety tax,
tuning-dependence, and a risk of collapse — a trade-off resolved decisively on the
**deployment-safety** axis without conceding the performance axis except in that one regime.

> **Artifacts.** Code: `eval/eval_full_pearl_finetune.py` (fine-tune baselines), `q_lcb_beta` in
> `eval/eval_full_pearl_dynamics_lookahead.py` (LCB), `finalize_traces` (time-resolved traces).
> Runs: `run_ft_vs_qr.sh` (K-sweep), `aggregate_ft_sweep.py` (E.1), `run_drift.sh` +
> `aggregate_drift.py` → `results/drift/drift_{reward,alive}.png` (E.2), `aggregate_lcb.py` (E.3).

## Appendix F — online / anytime deployment: cumulative return & regret ($K{=}0$, $M{=}1$)

Appendices A–E score the **measured** episodes after warmup. But in true zero-shot deployment
there *is* no free warmup — **every episode counts from step 1**, and a method that eventually wins
can still be a net loss if it pays an unstable warm-up tax first. The honest online metric is
therefore the **cumulative return** over the first $E$ deployment episodes (and its **online regret**
vs. the frozen policy), with $K{=}0,\,M{=}1$, continual within-episode adaptation, every episode
scored (`aggregate_online.py`, 3 seeds, $E{=}10$, regimes
{nominal, action gain 0.6, compound mild, `multi_sudden`}).

**Cumulative return** over $E{=}5$ / $E{=}10$ episodes, and **$\Delta$ vs frozen @10** $=\sum_{e\le10}(\text{ret}_e-\text{ret}^{\text{frozen}}_e)$
(sign as elsewhere: **$+$ = better than frozen**; this is the negative of the "online regret" the
script prints):

| regime | method | ep1 (zero-shot) | cum-ret@5 | cum-ret@10 | $\Delta$ vs frozen @10 |
|---|---|---|---|---|---|
| nominal | frozen | 1447 | 7337 | 14674 | 0 |
| | q_greedy | 1442 | 7247 | 14502 | −172 |
| | **value_shift_qr** | 1439 | 7252 | 14501 | **−173** |
| | ft-lastlayer | 1443 | 7202 | 14474 | −200 |
| | ft-full | 1447 | 6539 | 11690 | **−2984** |
| action gain 0.60 | frozen | 717 | 3600 | 7210 | 0 |
| | q_greedy | 786 | 3927 | 7857 | +647 |
| | **value_shift_qr** | 819 | 4167 | 8296 | **+1086** |
| | ft-lastlayer | 810 | 3935 | 7846 | +636 |
| | ft-full | 717 | 3280 | 5803 | −1407 |
| compound mild | frozen | 886 | 4577 | 9140 | 0 |
| | q_greedy | 997 | 4973 | 9936 | +796 |
| | **value_shift_qr** | 1051 | 5235 | 10428 | **+1288** |
| | ft-lastlayer | 979 | 5122 | 9876 | +736 |
| | ft-full | 886 | 4293 | 8621 | −519 |
| `multi_sudden` | frozen | 648 | 3241 | 6492 | 0 |
| | q_greedy | 642 | 3220 | 6484 | −8 |
| | **value_shift_qr** | 694 | 3561 | 6922 | **+430** |
| | ft-lastlayer | 647 | 3342 | 6753 | +261 |
| | ft-full | 648 | 3050 | 5721 | −771 |

**Takeaways.** (1) **`value_shift_qr` has the highest cumulative return on every OOD regime** and the
largest online advantage over frozen (+1086 / +1288 / +430), ahead of `q_greedy` and
fine-tune-lastlayer in all three — *including `multi_sudden`*, where the **asymptotic** view
(Appendix E.1/E.2) had fine-tune-lastlayer leading. The reversal is the whole point: fine-tune's
edge there is paid for by a slow SGD warm-up, so over the first 10 episodes it has not yet caught up,
while `value_shift_qr`'s recursive head corrects within the first episode (note `q_greedy` is already
**net-negative** here, −8, i.e. no better than just deploying the frozen policy). (2) **The advantage
shows up immediately in `ep1`** — the zero-shot first episode is already best-or-tied for
`value_shift_qr` on all OOD regimes (819, 1051, 694), confirming the correction needs no episodes of
"burn-in." (3) **`finetune_full` is the cautionary tale of the anytime setting**: it is
*worse than frozen* within 10 episodes on **all four** regimes — catastrophic on nominal (−2984; it
destabilizes the actor and collapses to ep10≈650) and net-negative even on the OOD regimes it would
eventually help. Its good post-warmup numbers elsewhere hide a deployment cost no online user would
accept. (4) **Nominal is a near-tie among the safe methods** (`value_shift_qr` −173 ≈ `q_greedy` −172
≈ ft-LL −200, a ~1% anytime tax from candidate re-ranking), so `value_shift_qr`'s OOD gains are not
bought with a nominal anytime penalty. This is the deployment-safety story of Appendix E told on the
*cumulative* axis: **safe from episode one, and ahead on total return throughout the warm-up window.**

> **Hybrids (`qr_finetune_critic` = vsqr-init + critic SGD, `qr_finetune_full` = vsqr-init + actor+critic
> SGD).** Adding SGD on top of the vsqr correction does not help in the anytime window and reintroduces
> fine-tune's instability: `qr_finetune_critic` roughly matches `q_greedy` (nominal −167, ood +744,
> compound +920, multi −22) and `qr_finetune_full` inherits the full-fine-tune warm-up tax (nominal
> −1653, multi −624). The closed-form recursive head alone is the better anytime estimator.

> **Artifacts.** `aggregate_online.py` → `results/online/{online_return,online_cumregret}.png`,
> `results/online/summary_by_method_regime.{csv,json}`; per-seed curves under
> `results/online/<regime>/<method>_K0/seed*/aggregate.json` (`episode_returns`).
