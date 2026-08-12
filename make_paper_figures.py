"""Generate paper figures:
  1) docs/figures/hopper_task.png            - Hopper rollout filmstrip (trained PEARL policy)
  2) docs/figures/bellman_residual_timeseries.png - |eta| over time: nominal vs OOD vs sudden(shift@100)
"""
from __future__ import annotations
import csv, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)


def fig_hopper_filmstrip():
    import gymnasium as gym
    import torch
    from pearl_brpc_action_adapter.eval.eval_full_pearl import load_checkpoint
    from pearl_brpc_action_adapter.eval.eval_full_pearl_dynamics_lookahead import infer_z
    from pearl_brpc_action_adapter.experiments.train_full_pearl import _make_context_item

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, meta, models = load_checkpoint(Path("checkpoints/full_pearl/full_pearl_best.pt"), device)
    env = gym.make("Hopper-v4", render_mode="rgb_array")
    from collections import deque
    ctx = deque(maxlen=50)
    s, _ = env.reset(seed=3)
    frames, ts = [], []
    capture = set(range(15, 15 + 6 * 11, 11))  # 6 frames across a hop cycle
    for t in range(120):
        z = infer_z(models["encoder"], list(ctx), meta["latent_dim"], 5, device)
        with torch.no_grad():
            a = models["actor"].deterministic_action(
                torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device), z)[0].cpu().numpy().astype(np.float32)
        s2, r, term, trunc, _ = env.step(a)
        if t in capture:
            frames.append(env.render()); ts.append(t)
        ctx.append(_make_context_item(s, a, r, s2, term or trunc))
        s = s2
        if term or trunc:
            s, _ = env.reset(seed=3); ctx.clear()
    env.close()
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.4))
    for ax, fr, t in zip(axes, frames, ts):
        ax.imshow(fr); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"t={t}", fontsize=10)
    fig.suptitle("Hopper-v4 — frozen full-PEARL policy (nominal dynamics)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "hopper_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "hopper_task.png")


def fig_bellman_timeseries():
    path = Path("results/full_pearl_bellman_diag/transition_bellman_residuals.csv")
    rows = list(csv.DictReader(open(path)))
    # EWMA |eta| (the gate input) averaged across episodes per t -> smooth, detector-relevant
    ew = collections.defaultdict(lambda: collections.defaultdict(list))   # regime -> t -> [ewma_abs_eta]
    by = collections.defaultdict(lambda: collections.defaultdict(list))   # regime -> t -> [abs_eta]
    epi = collections.defaultdict(lambda: collections.defaultdict(list))  # regime -> (seed,ep) -> [(t,abs_eta)]
    for r in rows:
        reg, t = r["regime"], int(r["t"])
        ew[reg][t].append(float(r["ewma_abs_eta"]))
        by[reg][t].append(float(r["abs_eta"]))
        epi[reg][(r["seed"], r["episode"])].append((t, float(r["abs_eta"])))

    def profile(d, min_n=22, tmax=185):
        # median across episodes per t: robust to the heavy right tail / survivorship
        ts = sorted(t for t in d if len(d[t]) >= min_n and t <= tmax)
        return np.array(ts), np.array([np.median(d[t]) for t in ts])

    def phase_bar(regime, lo=None, hi=None):
        # per-episode mean |eta| (optionally restricted to a t-window) -> mean +/- std across episodes
        vals = []
        for _, seq in epi[regime].items():
            xs = [a for (t, a) in seq if (lo is None or t >= lo) and (hi is None or t < hi)]
            if xs:
                vals.append(np.mean(xs))
        return (np.mean(vals), np.std(vals)) if vals else (0.0, 0.0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})

    # ---- left: windowed |eta| temporal profile (20-step windows -> low noise) ----
    # NOTE: cross-episode per-time averages are confounded by gait phase AND survivorship
    # (struggling sudden episodes terminate early); the rigorous, phase-controlled result is
    # panel (b). This panel is the qualitative temporal view only.
    def windowed(regime, w=20, tmax=200, min_n=15):
        d = by[regime]
        cs, ms, es = [], [], []
        for a in range(0, tmax, w):
            vals = [v for t in range(a, a + w) if t in d for v in d[t]]
            if len(vals) >= min_n:
                cs.append(a + w / 2); ms.append(np.mean(vals)); es.append(np.std(vals) / np.sqrt(len(vals)))
        return np.array(cs), np.array(ms), np.array(es)

    for regime, label, color in [
        ("nominal", "nominal", "#1b9e77"),
        ("sudden_actuator", r"sudden ($\xi_{act}{:}1.0{\to}0.6$ @ $t{=}100$)", "#1f6fd6"),
    ]:
        if regime in by:
            c, m, e = windowed(regime)
            axL.plot(c, m, "-o", label=label, color=color, lw=2, ms=4)
            axL.fill_between(c, m - e, m + e, color=color, alpha=0.18)
    axL.axvline(100, ls="--", color="k", alpha=0.6, lw=1.2)
    axL.text(103, axL.get_ylim()[1] * 0.92, "shift", fontsize=9)
    axL.set_xlabel("environment step  $t$")
    axL.set_ylabel(r"mean $|\eta_t|=|r+\gamma V(s')-Q_{\min}|$  (20-step windows)")
    axL.set_title("(a) Residual over time: nominal vs. sudden shift")
    axL.legend(loc="upper left", fontsize=8.5)
    axL.grid(alpha=0.25)

    # ---- right: PHASE-CONTROLLED bars (the change-point is the gait-phase-controlled gap) ----
    bars = [
        ("nominal\npre", phase_bar("nominal", hi=100), "#a6dba0"),
        ("nominal\npost", phase_bar("nominal", lo=100), "#1b9e77"),
        ("sudden\npre", phase_bar("sudden_actuator", hi=100), "#9ecae1"),
        ("sudden\npost", phase_bar("sudden_actuator", lo=100), "#1f6fd6"),
        ("OOD 0.6\n(all $t$)", phase_bar("ood_actuator_0.60"), "#d95f02"),
    ]
    xs = np.arange(len(bars))
    axR.bar(xs, [b[1][0] for b in bars], yerr=[b[1][1] for b in bars],
            color=[b[2] for b in bars], capsize=4, alpha=0.9)
    axR.set_xticks(xs); axR.set_xticklabels([b[0] for b in bars], fontsize=8)
    axR.set_ylabel(r"mean $|\eta|$ (per-episode, $\pm$ std)")
    axR.set_title("(b) Phase-controlled: shift adds residual\nbeyond the early/late-episode trend")
    axR.grid(alpha=0.25, axis="y")
    for x, b in zip(xs, bars):
        axR.text(x, b[1][0] + 0.02, f"{b[1][0]:.2f}", ha="center", fontsize=8)
    # annotate the phase-controlled shift effect: sudden-post vs nominal-post
    np_post = phase_bar("nominal", lo=100)[0]; sp_post = phase_bar("sudden_actuator", lo=100)[0]
    axR.annotate("", xy=(3, sp_post), xytext=(3, np_post),
                 arrowprops=dict(arrowstyle="<->", color="k"))
    axR.text(3.15, (sp_post + np_post) / 2, f"shift\n+{100*(sp_post/np_post-1):.0f}%", fontsize=8)

    fig.suptitle("Bellman residual as an unsupervised OOD / non-stationarity signal", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "bellman_residual_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "bellman_residual_timeseries.png")


def fig_residual_bars():
    """Standalone phase-controlled bar chart (no panel label in the title)."""
    path = Path("results/full_pearl_bellman_diag/transition_bellman_residuals.csv")
    rows = list(csv.DictReader(open(path)))
    epi = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        epi[r["regime"]][(r["seed"], r["episode"])].append((int(r["t"]), float(r["abs_eta"])))

    def phase_bar(regime, lo=None, hi=None):
        vals = []
        for _, seq in epi[regime].items():
            xs = [a for (t, a) in seq if (lo is None or t >= lo) and (hi is None or t < hi)]
            if xs:
                vals.append(np.mean(xs))
        return (np.mean(vals), np.std(vals)) if vals else (0.0, 0.0)

    bars = [
        ("nominal\npre", phase_bar("nominal", hi=100), "#a6dba0"),
        ("nominal\npost", phase_bar("nominal", lo=100), "#1b9e77"),
        ("sudden\npre", phase_bar("sudden_actuator", hi=100), "#9ecae1"),
        ("sudden\npost", phase_bar("sudden_actuator", lo=100), "#1f6fd6"),
        ("OOD 0.6\n(all $t$)", phase_bar("ood_actuator_0.60"), "#d95f02"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1][0] for b in bars], yerr=[b[1][1] for b in bars],
           color=[b[2] for b in bars], capsize=4, alpha=0.9)
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel(r"mean Bellman residual $|\eta|=|r+\gamma V(s')-Q_{\min}|$  (per-episode, $\pm$ std)")
    ax.set_title("Bellman residual rises out-of-distribution and after a change-point\n"
                 "(phase-controlled: the shift adds residual beyond the early/late-episode trend)",
                 fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    for x, b in zip(xs, bars):
        ax.text(x, b[1][0] + 0.02, f"{b[1][0]:.2f}", ha="center", fontsize=9)
    np_post = phase_bar("nominal", lo=100)[0]; sp_post = phase_bar("sudden_actuator", lo=100)[0]
    ax.annotate("", xy=(3, sp_post), xytext=(3, np_post), arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(3.18, (sp_post + np_post) / 2, f"shift\n+{100*(sp_post/np_post-1):.0f}%", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "bellman_residual_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "bellman_residual_bars.png")


if __name__ == "__main__":
    import sys
    if "--bars-only" in sys.argv:
        fig_residual_bars()
    else:
        try:
            fig_hopper_filmstrip()
        except Exception as ex:
            print("hopper figure failed:", repr(ex)[:300])
        fig_bellman_timeseries()
        fig_residual_bars()
