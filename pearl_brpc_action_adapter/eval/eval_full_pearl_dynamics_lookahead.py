"""Value-shift action re-ranking on top of a frozen full-PEARL policy.

The base score for a candidate action is PEARL's own critic, Q_pearl(s,a,z). A
frozen offline world model f(s,a,z)->delta_s provides the nominal next-state, and
an ONLINE BRPC residual Df(s,a,z) (fit continually to the realized one-step
dynamics error) carries the distribution shift. The shift is injected only as a
*value correction*:

    score(s,a) = Q_pearl(s,a,z)
               + w_fwd * Df(s,a,z)[vx]                          (immediate reward shift, optional)
               + gamma * ( V(s + f + Df) - V(s + f) )           (value-to-go shift, dV)

with V the deterministic critic value V(s',z) = min_i Q_i(s', pi_det(s',z), z).

Properties:
  * Nominal: Df -> 0  =>  dV -> 0  =>  score == Q_pearl (pure Q re-selection); a gate
    further interpolates a = (1-h) a0 + h a_sel so nominal is protected (h -> 0).
  * The world model's own bias cancels in the dV difference (both terms use f), so the
    OOD signal comes from the online Df, not from trusting f's absolute prediction.
  * Without BRPC (Df == 0) value_shift reduces exactly to q_greedy; the gain over
    q_greedy is therefore attributable to the few-shot online adaptation.

Methods:
  full_pearl_only : execute a0 (frozen PEARL deterministic action)
  q_greedy        : re-rank candidates by Q_pearl only (zero-shot control)
  value_shift     : re-rank by the full score above (few-shot, continual BRPC)
"""
from __future__ import annotations

import json
import math
from collections import deque
from itertools import zip_longest
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch

from brpc_latent_mpc.brpc.bayes_linear_residual import BRPCResidualCalibrator
from pearl_brpc_action_adapter.brpc.action_adapter import build_rff
from pearl_brpc_action_adapter.dynamics.dynamics_model import NormalizedLatentDynamicsModel
from pearl_brpc_action_adapter.envs.make_env import EvalDynamicsSchedule, make_env
from pearl_brpc_action_adapter.envs.action_perturb import maybe_wrap_action_perturb
from pearl_brpc_action_adapter.eval.eval_full_pearl import load_checkpoint
from pearl_brpc_action_adapter.eval.eval_full_pearl_residual_q import (
    feature_inputs,
    make_candidates,
    q_min_batch,
)
from pearl_brpc_action_adapter.experiments.train_full_pearl import _context_tensor, _make_context_item


METHODS = [
    "full_pearl_only", "q_greedy", "value_shift", "value_shift_h", "value_shift_qr",
    "oracle_pool", "finetune_lastlayer", "finetune_full", "qr_finetune_critic", "qr_finetune_full",
]

# Test-time fine-tuning baselines (handled in their own module / evaluate branch).
_FINETUNE_METHODS = ("finetune_lastlayer", "finetune_full")

# value_shift/value_shift_h fit the dynamics residual Δf; value_shift_qr fits the value
# (Bellman) residual η_value in Q's own feature space. value_shift_qr re-ranks by the
# corrected Q only (Q + h_value), isolating the "fix Q" effect from the dynamics correction.
_DYN_BRPC_METHODS = ("value_shift", "value_shift_h")
_Q_RESIDUAL_METHODS = ("value_shift_qr",)


def uses_dyn_brpc(method: str) -> bool:
    return method in _DYN_BRPC_METHODS


def uses_q_residual(method: str) -> bool:
    return method in _Q_RESIDUAL_METHODS


def uses_brpc(method: str) -> bool:
    """Any online adapter (gets warmup + a calibrator + can use fleet sharing)."""
    return uses_dyn_brpc(method) or uses_q_residual(method)


def is_finetune(method: str) -> bool:
    return method in _FINETUNE_METHODS


def fleet_capable(method: str) -> bool:
    # Fine-tune baselines also use the fleet (M-fold in-regime data) for parity.
    return uses_brpc(method) or is_finetune(method)


_DEFAULT_XI = {"mass": 1.0, "friction": 1.0, "damping": 1.0, "actuator": 1.0}


def analytic_reward(actions: torch.Tensor, s_next: torch.Tensor, lk_cfg: Dict) -> torch.Tensor:
    """Hopper analytic reward from (a, predicted next-state): healthy + fwd*vx(s') - ctrl*||a||^2.
    Used for multi-horizon rollouts; for H=1 the difference reduces to fwd*Δf[vx]."""
    healthy = float(lk_cfg.get("healthy_reward", 1.0))
    fwd = float(lk_cfg.get("forward_reward_weight", 1.0))
    ctrl = float(lk_cfg.get("ctrl_cost_weight", 1e-3))
    vx = int(lk_cfg.get("vx_index", 5))
    return healthy + fwd * s_next[:, vx] - ctrl * (actions ** 2).sum(dim=-1)


class MultiSuddenSchedule:
    """Piecewise-constant dynamics with multiple sudden changes.

    regime = {"type": "multi_sudden", "changes": [{"step": 0, "actuator": 1.0},
              {"step": 120, "actuator": 0.5}, {"step": 260, "actuator": 0.85}, ...]}
    At step t the active xi is the last change whose step <= t (default xi before the
    first change). Each change dict overrides only the latent keys it lists.
    """

    def __init__(self, regime: Dict):
        changes = sorted(regime.get("changes", []), key=lambda c: int(c.get("step", 0)))
        self.changes = changes

    def xi_at(self, t: int) -> Dict[str, float]:
        xi = dict(_DEFAULT_XI)
        for c in self.changes:
            if int(c.get("step", 0)) <= t:
                xi.update({k: float(v) for k, v in c.items() if k != "step"})
            else:
                break
        return xi


def make_schedule(regime: Dict, train_range: Dict):
    """Local schedule factory: adds multi_sudden on top of the shared EvalDynamicsSchedule
    (nominal / fixed / sudden / gradual) without modifying the read-only brpc_latent_mpc pkg."""
    if regime.get("type") == "multi_sudden":
        return MultiSuddenSchedule(regime)
    return EvalDynamicsSchedule(regime, train_range)


def make_candidates_local(actor, state, z, a0, action_low, action_high, lk_cfg, rng, device):
    """Candidate pool around a0. Supports multi-scale Gaussian shells via
    `local_action_stds` (list) for a wider search, plus policy samples; falls back to a
    single `local_action_std`. Always includes a0."""
    n = int(lk_cfg.get("num_candidates", 64))
    policy_frac = float(lk_cfg.get("policy_candidate_frac", 0.25))
    n_policy = int(round(max(n - 1, 0) * policy_frac))
    n_local = max(n - 1 - n_policy, 0)
    stds = lk_cfg.get("local_action_stds") or [float(lk_cfg.get("local_action_std", 0.1))]
    stds = [float(x) for x in stds]
    locals_ = []
    if n_local > 0:
        per = [n_local // len(stds)] * len(stds)
        for j in range(n_local - sum(per)):
            per[j % len(stds)] += 1
        for std, cnt in zip(stds, per):
            if cnt > 0:
                locals_.append(a0[None, :] + rng.normal(0.0, std, size=(cnt, a0.shape[0])).astype(np.float32))
    local = np.concatenate(locals_, axis=0) if locals_ else np.zeros((0, a0.shape[0]), np.float32)
    policy = np.zeros((0, a0.shape[0]), dtype=np.float32)
    if n_policy > 0:
        st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device).expand(n_policy, -1)
        with torch.no_grad():
            policy = actor.sample_action_logprob(st, z.expand(n_policy, -1))[0].cpu().numpy().astype(np.float32)
    candidates = np.concatenate([a0[None, :], local, policy], axis=0)
    return np.clip(candidates, action_low, action_high).astype(np.float32)


def brpc_measurement_update(brpc: BRPCResidualCalibrator, feat: np.ndarray, resid: np.ndarray) -> None:
    """Kalman measurement update WITHOUT the AR(1) predict_step. Used for fleet/batch
    updates where many simultaneous observations share a single time step: call
    brpc._predict_step() once per wall-clock step, then this once per observation."""
    b = np.asarray(feat, dtype=np.float64).reshape(-1)
    r = brpc._clip_residual(np.asarray(resid, dtype=np.float64).reshape(-1))
    if np.ndim(brpc.obs_noise) == 0:
        obs_var = brpc.obs_noise ** 2
    else:
        obs_var = float(np.mean(brpc.obs_noise ** 2))
    S = obs_var + b @ brpc.P @ b
    K = (brpc.P @ b) / max(S, 1e-12)
    err = r - b @ brpc.M
    brpc.M = brpc.M + np.outer(K, err)
    brpc.P = brpc.P - np.outer(K, b) @ brpc.P
    brpc.P = 0.5 * (brpc.P + brpc.P.T) + 1e-6 * np.eye(brpc.feature_dim)


def load_dynamics(path: Path, meta: Dict, device: torch.device) -> NormalizedLatentDynamicsModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = NormalizedLatentDynamicsModel(
        meta["state_dim"], meta["action_dim"], meta["latent_dim"], tuple(payload["hidden"])
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload.get("nominal_resid", {})


def infer_z(encoder, context: List[np.ndarray], latent_dim: int, min_context: int, device: torch.device) -> torch.Tensor:
    if len(context) < min_context:
        return torch.zeros(1, latent_dim, device=device)
    with torch.no_grad():
        return encoder.infer_mean(_context_tensor(context, device))


def v_det_batch(actor, q1, q2, next_states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Deterministic critic value V(s',z) = min_i Q_i(s', pi_det(s',z), z), batched."""
    z_b = z.expand(next_states.shape[0], -1)
    with torch.no_grad():
        a_det = actor.deterministic_action(next_states, z_b)
        v = torch.minimum(q1(next_states, a_det, z_b), q2(next_states, a_det, z_b)).squeeze(-1)
    return v


def q_min_single(models: Dict, state: np.ndarray, action: np.ndarray, z: torch.Tensor) -> float:
    device = next(models["actor"].parameters()).device
    st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    at = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        q = torch.minimum(models["q1"](st, at, z), models["q2"](st, at, z))
    return float(q.cpu().item())


def brpc_predict_batch_vec(brpc: BRPCResidualCalibrator, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vector BRPC prediction. Returns mean (N, output_dim) and scalar var (N,)."""
    b = np.asarray(features, dtype=np.float64)
    mean = b @ brpc.M  # (N, output_dim)
    var = np.einsum("bi,ij,bj->b", b, brpc.P, b)  # shared posterior cov across outputs
    return mean.astype(np.float32), np.maximum(var, 0.0).astype(np.float32)


def select_index(scores: np.ndarray, lk_cfg: Dict, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
    selection = lk_cfg.get("selection", "argmax")
    if selection == "argmax":
        weights = np.zeros_like(scores, dtype=np.float64)
        idx = int(np.argmax(scores))
        weights[idx] = 1.0
        return idx, weights
    if selection == "topk_sample":
        tau = max(float(lk_cfg.get("topk_softmax_tau", 1.0)), 1e-6)
        topk = max(1, min(int(lk_cfg.get("topk", 8)), scores.shape[0]))
        top_idx = np.argpartition(scores, -topk)[-topk:]
        top_scores = scores[top_idx]
        w = np.exp((top_scores - np.max(top_scores)) / tau)
        w = w / max(np.sum(w), 1e-12)
        weights = np.zeros_like(scores, dtype=np.float64)
        weights[top_idx] = w
        idx = int(rng.choice(scores.shape[0], p=weights / max(weights.sum(), 1e-12)))
        return idx, weights
    raise ValueError(f"Unknown selection mode: {selection}")


def compute_scores(
    method: str,
    cfg: Dict,
    lk_cfg: Dict,
    models: Dict,
    dyn: NormalizedLatentDynamicsModel,
    brpc: BRPCResidualCalibrator,
    rff,
    state: np.ndarray,
    z: torch.Tensor,
    candidates: np.ndarray,
    a0: np.ndarray,
    prev_action: np.ndarray,
    action_scale: float,
    device: torch.device,
    q_brpc: BRPCResidualCalibrator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, q_scores) for the candidate set."""
    n = candidates.shape[0]
    states_t = torch.from_numpy(np.repeat(state[None, :].astype(np.float32), n, axis=0)).to(device)
    actions_t = torch.from_numpy(candidates.astype(np.float32)).to(device)
    z_b = z.expand(n, -1)
    with torch.no_grad():
        q_scores = q_min_batch(models["q1"], models["q2"], states_t, actions_t, z).cpu().numpy()

    lambda_a = float(lk_cfg.get("lambda_action", 1.0))
    lambda_sw = float(lk_cfg.get("lambda_switch", 0.0))
    trust = lambda_a * np.sum((candidates - a0[None, :]) ** 2, axis=1)
    switch = lambda_sw * np.sum((candidates - prev_action[None, :]) ** 2, axis=1)

    if method == "q_greedy":
        return q_scores - trust - switch, q_scores

    if method == "value_shift_qr":
        # Correct Q's own value error with an online Bellman-residual head fit in Q's
        # penultimate feature space (where Q is linear), then re-rank by Q + h_value.
        with torch.no_grad():
            phi = models["q1"].features(states_t, actions_t, z_b).cpu().numpy()
        h_value, h_var = brpc_predict_batch_vec(q_brpc, phi)  # mean (N, 1), var (N,)
        # Optional risk-averse LCB: distrust the correction where the posterior is wide
        # (phi in an under-observed region). beta=0 recovers the plain mean correction.
        # The posterior std sigma = sqrt(phi^T P phi) is information fine-tune has no analog of.
        beta = float(lk_cfg.get("q_lcb_beta", 0.0))
        score = q_scores + h_value[:, 0]
        if beta != 0.0:
            score = score - beta * np.sqrt(np.maximum(h_var, 0.0))
        return score - trust - switch, q_scores

    gamma = float(cfg["pearl"]["gamma"])
    z_np = z[0].detach().cpu().numpy().astype(np.float32)

    if method == "value_shift_h":
        # Multi-horizon: roll nominal (f) and corrected (f+Δf) trajectories H steps under the
        # policy, take the discounted-return difference (+ terminal V bootstrap), add to Q.
        # Model bias largely cancels in the nominal-vs-corrected difference; the OOD signal is Δf.
        H = max(int(lk_cfg.get("horizon", 5)), 1)
        s_nom = states_t.clone()
        s_corr = states_t.clone()
        a_nom = actions_t.clone()
        a_corr = actions_t.clone()
        ret_nom = torch.zeros(n, device=device)
        ret_corr = torch.zeros(n, device=device)
        for k in range(H):
            with torch.no_grad():
                d_nom = dyn.predict(s_nom, a_nom, z_b)
                d_corr_base = dyn.predict(s_corr, a_corr, z_b)
            feats_k = rff.transform(
                feature_inputs(s_corr.cpu().numpy(), a_corr.cpu().numpy(), z_np, lk_cfg, action_scale)
            )
            mu_k, _ = brpc_predict_batch_vec(brpc, feats_k)
            s_nom_n = s_nom + d_nom
            s_corr_n = s_corr + d_corr_base + torch.from_numpy(mu_k).to(device)
            ret_nom = ret_nom + (gamma ** k) * analytic_reward(a_nom, s_nom_n, lk_cfg)
            ret_corr = ret_corr + (gamma ** k) * analytic_reward(a_corr, s_corr_n, lk_cfg)
            s_nom, s_corr = s_nom_n, s_corr_n
            with torch.no_grad():
                a_nom = models["actor"].deterministic_action(s_nom, z_b)
                a_corr = models["actor"].deterministic_action(s_corr, z_b)
        v_nom = v_det_batch(models["actor"], models["q1"], models["q2"], s_nom, z)
        v_corr = v_det_batch(models["actor"], models["q1"], models["q2"], s_corr, z)
        ret_nom = ret_nom + (gamma ** H) * v_nom
        ret_corr = ret_corr + (gamma ** H) * v_corr
        dReturn = (ret_corr - ret_nom).cpu().numpy()
        return q_scores + dReturn - trust - switch, q_scores

    # value_shift (H=1): Q + reward-shift + gamma * dV
    with torch.no_grad():
        delta_nom = dyn.predict(states_t, actions_t, z_b)  # (N, sd) real units
    feats = rff.transform(feature_inputs(np.repeat(state[None, :], n, axis=0), candidates, z_np, lk_cfg, action_scale))
    mu_shift, _ = brpc_predict_batch_vec(brpc, feats)  # (N, sd)
    mu_shift_t = torch.from_numpy(mu_shift).to(device)
    s_nom = states_t + delta_nom
    s_shift = states_t + delta_nom + mu_shift_t
    v_nom = v_det_batch(models["actor"], models["q1"], models["q2"], s_nom, z)
    v_shift = v_det_batch(models["actor"], models["q1"], models["q2"], s_shift, z)
    dV = (v_shift - v_nom).cpu().numpy()

    reward_shift = np.zeros(n, dtype=np.float32)
    if bool(lk_cfg.get("use_reward_shift", True)):
        vx = int(lk_cfg.get("vx_index", 5))
        reward_shift = float(lk_cfg.get("forward_reward_weight", 1.0)) * mu_shift[:, vx]

    scores = q_scores + reward_shift + gamma * dV - trust - switch
    return scores, q_scores


def pick_action(
    method: str,
    cfg: Dict,
    lk_cfg: Dict,
    models: Dict,
    dyn,
    brpc,
    rff,
    s: np.ndarray,
    z: torch.Tensor,
    a0: np.ndarray,
    prev_action: np.ndarray,
    gate_h: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_scale: float,
    rng: np.random.Generator,
    device: torch.device,
    q_brpc=None,
) -> Tuple[np.ndarray, float]:
    """Return (action, dV_sel) for one decision. Shared by single-agent and fleet paths."""
    if method == "full_pearl_only":
        return a0, 0.0
    candidates = make_candidates_local(models["actor"], s, z, a0, action_low, action_high, lk_cfg, rng, device)
    scores, q_scores = compute_scores(
        method, cfg, lk_cfg, models, dyn, brpc, rff, s, z, candidates, a0, prev_action, action_scale, device, q_brpc
    )
    idx, _ = select_index(scores, lk_cfg, rng)
    a_sel = candidates[idx]
    action = np.clip((1.0 - gate_h) * a0 + gate_h * a_sel, action_low, action_high).astype(np.float32)
    return action, float(scores[idx] - q_scores[idx])


def world_residual(dyn, s: np.ndarray, action: np.ndarray, s_next: np.ndarray, z: torch.Tensor, device: torch.device):
    """Raw one-step residual of the frozen world model on the executed action.
    Returns (raw_resid, ||raw_resid||, delta_pred) where delta_pred = f(s,a,z)."""
    with torch.no_grad():
        delta_pred = dyn.predict(
            torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device),
            torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device),
            z,
        )[0].cpu().numpy().astype(np.float32)
    raw_resid = (s_next - s).astype(np.float32) - delta_pred
    return raw_resid, float(np.linalg.norm(raw_resid)), delta_pred


def step_adaptation(
    method, lk_cfg, models, dyn, dyn_brpc, q_brpc, rff,
    s, action, s_next, reward, terminated, z, action_scale, gamma, device,
):
    """Compute the gate residual norm and the pending online-adapter update(s) for the
    executed transition. Returns (resid_norm, pending, qr_record):
      * pending = list of (calibrator, feature, target) applied immediately / batched (the
        1-step path; qr target uses the FROZEN-NOMINAL bootstrap V(s+f), value-error only).
      * qr_record = per-step record for the n-step / lambda-return QR target (uses the REAL
        next-state bootstrap V(s'), assembled at episode end); None unless uses_q_residual."""
    raw_resid, resid_norm, delta_pred = world_residual(dyn, s, action, s_next, z, device)
    z_np = z[0].detach().cpu().numpy().astype(np.float32)
    pending = []
    qr_record = None
    if uses_dyn_brpc(method):
        feat = rff.transform(feature_inputs(s[None, :], action[None, :], z_np, lk_cfg, action_scale))[0]
        pending.append((dyn_brpc, feat, raw_resid))
    if uses_q_residual(method):
        # η_value = r + γ V(s'_nom) − Q_min(s,a), with s'_nom = s + f (dynamics-shift part excluded).
        s_nom = torch.from_numpy((s + delta_pred).astype(np.float32)).unsqueeze(0).to(device)
        v_nom = 0.0 if terminated else float(v_det_batch(models["actor"], models["q1"], models["q2"], s_nom, z)[0].item())
        q_min = q_min_single(models, s, action, z)
        eta_value = float(reward) + gamma * v_nom - q_min
        with torch.no_grad():
            phi = models["q1"].features(
                torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device),
                torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device),
                z,
            )[0].cpu().numpy().astype(np.float32)
        pending.append((q_brpc, phi, np.asarray([eta_value], dtype=np.float32)))
        # n-step/lambda use the REAL next-state bootstrap (so n real rewards reduce reliance
        # on the possibly-OOD-wrong frozen V — the single-step bias the head otherwise inherits).
        sn = torch.from_numpy(s_next.astype(np.float32)).unsqueeze(0).to(device)
        v_real_next = 0.0 if terminated else float(v_det_batch(models["actor"], models["q1"], models["q2"], sn, z)[0].item())
        qr_record = {"phi": phi, "r": float(reward), "q": float(q_min), "v_next": v_real_next}
    return resid_norm, pending, qr_record


def qr_nstep_mode(lk_cfg: Dict) -> bool:
    """True when the QR head should use multi-step (n-step or lambda) returns instead of the
    immediate 1-step backup."""
    return int(lk_cfg.get("q_nstep", 1)) > 1 or float(lk_cfg.get("q_lambda", 0.0)) > 0.0


def qr_traj_targets(traj: List[Dict], gamma: float, lk_cfg: Dict) -> List[tuple]:
    """Assemble (phi, eta) QR targets from one episode's per-step records using the REAL
    observed return. lambda-return (q_lambda>0) takes precedence over fixed n-step (q_nstep).
    Each record: {phi, r, q, v_next} with v_next = V(s') (0 at terminal). eta = G - Q(s,a)."""
    L = len(traj)
    if L == 0:
        return []
    lam = float(lk_cfg.get("q_lambda", 0.0))
    out = []
    if lam > 0.0:
        # Backward lambda-return: G_t = r_t + γ[(1-λ) V(s_{t+1}) + λ G_{t+1}].
        G_next = 0.0
        Gs = [0.0] * L
        for t in range(L - 1, -1, -1):
            r, v = traj[t]["r"], traj[t]["v_next"]
            G = r + gamma * ((1.0 - lam) * v + lam * G_next) if t < L - 1 else r + gamma * v
            Gs[t] = G
            G_next = G
        for t in range(L):
            out.append((traj[t]["phi"], np.asarray([Gs[t] - traj[t]["q"]], dtype=np.float32)))
    else:
        n = int(lk_cfg.get("q_nstep", 1))
        for t in range(L):
            h = min(n, L - t)
            G = 0.0
            for i in range(h):
                G += (gamma ** i) * traj[t + i]["r"]
            G += (gamma ** h) * traj[t + h - 1]["v_next"]  # v_next is 0 at a terminal step
            out.append((traj[t]["phi"], np.asarray([G - traj[t]["q"]], dtype=np.float32)))
    return out


def _find_timelimit(env):
    """Walk the wrapper chain to the TimeLimit wrapper (holds _elapsed_steps)."""
    e = env
    while e is not None:
        if e.__class__.__name__ == "TimeLimit":
            return e
        e = getattr(e, "env", None)
    return None


def oracle_pool_pick(
    env, models: Dict, s: np.ndarray, z: torch.Tensor, a0: np.ndarray,
    action_low: np.ndarray, action_high: np.ndarray, lk_cfg: Dict, gamma: float,
    rng: np.random.Generator, device: torch.device,
) -> np.ndarray:
    """DIAGNOSTIC (not deployable): pick the candidate with the best TRUE one-step outcome
    by save/stepping/restoring the real (shifted) MuJoCo env. Upper bound for one-step
    re-ranking of the current candidate pool given the critic's terminal value V.

    We snapshot+restore both the physics (qpos/qvel) and the TimeLimit step counter so the
    counterfactual candidate steps do not advance the episode's truncation budget."""
    candidates = make_candidates_local(models["actor"], s, z, a0, action_low, action_high, lk_cfg, rng, device)
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    tl = _find_timelimit(env)
    elapsed = getattr(tl, "_elapsed_steps", None) if tl is not None else None
    best_score, best = -1e18, a0
    for cand in candidates:
        s_next, r, term, trunc, _ = env.step(cand)
        if term:
            v = 0.0
        else:
            st = torch.from_numpy(s_next.astype(np.float32)).unsqueeze(0).to(device)
            v = float(v_det_batch(models["actor"], models["q1"], models["q2"], st, z)[0].item())
        score = float(r) + gamma * v
        env.unwrapped.set_state(qpos, qvel)
        if tl is not None and elapsed is not None:
            tl._elapsed_steps = elapsed
        if score > best_score:
            best_score, best = score, cand.astype(np.float32)
    return np.clip(best, action_low, action_high).astype(np.float32)


def finalize_traces(episodes: List[Dict], max_steps: int, n_bins: int) -> Dict:
    """Pop the per-step 'reward_trace' from each measured episode and reduce to fixed
    time-bins so non-stationary TRACKING can be plotted (per-bin mean reward) and the
    survivorship confound made explicit (per-bin alive fraction). Episodes terminate at
    different lengths; binning by absolute step index keeps cross-seed averaging trivial.
    Returns {} if no traces were recorded. Mutates episodes (removes reward_trace)."""
    traces = [ep.pop("reward_trace", None) for ep in episodes]
    traces = [t for t in traces if t]
    if not traces:
        return {}
    width = max(max_steps / n_bins, 1.0)
    rew_sum = np.zeros(n_bins)
    rew_cnt = np.zeros(n_bins)
    alive = np.zeros(n_bins)
    for tr in traces:
        for k, r in enumerate(tr):
            b = min(int(k / width), n_bins - 1)
            rew_sum[b] += r
            rew_cnt[b] += 1
        reached = len(tr)
        for b in range(n_bins):
            if reached > b * width:
                alive[b] += 1
    mean_rew = np.where(rew_cnt > 0, rew_sum / np.maximum(rew_cnt, 1), np.nan)
    return {
        "bin_centers": [float((b + 0.5) * width) for b in range(n_bins)],
        "bin_mean_reward": [float(x) for x in mean_rew],
        "bin_alive_frac": [float(a / len(traces)) for a in alive],
        "n_traces": len(traces),
    }


def run_episode(
    cfg: Dict,
    meta: Dict,
    models: Dict,
    dyn: NormalizedLatentDynamicsModel,
    brpc: BRPCResidualCalibrator,
    rff,
    method: str,
    regime: Dict,
    seed: int,
    episode_idx: int,
    nominal_resid_p95: float,
    measure: bool,
    q_brpc: BRPCResidualCalibrator = None,
    context: Deque[np.ndarray] = None,
    carry: Dict = None,
) -> Dict:
    """Run one episode. Calibrators (passed in) keep adapting across episodes (continual
    few-shot). `measure=False` is a warmup episode (still records nothing).

    If `context` (a deque) is passed it PERSISTS across episodes (PEARL-style cross-episode
    z adaptation); otherwise a fresh per-episode context is used. `carry` (a dict) persists
    the gate EWMA and previous z across episodes when provided."""
    device = next(models["actor"].parameters()).device
    lk_cfg = cfg["lookahead"]
    env = make_env(cfg["env"]["name"], seed + episode_idx)
    env = maybe_wrap_action_perturb(env, regime, seed + episode_idx)
    schedule = make_schedule(regime, cfg["dynamics_randomization"]["train_range"])
    if context is None:
        context = deque(maxlen=int(cfg["latent"].get("eval_context_max", 50)))
    min_context = int(cfg["latent"].get("eval_context_min", 5))
    max_steps = int(cfg["env"]["max_episode_steps"])
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)
    action_scale = float(np.max(np.abs(action_high)))
    rng = np.random.default_rng(seed + 100000 * episode_idx)

    use_gate = bool(lk_cfg.get("use_gate", True))
    gate_slope = float(lk_cfg.get("gate_slope", 8.0))
    gate_threshold = float(lk_cfg.get("gate_threshold", 0.55))
    ewma_rho = float(lk_cfg.get("ewma_rho", 0.1))
    gamma = float(cfg["pearl"]["gamma"])
    p95 = max(nominal_resid_p95, 1e-6)
    rerank_methods = ("q_greedy", "value_shift", "value_shift_h", "value_shift_qr")

    s, _ = env.reset(seed=seed + episode_idx)
    prev_action = np.zeros(meta["action_dim"], dtype=np.float32)
    carry = carry if carry is not None else {}
    ewma_ratio = float(carry.get("ewma", 0.0))
    prev_z = carry.get("prev_z", None)
    ret = 0.0
    q_lift_vals, dV_vals, gate_vals, resid_vals, correction_vals, reward_vals = [], [], [], [], [], []
    nstep = uses_q_residual(method) and qr_nstep_mode(lk_cfg)
    qr_traj = []  # per-episode QR records for n-step/lambda targets (flushed at episode end)

    for t in range(max_steps):
        env.set_dynamics(schedule.xi_at(t))
        z = infer_z(models["encoder"], list(context), meta["latent_dim"], min_context, device)
        # Two-timescale stability: if z (slow) jumps, re-open the fast Q-residual head by
        # inflating its posterior covariance, since its feature basis φ_Q(·,·,z) just moved.
        if q_brpc is not None and prev_z is not None:
            z_jump = float(np.linalg.norm(z[0].detach().cpu().numpy() - prev_z))
            thresh = float(lk_cfg.get("q_z_jump_thresh", 0.0))
            if thresh > 0.0 and z_jump > thresh:
                q_brpc.P = q_brpc.P + float(lk_cfg.get("q_z_inflate", 1.0)) * q_brpc.prior_var * np.eye(q_brpc.feature_dim)
        prev_z = z[0].detach().cpu().numpy().copy()
        with torch.no_grad():
            a0 = models["actor"].deterministic_action(
                torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device), z
            )[0].cpu().numpy().astype(np.float32)
        q_base = q_min_single(models, s, a0, z)
        gate_h = 1.0 / (1.0 + math.exp(-gate_slope * (ewma_ratio - gate_threshold))) if use_gate else 1.0

        if method == "oracle_pool":
            action = oracle_pool_pick(env, models, s, z, a0, action_low, action_high, lk_cfg, gamma, rng, device)
            dV_sel = 0.0
        else:
            action, dV_sel = pick_action(
                method, cfg, lk_cfg, models, dyn, brpc, rff, s, z, a0, prev_action,
                gate_h, action_low, action_high, action_scale, rng, device, q_brpc,
            )

        q_exec = q_min_single(models, s, action, z)
        s_next, reward, term, trunc, _ = env.step(action)
        done = bool(term or trunc)

        # Shift gate (shared by all re-ranking methods) + online adapter update(s).
        if method in rerank_methods:
            resid_norm, pending, qr_record = step_adaptation(
                method, lk_cfg, models, dyn, brpc, q_brpc, rff,
                s, action, s_next, reward, bool(term), z, action_scale, gamma, device,
            )
            ewma_ratio = (1.0 - ewma_rho) * ewma_ratio + ewma_rho * (resid_norm / p95)
            resid_vals.append(resid_norm)
            for calibrator, feat, target in pending:
                # In n-step/lambda mode the QR head is updated at episode end from real returns;
                # skip its immediate 1-step update (other calibrators, e.g. dyn, still apply now).
                if nstep and calibrator is q_brpc:
                    continue
                calibrator.update(feat, target)
            if nstep and qr_record is not None:
                qr_traj.append(qr_record)

        ret += float(reward)
        q_lift_vals.append(float(q_exec - q_base))
        dV_vals.append(dV_sel)
        gate_vals.append(float(gate_h))
        correction_vals.append(float(np.sum((action - a0) ** 2)))
        reward_vals.append(float(reward))
        context.append(_make_context_item(s, action, reward, s_next, done))
        s = s_next
        prev_action = action
        if done:
            break

    # Flush episode-end n-step / lambda-return QR targets (real observed returns).
    if nstep and q_brpc is not None and qr_traj:
        for feat, target in qr_traj_targets(qr_traj, gamma, lk_cfg):
            q_brpc.update(feat, target)

    env.close()
    carry["ewma"] = ewma_ratio
    if prev_z is not None:
        carry["prev_z"] = prev_z
    return {
        "return": ret,
        "length": len(q_lift_vals),
        "mean_q_lift": float(np.mean(q_lift_vals)) if q_lift_vals else 0.0,
        "mean_value_shift_term": float(np.mean(dV_vals)) if dV_vals else 0.0,
        "mean_gate": float(np.mean(gate_vals)) if gate_vals else 0.0,
        "mean_raw_resid_norm": float(np.mean(resid_vals)) if resid_vals else 0.0,
        "mean_correction_sq": float(np.mean(correction_vals)) if correction_vals else 0.0,
        "reward_trace": reward_vals,
    }


def run_fleet_episode(
    cfg: Dict,
    meta: Dict,
    models: Dict,
    dyn: NormalizedLatentDynamicsModel,
    brpc: BRPCResidualCalibrator,
    rff,
    regime: Dict,
    seed: int,
    episode_idx: int,
    num_agents: int,
    nominal_p95: float,
    method: str = "value_shift",
    q_brpc: BRPCResidualCalibrator = None,
    contexts: List = None,
    ewma: List = None,
) -> List[Dict]:
    """One fleet episode: `num_agents` agents explore the SAME regime in lockstep and
    share a single online calibrator. Each wall-clock step advances the calibrator by one
    AR(1) predict_step and then absorbs all agents' residuals as a batch, so M parallel
    agents yield M x adaptation data per step without over-decaying. Works for any BRPC
    method (dynamics residual or Q value residual).

    `contexts` (list of deques) persists across episodes for cross-episode z adaptation;
    with lookahead.fleet_share_context the fleet pools one shared context for a common z
    (fleet z-adaptation, parallel to the shared calibrator). `ewma` (list) persists the gate.
    Returns one metric dict per agent (treated as independent measured episodes)."""
    device = next(models["actor"].parameters()).device
    lk_cfg = cfg["lookahead"]
    gamma = float(cfg["pearl"]["gamma"])
    train_range = cfg["dynamics_randomization"]["train_range"]
    M = num_agents
    context_max = int(cfg["latent"].get("eval_context_max", 50))
    min_context = int(cfg["latent"].get("eval_context_min", 5))
    max_steps = int(cfg["env"]["max_episode_steps"])
    use_gate = bool(lk_cfg.get("use_gate", True))
    gate_slope = float(lk_cfg.get("gate_slope", 8.0))
    gate_threshold = float(lk_cfg.get("gate_threshold", 0.55))
    ewma_rho = float(lk_cfg.get("ewma_rho", 0.1))
    share_ctx = bool(lk_cfg.get("fleet_share_context", False))
    p95 = max(nominal_p95, 1e-6)

    envs = [maybe_wrap_action_perturb(make_env(cfg["env"]["name"], seed + 1000 * episode_idx + i), regime, seed + 1000 * episode_idx + i) for i in range(M)]
    schedule = make_schedule(regime, train_range)
    action_low = envs[0].action_space.low.astype(np.float32)
    action_high = envs[0].action_space.high.astype(np.float32)
    action_scale = float(np.max(np.abs(action_high)))
    rngs = [np.random.default_rng(seed + 100000 * episode_idx + 7 * i) for i in range(M)]
    if contexts is None:
        shared = deque(maxlen=context_max)
        contexts = [shared for _ in range(M)] if share_ctx else [deque(maxlen=context_max) for _ in range(M)]
    states = [envs[i].reset(seed=seed + 1000 * episode_idx + i)[0] for i in range(M)]
    prev_actions = [np.zeros(meta["action_dim"], dtype=np.float32) for _ in range(M)]
    if ewma is None:
        ewma = [0.0 for _ in range(M)]
    alive = [True for _ in range(M)]
    agg = [{"return": 0.0, "q_lift": [], "dV": [], "gate": [], "resid": [], "corr": [], "rew": []} for _ in range(M)]
    nstep = uses_q_residual(method) and qr_nstep_mode(lk_cfg)
    qr_trajs = [[] for _ in range(M)]  # per-agent QR records for n-step/lambda targets

    for t in range(max_steps):
        if not any(alive):
            break
        batch_obs = []
        for i in range(M):
            if not alive[i]:
                continue
            env, s = envs[i], states[i]
            env.set_dynamics(schedule.xi_at(t))
            z = infer_z(models["encoder"], list(contexts[i]), meta["latent_dim"], min_context, device)
            with torch.no_grad():
                a0 = models["actor"].deterministic_action(
                    torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device), z
                )[0].cpu().numpy().astype(np.float32)
            q_base = q_min_single(models, s, a0, z)
            gate_h = 1.0 / (1.0 + math.exp(-gate_slope * (ewma[i] - gate_threshold))) if use_gate else 1.0
            action, dV_sel = pick_action(
                method, cfg, lk_cfg, models, dyn, brpc, rff, s, z, a0, prev_actions[i],
                gate_h, action_low, action_high, action_scale, rngs[i], device, q_brpc,
            )
            q_exec = q_min_single(models, s, action, z)
            s_next, reward, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            resid_norm, pending, qr_record = step_adaptation(
                method, lk_cfg, models, dyn, brpc, q_brpc, rff,
                s, action, s_next, reward, bool(term), z, action_scale, gamma, device,
            )
            ewma[i] = (1.0 - ewma_rho) * ewma[i] + ewma_rho * (resid_norm / p95)
            for cal, feat, target in pending:
                # n-step/lambda: defer the QR head to episode end (real returns); skip its 1-step.
                if nstep and cal is q_brpc:
                    continue
                batch_obs.append((cal, feat, target))
            if nstep and qr_record is not None:
                qr_trajs[i].append(qr_record)
            agg[i]["return"] += float(reward)
            agg[i]["q_lift"].append(float(q_exec - q_base))
            agg[i]["dV"].append(dV_sel)
            agg[i]["gate"].append(float(gate_h))
            agg[i]["resid"].append(resid_norm)
            agg[i]["corr"].append(float(np.sum((action - a0) ** 2)))
            agg[i]["rew"].append(float(reward))
            contexts[i].append(_make_context_item(s, action, reward, s_next, done))
            states[i] = s_next
            prev_actions[i] = action
            if done:
                alive[i] = False
                env.close()
        # Batch update: one AR(1) process step per distinct calibrator, then all measurements.
        if batch_obs:
            for cal in {id(c): c for c, _, _ in batch_obs}.values():
                cal._predict_step()
            for cal, feat, target in batch_obs:
                brpc_measurement_update(cal, feat, target)
    # Flush episode-end n-step / lambda QR targets: one AR(1) predict per wall-clock-equivalent
    # then the fleet's targets for that step as a batch (mirrors the shared-calibrator cadence).
    if nstep and q_brpc is not None:
        per_agent = [qr_traj_targets(tr, gamma, lk_cfg) for tr in qr_trajs]
        for step_targets in zip_longest(*per_agent):
            obs = [st for st in step_targets if st is not None]
            if not obs:
                continue
            q_brpc._predict_step()
            for feat, target in obs:
                brpc_measurement_update(q_brpc, feat, target)
    for i in range(M):
        if alive[i]:
            envs[i].close()

    out = []
    for i in range(M):
        a = agg[i]
        out.append({
            "return": a["return"],
            "length": len(a["q_lift"]),
            "mean_q_lift": float(np.mean(a["q_lift"])) if a["q_lift"] else 0.0,
            "mean_value_shift_term": float(np.mean(a["dV"])) if a["dV"] else 0.0,
            "mean_gate": float(np.mean(a["gate"])) if a["gate"] else 0.0,
            "mean_raw_resid_norm": float(np.mean(a["resid"])) if a["resid"] else 0.0,
            "mean_correction_sq": float(np.mean(a["corr"])) if a["corr"] else 0.0,
            "reward_trace": a["rew"],
        })
    return out


def make_brpc_rff(cfg: Dict, meta: Dict, nominal_resid: Dict, seed: int):
    lk_cfg = cfg["lookahead"]
    input_dim = meta["state_dim"] + meta["action_dim"] + meta["latent_dim"]
    rff = build_rff(input_dim, lk_cfg, seed=seed)
    # obs_noise: per-dim nominal residual std scaled, or scalar fallback.
    scale = float(lk_cfg.get("obs_noise_scale", 1.0))
    per_dim = nominal_resid.get("resid_std_per_dim")
    if per_dim is not None and lk_cfg.get("obs_noise_mode", "nominal_std") == "nominal_std":
        obs_noise = np.maximum(np.asarray(per_dim, dtype=np.float64) * scale, 1e-3)
    else:
        obs_noise = float(lk_cfg.get("obs_noise", 0.2))
    brpc = BRPCResidualCalibrator(
        feature_dim=int(lk_cfg.get("feature_dim", 128)),
        output_dim=meta["state_dim"],
        prior_var=float(lk_cfg.get("prior_var", 0.25)),
        obs_noise=obs_noise,
        rho=float(lk_cfg.get("rho", 0.999)),
        q_alpha=float(lk_cfg.get("q_alpha", 1e-5)),
        residual_clip_sigma=float(lk_cfg.get("residual_clip_sigma", 5.0)),
    )
    return brpc, rff


def make_q_brpc(cfg: Dict, meta: Dict) -> BRPCResidualCalibrator:
    """Calibrator for the value (Bellman) residual head over Q's penultimate features."""
    lk_cfg = cfg["lookahead"]
    q_hidden = int(cfg.get("networks", {}).get("q_hidden", [256, 256])[-1])
    return BRPCResidualCalibrator(
        feature_dim=q_hidden,
        output_dim=1,
        prior_var=float(lk_cfg.get("q_prior_var", lk_cfg.get("prior_var", 0.25))),
        obs_noise=float(lk_cfg.get("q_obs_noise", 0.5)),
        rho=float(lk_cfg.get("q_rho", lk_cfg.get("rho", 0.999))),
        q_alpha=float(lk_cfg.get("q_q_alpha", lk_cfg.get("q_alpha", 1e-5))),
        residual_clip_sigma=float(lk_cfg.get("residual_clip_sigma", 5.0)),
    )


def evaluate(
    cfg: Dict,
    meta: Dict,
    models: Dict,
    dyn: NormalizedLatentDynamicsModel,
    nominal_resid: Dict,
    method: str,
    regime: Dict,
    seed: int,
    output_dir: Path,
    warmup_override: int = None,
    num_agents: int = 1,
) -> Dict:
    """Few-shot, continual: `warmup` episodes adapt BRPC in this regime, then measure
    test episodes (BRPC keeps adapting through them).

    num_agents > 1 (value_shift only) runs a fleet: agents share one BRPC, so K warmup
    fleet-episodes give K*num_agents episode-equivalents of adaptation data, and each
    measured episode is M agents adapting in parallel from step 0.

    PEARL-style z adaptation: when lookahead.persist_context is set, the context (and gate
    EWMA) persists across the warmup + test episodes so z is inferred from accumulated
    in-regime data (applies to ALL methods incl. baselines, for a fair comparison)."""
    if is_finetune(method):
        # Test-time fine-tuning: deep-copies the frozen backbone (checkpoint untouched),
        # adapts weights via SAC gradient steps. Ignores the dynamics model / nominal_resid.
        from pearl_brpc_action_adapter.eval.eval_full_pearl_finetune import evaluate_finetune
        return evaluate_finetune(
            cfg, meta, models, method, regime, seed, output_dir,
            warmup_override=warmup_override, num_agents=num_agents,
        )
    if method in ("qr_finetune_critic", "qr_finetune_full"):
        # Hybrid: fast QR head (safe corrector) alternated with slow critic (qr_finetune_critic)
        # or critic+actor (qr_finetune_full) fine-tune; needs the frozen world model for the gate.
        # Checkpoint untouched (backbone deep-copied per run).
        from pearl_brpc_action_adapter.eval.eval_full_pearl_qr_finetune import evaluate_qr_finetune
        return evaluate_qr_finetune(
            cfg, meta, models, dyn, nominal_resid, method, regime, seed, output_dir,
            warmup_override=warmup_override, num_agents=num_agents,
        )
    lk_cfg = cfg["lookahead"]
    cfg_warmup = int(lk_cfg.get("warmup_episodes", 0)) if warmup_override is None else int(warmup_override)
    # Warmup applies to ALL methods now (it builds the persistent z-context); BRPC methods
    # additionally adapt their calibrators during it.
    warmup = cfg_warmup
    n_test = int(cfg["eval"]["num_episodes"])
    nominal_p95 = float(nominal_resid.get("resid_norm_p95", 1.0))
    fleet = fleet_capable(method) and num_agents > 1
    persist = bool(lk_cfg.get("persist_context", True))
    ctx_max = int(cfg["latent"].get("eval_context_max", 50))

    dyn_brpc, rff = make_brpc_rff(cfg, meta, nominal_resid, seed) if uses_dyn_brpc(method) else (None, None)
    q_brpc = make_q_brpc(cfg, meta) if uses_q_residual(method) else None

    ep = 0
    episodes = []
    if fleet:
        share_ctx = bool(lk_cfg.get("fleet_share_context", False))
        if persist:
            shared = deque(maxlen=ctx_max)
            contexts = [shared] * num_agents if share_ctx else [deque(maxlen=ctx_max) for _ in range(num_agents)]
            ewma = [0.0 for _ in range(num_agents)]
        else:
            contexts, ewma = None, None
        for _ in range(warmup):
            run_fleet_episode(cfg, meta, models, dyn, dyn_brpc, rff, regime, seed, ep, num_agents, nominal_p95, method, q_brpc, contexts, ewma)
            ep += 1
        for _ in range(n_test):
            episodes.extend(
                run_fleet_episode(cfg, meta, models, dyn, dyn_brpc, rff, regime, seed, ep, num_agents, nominal_p95, method, q_brpc, contexts, ewma)
            )
            ep += 1
    else:
        context = deque(maxlen=ctx_max) if persist else None
        carry = {} if persist else None
        for _ in range(warmup):
            run_episode(cfg, meta, models, dyn, dyn_brpc, rff, method, regime, seed, ep, nominal_p95, measure=False, q_brpc=q_brpc, context=context, carry=carry)
            ep += 1
        for _ in range(n_test):
            episodes.append(
                run_episode(cfg, meta, models, dyn, dyn_brpc, rff, method, regime, seed, ep, nominal_p95, measure=True, q_brpc=q_brpc, context=context, carry=carry)
            )
            ep += 1

    out = {
        "method": method, "regime": regime["name"], "seed": seed,
        "warmup_episodes": warmup, "num_agents": num_agents,
    }
    # Reduce per-step reward traces to fixed time-bins (tracking + survivorship) before the
    # scalar-mean loop below (which would choke on the list-valued reward_trace).
    trace_summary = finalize_traces(episodes, int(cfg["env"]["max_episode_steps"]), int(lk_cfg.get("trace_bins", 20)))
    out.update(trace_summary)
    for key in episodes[0]:
        vals = [x[key] for x in episodes]
        out[f"mean_{key}"] = float(np.mean(vals))
        out[f"std_{key}"] = float(np.std(vals))
    out["num_episodes"] = len(episodes)
    # Ordered per-episode returns (at M=1 this is the in-order online deployment curve;
    # used for anytime / cumulative-return / online-regret analysis).
    out["episode_returns"] = [float(e["return"]) for e in episodes]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
