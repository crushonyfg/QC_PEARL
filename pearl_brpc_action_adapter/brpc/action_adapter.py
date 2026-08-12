from __future__ import annotations

from typing import Tuple

import numpy as np

from brpc_latent_mpc.brpc.bayes_linear_residual import BRPCResidualCalibrator
from brpc_latent_mpc.brpc.rff import RFFFeatures


class BRPCActionAdapter:
    """Online Bayesian action correction with uncertainty gate."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        prior_var: float = 1.0,
        obs_noise: float = 0.05,
        rho: float = 0.99,
        q_alpha: float = 1e-4,
        u_max: float = 0.2,
        kappa: float = 0.5,
    ):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.u_max = u_max
        self.kappa = kappa
        self._brpc = BRPCResidualCalibrator(
            feature_dim=feature_dim,
            output_dim=action_dim,
            prior_var=prior_var,
            obs_noise=obs_noise,
            rho=rho,
            q_alpha=q_alpha,
            residual_clip_sigma=5.0,
        )

    def reset(self) -> None:
        self._brpc.reset()

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        return self._brpc.predict(features)

    def predict_step(self) -> None:
        self._brpc._predict_step()

    def update(self, features: np.ndarray, u_target: np.ndarray) -> None:
        self._brpc.update(features, u_target)

    def act_correction(self, features: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        u_mean, unc = self.predict(features)
        gate = 1.0 / (1.0 + self.kappa * unc)
        u_exec = gate * u_mean
        u_exec = np.clip(u_exec, -self.u_max, self.u_max)
        return u_exec.astype(np.float32), float(unc), u_mean.astype(np.float32)


class RLSActionAdapter:
    """RLS action adapter without dynamic prior or uncertainty gate."""

    def __init__(self, feature_dim: int, action_dim: int, prior_var: float = 1.0, obs_noise: float = 0.05):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.obs_noise = obs_noise
        self.reset(prior_var)

    def reset(self, prior_var: float = 1.0) -> None:
        self.M = np.zeros((self.feature_dim, self.action_dim), dtype=np.float64)
        self.P = np.eye(self.feature_dim, dtype=np.float64) * prior_var

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        b = np.asarray(features, dtype=np.float64).reshape(-1)
        return (b @ self.M).astype(np.float32), 0.0

    def predict_step(self) -> None:
        pass

    def update(self, features: np.ndarray, u_target: np.ndarray) -> None:
        b = np.asarray(features, dtype=np.float64).reshape(-1)
        y = np.asarray(u_target, dtype=np.float64).reshape(-1)
        S = self.obs_noise ** 2 + b @ self.P @ b
        K = (self.P @ b) / max(S, 1e-12)
        pred = b @ self.M
        self.M = self.M + np.outer(K, y - pred)
        self.P = self.P - np.outer(K, b) @ self.P
        self.P = 0.5 * (self.P + self.P.T) + 1e-6 * np.eye(self.feature_dim)

    def act_correction(self, features: np.ndarray, u_max: float) -> np.ndarray:
        u_mean, _ = self.predict(features)
        return np.clip(u_mean, -u_max, u_max).astype(np.float32)


class ConstantBiasAdapter:
    """Global online action bias (no state features)."""

    def __init__(self, action_dim: int, lr: float = 0.05, u_max: float = 0.2):
        self.action_dim = action_dim
        self.lr = lr
        self.u_max = u_max
        self.bias = np.zeros(action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.bias = np.zeros(self.action_dim, dtype=np.float32)

    def act_correction(self) -> np.ndarray:
        return np.clip(self.bias, -self.u_max, self.u_max)

    def update(self, u_target: np.ndarray) -> None:
        self.bias = np.clip(
            self.bias + self.lr * (u_target - self.bias),
            -self.u_max,
            self.u_max,
        )


def make_feature_input(state: np.ndarray, action: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.concatenate([state, action, z], axis=-1).astype(np.float32)


def build_rff(input_dim: int, cfg: dict, seed: int = 0) -> RFFFeatures:
    return RFFFeatures(
        input_dim=input_dim,
        feature_dim=int(cfg.get("feature_dim", 128)),
        lengthscale=float(cfg.get("lengthscale", 1.0)),
        seed=seed,
    )
