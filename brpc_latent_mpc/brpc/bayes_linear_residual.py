from __future__ import annotations

from typing import Tuple, Union

import numpy as np


class BRPCResidualCalibrator:
    """Bayesian linear regression with AR(1) dynamic prior on coefficients."""

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        prior_var: float = 1.0,
        obs_noise: Union[float, np.ndarray] = 0.01,
        rho: float = 0.98,
        q_alpha: float = 1e-4,
        residual_clip_sigma: float = 5.0,
    ):
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.prior_var = prior_var
        self.rho = rho
        self.q_alpha = q_alpha
        self.residual_clip_sigma = residual_clip_sigma
        if np.ndim(obs_noise) == 0:
            self.obs_noise = float(obs_noise)
        else:
            self.obs_noise = np.asarray(obs_noise, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.M = np.zeros((self.feature_dim, self.output_dim), dtype=np.float64)
        self.P = np.eye(self.feature_dim, dtype=np.float64) * self.prior_var

    def _predict_step(self) -> None:
        self.M *= self.rho
        self.P = (self.rho ** 2) * self.P + self.q_alpha * np.eye(self.feature_dim)

    def _clip_residual(self, residual: np.ndarray) -> np.ndarray:
        if np.ndim(self.obs_noise) == 0:
            sigma = self.obs_noise
        else:
            sigma = self.obs_noise
        return np.clip(
            residual,
            -self.residual_clip_sigma * sigma,
            self.residual_clip_sigma * sigma,
        )

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        b = np.asarray(features, dtype=np.float64).reshape(-1)
        mean = b @ self.M
        var = float(b @ self.P @ b)
        return mean.astype(np.float32), var

    def update(self, features: np.ndarray, residual: np.ndarray) -> None:
        self._predict_step()
        b = np.asarray(features, dtype=np.float64).reshape(-1)
        r = self._clip_residual(np.asarray(residual, dtype=np.float64).reshape(-1))

        if np.ndim(self.obs_noise) == 0:
            obs_var = self.obs_noise ** 2
        else:
            obs_var = float(np.mean(self.obs_noise ** 2))

        S = obs_var + b @ self.P @ b
        K = (self.P @ b) / max(S, 1e-12)
        pred = b @ self.M
        err = r - pred
        self.M = self.M + np.outer(K, err)
        self.P = self.P - np.outer(K, b) @ self.P
        self.P = 0.5 * (self.P + self.P.T) + 1e-6 * np.eye(self.feature_dim)

    def predict_then_update(self, features: np.ndarray, residual: np.ndarray):
        pred, unc = self.predict(features)
        self.update(features, residual)
        return pred, unc

    def uncertainty_score(self, features: np.ndarray) -> float:
        b = np.asarray(features, dtype=np.float64).reshape(-1)
        return float(self.output_dim * (b @ self.P @ b))
