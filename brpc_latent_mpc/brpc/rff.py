from __future__ import annotations

import numpy as np


class RFFFeatures:
    """Random Fourier Features for BRPC residual model."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        lengthscale: float = 1.0,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.omega = rng.normal(0.0, 1.0 / lengthscale, size=(feature_dim, input_dim)).astype(np.float32)
        self.bias = rng.uniform(0.0, 2 * np.pi, size=(feature_dim,)).astype(np.float32)
        self.scale = np.sqrt(2.0 / feature_dim).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        proj = x @ self.omega.T + self.bias
        return (self.scale * np.cos(proj)).astype(np.float32)
