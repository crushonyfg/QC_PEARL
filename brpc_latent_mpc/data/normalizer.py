from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class Normalizer:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    eps: float = 1e-8
    _count: int = 0
    _m2: Optional[np.ndarray] = field(default=None, repr=False)

    def fit_partial(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if self.mean is None:
            self.mean = x.copy()
            self._m2 = np.zeros_like(x)
            self._count = 1
            return
        self._count += 1
        delta = x - self.mean
        self.mean += delta / self._count
        delta2 = x - self.mean
        self._m2 += delta * delta2

    def finalize(self) -> None:
        if self.mean is None:
            raise ValueError("Normalizer has no data.")
        if self._count < 2:
            self.std = np.ones_like(self.mean)
        else:
            var = self._m2 / max(self._count - 1, 1)
            self.std = np.sqrt(np.maximum(var, self.eps))

    def fit(self, x: np.ndarray) -> "Normalizer":
        for row in np.asarray(x):
            self.fit_partial(row)
        self.finalize()
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return np.asarray(x, dtype=np.float32)
        return ((np.asarray(x) - self.mean) / (self.std + self.eps)).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return np.asarray(x, dtype=np.float32)
        return (np.asarray(x) * (self.std + self.eps) + self.mean).astype(np.float32)

    def to_dict(self) -> Dict:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Normalizer":
        n = cls(eps=d.get("eps", 1e-8))
        if d.get("mean") is not None:
            n.mean = np.asarray(d["mean"], dtype=np.float64)
            n.std = np.asarray(d["std"], dtype=np.float64)
        return n
