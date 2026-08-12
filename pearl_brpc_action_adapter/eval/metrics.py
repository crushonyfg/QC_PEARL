from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    return_: float = 0.0
    length: int = 0
    action_move_cost: float = 0.0
    correction_norm: float = 0.0
    e_base_sum: float = 0.0
    e_base_count: int = 0
    latent_error_sum: float = 0.0
    latent_error_count: int = 0
    uncertainties: List[float] = field(default_factory=list)
    e_norms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "return": self.return_,
            "length": self.length,
            "action_move_cost": self.action_move_cost,
            "correction_norm": self.correction_norm,
            "mean_e_base": self.e_base_sum / max(self.e_base_count, 1),
            "mean_latent_error": self.latent_error_sum / max(self.latent_error_count, 1),
            "mean_uncertainty": float(np.mean(self.uncertainties)) if self.uncertainties else 0.0,
            "mean_e_norm": float(np.mean(self.e_norms)) if self.e_norms else 0.0,
        }


def aggregate_episodes(episodes: List[EpisodeMetrics]) -> Dict:
    d = [e.to_dict() for e in episodes]
    out = {}
    for key in d[0]:
        vals = [x[key] for x in d]
        out[f"mean_{key}"] = float(np.mean(vals))
        out[f"std_{key}"] = float(np.std(vals))
    out["num_episodes"] = len(episodes)
    return out
