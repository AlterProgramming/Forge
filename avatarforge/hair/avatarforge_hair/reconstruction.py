from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .texture_model import TextureState


@dataclass(frozen=True)
class ReconstructionBudget:
    """Separate interactive guide density from beauty reconstruction density."""

    guide_strands: int = 240
    render_followers: int = 2400
    minimum_samples: int = 72
    maximum_samples: int = 384
    samples_per_cycle: float = 11.0


def adaptive_longitudinal_samples(
    texture: TextureState,
    *,
    length_scale: float = 1.0,
    budget: ReconstructionBudget = ReconstructionBudget(),
) -> int:
    """Return enough points to represent the requested texture without aliasing.

    The old v11 atlas forced every 1A-4C state through the same ~49 point
    guide polyline. High-curvature 4A-4C states therefore collapsed into zig-zag
    geometry. Guide topology remains low-frequency; beauty followers receive a
    texture-dependent longitudinal budget.
    """

    cycles = max(0.0, texture.cycles_per_10cm * float(length_scale))
    samples = math.ceil(cycles * budget.samples_per_cycle + 28)
    return int(np.clip(samples, budget.minimum_samples, budget.maximum_samples))


def tangent_flow(desired: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project a desired root-flow vector into the real scalp tangent plane."""

    desired = np.asarray(desired, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    tangent = desired - normal * float(np.dot(desired, normal))
    norm = np.linalg.norm(tangent)
    if norm < 1e-10:
        # Deterministic fallback perpendicular to the normal.
        ref = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(ref, normal))) > 0.92:
            ref = np.array([1.0, 0.0, 0.0])
        tangent = np.cross(normal, ref)
        norm = np.linalg.norm(tangent)
    return tangent / (norm + 1e-12)


def root_flow_target(root: np.ndarray) -> np.ndarray:
    """Default gravity/back/side flow used before a strand leaves the scalp.

    This field is deliberately evaluated on the scalp rather than drawing a
    free-space chord from root to hairstyle endpoint. Front roots are therefore
    routed around the head instead of cutting visually across the forehead.
    """

    root = np.asarray(root, dtype=np.float64)
    side = -1.0 if root[0] < 0 else 1.0
    frontness = float(np.clip((root[2] + 0.015) / 0.11, 0.0, 1.0))
    crownness = float(np.clip((root[1] - 1.61) / 0.17, 0.0, 1.0))
    return np.array(
        [
            side * (0.42 + 0.26 * frontness),
            -0.48 - 0.32 * (1.0 - crownness),
            -0.88 + 0.18 * (1.0 - frontness),
        ],
        dtype=np.float64,
    )
