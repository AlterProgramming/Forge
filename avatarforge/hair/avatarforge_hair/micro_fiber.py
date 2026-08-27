from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class MicroFiberState:
    """Coarse-grained microscopic state of one hair fiber.

    Geometry is NOT prescribed here. The state describes cross-section and
    cortex-scale differential strain/orientation. A Cosserat-style rest-strain
    integration produces the natural 3D curve.
    """
    diameter_m: float = 70e-6
    ellipticity: float = 1.0
    cortical_strain_bias: float = 5e-5
    cortical_orientation_rad: float = 0.0
    orientation_rotation_per_m: float = 0.0
    torsional_bias_per_m: float = 0.0
    heterogeneity: float = 0.08
    heterogeneity_scale_m: float = 0.0015
    hydration: float = 0.2
    damage: float = 0.05
    cuticle_integrity: float = 0.95
    seed: int = 1

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def microstate_axis(x: float, *, seed: int = 1, diameter_m: float = 70e-6) -> MicroFiberState:
    """Continuous *microstructure* axis used by the editor.

    x does not encode curl radius/cycles/kinks. It only changes quantities that
    exist at the fiber/microstructure level. The resulting curve is classified
    after integration.
    """
    x = _clamp(x, 0.0, 1.0)
    strain_lo, strain_hi = 4e-5, 0.055
    strain = math.exp(math.log(strain_lo) * (1-x) + math.log(strain_hi) * x)
    orientation_rate = (0.3 + 20.0 * x**1.7) * 2.0 * math.pi
    torsional_bias = (0.05 + 3.5 * x**1.8) * 2.0 * math.pi
    ellipticity = 1.0 - 0.34 * x
    heterogeneity = 0.03 + 0.30 * x**1.4
    scale = 0.0025 - 0.0017*x
    return MicroFiberState(
        diameter_m=diameter_m,
        ellipticity=ellipticity,
        cortical_strain_bias=strain,
        cortical_orientation_rad=0.0,
        orientation_rotation_per_m=orientation_rate,
        torsional_bias_per_m=torsional_bias,
        heterogeneity=heterogeneity,
        heterogeneity_scale_m=scale,
        hydration=0.18,
        damage=0.06,
        cuticle_integrity=0.94,
        seed=seed,
    )


def apply_environment(state: MicroFiberState, *, water: float = 0.0, conditioner: float = 0.0,
                      gel: float = 0.0, oil: float = 0.0) -> MicroFiberState:
    """Update micro/material state, not final geometry.

    These transfer laws are provisional response laws. They alter differential
    strain/heterogeneity/torsion; the rest shape is always recomputed afterward.
    """
    water = _clamp(water, 0, 1); conditioner = _clamp(conditioner, 0, 1)
    gel = _clamp(gel, 0, 1); oil = _clamp(oil, 0, 1)
    hydration = _clamp(state.hydration + 0.72*water + 0.08*conditioner, 0, 1)

    swelling = 1.0 + 0.055*hydration
    strain_scale = 1.0 - 0.18*hydration
    hetero_scale = 1.0 - 0.22*conditioner - 0.14*oil
    torsion_scale = 1.0 - 0.10*hydration
    return MicroFiberState(
        diameter_m=state.diameter_m*swelling,
        ellipticity=min(1.0, state.ellipticity + 0.06*hydration),
        cortical_strain_bias=state.cortical_strain_bias*strain_scale,
        cortical_orientation_rad=state.cortical_orientation_rad,
        orientation_rotation_per_m=state.orientation_rotation_per_m*torsion_scale,
        torsional_bias_per_m=state.torsional_bias_per_m*torsion_scale,
        heterogeneity=max(0.0, state.heterogeneity*hetero_scale),
        heterogeneity_scale_m=state.heterogeneity_scale_m,
        hydration=hydration,
        damage=state.damage,
        cuticle_integrity=state.cuticle_integrity,
        seed=state.seed,
    )


def _correlated_noise(n: int, ds: float, corr_m: float, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=n)
    sigma = max(1.0, corr_m / max(ds, 1e-9))
    radius = int(min(max(2, round(3*sigma)), max(2, n//4)))
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5*(x/sigma)**2)
    kernel /= kernel.sum()
    out = np.convolve(raw, kernel, mode='same')
    std = out.std()
    return out / (std if std > 1e-8 else 1.0)


def microscopic_rest_strains(state: MicroFiberState, length_m: float, ds: float):
    """Return s, kappa1, kappa2, tau generated from the microstate."""
    n = max(3, int(math.ceil(length_m/ds)) + 1)
    s = np.linspace(0.0, length_m, n)
    ds_eff = s[1] - s[0]
    rng = np.random.default_rng(state.seed)
    noise_a = _correlated_noise(n, ds_eff, state.heterogeneity_scale_m, rng)
    noise_phi = _correlated_noise(n, ds_eff, state.heterogeneity_scale_m*1.7, rng)

    minor_d = state.diameter_m * state.ellipticity
    kappa0 = 2.0 * state.cortical_strain_bias / max(minor_d, 1e-8)
    amplitude = kappa0 * np.clip(1.0 + state.heterogeneity*noise_a, 0.15, 2.5)

    phase = (state.cortical_orientation_rad
             + state.orientation_rotation_per_m*s
             + state.heterogeneity*0.55*noise_phi)
    k1 = amplitude * np.cos(phase)
    k2 = amplitude * np.sin(phase) * (0.72 + 0.28*state.ellipticity)
    tau = np.full(n, state.torsional_bias_per_m)
    tau += state.heterogeneity * state.torsional_bias_per_m * 0.15 * noise_phi
    return s, k1, k2, tau


def _hat(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def _exp_so3(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3) + _hat(w)
    a = w/theta
    K = _hat(a)
    return np.eye(3) + math.sin(theta)*K + (1-math.cos(theta))*(K@K)


def integrate_rest_shape(state: MicroFiberState, length_m: float = 0.10, ds: float = 1e-4,
                         initial_frame: np.ndarray | None = None) -> np.ndarray:
    """Integrate the natural fiber centerline from microscopic rest strains."""
    s, k1, k2, tau = microscopic_rest_strains(state, length_m, ds)
    n = len(s)
    pts = np.zeros((n, 3), dtype=np.float64)
    R = np.eye(3) if initial_frame is None else np.asarray(initial_frame, dtype=np.float64).copy()
    for i in range(1, n):
        h = s[i] - s[i-1]
        omega_body = np.array([k1[i-1], k2[i-1], tau[i-1]], dtype=np.float64)
        R = R @ _exp_so3(omega_body*h)
        pts[i] = pts[i-1] + R[:, 2]*h
    return pts


def curve_metrics(points: np.ndarray) -> dict:
    seg = np.diff(points, axis=0)
    arc = float(np.linalg.norm(seg, axis=1).sum())
    end = float(np.linalg.norm(points[-1]-points[0]))
    tang = seg/(np.linalg.norm(seg, axis=1, keepdims=True)+1e-12)
    if len(tang) >= 2:
        angles = np.arccos(np.clip((tang[:-1]*tang[1:]).sum(axis=1), -1, 1))
        ds = np.linalg.norm(seg[:-1], axis=1)+1e-12
        curv = angles/ds
        median_k = float(np.median(curv))
        p90_k = float(np.percentile(curv, 90))
    else:
        median_k = p90_k = 0.0
    return {
        'arc_length_m': arc,
        'end_to_end_m': end,
        'visible_length_ratio': end/max(arc, 1e-12),
        'median_curvature_per_m': median_k,
        'p90_curvature_per_m': p90_k,
    }
