from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

TEXTURE_LABELS = (
    "1A", "1B", "1C",
    "2A", "2B", "2C",
    "3A", "3B", "3C",
    "4A", "4B", "4C",
)

@dataclass(frozen=True)
class TextureState:
    """Editor-facing strand texture anchor.

    The labels are conventional visual vocabulary. The simulator operates on the
    continuous parameters below; labels are just stable anchor points.
    """
    label: str
    family: str
    curl_coordinate: float
    coil_radius_m: float
    cycles_per_10cm: float
    cross_plane: float
    kink: float
    visible_length_ratio: float
    lock_phase_coherence: float
    rest_curvature_per_m: float
    rest_twist_per_m: float
    tip_response_power: float
    base_damping: float
    roughness: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)

_ANCHORS: Dict[str, TextureState] = {
    "1A": TextureState("1A", "straight", 0.00, 0.00008, 0.05, 0.05, 0.00, 1.00, 0.32, 0.8,   0.1, 1.00, 0.024, 0.34),
    "1B": TextureState("1B", "straight", 0.07, 0.00015, 0.18, 0.06, 0.00, 0.995,0.34, 1.4,   0.2, 1.02, 0.024, 0.35),
    "1C": TextureState("1C", "straight", 0.14, 0.00028, 0.42, 0.08, 0.01, 0.985,0.36, 2.5,   0.5, 1.04, 0.025, 0.36),
    "2A": TextureState("2A", "wavy",     0.23, 0.00390, 1.25, 0.10, 0.01, 0.965,0.48, 7.0,   1.5, 1.08, 0.027, 0.37),
    "2B": TextureState("2B", "wavy",     0.32, 0.00460, 2.10, 0.16, 0.02, 0.935,0.56, 12.0,  3.0, 1.12, 0.029, 0.38),
    "2C": TextureState("2C", "wavy",     0.41, 0.00480, 3.20, 0.24, 0.04, 0.895,0.62, 19.0,  6.0, 1.17, 0.031, 0.39),
    "3A": TextureState("3A", "curly",    0.51, 0.00450, 4.80, 0.46, 0.06, 0.835,0.70, 30.0, 12.0, 1.25, 0.034, 0.40),
    "3B": TextureState("3B", "curly",    0.60, 0.00370, 7.10, 0.62, 0.09, 0.770,0.76, 45.0, 20.0, 1.34, 0.037, 0.42),
    "3C": TextureState("3C", "curly",    0.69, 0.00290,10.30, 0.76, 0.14, 0.700,0.81, 65.0, 31.0, 1.44, 0.041, 0.44),
    "4A": TextureState("4A", "coily",    0.78, 0.00220,14.60, 0.88, 0.24, 0.620,0.86, 92.0, 46.0, 1.57, 0.046, 0.46),
    "4B": TextureState("4B", "coily",    0.89, 0.00155,19.50, 0.78, 0.50, 0.535,0.90,125.0, 68.0, 1.72, 0.052, 0.48),
    "4C": TextureState("4C", "coily",    1.00, 0.00105,25.00, 0.72, 0.78, 0.450,0.93,165.0, 92.0, 1.90, 0.058, 0.50),
}

@dataclass(frozen=True)
class ProductState:
    """Visible/mechanical product state, 0..1 per channel.

    This is an editor response model, not a chemical reaction simulation.
    """
    water: float = 0.0
    conditioner: float = 0.0
    gel: float = 0.0
    oil: float = 0.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def texture_state(label: str) -> TextureState:
    key = str(label).upper()
    if key not in _ANCHORS:
        raise KeyError(f"unknown texture {label!r}; choose one of {', '.join(TEXTURE_LABELS)}")
    return _ANCHORS[key]


def apply_products(texture: TextureState, products: ProductState) -> dict:
    water = _clamp01(products.water)
    conditioner = _clamp01(products.conditioner)
    gel = _clamp01(products.gel)
    oil = _clamp01(products.oil)

    elongation = 1.0 + 0.16 * water * (0.35 + 0.65 * texture.curl_coordinate)
    definition = 1.0 + 0.18*conditioner + 0.34*gel + 0.08*water
    frizz_scale = max(0.22, 1.0 - 0.28*water - 0.34*conditioner - 0.48*gel - 0.20*oil)
    cohesion = min(1.0, 0.30 + 0.45*texture.lock_phase_coherence + 0.20*conditioner + 0.38*gel)
    mass_scale = 1.0 + 0.38*water + 0.06*oil
    damping_scale = 1.0 + 0.26*water + 0.18*conditioner + 0.46*gel + 0.08*oil
    stiffness_scale = 1.0 + 0.52*gel - 0.10*water
    roughness = max(0.12, texture.roughness - 0.12*conditioner - 0.10*oil - 0.04*water)
    gloss = min(1.0, 0.18 + 0.30*conditioner + 0.45*oil + 0.18*water)
    slip = min(1.0, 0.18 + 0.38*conditioner + 0.48*oil + 0.12*water)

    return {
        "elongation": elongation,
        "definition": definition,
        "frizz_scale": frizz_scale,
        "cohesion": cohesion,
        "mass_scale": mass_scale,
        "damping_scale": damping_scale,
        "stiffness_scale": stiffness_scale,
        "roughness": roughness,
        "gloss": gloss,
        "slip": slip,
    }
