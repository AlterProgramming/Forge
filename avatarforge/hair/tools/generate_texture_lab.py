from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from avatarforge_hair.texture_model import TEXTURE_LABELS, texture_state
from tools.generate_hair_groom import (
    COMPONENT_FLOAT,
    COMPONENT_UINT32,
    TARGET_ARRAY,
    TARGET_ELEMENT,
    Glb,
    choose_head_and_bone,
    global_matrix,
    lock_key,
    mesh_world_geometry,
    sample_scalp,
    smooth_curve,
    smoothstep,
    transform_points,
    transport_frames,
)


class CollisionProjector:
    def __init__(self, head_points, head_normals, body_points=None, body_normals=None):
        from scipy.spatial import cKDTree
        self.head_points = head_points
        self.head_normals = head_normals
        self.head_tree = cKDTree(head_points)
        self.body_points = body_points
        self.body_normals = body_normals
        self.body_tree = cKDTree(body_points) if body_points is not None else None

    def project(self, curve):
        result = curve.copy()
        if len(result) <= 1:
            return result
        points = result[1:]
        high = points[:, 1] > 1.42
        if np.any(high):
            subset = points[high]
            dist, idx = self.head_tree.query(subset, k=1)
            near = dist < 0.024
            if np.any(near):
                hp = self.head_points[idx[near]]
                hn = self.head_normals[idx[near]]
                signed = np.sum((subset[near] - hp) * hn, axis=1)
                global_idx = np.flatnonzero(high)[near]
                q = (global_idx + 1) / (len(result) - 1)
                clearance = 0.0011 + 0.0012 * (1.0 - q)
                push = np.maximum(0.0, clearance - signed)
                subset2 = subset.copy()
                subset2[near] += hn * push[:, None]
                points[high] = subset2

        if self.body_tree is not None:
            low_mask = (points[:, 1] <= 1.42) & (points[:, 1] > 1.12) & (np.abs(points[:, 0]) < 0.28)
            if np.any(low_mask):
                subset = points[low_mask]
                dist, idx = self.body_tree.query(subset, k=1)
                near = dist < 0.020
                if np.any(near):
                    bp = self.body_points[idx[near]]
                    bn = self.body_normals[idx[near]]
                    signed = np.sum((subset[near] - bp) * bn, axis=1)
                    push = np.maximum(0.0, 0.0018 - signed)
                    subset2 = subset.copy()
                    subset2[near] += bn * push[:, None]
                    points[low_mask] = subset2
        result[1:] = points
        return result


LENGTH_PROFILES = {
    "short": {"base": 0.064, "crown": 0.024, "back": 0.008, "samples": 43},
    "medium": {"base": 0.165, "crown": 0.050, "back": 0.022, "samples": 49},
    "long": {"base": 0.285, "crown": 0.075, "back": 0.040, "samples": 55},
}


def blue_noise_scalp(glb: Glb, count: int, rng: np.random.Generator):
    """Select stable, evenly distributed roots from a dense area-weighted pool."""
    pool_count = max(count * 6, count + 512)
    candidates, normals = sample_scalp(glb, pool_count, rng)
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = int(np.argmax(candidates[:, 1]))
    min_d2 = np.sum((candidates - candidates[chosen[0]]) ** 2, axis=1)
    min_d2[chosen[0]] = -1.0
    for i in range(1, count):
        idx = int(np.argmax(min_d2))
        chosen[i] = idx
        d2 = np.sum((candidates - candidates[idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[chosen[: i + 1]] = -1.0
    return candidates[chosen], normals[chosen]


def style_guide(root, normal, key, rng, length_profile: str, samples: int):
    profile = LENGTH_PROFILES[length_profile]
    side = -1.0 if key[0] == 0 else 1.0
    frontness = float(np.clip((root[2] + 0.02) / 0.115, 0.0, 1.0))
    crownness = float(np.clip((root[1] - 1.59) / 0.19, 0.0, 1.0))
    backness = 1.0 - frontness
    length = profile["base"] + profile["crown"] * crownness + profile["back"] * backness
    length *= rng.uniform(0.96, 1.04)

    t = np.linspace(0.0, 1.0, samples)
    s = smoothstep(t)
    part = side * (0.008 + 0.028 * frontness) * (1.0 - np.exp(-7.0 * t))
    sweep = (0.010 + 0.038 * crownness + 0.012 * frontness) * s

    if length_profile == "short":
        outward = normal[None, :] * (length * (0.72 * t + 0.28 * s))[:, None]
        curve = root[None, :] + outward
        curve[:, 0] += part
        curve[:, 1] -= length * 0.20 * t**1.7
        curve[:, 2] -= sweep * 0.30
    else:
        fall = length * (0.58 * t + 0.42 * t**1.55)
        curve = np.column_stack([
            root[0] + part + side * (0.008 + 0.018 * frontness) * s,
            root[1] - fall,
            root[2] - sweep,
        ])

    lift = 0.0022 * np.exp(-14.0 * t)
    curve += normal[None, :] * lift[:, None]
    curve[0] = root + normal * 0.00035
    return smooth_curve(curve, passes=2), length


def sample_curve(curve: np.ndarray, u: np.ndarray) -> np.ndarray:
    old = np.linspace(0.0, 1.0, len(curve))
    return np.column_stack([np.interp(u, old, curve[:, k]) for k in range(3)])


def follower_identity(rng: np.random.Generator):
    return {
        "phase_jitter": rng.normal(0.0, 1.0),
        "radius_jitter": rng.normal(0.0, 0.075),
        "freq_jitter": rng.normal(0.0, 0.035),
        "residual_phase": rng.uniform(-math.pi, math.pi, 3),
        "residual_coeff": rng.normal(0.0, 1.0, 3),
        "length_jitter": rng.normal(0.0, 0.018),
    }


def make_texture_curve(root, normal, guide_root, guide, grown_length, texture_label, identity, lock_phase, lock_coherence, samples, root_offset):
    tex = texture_state(texture_label)
    t = np.linspace(0.0, 1.0, samples)
    s = smoothstep(t)
    visible_ratio = tex.visible_length_ratio * (1.0 + 0.10 * identity["length_jitter"])
    visible_ratio = float(np.clip(visible_ratio, 0.38, 1.02))
    u = np.clip(t * visible_ratio, 0.0, 1.0)
    base = sample_curve(guide, u)
    tangent, frame_n, frame_b = transport_frames(base)

    guide_tangent, guide_n, guide_b = transport_frames(guide)
    a = float(np.dot(root_offset, guide_n[0]))
    b = float(np.dot(root_offset, guide_b[0]))
    along = float(np.dot(root_offset, guide_tangent[0]))
    width = 1.0 - 0.82 * s**0.90
    curve = base + frame_n * (a * width)[:, None] + frame_b * (b * width)[:, None]
    curve += tangent * (along * (1.0 - s) * 0.30)[:, None]

    length_factor = max(0.55, min(1.45, grown_length / 0.16))
    cycles = tex.cycles_per_10cm * (0.72 + 0.28 * length_factor)
    cycles *= 1.0 + identity["freq_jitter"]
    cycles = float(np.clip(cycles, 0.03, 18.0))
    radius = max(0.00004, tex.coil_radius_m * (1.0 + identity["radius_jitter"]))
    phase_noise = identity["phase_jitter"] * (1.0 - lock_coherence) * 0.85
    theta = 2.0 * math.pi * cycles * t + lock_phase + phase_noise

    envelope = 0.22 + 0.78 * smoothstep(np.clip(t / 0.08, 0.0, 1.0))
    envelope *= 0.92 + 0.08 * np.sin(math.pi * t)
    n_disp = radius * np.sin(theta)
    b_disp = radius * tex.cross_plane * np.cos(theta + 0.33)
    if tex.kink > 0.0:
        n_disp += radius * tex.kink * (0.38 * np.sin(3.0 * theta + 0.42) + 0.16 * np.sin(5.0 * theta - 0.35))
        b_disp += radius * tex.kink * 0.22 * np.sign(np.sin(theta + 0.18)) * np.sin(2.0 * theta)
    curve += frame_n * (n_disp * envelope)[:, None]
    curve += frame_b * (b_disp * envelope)[:, None]

    residual_env = np.sin(math.pi * t) ** 0.40 * (0.15 + 0.85 * t)
    residual_scale = 0.00010 + 0.00042 * tex.curl_coordinate
    for band in range(3):
        freq = (2.2 + band * 1.85) * (0.8 + 0.6 * tex.curl_coordinate)
        ph = identity["residual_phase"][band]
        coeff = identity["residual_coeff"][band] * residual_scale / (1.0 + band)
        curve += frame_n * (coeff * np.sin(2 * math.pi * freq * t + ph) * residual_env)[:, None]
        curve += frame_b * (0.55 * coeff * np.cos(2 * math.pi * freq * 0.71 * t + ph * 0.8) * residual_env)[:, None]
    curve[0] = root + normal * 0.00035
    return smooth_curve(curve, passes=1)


def tube_vertices(curve, base_radius, tip_radius, sides=3):
    n = len(curve)
    _, frame_n, frame_b = transport_frames(curve)
    vertices, normals = [], []
    for i in range(n):
        q = i / (n - 1)
        radius = base_radius * (1.0 - q) ** 0.76 + tip_radius * q
        for j in range(sides):
            angle = 2.0 * math.pi * j / sides
            radial = math.cos(angle) * frame_n[i] + math.sin(angle) * frame_b[i]
            vertices.append(curve[i] + radial * radius)
            normals.append(radial)
    return np.asarray(vertices, np.float32), np.asarray(normals, np.float32)


def tube_indices(samples: int, sides=3):
    indices = []
    for i in range(samples - 1):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = (i + 1) * sides + (j + 1) % sides
            d = (i + 1) * sides + j
            indices += [a, b, c, a, c, d]
    return np.asarray(indices, np.uint32)


def build_texture_lab(glb: Glb, *, strands: int, seed: int, length_profile: str):
    rng = np.random.default_rng(seed)
    samples = LENGTH_PROFILES[length_profile]["samples"]
    roots, root_normals = blue_noise_scalp(glb, strands, rng)
    head_node = next(i for i, n in enumerate(glb.doc["nodes"]) if n.get("name") == "AvatarHead")
    body_node = next((i for i, n in enumerate(glb.doc["nodes"]) if n.get("name") == "AvatarBody"), None)
    head_points, head_normals = mesh_world_geometry(glb, head_node)
    body_points, body_normals = (None, None)
    if body_node is not None:
        body_points, body_normals = mesh_world_geometry(glb, body_node)
    projector = CollisionProjector(head_points, head_normals, body_points, body_normals)

    keys = [lock_key(r) for r in roots]
    grouped = {}
    for i, key in enumerate(keys): grouped.setdefault(key, []).append(i)
    ordered_keys = sorted(grouped)
    key_to_id = {key: i for i, key in enumerate(ordered_keys)}
    locks = {}
    for key in ordered_keys:
        ids = grouped[key]
        root = roots[ids].mean(axis=0)
        normal = root_normals[ids].mean(axis=0); normal /= np.linalg.norm(normal) + 1e-12
        lrng = np.random.default_rng(seed + key[0] * 10007 + key[1] * 1009 + key[2] * 101)
        guide, grown_length = style_guide(root, normal, key, lrng, length_profile, samples)
        locks[key] = {"root": root, "normal": normal, "guide": projector.project(guide), "grown_length": grown_length, "phase": lrng.uniform(-math.pi, math.pi)}

    identities = [follower_identity(rng) for _ in range(strands)]
    base_radii = rng.uniform(0.000030, 0.000043, strands)
    tip_radii = rng.uniform(0.000006, 0.000011, strands)
    state_world_vertices = {label: [] for label in TEXTURE_LABELS}
    state_centerlines = {label: [] for label in TEXTURE_LABELS}
    base_normals, uv0, uv1, indices = [], [], [], []
    index_offset = 0
    per_strand_indices = tube_indices(samples)

    for si, (root, normal, key) in enumerate(zip(roots, root_normals, keys)):
        lock = locks[key]
        root_offset = root - lock["root"]
        lock_id = key_to_id[key]
        for label in TEXTURE_LABELS:
            tex = texture_state(label)
            curve = make_texture_curve(root, normal, lock["root"], lock["guide"], lock["grown_length"], label, identities[si], lock["phase"], tex.lock_phase_coherence, samples, root_offset)
            curve = projector.project(curve)
            state_centerlines[label].append(curve)
            vertices, normals = tube_vertices(curve, base_radii[si], tip_radii[si])
            state_world_vertices[label].append(vertices)
            if label == "1A": base_normals.append(normals)
        for i in range(samples):
            q = i / (samples - 1)
            for side in range(3):
                uv0.append([q, lock_id / max(1, len(ordered_keys) - 1)])
                uv1.append([si / max(1, strands - 1), side / 2.0])
        indices.append(per_strand_indices + index_offset)
        index_offset += samples * 3

    for label in TEXTURE_LABELS: state_world_vertices[label] = np.concatenate(state_world_vertices[label]).astype(np.float32)
    base_normals = np.concatenate(base_normals).astype(np.float32)
    uv0 = np.asarray(uv0, np.float32); uv1 = np.asarray(uv1, np.float32); indices = np.concatenate(indices).astype(np.uint32)
    _, head_bone = choose_head_and_bone(glb)
    inv = np.linalg.inv(global_matrix(glb.doc, head_bone))
    state_local = {label: transform_points(vertices.astype(np.float64), inv).astype(np.float32) for label, vertices in state_world_vertices.items()}
    normal_m = inv[:3, :3]
    normals_local = base_normals @ normal_m.T; normals_local /= np.linalg.norm(normals_local, axis=1, keepdims=True) + 1e-12
    stats = {"version":"0.11","strands":strands,"locks":len(ordered_keys),"samples":samples,"length_profile":length_profile,"seed":seed,"textures":list(TEXTURE_LABELS),"root_sampler":"area candidates + farthest-point canonical root selection"}
    return state_local, normals_local.astype(np.float32), uv0, uv1, indices, state_centerlines, stats


def install_texture_lab(glb: Glb, state_local, normals, uv0, uv1, indices, stats):
    base = state_local["1A"]
    pos_acc = glb.append_array(base, target=TARGET_ARRAY, component_type=COMPONENT_FLOAT, accessor_type="VEC3")
    norm_acc = glb.append_array(normals, target=TARGET_ARRAY, component_type=COMPONENT_FLOAT, accessor_type="VEC3")
    uv0_acc = glb.append_array(uv0, target=TARGET_ARRAY, component_type=COMPONENT_FLOAT, accessor_type="VEC2")
    uv1_acc = glb.append_array(uv1, target=TARGET_ARRAY, component_type=COMPONENT_FLOAT, accessor_type="VEC2")
    idx_acc = glb.append_array(indices.reshape(-1, 1), target=TARGET_ELEMENT, component_type=COMPONENT_UINT32, accessor_type="SCALAR")
    targets = []
    for label in TEXTURE_LABELS[1:]:
        delta = (state_local[label] - base).astype(np.float32)
        targets.append({"POSITION": glb.append_array(delta, target=TARGET_ARRAY, component_type=COMPONENT_FLOAT, accessor_type="VEC3")})
    material_index = len(glb.doc.setdefault("materials", []))
    glb.doc["materials"].append({"name":"Avatar Forge Hair Texture Lab","pbrMetallicRoughness":{"baseColorFactor":[0.055,0.016,0.007,1.0],"metallicFactor":0.0,"roughnessFactor":0.36},"doubleSided":True,"extensions":{"KHR_materials_anisotropy":{"anisotropyStrength":0.85,"anisotropyRotation":1.57079632679}}})
    used = glb.doc.setdefault("extensionsUsed", [])
    if "KHR_materials_anisotropy" not in used: used.append("KHR_materials_anisotropy")
    mesh_index = len(glb.doc.setdefault("meshes", []))
    glb.doc["meshes"].append({"name":"AvatarForgeHairTextureLab","weights":[0.0]*(len(TEXTURE_LABELS)-1),"primitives":[{"attributes":{"POSITION":pos_acc,"NORMAL":norm_acc,"TEXCOORD_0":uv0_acc,"TEXCOORD_1":uv1_acc},"indices":idx_acc,"material":material_index,"mode":4,"targets":targets}],"extras":{"avatarforge_role":"hair_texture_lab","texture_base":"1A","texture_targets":list(TEXTURE_LABELS[1:]),"texture_parameters":{label:texture_state(label).to_dict() for label in TEXTURE_LABELS},"dynamics":{"uv0_x":"root_to_tip","uv0_y":"lock_id_normalized","lock_count":stats["locks"]},**stats}})
    node_index = len(glb.doc.setdefault("nodes", [])); glb.doc["nodes"].append({"name":"AvatarForgeHairTextureLab","mesh":mesh_index})
    _, head_bone = choose_head_and_bone(glb); glb.doc["nodes"][head_bone].setdefault("children", []).append(node_index)
    glb.doc.setdefault("asset", {}).setdefault("extras", {})["avatarforge_hair"] = {"version":"0.11","node":node_index,"mesh":mesh_index,"generator":"tools/generate_texture_lab.py","texture_labels":list(TEXTURE_LABELS),"length_profile":stats["length_profile"],"root_sampler":stats["root_sampler"]}


def main():
    ap = argparse.ArgumentParser(description="Build one Avatar Forge groom with 1A→4C morph targets.")
    ap.add_argument("source", type=Path); ap.add_argument("output", type=Path)
    ap.add_argument("--strands", type=int, default=600); ap.add_argument("--seed", type=int, default=73129)
    ap.add_argument("--length", choices=sorted(LENGTH_PROFILES), default="medium")
    args = ap.parse_args()
    glb = Glb.load(args.source)
    state_local, normals, uv0, uv1, indices, _, stats = build_texture_lab(glb, strands=args.strands, seed=args.seed, length_profile=args.length)
    install_texture_lab(glb, state_local, normals, uv0, uv1, indices, stats)
    args.output.parent.mkdir(parents=True, exist_ok=True); glb.save(args.output)
    print(json.dumps({"source":str(args.source),"output":str(args.output),**stats,"vertices":int(len(state_local["1A"])),"triangles":int(len(indices)//3),"morph_targets":len(TEXTURE_LABELS)-1,"bytes":args.output.stat().st_size}, indent=2))

if __name__ == "__main__": main()
