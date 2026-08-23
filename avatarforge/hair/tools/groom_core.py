from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
COMPONENT_FLOAT = 5126
COMPONENT_UINT32 = 5125
TARGET_ARRAY = 34962
TARGET_ELEMENT = 34963


def align4(data: bytearray) -> None:
    while len(data) % 4:
        data.append(0)


@dataclass
class Glb:
    doc: dict
    binary: bytearray

    @classmethod
    def load(cls, path):
        raw = path.read_bytes()
        magic, version, length = struct.unpack_from('<4sII', raw, 0)
        if magic != b'glTF' or version != 2:
            raise ValueError('expected glTF 2.0 GLB')
        off = 12
        doc = None
        binary = bytearray()
        while off < length:
            chunk_len, chunk_type = struct.unpack_from('<II', raw, off)
            off += 8
            chunk = raw[off:off + chunk_len]
            off += chunk_len
            if chunk_type == JSON_CHUNK:
                doc = json.loads(chunk.decode('utf-8').rstrip('\x00 '))
            elif chunk_type == BIN_CHUNK:
                binary = bytearray(chunk)
        if doc is None:
            raise ValueError('GLB has no JSON chunk')
        return cls(doc, binary)

    def save(self, path) -> None:
        self.doc.setdefault('buffers', [{}])[0]['byteLength'] = len(self.binary)
        j = json.dumps(self.doc, separators=(',', ':')).encode('utf-8')
        while len(j) % 4:
            j += b' '
        b = bytes(self.binary)
        while len(b) % 4:
            b += b'\x00'
        total = 12 + 8 + len(j) + 8 + len(b)
        out = bytearray(struct.pack('<4sII', b'glTF', 2, total))
        out += struct.pack('<II', len(j), JSON_CHUNK) + j
        out += struct.pack('<II', len(b), BIN_CHUNK) + b
        path.write_bytes(out)

    def accessor(self, index: int) -> np.ndarray:
        a = self.doc['accessors'][index]
        bv = self.doc['bufferViews'][a['bufferView']]
        dtype = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a['componentType']]
        width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
        start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        item = np.dtype(dtype).itemsize * width
        stride = bv.get('byteStride', item)
        count = a['count']
        if stride == item:
            return np.frombuffer(self.binary, dtype=dtype, count=count * width, offset=start).reshape(count, width).copy()
        arr = np.empty((count, width), dtype=dtype)
        view = memoryview(self.binary)
        for i in range(count):
            arr[i] = np.frombuffer(view, dtype=dtype, count=width, offset=start + i * stride)
        return arr

    def append_array(self, arr: np.ndarray, *, target: int, component_type: int, accessor_type: str) -> int:
        arr = np.ascontiguousarray(arr)
        align4(self.binary)
        offset = len(self.binary)
        payload = arr.tobytes()
        self.binary.extend(payload)
        bv_index = len(self.doc.setdefault('bufferViews', []))
        self.doc['bufferViews'].append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(payload), 'target': target})
        acc = {'bufferView': bv_index, 'componentType': component_type, 'count': int(arr.shape[0]), 'type': accessor_type}
        if np.issubdtype(arr.dtype, np.floating):
            acc['min'] = arr.min(axis=0).astype(float).tolist()
            acc['max'] = arr.max(axis=0).astype(float).tolist()
        acc_index = len(self.doc.setdefault('accessors', []))
        self.doc['accessors'].append(acc)
        return acc_index


def quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w, 0],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w, 0],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def local_matrix(node):
    if 'matrix' in node:
        return np.asarray(node['matrix'], dtype=np.float64).reshape(4, 4).T
    t = np.eye(4)
    t[:3, 3] = node.get('translation', [0, 0, 0])
    r = quat_matrix(node.get('rotation', [0, 0, 0, 1]))
    s = np.diag([*node.get('scale', [1, 1, 1]), 1.0])
    return t @ r @ s


def global_matrix(doc, node_index):
    parents = {}
    for i, node in enumerate(doc.get('nodes', [])):
        for child in node.get('children', []):
            parents[child] = i
    chain = [node_index]
    while chain[-1] in parents:
        chain.append(parents[chain[-1]])
    matrix = np.eye(4)
    for idx in reversed(chain):
        matrix = matrix @ local_matrix(doc['nodes'][idx])
    return matrix


def transform_points(points, matrix):
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def choose_head_and_bone(glb: Glb):
    head_node = next(i for i, node in enumerate(glb.doc['nodes']) if node.get('name') == 'AvatarHead')
    head_mesh_index = glb.doc['nodes'][head_node]['mesh']
    bone_node = next(i for i, node in enumerate(glb.doc['nodes']) if node.get('name') == 'Head')
    return head_mesh_index, bone_node


def find_mesh_node(glb: Glb, name: str):
    for i, node in enumerate(glb.doc.get('nodes', [])):
        if node.get('name') == name and 'mesh' in node:
            return i
    return None


def mesh_world_geometry(glb: Glb, node_index: int):
    node = glb.doc['nodes'][node_index]
    primitive = glb.doc['meshes'][node['mesh']]['primitives'][0]
    positions = glb.accessor(primitive['attributes']['POSITION']).astype(np.float64)
    normals = glb.accessor(primitive['attributes']['NORMAL']).astype(np.float64) if 'NORMAL' in primitive['attributes'] else None
    matrix = global_matrix(glb.doc, node_index)
    world = transform_points(positions, matrix)
    if normals is not None:
        normal_matrix = np.linalg.inv(matrix[:3, :3]).T
        normals = normals @ normal_matrix.T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return world, normals


def sample_scalp(glb: Glb, count: int, rng: np.random.Generator):
    head_mesh_index, _ = choose_head_and_bone(glb)
    primitive = glb.doc['meshes'][head_mesh_index]['primitives'][0]
    verts = glb.accessor(primitive['attributes']['POSITION']).astype(np.float64)
    norms = glb.accessor(primitive['attributes']['NORMAL']).astype(np.float64)
    indices = glb.accessor(primitive['indices']).reshape(-1).astype(np.int64)
    triangles = indices.reshape(-1, 3)
    tv = verts[triangles]
    tn = norms[triangles]
    centers = tv.mean(axis=1)
    area = 0.5 * np.linalg.norm(np.cross(tv[:, 1]-tv[:, 0], tv[:, 2]-tv[:, 0]), axis=1)
    xabs = np.abs(centers[:, 0])
    scalp = (centers[:, 1] > 1.595) & ((centers[:, 2] < 0.018) | (centers[:, 1] > 1.695) | ((xabs > 0.052) & (centers[:, 2] < 0.058)))
    ids = np.flatnonzero(scalp)
    weights = area[ids] / area[ids].sum()
    picks = rng.choice(ids, size=count, replace=True, p=weights)
    u, v = rng.random(count), rng.random(count)
    swap = u + v > 1
    u[swap], v[swap] = 1-u[swap], 1-v[swap]
    w = 1-u-v
    roots = tv[picks, 0]*w[:, None] + tv[picks, 1]*u[:, None] + tv[picks, 2]*v[:, None]
    ns = tn[picks, 0]*w[:, None] + tn[picks, 1]*u[:, None] + tn[picks, 2]*v[:, None]
    ns /= np.linalg.norm(ns, axis=1, keepdims=True) + 1e-12
    return roots, ns


def lock_key(root):
    side = 0 if root[0] < 0 else 1
    depth = int(np.clip(np.floor((root[2] + 0.095) / 0.205 * 4.0), 0, 3))
    height = int(np.clip(np.floor((root[1] - 1.585) / 0.200 * 3.0), 0, 2))
    return side, depth, height


def smoothstep(x):
    return x*x*(3-2*x)


def smooth_curve(curve, passes=2):
    result = curve.copy()
    for _ in range(passes):
        next_curve = result.copy()
        next_curve[1:-1] = 0.22*result[:-2] + 0.56*result[1:-1] + 0.22*result[2:]
        result = next_curve
    return result


def transport_frames(curve):
    tangent = np.empty_like(curve)
    tangent[1:-1] = curve[2:] - curve[:-2]
    tangent[0] = curve[1] - curve[0]
    tangent[-1] = curve[-1] - curve[-2]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    normal = np.empty_like(curve)
    binormal = np.empty_like(curve)
    ref = np.array([0.0, 0.0, 1.0]) if abs(tangent[0, 2]) < 0.82 else np.array([1.0, 0.0, 0.0])
    n0 = np.cross(tangent[0], ref); n0 /= np.linalg.norm(n0) + 1e-12
    normal[0] = n0
    binormal[0] = np.cross(tangent[0], normal[0]); binormal[0] /= np.linalg.norm(binormal[0]) + 1e-12
    for i in range(1, len(curve)):
        n = normal[i-1] - tangent[i] * np.dot(normal[i-1], tangent[i])
        if np.linalg.norm(n) < 1e-9:
            n = np.cross(tangent[i], binormal[i-1])
        n /= np.linalg.norm(n) + 1e-12
        b = np.cross(tangent[i], n); b /= np.linalg.norm(b) + 1e-12
        normal[i], binormal[i] = n, b
    return tangent, normal, binormal
