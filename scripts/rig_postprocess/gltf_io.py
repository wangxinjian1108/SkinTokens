"""Minimal binary glTF (GLB) reader / writer with streaming buffer access.

Only what rig post-processing needs:

* read the JSON chunk without ever loading the mesh buffers,
* read individual accessors (optionally a prefix of one),
* rewrite a GLB by patching the BIN chunk in place, streaming every untouched
  region straight from the source file so multi-hundred-MB assets neither blow
  up memory nor grow in size.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
GLB_MAGIC = 0x46546C67

COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass(frozen=True)
class BinChunk:
    """Location of the BIN chunk inside a GLB file."""

    path: str
    offset: int
    length: int


def read_glb(path: str) -> tuple[dict, BinChunk | None]:
    """Return ``(gltf_json, bin_chunk)`` without reading the BIN bytes."""
    with open(path, "rb") as fh:
        header = fh.read(12)
        if len(header) < 12:
            raise ValueError(f"{path}: not a GLB file")
        magic, version, _total = struct.unpack("<III", header)
        if magic != GLB_MAGIC:
            raise ValueError(f"{path}: bad GLB magic")
        if version != 2:
            raise ValueError(f"{path}: unsupported GLB version {version}")
        gltf = None
        bin_chunk = None
        while True:
            chunk_header = fh.read(8)
            if len(chunk_header) < 8:
                break
            length, kind = struct.unpack("<II", chunk_header)
            start = fh.tell()
            if kind == JSON_CHUNK:
                gltf = json.loads(fh.read(length))
            elif kind == BIN_CHUNK:
                bin_chunk = BinChunk(path, start, length)
                fh.seek(length, 1)
            else:
                fh.seek(length, 1)
    if gltf is None:
        raise ValueError(f"{path}: no JSON chunk")
    return gltf, bin_chunk


def accessor_layout(gltf: dict, index: int) -> tuple[int, int, np.dtype, int]:
    """``(byte_offset, count, dtype, components)`` of accessor ``index``."""
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPE[acc["componentType"]])
    comps = TYPE_SIZE[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return offset, acc["count"], dtype, comps


def read_accessor(gltf: dict, bin_chunk: BinChunk, index: int, limit: int | None = None) -> np.ndarray:
    """Read accessor ``index`` (at most ``limit`` elements) as a numpy array."""
    if bin_chunk is None:
        raise ValueError("accessor read requires a BIN chunk")
    offset, count, dtype, comps = accessor_layout(gltf, index)
    if limit is not None:
        count = min(count, limit)
    stride = gltf["bufferViews"][gltf["accessors"][index]["bufferView"]].get("byteStride")
    with open(bin_chunk.path, "rb") as fh:
        fh.seek(bin_chunk.offset + offset)
        if stride in (None, dtype.itemsize * comps):
            raw = fh.read(count * dtype.itemsize * comps)
            arr = np.frombuffer(raw, dtype=dtype, count=count * comps)
        else:
            raw = fh.read(stride * count)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(count, stride)
            arr = arr[:, : dtype.itemsize * comps].copy().view(dtype)
    return arr.reshape(count, comps) if comps > 1 else arr.reshape(count)


@dataclass(frozen=True)
class Copy:
    """Stream ``length`` bytes from ``path`` starting at ``offset``."""

    path: str
    offset: int
    length: int


@dataclass(frozen=True)
class Literal:
    """In-line bytes (a patch)."""

    data: bytes


BinPart = Union[Copy, Literal]


def patch_bin(bin_chunk: BinChunk, patches: list[tuple[int, bytes]]) -> list[BinPart]:
    """Rewrite ``bin_chunk`` with ``(relative_offset, bytes)`` patches, streaming
    every untouched region straight from the source file (no size growth)."""
    parts: list[BinPart] = []
    cursor = 0
    for offset, payload in sorted(patches, key=lambda item: item[0]):
        if offset < cursor:
            raise ValueError(f"overlapping BIN patches at {offset}")
        if offset + len(payload) > bin_chunk.length:
            raise ValueError(f"BIN patch at {offset} runs past the chunk")
        if offset > cursor:
            parts.append(Copy(bin_chunk.path, bin_chunk.offset + cursor, offset - cursor))
        parts.append(Literal(payload))
        cursor = offset + len(payload)
    if cursor < bin_chunk.length:
        parts.append(Copy(bin_chunk.path, bin_chunk.offset + cursor, bin_chunk.length - cursor))
    return parts


def _iter_bin_bytes(parts: Iterable[BinPart], chunk_size: int = 8 << 20):
    for part in parts:
        if isinstance(part, Literal):
            yield part.data
            continue
        with open(part.path, "rb") as fh:
            fh.seek(part.offset)
            remaining = part.length
            while remaining > 0:
                block = fh.read(min(chunk_size, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block


def write_glb(path: str, gltf: dict, bin_parts: list[BinPart]) -> None:
    """Write ``gltf`` as a GLB whose BIN chunk is the concatenation of ``bin_parts``."""
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)

    bin_length = sum(len(p.data) if isinstance(p, Literal) else p.length for p in bin_parts)
    padded_bin_length = bin_length + (-bin_length % 4)
    gltf.setdefault("buffers", [{}])[0]["byteLength"] = bin_length

    total = 12 + 8 + len(json_bytes) + (8 + padded_bin_length if bin_parts else 0)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", GLB_MAGIC, 2, total))
        fh.write(struct.pack("<II", len(json_bytes), JSON_CHUNK))
        fh.write(json_bytes)
        if bin_parts:
            fh.write(struct.pack("<II", padded_bin_length, BIN_CHUNK))
            written = 0
            for block in _iter_bin_bytes(bin_parts):
                fh.write(block)
                written += len(block)
            if written != bin_length:
                raise IOError(f"short BIN write: {written} != {bin_length}")
            fh.write(b"\x00" * (padded_bin_length - bin_length))
