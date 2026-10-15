"""
Data loading layer for the Biohub Cell Tracking During Development competition.

Kaggle paths (adjust COMPETITION_SLUG if Kaggle renames the mount point):

    /kaggle/input/competitions/biohub-cell-tracking-during-development/train/
    /kaggle/input/competitions/biohub-cell-tracking-during-development/test/

Three responsibilities, kept separate so every downstream stage (EDA, target
generation, training, local metric scoring) goes through the same code path:

1. open_zarr_volume(path)   -> array-like (T, Z, Y, X), lazy per-timepoint access
2. load_geff(path)          -> GeffData(nodes_df, edges_df, estimated_number_of_nodes)
3. list_samples(root)       -> one entry per {embryo_id}_{fov} folder found
   embryo_grouped_split(...)-> train/val split that never puts two FOVs of the
                               same embryo on opposite sides of the split

Only 0.2 from the workflow -- no target generation, no metric, no training here.

Backend selection (important for Kaggle specifically)
-------------------------------------------------------
This data is zarr **v3** format. zarr-python only gained v3 support in its
own 3.x releases -- if the Kaggle base image ships an older zarr (2.x), it
cannot read this data at all, and will fail with an error that has nothing
to do with "not installed" (so a bare `except ImportError` won't catch it).
`_zarr_v3_available()` checks the installed version explicitly and routes
to the manual reader whenever it's not new enough, not just when the
import itself fails.

If you want the real `zarr` library path (nicer error messages, handles
edge cases in codecs we haven't manually implemented), pip install
`zarr>=3` in a cell with internet enabled, then re-run. Note the
competition's own submission notebooks run with **internet disabled**, so
for an actual submission you'd need to vendor a `zarr>=3` wheel via an
attached Kaggle Dataset and `pip install --no-index` from it -- the manual
path below exists specifically so you don't have to.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Kaggle paths
# ---------------------------------------------------------------------------

COMPETITION_SLUG = "biohub-cell-tracking-during-development"
KAGGLE_INPUT_ROOT = f"/kaggle/input/competitions/{COMPETITION_SLUG}"
TRAIN_DIR = os.path.join(KAGGLE_INPUT_ROOT, "train")
TEST_DIR = os.path.join(KAGGLE_INPUT_ROOT, "test")

# Physical voxel scale (Z, Y, X) in um/voxel, per the dataset card. Confirmed
# against the official repo's io.py DEFAULT_SCALE constant -- same values.
NATIVE_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _zarr_v3_available() -> bool:
    """
    True only if `zarr` is importable AND its major version is >= 3.
    zarr 2.x imports fine but silently can't handle this v3-format data,
    so checking import success alone isn't enough.
    """
    try:
        import zarr
    except ImportError:
        return False
    major = int(zarr.__version__.split(".")[0])
    return major >= 3


# ---------------------------------------------------------------------------
# Generic manual zarr v3 array reading (shared by the volume and geff
# fallback paths -- the two differ only in which chunk codec they use:
# image volumes are "compressed with blosc/zstd" i.e. the blosc codec
# (blosc-framed bytes, decoded with the `blosc2` package), geff arrays
# "use zstd compression" i.e. the plain zstd codec (raw zstd frames,
# decoded with the `zstandard` package) -- confirmed against a real
# zarr.json written by zarr-python itself: codecs is a list like
# [{"name": "bytes", ...}, {"name": "zstd"|"blosc", ...}].
# ---------------------------------------------------------------------------

def _read_zarr_v3_json(array_dir: str) -> dict:
    with open(os.path.join(array_dir, "zarr.json")) as f:
        return json.load(f)


def _decompress_chunk(raw: bytes, codec_name: Optional[str]) -> bytes:
    if codec_name == "blosc":
        import blosc2
        return blosc2.decompress(raw)
    if codec_name == "zstd":
        import zstandard
        return zstandard.ZstdDecompressor().decompress(raw)
    if codec_name is None:
        return raw  # uncompressed ("bytes" codec only)
    raise NotImplementedError(
        f"No manual decoder implemented for zarr codec '{codec_name}'. "
        f"Either pip install zarr>=3 to use the real library path, or "
        f"extend _decompress_chunk for this codec."
    )


def _compression_codec_name(meta: dict) -> Optional[str]:
    for codec in meta.get("codecs", []):
        if codec["name"] in ("blosc", "zstd"):
            return codec["name"]
    return None  # only a "bytes" codec (or none) -- data is uncompressed


def _read_full_zarr_v3_array(array_dir: str) -> np.ndarray:
    """
    Manually read an entire small zarr v3 array into memory. Used for
    `.geff` arrays (node ids, per-node t/z/y/x properties, edge ids) --
    all small enough (at most low thousands of entries per sample) to load
    eagerly rather than chunk-by-chunk like the much larger image volumes.
    """
    meta = _read_zarr_v3_json(array_dir)
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["data_type"])
    codec_name = _compression_codec_name(meta)

    if any(s == 0 for s in shape):
        return np.zeros(shape, dtype=dtype)

    chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    separator = meta.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
    n_chunks_per_dim = [-(-s // c) for s, c in zip(shape, chunk_shape)]  # ceil division

    out = np.empty(shape, dtype=dtype)
    for chunk_idx in itertools.product(*[range(n) for n in n_chunks_per_dim]):
        chunk_key = separator.join(str(i) for i in chunk_idx)
        chunk_path = os.path.join(array_dir, "c", *chunk_key.split(separator))
        with open(chunk_path, "rb") as f:
            raw = f.read()
        decompressed = _decompress_chunk(raw, codec_name)

        starts = [i * c for i, c in zip(chunk_idx, chunk_shape)]
        ends = [min(s + c, dim) for s, c, dim in zip(starts, chunk_shape, shape)]
        actual_shape = tuple(e - s for s, e in zip(starts, ends))

        full_chunk = np.frombuffer(decompressed, dtype=dtype).reshape(chunk_shape)
        trimmed = full_chunk[tuple(slice(0, a) for a in actual_shape)]
        out[tuple(slice(s, e) for s, e in zip(starts, ends))] = trimmed

    return out


# ---------------------------------------------------------------------------
# 1. Image volume reading
# ---------------------------------------------------------------------------

class _ManualBloscVolume:
    """
    Fallback reader used when a v3-capable `zarr` isn't available. Mirrors
    the starter notebook's manual chunk-decompression approach, but wrapped
    so callers get the same __getitem__-by-timepoint / .shape / .dtype
    interface as a real zarr array. Kept as its own lightweight class
    (rather than routing through `_read_full_zarr_v3_array`) because
    volumes are large (typically 100 x 64 x 256 x 256 uint16 =~ 1.6 GB) --
    we specifically want lazy, one-chunk-at-a-time access, not an eager
    full-array read.

    Assumes the layout described in the dataset card: array metadata at
    `{path}/zarr.json`, one chunk per timepoint at `{path}/c/{t}/0/0/0`
    (chunk shape (1, Z, Y, X) means only the T axis is ever split across
    more than one chunk).
    """

    def __init__(self, array_dir: str):
        self._array_dir = array_dir
        meta = _read_zarr_v3_json(array_dir)
        self.shape = tuple(meta["shape"])  # (T, Z, Y, X)
        self.dtype = np.dtype(meta["data_type"])
        self._frame_shape = self.shape[1:]  # (Z, Y, X)
        self._codec_name = _compression_codec_name(meta)

    def __len__(self) -> int:
        return self.shape[0]

    def _read_frame(self, t: int) -> np.ndarray:
        chunk_path = os.path.join(self._array_dir, "c", str(t), "0", "0", "0")
        with open(chunk_path, "rb") as f:
            raw = f.read()
        decompressed = _decompress_chunk(raw, self._codec_name)
        return np.frombuffer(decompressed, dtype=self.dtype).reshape(self._frame_shape)

    def __getitem__(self, key):
        # Only the access pattern the rest of this pipeline needs: a single
        # timepoint index, or a slice across timepoints.
        if isinstance(key, slice):
            return np.stack([self._read_frame(t) for t in range(*key.indices(self.shape[0]))])
        if isinstance(key, int):
            return self._read_frame(key)
        raise TypeError(
            f"_ManualBloscVolume only supports int or slice indexing on the "
            f"time axis, got {type(key)}"
        )


def open_zarr_volume(zarr_path: str):
    """
    Open a sample's image volume, shape (T, Z, Y, X).

    Uses the real `zarr` library when a v3-capable version (>=3) is
    installed -- correct, handles codec/edge cases we haven't manually
    implemented. Otherwise falls back to manual chunk reads, so this still
    runs against a Kaggle base image that only ships zarr 2.x.

    `zarr_path` is the sample's `.zarr` directory (e.g.
    "{TRAIN_DIR}/44b6_0113de3b.zarr"). Per the dataset card the array's
    own metadata lives at "{zarr_path}/0/zarr.json" -- the outer ".zarr"
    directory is *not* itself a zarr group (no top-level zarr.json), it's
    just a plain directory holding the array at sub-path "0". Confirmed
    against a fixture built to the dataset card's exact layout: opening
    `zarr_path` itself as a group fails with GroupNotFoundError; opening
    "{zarr_path}/0" directly as an array works.
    """
    array_path = os.path.join(zarr_path, "0")

    if not _zarr_v3_available():
        return _ManualBloscVolume(array_path)

    import zarr
    return zarr.open_array(array_path, mode="r")


# ---------------------------------------------------------------------------
# 2. .geff ground-truth reading
# ---------------------------------------------------------------------------

def _find_estimated_number_of_nodes_zarr(group) -> Optional[int]:
    """Real-zarr-library path: check root attrs, then nodes/ attrs."""
    for candidate in (group, group.get("nodes", None)):
        if candidate is None:
            continue

        geff_attrs = candidate.attrs.get("geff", {})
        extra = geff_attrs.get("extra", {})

        value = extra.get("estimated_number_of_nodes")

        if value is not None:
            return int(value)

    return None


def _find_estimated_number_of_nodes_manual(geff_path: str) -> Optional[int]:
    """Read estimated_number_of_nodes from GEFF metadata."""
    for candidate_dir in (geff_path, os.path.join(geff_path, "nodes")):
        json_path = os.path.join(candidate_dir, "zarr.json")

        if not os.path.isfile(json_path):
            continue

        with open(json_path) as f:
            meta = json.load(f)

        value = (
            meta.get("attributes", {})
                .get("geff", {})
                .get("extra", {})
                .get("estimated_number_of_nodes")
        )

        if value is not None:
            return int(value)

    return None


@dataclass
class GeffData:
    nodes: pd.DataFrame          # columns: node_id, t, z, y, x
    edges: pd.DataFrame          # columns: source_id, target_id
    estimated_number_of_nodes: Optional[int]


def _load_geff_zarr(geff_path: str) -> GeffData:
    import zarr

    root = zarr.open(geff_path, mode="r")

    node_ids = np.asarray(root["nodes"]["ids"][:])
    props = root["nodes"]["props"]
    nodes = pd.DataFrame({
        "node_id": node_ids,
        "t": np.asarray(props["t"]["values"][:]),
        "z": np.asarray(props["z"]["values"][:]),
        "y": np.asarray(props["y"]["values"][:]),
        "x": np.asarray(props["x"]["values"][:]),
    })

    edge_ids = np.asarray(root["edges"]["ids"][:])
    if edge_ids.size == 0:
        edges = pd.DataFrame({"source_id": pd.Series(dtype=node_ids.dtype),
                               "target_id": pd.Series(dtype=node_ids.dtype)})
    else:
        edges = pd.DataFrame({"source_id": edge_ids[:, 0], "target_id": edge_ids[:, 1]})

    estimated_number_of_nodes = _find_estimated_number_of_nodes_zarr(root)
    return GeffData(nodes=nodes, edges=edges, estimated_number_of_nodes=estimated_number_of_nodes)


def _load_geff_manual(geff_path: str) -> GeffData:
    node_ids = _read_full_zarr_v3_array(os.path.join(geff_path, "nodes", "ids"))
    props_dir = os.path.join(geff_path, "nodes", "props")
    nodes = pd.DataFrame({
        "node_id": node_ids,
        "t": _read_full_zarr_v3_array(os.path.join(props_dir, "t", "values")),
        "z": _read_full_zarr_v3_array(os.path.join(props_dir, "z", "values")),
        "y": _read_full_zarr_v3_array(os.path.join(props_dir, "y", "values")),
        "x": _read_full_zarr_v3_array(os.path.join(props_dir, "x", "values")),
    })

    edge_ids = _read_full_zarr_v3_array(os.path.join(geff_path, "edges", "ids"))
    if edge_ids.size == 0:
        edges = pd.DataFrame({"source_id": pd.Series(dtype=node_ids.dtype),
                               "target_id": pd.Series(dtype=node_ids.dtype)})
    else:
        edges = pd.DataFrame({"source_id": edge_ids[:, 0], "target_id": edge_ids[:, 1]})

    estimated_number_of_nodes = _find_estimated_number_of_nodes_manual(geff_path)
    return GeffData(nodes=nodes, edges=edges, estimated_number_of_nodes=estimated_number_of_nodes)


def load_geff(geff_path: str) -> GeffData:
    """
    Read a `.geff` ground-truth graph into plain pandas DataFrames.

    Uses the real `zarr` library when a v3-capable version is installed,
    otherwise a manual reader (zstd-codec, per the dataset card's "All
    arrays within .geff use zstd compression" -- a different codec from
    the image volumes' blosc/zstd, so this is a separate manual path from
    `_ManualBloscVolume`, not a reuse of it).

    Returns
    -------
    GeffData(nodes, edges, estimated_number_of_nodes)
        nodes : DataFrame[node_id, t, z, y, x]
        edges : DataFrame[source_id, target_id]
        estimated_number_of_nodes : the coarse true-cell-count estimate used
            by the edge-Jaccard node-count penalty, or None if not found in
            either the root or nodes/ group's metadata.
    """
    if _zarr_v3_available():
        return _load_geff_zarr(geff_path)
    return _load_geff_manual(geff_path)


# ---------------------------------------------------------------------------
# 3. Sample listing + embryo-grouped split
# ---------------------------------------------------------------------------

# Folder names look like "{embryo_id}_{field_of_view}", e.g.
# "44b6_0049_0438_1330_1273" or the shorter "44b6_0113de3b" -- embryo_id is
# always the first underscore-delimited segment; everything after the
# first underscore (however many segments) is the field-of-view id.
_SAMPLE_NAME_RE = re.compile(r"^(?P<embryo_id>[^_]+)_(?P<fov>.+)$")


@dataclass
class Sample:
    name: str                 # full folder stem, e.g. "44b6_0113de3b"
    embryo_id: str
    fov: str
    zarr_path: str
    geff_path: Optional[str]  # None for test-set samples (no ground truth)


def list_samples(root_dir: str) -> list[Sample]:
    """
    Scan a train/ or test/ directory (e.g. TRAIN_DIR / TEST_DIR above) and
    return one Sample per {embryo_id}_{fov}.zarr found, paired with its
    .geff if present.
    """
    samples = []
    for entry in sorted(os.listdir(root_dir)):
        if not entry.endswith(".zarr"):
            continue
        stem = entry[: -len(".zarr")]
        match = _SAMPLE_NAME_RE.match(stem)
        if not match:
            raise ValueError(
                f"Folder name '{stem}' doesn't match the expected "
                f"'{{embryo_id}}_{{field_of_view}}' pattern"
            )
        geff_path = os.path.join(root_dir, stem + ".geff")
        samples.append(Sample(
            name=stem,
            embryo_id=match.group("embryo_id"),
            fov=match.group("fov"),
            zarr_path=os.path.join(root_dir, entry),
            geff_path=geff_path if os.path.isdir(geff_path) else None,
        ))
    return samples


def embryo_grouped_split(
    samples: list[Sample],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample]]:
    """
    Split samples into train/val by embryo_id, so every FOV from a given
    embryo lands entirely on one side. Mirrors the competition's own
    train/test split (embryo-disjoint) -- validating across embryos your
    model has already seen FOVs from gives an optimistic local score that
    won't predict leaderboard behavior.

    Splits at the embryo level by count of embryos, not by sample count --
    with only a handful of embryos this can swing away from exactly
    `val_fraction`; check the resulting sample counts per split rather than
    assuming they land close to the requested fraction.
    """
    embryo_ids = sorted({s.embryo_id for s in samples})
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(embryo_ids)

    n_val_embryos = max(1, round(len(embryo_ids) * val_fraction))
    val_embryo_ids = set(shuffled[:n_val_embryos])

    train_samples = [s for s in samples if s.embryo_id not in val_embryo_ids]
    val_samples = [s for s in samples if s.embryo_id in val_embryo_ids]
    return train_samples, val_samples


if __name__ == "__main__":
    # Quick sanity check when run directly in a Kaggle notebook cell:
    # tells you immediately which backend is active and how many samples
    # were found, before you build anything on top of it.
    backend = "real zarr>=3" if _zarr_v3_available() else "manual fallback (zarr<3 or missing)"
    print(f"backend: {backend}")
    if os.path.isdir(TRAIN_DIR):
        samples = list_samples(TRAIN_DIR)
        print(f"found {len(samples)} train samples, {len({s.embryo_id for s in samples})} embryos")
    else:
        print(f"TRAIN_DIR not found at {TRAIN_DIR} -- update COMPETITION_SLUG or run this in the Kaggle notebook")
