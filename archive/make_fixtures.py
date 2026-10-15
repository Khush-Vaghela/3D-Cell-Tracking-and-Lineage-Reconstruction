"""
Builds a tiny synthetic dataset that mirrors the real layout:
  - {embryo_id}_{fov}.zarr : (T, Z, Y, X) uint16, chunked (1, Z, Y, X)
  - {embryo_id}_{fov}.geff : nodes/ids, nodes/props/{t,z,y,x}/values,
                             edges/ids, root attrs incl. estimated_number_of_nodes

Two embryos, two FOVs each, so embryo_grouped_split has something to split.
Small shapes (T=5, Z=8, Y=16, X=16) -- this is purely a plumbing test, not
a scale test.
"""

import os
import shutil

import numpy as np
import zarr
from zarr.codecs import BloscCodec

ROOT = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"


def make_zarr_volume(path, T=5, Z=8, Y=16, X=16, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4000, size=(T, Z, Y, X), dtype=np.uint16)
    arr = zarr.create_array(
        store=os.path.join(path, "0"),
        shape=(T, Z, Y, X),
        chunks=(1, Z, Y, X),
        dtype=np.uint16,
        # the dataset card says chunks are "compressed with blosc/zstd" --
        # use the real blosc codec so the manual blosc2-decompress fallback
        # path in data_io.py is tested against genuinely blosc-framed
        # bytes, not zarr's zstd-only default codec (which blosc2.decompress
        # can't read -- caught by this exact test).
        compressors=[BloscCodec(cname="zstd")],
    )
    arr[:] = data
    return data


def make_geff(path, n_nodes=6, estimated_number_of_nodes=10, seed=0):
    from zarr.codecs import ZstdCodec

    rng = np.random.default_rng(seed)
    root = zarr.open_group(path, mode="w")
    root.attrs["estimated_number_of_nodes"] = estimated_number_of_nodes

    node_ids = np.arange(1, n_nodes + 1)
    nodes = root.create_group("nodes")
    nodes.create_array("ids", shape=node_ids.shape, dtype=node_ids.dtype,
                        compressors=[ZstdCodec()])
    nodes["ids"][:] = node_ids

    props = nodes.create_group("props")
    t_vals = np.sort(rng.integers(0, 5, size=n_nodes))
    z_vals = rng.integers(0, 8, size=n_nodes)
    y_vals = rng.integers(0, 16, size=n_nodes)
    x_vals = rng.integers(0, 16, size=n_nodes)
    for name, vals in [("t", t_vals), ("z", z_vals), ("y", y_vals), ("x", x_vals)]:
        g = props.create_group(name)
        g.create_array("values", shape=vals.shape, dtype=vals.dtype,
                        compressors=[ZstdCodec()])
        g["values"][:] = vals

    # a couple of edges, including one division (node 1 -> node 2 and node 3)
    edges_arr = np.array([[1, 2], [1, 3], [2, 4]])
    edges = root.create_group("edges")
    edges.create_array("ids", shape=edges_arr.shape, dtype=edges_arr.dtype,
                        compressors=[ZstdCodec()])
    edges["ids"][:] = edges_arr


def main():
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    os.makedirs(ROOT)

    samples = [
        ("44b6", "0001_0190_1651_0467"),
        ("44b6", "0002_0300_1000_0900"),
        ("6bba", "0000_0195_0050_0532"),
        ("44b6", "0113de3b"),  # matches the short-fov style from the real competition path
    ]
    for i, (embryo_id, fov) in enumerate(samples):
        stem = f"{embryo_id}_{fov}"
        make_zarr_volume(os.path.join(ROOT, f"{stem}.zarr"), seed=i)
        make_geff(os.path.join(ROOT, f"{stem}.geff"), seed=i)
        print(f"built {stem}")


if __name__ == "__main__":
    main()
