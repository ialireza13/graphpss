"""GraphPSS storage, splits and batching.

All graphs live in one compact ``.npz`` store built once from the CSV release:

    ids       [G]     PDB ids
    node_ptr  [G+1]   node offsets into ``y``
    edge_ptr  [G+1]   edge offsets into ``edges``
    y         [N]     int8 labels (0 = coil, 1 = helix, 2 = sheet)
    edges     [E, 2]  int16 local (per-graph) directed edges, both directions

Node features are the 1-based residue index, so they are not stored.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

NUM_CLASSES = 3
CLASS_NAMES = ["coil", "helix", "sheet"]


# ---------------------------------------------------------------- building

def _read_one(args):
    csv_dir, pdb_id = args
    pos = np.loadtxt(f"{csv_dir}/{pdb_id}.node_feature", delimiter=",", dtype=np.int64, ndmin=1)
    y = np.loadtxt(f"{csv_dir}/{pdb_id}.node_labels", dtype=np.int8, ndmin=1)
    edges = np.loadtxt(f"{csv_dir}/{pdb_id}.edges", delimiter=",", dtype=np.int64, ndmin=2)
    n = len(y)
    ok = len(pos) == n and np.array_equal(pos, np.arange(1, n + 1))
    if edges.size == 0:
        edges = np.zeros((0, 2), dtype=np.int64)
    return pdb_id, y, edges.astype(np.int16), ok


def _read_one_pt(args):
    pt_dir, name = args
    d = torch.load(f"{pt_dir}/{name}", weights_only=False)
    d = d.to_dict() if hasattr(d, "to_dict") else d
    y = np.asarray(d["y"]).ravel().astype(np.int8)
    pos = np.asarray(d["x"]).ravel()
    edges = np.asarray(d["edge_index"]).T.astype(np.int16)
    return name.split(".")[0].upper(), y, edges, np.array_equal(pos, np.arange(1, len(y) + 1))


def build_store(src_dir: str, out_path: str, workers: int = 64, from_pt: bool = False) -> dict:
    """Build the store from the CSV release directory, or from per-protein .pt files."""
    if from_pt:
        names = sorted(os.listdir(src_dir), key=lambda f: f.split(".")[0].upper())
        reader, jobs = _read_one_pt, [(src_dir, f) for f in names]
    else:
        ids = sorted({f.split(".")[0] for f in os.listdir(src_dir) if f.endswith(".node_labels")})
        reader, jobs = _read_one, [(src_dir, i) for i in ids]
    ids, ys, es, bad = [], [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for pdb_id, y, e, ok in ex.map(reader, jobs, chunksize=256):
            ids.append(pdb_id)
            ys.append(y)
            es.append(e)
            if not ok:
                bad.append(pdb_id)
    node_ptr = np.concatenate([[0], np.cumsum([len(y) for y in ys])])
    edge_ptr = np.concatenate([[0], np.cumsum([len(e) for e in es])])
    np.savez(out_path, ids=np.array(ids), node_ptr=node_ptr, edge_ptr=edge_ptr,
             y=np.concatenate(ys), edges=np.concatenate(es))
    return {"graphs": len(ids), "nodes": int(node_ptr[-1]), "edges": int(edge_ptr[-1]),
            "non_arange_features": bad}


@dataclass
class GraphStore:
    ids: np.ndarray
    node_ptr: np.ndarray
    edge_ptr: np.ndarray
    y: np.ndarray
    edges: np.ndarray

    @classmethod
    def load(cls, path: str) -> "GraphStore":
        z = np.load(path)
        return cls(**{k: z[k] for k in ("ids", "node_ptr", "edge_ptr", "y", "edges")})

    def __len__(self):
        return len(self.ids)

    def num_nodes(self, i: int) -> int:
        return int(self.node_ptr[i + 1] - self.node_ptr[i])

    def index_of(self, pdb_id: str) -> int:
        return int(np.searchsorted(self.ids, pdb_id.upper()))


# ---------------------------------------------------------------- splits

def make_split(n_graphs: int, seed: int = 0, ratios=(0.8, 0.1, 0.1)) -> dict:
    perm = np.random.default_rng(seed).permutation(n_graphs)
    a = int(ratios[0] * n_graphs)
    b = a + int(ratios[1] * n_graphs)
    return {"train": perm[:a].tolist(), "val": perm[a:b].tolist(), "test": perm[b:].tolist()}


def load_or_make_split(path: str, n_graphs: int, seed: int = 0) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    if "cluster" in os.path.basename(path):
        raise FileNotFoundError(f"{path}: build it with scripts/make_cluster_split.py")
    split = make_split(n_graphs, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f)
    return split


# ---------------------------------------------------------------- batching

def exact_hop_indices(edge_index: np.ndarray, n: int, k_max: int) -> list[np.ndarray]:
    """Edge lists of node pairs at shortest-path distance exactly k, for k = 2..k_max."""
    a = sp.csr_matrix((np.ones(edge_index.shape[1], dtype=np.float32), edge_index), shape=(n, n))
    a.data[:] = 1.0
    reach = (a + sp.eye(n, dtype=np.float32, format="csr")).astype(bool).astype(np.float32)
    frontier = a
    out = []
    for _ in range(2, k_max + 1):
        nxt = (frontier @ a).astype(bool).astype(np.float32)
        exact = (nxt - nxt.multiply(reach)).tocoo()
        exact.eliminate_zeros()
        out.append(np.vstack([exact.row, exact.col]).astype(np.int64))
        reach = (reach + nxt).astype(bool).astype(np.float32)
        frontier = exact.tocsr()
    return out


def _hops_one(args):
    edges, n, k_max = args
    return [h.T.astype(np.int16) for h in exact_hop_indices(edges.T.astype(np.int64), n, k_max)]


def build_hop_store(store: "GraphStore", out_path: str, k_max: int = 3, workers: int = 24):
    """Precompute exact-k-hop pair lists (k = 2..k_max) for every graph, local int16 indices."""
    jobs = ((store.edges[store.edge_ptr[g]:store.edge_ptr[g + 1]], store.num_nodes(g), k_max)
            for g in range(len(store)))
    per_k = [[] for _ in range(2, k_max + 1)]
    with ProcessPoolExecutor(workers) as ex:
        for hops in ex.map(_hops_one, jobs, chunksize=256):
            for lst, h in zip(per_k, hops):
                lst.append(h)
    # One .npy per array so training processes can memory-map them (they are ~14 GB for k <= 3).
    os.makedirs(out_path, exist_ok=True)
    counts = {}
    for k, lst in zip(range(2, k_max + 1), per_k):
        ptr = np.concatenate([[0], np.cumsum([len(h) for h in lst])])
        np.save(f"{out_path}/hop{k}_ptr.npy", ptr)
        np.save(f"{out_path}/hop{k}.npy", np.concatenate(lst))
        counts[f"hop{k}_pairs"] = int(ptr[-1])
    return counts


class HopStore:
    def __init__(self, path):
        ks = sorted(int(f[3:-4]) for f in os.listdir(path) if f.startswith("hop") and "ptr" not in f)
        self.k_max = max(ks)
        self.ptr = {k: np.load(f"{path}/hop{k}_ptr.npy") for k in ks}
        self.pairs = {k: np.load(f"{path}/hop{k}.npy", mmap_mode="r") for k in ks}

    def get(self, g, k):
        return self.pairs[k][self.ptr[k][g]:self.ptr[k][g + 1]]


class Batch:
    """Minimal PyG-compatible batch: x (positions), edge_index, y, batch, optional hop_index."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to(self, device, non_blocking=True):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device, non_blocking=non_blocking))
            elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                setattr(self, k, [t.to(device, non_blocking=non_blocking) for t in v])
        return self


def perturb_edges(edges: np.ndarray, n: int, rng, drop_long=0.0, add_long=0.0, min_sep=3,
                  chain_only=False) -> np.ndarray:
    """Perturb one graph's local edge list [E, 2], keeping it symmetric.

    Long-range edges are those with sequence separation >= min_sep.
    drop_long: fraction of long-range contacts removed.
    add_long:  random long-range contacts added, as a fraction of the existing ones.
    chain_only: keep only edges between consecutive residues.
    """
    u, v = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    keep = u < v
    u, v = u[keep], v[keep]
    sep = v - u
    if chain_only:
        sel = sep == 1
    else:
        long = sep >= min_sep
        sel = ~long | (rng.random(len(u)) >= drop_long)
        n_add = int(round(add_long * long.sum()))
        if n_add and n > min_sep + 1:
            a = rng.integers(0, n - min_sep, n_add)
            b = a + rng.integers(min_sep, np.maximum(n - a, min_sep + 1))
            ok = b < n
            u = np.concatenate([u, a[ok]])
            v = np.concatenate([v, b[ok]])
            sel = np.concatenate([sel, np.ones(ok.sum(), dtype=bool)])
    u, v = u[sel], v[sel]
    pairs = np.unique(np.stack([u, v], 1), axis=0)
    return np.concatenate([pairs, pairs[:, ::-1]]).astype(np.int16)


class Collate:
    def __init__(self, store: GraphStore, hops: int = 1, perturb: dict | None = None, seed: int = 0,
                 hop_store: HopStore | None = None):
        self.store = store
        self.hops = hops
        self.perturb = perturb
        self.seed = seed
        # Precomputed hops only describe the unperturbed graphs.
        self.hop_store = hop_store if (hop_store is not None and not perturb
                                       and hop_store.k_max >= hops) else None

    def __call__(self, graph_ids):
        s = self.store
        pos, ys, eis, batch = [], [], [], []
        offset = 0
        for b, g in enumerate(graph_ids):
            n0, n1 = s.node_ptr[g], s.node_ptr[g + 1]
            e0, e1 = s.edge_ptr[g], s.edge_ptr[g + 1]
            n = int(n1 - n0)
            edges = s.edges[e0:e1]
            if self.perturb:
                rng = np.random.default_rng((self.seed, int(g)))
                edges = perturb_edges(edges, n, rng, **self.perturb)
            pos.append(np.arange(1, n + 1))
            ys.append(s.y[n0:n1])
            eis.append(edges.T.astype(np.int64) + offset)
            batch.append(np.full(n, b))
            offset += n
        ei = np.concatenate(eis, axis=1)
        offsets = np.concatenate([[0], np.cumsum([s.num_nodes(g) for g in graph_ids])])
        out = Batch(
            x=torch.from_numpy(np.concatenate(pos)),
            y=torch.from_numpy(np.concatenate(ys).astype(np.int64)),
            edge_index=torch.from_numpy(ei),
            batch=torch.from_numpy(np.concatenate(batch)),
            num_nodes=offset,
            num_graphs=len(graph_ids),
            graph_ids=list(graph_ids),
        )
        if self.hops > 1 and self.hop_store is not None:
            out.hop_index = [
                torch.from_numpy(np.concatenate(
                    [self.hop_store.get(g, k).T.astype(np.int64) + o for g, o in zip(graph_ids, offsets)], axis=1))
                for k in range(2, self.hops + 1)
            ]
        elif self.hops > 1:
            out.hop_index = [torch.from_numpy(h) for h in exact_hop_indices(ei, offset, self.hops)]
        return out


def make_loader(store, graph_ids, batch_size, shuffle, hops=1, workers=8, seed=0, drop_last=False,
                perturb=None, hop_store=None):
    ids = np.asarray(graph_ids)

    class _Sampler(torch.utils.data.Sampler):
        def __init__(self):
            self.epoch = 0

        def __iter__(self):
            order = np.random.default_rng(seed + self.epoch).permutation(ids) if shuffle else ids
            self.epoch += 1
            stop = len(order) - (len(order) % batch_size if drop_last else 0)
            for i in range(0, stop, batch_size):
                yield order[i:i + batch_size].tolist()

        def __len__(self):
            return len(ids) // batch_size if drop_last else -(-len(ids) // batch_size)

    # Dataset items are the graph ids themselves, so the collate receives ids.
    return torch.utils.data.DataLoader(
        range(len(store)), batch_sampler=_Sampler(), collate_fn=Collate(store, hops, perturb, seed, hop_store),
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
