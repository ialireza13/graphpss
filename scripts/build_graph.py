"""Build GraphPSS samples from mmCIF files.

Each sample has one node per Cα atom of the first chain in model 1, an edge for Cα pairs
closer than 8 Å, and the residue index 1..N as x, stored as data/pt_cache/<id>.pt for
scripts/build_store.py. The labels are the PDB's own secondary-structure records, not DSSP output. Residues are
the Cα atoms of the first chain in model 1, identified by (label_asym_id, label_seq_id):
helix if inside a `_struct_conf` HELX range, sheet if inside a `_struct_sheet_range`, and
coil otherwise.

    python scripts/build_graph.py data/cif/*.cif --out data/pt_cache
"""
import argparse
import os

import gemmi
import numpy as np
import torch
from torch_geometric.data import Data

COIL, HELIX, SHEET = 0, 1, 2


def observed_residues(block):
    """(label_asym_id, label_seq_id) of the Cα atoms of the first chain, model 1."""
    rows = block.find("_atom_site.", ["label_atom_id", "label_asym_id", "label_seq_id",
                                      "auth_asym_id", "pdbx_PDB_model_num", "type_symbol"])
    residues, seen, first = [], set(), None
    for atom, asym, seq, auth, model, element in rows:
        if atom != "CA" or element != "C" or seq in ".?":  # element check skips calcium ions
            continue
        first = first or (auth, model)
        if (auth, model) == first and (asym, seq) not in seen:
            seen.add((asym, seq))
            residues.append((asym, seq))
    return residues


def expand(rows):
    return {(asym, str(i)) for asym, beg, end in rows for i in range(int(beg), int(end) + 1)}


def labels(block):
    conf = block.find("_struct_conf.", ["conf_type_id", "beg_label_asym_id", "beg_label_seq_id",
                                        "end_label_seq_id"])
    helix = expand((row[1], row[2], row[3]) for row in conf if row[0].startswith("HELX"))
    sheet = expand(block.find("_struct_sheet_range.", ["beg_label_asym_id", "beg_label_seq_id",
                                                       "end_label_seq_id"]))
    return np.array([HELIX if r in helix else SHEET if r in sheet else COIL
                     for r in observed_residues(block)], dtype=np.int8)


def contacts(block, threshold=8.0):
    """Directed edge list [2, E] of Cα pairs closer than threshold, without self-loops."""
    rows = block.find("_atom_site.", ["label_atom_id", "label_asym_id", "label_seq_id", "auth_asym_id",
                                      "pdbx_PDB_model_num", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z"])
    xyz, seen, first = [], set(), None
    for atom, asym, seq, auth, model, element, x, y, z in rows:
        if atom != "CA" or element != "C" or seq in ".?":
            continue
        first = first or (auth, model)
        if (auth, model) == first and (asym, seq) not in seen:
            seen.add((asym, seq))
            xyz.append((float(x), float(y), float(z)))
    xyz = np.array(xyz)
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return np.array(np.nonzero((dist < threshold) & ~np.eye(len(xyz), dtype=bool)))


def build(path):
    block = gemmi.cif.read(path).sole_block()
    y = labels(block)
    return Data(x=torch.arange(1, len(y) + 1), edge_index=torch.from_numpy(contacts(block)),
                y=torch.from_numpy(y).long())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cif", nargs="+")
    ap.add_argument("--out", default="data/pt_cache")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for path in args.cif:
        name = os.path.basename(path).replace(".cif", "")
        torch.save(build(path), f"{args.out}/{name}.pt")
