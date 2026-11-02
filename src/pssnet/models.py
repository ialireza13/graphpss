"""PSS-net and baselines as one configurable node classifier."""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINConv, SAGEConv

from .data import NUM_CLASSES

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")


class PositionalInput(nn.Module):
    """Node input features from the residue index.

    learned:    nn.Embedding over the index (what the original PSS-net used)
    sinusoidal: fixed transformer-style encoding of the index
    none:       one shared learned vector, i.e. no positional information at all
    contacts:   like none here; NodeClassifier adds the sum of embeddings of the node's
                contact separations (needs rel_pos), a translation-invariant input
    """

    def __init__(self, kind: str, dim: int, max_len: int = 2001):
        super().__init__()
        self.kind = kind
        if kind == "learned":
            self.emb = nn.Embedding(max_len, dim)
        elif kind in ("none", "contacts"):
            self.emb = nn.Embedding(1, dim)
        elif kind == "sinusoidal":
            pos = torch.arange(max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
            table = torch.zeros(max_len, dim)
            table[:, 0::2] = torch.sin(pos * div)
            table[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("table", table)
        else:
            raise ValueError(kind)

    def forward(self, pos):
        if self.kind == "learned":
            return self.emb(pos)
        if self.kind in ("none", "contacts"):
            return self.emb(torch.zeros_like(pos))
        return self.table[pos]


def separations(batch, rel_pos):
    """Index (0..2*rel_pos) of each edge's sequence separation j - i, clipped to +-rel_pos."""
    src, dst = batch.edge_index
    return (batch.x[src] - batch.x[dst]).clamp(-rel_pos, rel_pos) + rel_pos


def contact_sum(emb, batch, sep):
    """Per-node sum of emb(separation) over the node's contacts."""
    out = torch.zeros(batch.x.size(0), emb.embedding_dim, device=batch.x.device)
    return out.index_add(0, batch.edge_index[1], emb(sep))


class ExpandedGCNConv(nn.Module):
    """Expanded GCN layer (Hashemi & Makse, arXiv:2301.10960).

    h' = sum_{k=0..K} gamma_k * D_k^-1 A_k (h W) + b, where A_0 = I and A_k connects
    node pairs at shortest-path distance exactly k. gamma is learned per layer.
    """

    def __init__(self, in_dim, out_dim, hops):
        super().__init__()
        self.hops = hops
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.gamma = nn.Parameter(torch.empty(hops + 1).uniform_(0, 1))
        stdv = 1.0 / math.sqrt(out_dim)
        nn.init.uniform_(self.lin.weight, -stdv, stdv)

    @staticmethod
    def _mean_aggregate(h, index, n):
        # D^-1 A h as a sparse-dense product; avoids materialising an [E, d] gather.
        row, col = index
        deg = torch.bincount(row, minlength=n).clamp(min=1)
        a = torch.sparse_coo_tensor(index, (1.0 / deg.float())[row], (n, n)).coalesce().to_sparse_csr()
        with torch.autocast("cuda", enabled=False):
            return torch.sparse.mm(a, h.float()).to(h.dtype)

    def forward(self, x, edge_index, hop_index):
        h = self.lin(x)
        n = h.size(0)
        out = self.gamma[0] * h + self.gamma[1] * self._mean_aggregate(h, edge_index, n)
        for k in range(2, self.hops + 1):
            out = out + self.gamma[k] * self._mean_aggregate(h, hop_index[k - 2], n)
        return out + self.bias


class GNNBlock(nn.Module):
    def __init__(self, conv: str, in_dim: int, hidden: int, heads: int, hops: int, edge_dim=None):
        super().__init__()
        self.conv_type = conv
        if edge_dim and conv not in ("gat", "gatv2"):
            raise ValueError(f"edge features are only supported for gat/gatv2, not {conv}")
        if conv == "gatv2":
            self.conv, out = GATv2Conv(in_dim, hidden, heads=heads, concat=True, edge_dim=edge_dim), hidden * heads
        elif conv == "gat":
            self.conv, out = GATConv(in_dim, hidden, heads=heads, concat=True, edge_dim=edge_dim), hidden * heads
        elif conv == "gcn":
            self.conv, out = GCNConv(in_dim, hidden), hidden
        elif conv == "sage":
            self.conv, out = SAGEConv(in_dim, hidden), hidden
        elif conv == "gin":
            mlp = nn.Sequential(nn.Linear(in_dim, 2 * in_dim), nn.ReLU(),
                                nn.Linear(2 * in_dim, 2 * in_dim), nn.ReLU(),
                                nn.Linear(2 * in_dim, hidden))
            self.conv, out = GINConv(mlp), hidden
        elif conv == "expanded_gcn":
            self.conv, out = ExpandedGCNConv(in_dim, hidden, hops), hidden
        else:
            raise ValueError(conv)
        self.norm = nn.LayerNorm(out)
        self.out_dim = out

    def forward(self, x, batch, return_attention=False, edge_attr=None):
        att = None
        if self.conv_type == "expanded_gcn":
            x = self.conv(x, batch.edge_index, getattr(batch, "hop_index", []))
        elif self.conv_type in ("gat", "gatv2"):
            if return_attention:
                x, att = self.conv(x, batch.edge_index, edge_attr=edge_attr, return_attention_weights=True)
            else:
                x = self.conv(x, batch.edge_index, edge_attr=edge_attr)
        else:
            x = self.conv(x, batch.edge_index)
        return F.relu(self.norm(x)), att


class NodeClassifier(nn.Module):
    def __init__(self, conv="gatv2", pos_enc="learned", in_dim=32, hidden=512, layers=2,
                 heads=4, fc_layers=1, hops=1, dropout=0.0, max_len=2001, rel_pos=0, edge_dim=16, **_):
        super().__init__()
        self.input = PositionalInput(pos_enc, in_dim, max_len)
        # rel_pos > 0: GAT/GATv2 edges carry an embedding of the sequence separation j - i,
        # clipped to [-rel_pos, rel_pos]. Translation-invariant, unlike the index embedding.
        self.rel_pos = rel_pos
        edge_dim = edge_dim if rel_pos and conv in ("gat", "gatv2") else None
        if edge_dim:
            self.rel_emb = nn.Embedding(2 * rel_pos + 1, edge_dim)
        # GATv2 uses edge features only in the attention logits, so with identical node
        # inputs every node stays identical; "contacts" breaks that symmetry and works for
        # every conv type.
        if pos_enc == "contacts":
            if not rel_pos:
                raise ValueError("pos_enc=contacts needs rel_pos > 0")
            self.contact_emb = nn.Embedding(2 * rel_pos + 1, in_dim)
        self.blocks = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            block = GNNBlock(conv, d, hidden, heads, hops, edge_dim)
            self.blocks.append(block)
            d = block.out_dim
        head = [nn.Linear(d, d) for _ in range(fc_layers - 1)] + [nn.Linear(d, hidden)]
        self.fc = nn.ModuleList(head)
        self.out = nn.Linear(hidden, NUM_CLASSES)
        self.dropout = dropout

    def forward(self, batch, return_attention=False, return_embed=False):
        x = self.input(batch.x)
        edge_attr = None
        if self.rel_pos:
            sep = separations(batch, self.rel_pos)
            if hasattr(self, "rel_emb"):
                edge_attr = self.rel_emb(sep)
            if hasattr(self, "contact_emb"):
                x = x + contact_sum(self.contact_emb, batch, sep)
        atts = []
        for block in self.blocks:
            x, att = block(x, batch, return_attention, edge_attr)
            x = F.dropout(x, self.dropout, self.training)
            atts.append(att)
        embed = {"gnn": x}
        for lin in self.fc:
            x = F.dropout(F.relu(lin(x)), self.dropout, self.training)
        embed["fc"] = x
        logits = self.out(x)
        if not (return_attention or return_embed):
            return logits
        extra = {}
        if return_attention:
            extra["attention"] = atts
        if return_embed:
            extra["embed"] = embed
        return logits, extra


class PositionMLP(nn.Module):
    """Baseline without message passing: the same input encoding and an MLP per residue.

    With pos_enc=contacts the input is the residue's own contact-separation profile, so
    this measures what the GNN adds beyond each residue's own contacts.
    """

    def __init__(self, pos_enc="learned", in_dim=32, hidden=512, fc_layers=2, max_len=2001,
                 rel_pos=0, **_):
        super().__init__()
        self.input = PositionalInput(pos_enc, in_dim, max_len)
        self.rel_pos = rel_pos if pos_enc == "contacts" else 0
        if self.rel_pos:
            self.contact_emb = nn.Embedding(2 * rel_pos + 1, in_dim)
        dims = [in_dim] + [hidden] * fc_layers
        self.fc = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, batch, **_):
        x = self.input(batch.x)
        if self.rel_pos:
            x = x + contact_sum(self.contact_emb, batch, separations(batch, self.rel_pos))
        for lin in self.fc:
            x = F.relu(lin(x))
        return self.out(x)


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m.get("conv") == "mlp":
        return PositionMLP(**m)
    return NodeClassifier(**m)
