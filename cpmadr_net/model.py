"""Self-contained CPMADR-Net encoder and path-readout model.

This module is the single source of truth for the released model architecture.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import inspect
from typing import Dict, List, Literal, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear

from baselines.common.data import DOSAGE_EDGE_TYPE, TARGET_ETYPE


class BimodalFiLM(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        h = int(hidden_channels)
        self.mlp = torch.nn.Sequential(
            Linear(h * 2, h),
            torch.nn.ReLU(),
            Linear(h, h * 2),
        )

    def forward(self, h_bert: torch.Tensor, h_struct: torch.Tensor, h_fused: torch.Tensor) -> torch.Tensor:
        gb = self.mlp(torch.cat([h_bert, h_struct], dim=-1))
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.sigmoid(gamma) * 2.0
        return gamma * h_fused + beta


class HSPLiteFFN(torch.nn.Module):
    def __init__(self, hidden_channels: int, ffn_mult: float = 2.0, dropout: float = 0.2):
        super().__init__()
        mid = max(8, int(round(hidden_channels * float(ffn_mult))))
        self.net = torch.nn.Sequential(
            Linear(hidden_channels, mid),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
            Linear(mid, hidden_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ESAGateEdgeScale13(torch.nn.Module):
    def __init__(self, in_dim: int, mlp_hidden: int = 16):
        super().__init__()
        self.shortcut = Linear(in_dim, 1)
        self.mlp = torch.nn.Sequential(
            Linear(in_dim, mlp_hidden),
            torch.nn.ReLU(),
            Linear(mlp_hidden, 1),
        )
        self.mix_logit = torch.nn.Parameter(torch.zeros(1))
        self.scale_strength = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.mix_logit)
        beta = g * self.mlp(edge_feat) + (1.0 - g) * self.shortcut(edge_feat)
        s = torch.tanh(self.scale_strength)
        return 1.0 + s * torch.tanh(beta)


class DoseAwareRGCNLayerESA13(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_relations: int, rel_emb_dim: int = 8, esa_mlp_hidden: int = 16):
        super().__init__()
        self.rel_lins = torch.nn.ModuleList(
            [Linear(hidden_channels, hidden_channels, bias=False) for _ in range(num_relations)]
        )
        self.self_lin = Linear(hidden_channels, hidden_channels)
        self.residual_alpha = torch.nn.Parameter(torch.tensor(0.5))
        edge_in = 2 + rel_emb_dim
        self.gate_mlp = torch.nn.Sequential(
            Linear(edge_in, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, 1),
        )
        self.esa = ESAGateEdgeScale13(edge_in, mlp_hidden=esa_mlp_hidden)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
        dosage_mask: torch.Tensor,
        rel_feat: torch.Tensor,
        edge_keep_prob: float = 1.0,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        num_edges = edge_index.size(1)
        hidden = x.size(1)

        msg = x.new_zeros((num_edges, hidden))
        for rel_id, rel_lin in enumerate(self.rel_lins):
            rel_mask = edge_type == rel_id
            if rel_mask.any():
                msg[rel_mask] = rel_lin(x[src[rel_mask]])

        gate_input = torch.cat(
            [torch.stack([edge_weight, dosage_mask.float()], dim=-1), rel_feat],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        if self.training and edge_keep_prob < 1.0:
            keep_mask = (torch.rand(num_edges, device=x.device) < edge_keep_prob).float().view(-1, 1)
            gate = gate * keep_mask
        msg = msg * gate
        msg = msg * self.esa(gate_input)

        out = x.new_zeros((x.size(0), hidden))
        out.index_add_(0, dst, msg)
        deg = x.new_zeros(x.size(0))
        deg.index_add_(0, dst, torch.ones(num_edges, device=x.device))
        out = out / deg.clamp_min(1.0).unsqueeze(-1)
        alpha = torch.sigmoid(self.residual_alpha)
        out = alpha * out + (1.0 - alpha) * self.self_lin(x)
        return out


class ResMLPBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, dropout: float):
        super().__init__()
        self.fc1 = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels)
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return self.ln(x + h)


class FeatureInteractionResNetHead(torch.nn.Module):
    def __init__(self, hidden_channels: int, dropout: float = 0.25, num_res_blocks: int = 1):
        super().__init__()
        in_dim = hidden_channels * 4
        self.in_proj = Linear(in_dim, hidden_channels)
        self.blocks = torch.nn.ModuleList(
            [ResMLPBlock(hidden_channels, dropout=dropout) for _ in range(max(1, int(num_res_blocks)))]
        )
        self.dropout = torch.nn.Dropout(float(dropout))
        self.out = Linear(hidden_channels, 1)

    def forward(self, med_h: torch.Tensor, reac_h: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([med_h, reac_h, torch.abs(med_h - reac_h), med_h * reac_h], dim=-1)
        h = F.relu(self.in_proj(feat))
        for block in self.blocks:
            h = block(h)
        h = self.dropout(h)
        return self.out(h).view(-1)


class DoseAwareHSPESARGCNv13(torch.nn.Module):
    def __init__(
        self,
        metadata,
        data_obj,
        init_med_x: torch.Tensor,
        init_reac_x: torch.Tensor,
        init_target_x: torch.Tensor,
        init_herb_x: torch.Tensor,
        hidden_channels: int = 64,
        num_conv_layers: int = 2,
        dropout: float = 0.25,
        rel_emb_dim: int = 8,
        edge_drop: float = 0.1,
        ffn_mult: float = 2.0,
        esa_mlp_hidden: int = 16,
        ffn_dropout: Optional[float] = None,
        use_bimodal_film: bool = True,
        head_dropout: Optional[float] = None,
        head_res_blocks: int = 1,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.ffn_dropout = float(ffn_dropout) if ffn_dropout is not None else float(dropout)
        self.edge_keep_prob = 1.0 - float(edge_drop)
        self.num_conv_layers = int(num_conv_layers)
        self.use_bimodal_film = bool(use_bimodal_film)

        self.rel_emb = torch.nn.Embedding(len(metadata[1]), rel_emb_dim)
        self.dosage_rel_ids: List[int] = []
        for i, etype in enumerate(metadata[1]):
            src, rel, dst = etype
            is_dosage_pair = {src, dst} == {"medicine", "herb"}
            if is_dosage_pair and "contains" in rel:
                self.dosage_rel_ids.append(i)

        self.med_lin = Linear(init_med_x.size(1), hidden_channels)
        self.reac_lin = Linear(init_reac_x.size(1), hidden_channels)
        self.target_lin = Linear(init_target_x.size(1), hidden_channels)
        self.herb_struct_lin = Linear(init_herb_x.size(1), hidden_channels)
        self.herb_bert_lin = Linear(data_obj["herb"].x.size(1), hidden_channels)
        self.reac_bert_lin = Linear(data_obj["reaction"].x.size(1), hidden_channels)
        self.target_bert_lin = Linear(data_obj["target"].x.size(1), hidden_channels)
        self.ingred_lin = Linear(data_obj["ingredient"].x.size(-1), hidden_channels)
        self.herb_fuse = Linear(hidden_channels * 2, hidden_channels)
        self.reac_fuse = Linear(hidden_channels * 2, hidden_channels)
        self.target_fuse = Linear(hidden_channels * 2, hidden_channels)

        if self.use_bimodal_film:
            self.herb_film = BimodalFiLM(hidden_channels)
            self.reac_film = BimodalFiLM(hidden_channels)
            self.target_film = BimodalFiLM(hidden_channels)
            self.med_ln = torch.nn.LayerNorm(hidden_channels)
            self.ingred_ln = torch.nn.LayerNorm(hidden_channels)

        self.register_buffer("raw_med_x", init_med_x)
        self.register_buffer("raw_reac_x", init_reac_x)
        self.register_buffer("raw_target_x", init_target_x)
        self.register_buffer("raw_herb_x", init_herb_x)

        num_rel = len(metadata[1])
        self.layers = torch.nn.ModuleList(
            [
                DoseAwareRGCNLayerESA13(
                    hidden_channels=hidden_channels,
                    num_relations=num_rel,
                    rel_emb_dim=rel_emb_dim,
                    esa_mlp_hidden=esa_mlp_hidden,
                )
                for _ in range(self.num_conv_layers)
            ]
        )
        self.hsps = torch.nn.ModuleList(
            [HSPLiteFFN(hidden_channels, ffn_mult=ffn_mult, dropout=self.ffn_dropout) for _ in range(self.num_conv_layers)]
        )
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(self.num_conv_layers)])
        self.jk_proj = (
            Linear(hidden_channels * self.num_conv_layers, hidden_channels) if self.num_conv_layers > 1 else None
        )
        hd = float(self.dropout if head_dropout is None else head_dropout)
        self.link_head = FeatureInteractionResNetHead(
            hidden_channels=hidden_channels,
            dropout=hd,
            num_res_blocks=head_res_blocks,
        )

    @staticmethod
    def _homogeneous_x_to_dict(x: torch.Tensor, data) -> Dict[str, torch.Tensor]:
        offset = 0
        out: Dict[str, torch.Tensor] = {}
        for ntype in data.node_types:
            n = data[ntype].num_nodes
            out[ntype] = x[offset : offset + n]
            offset += n
        return out

    @staticmethod
    def _to_homogeneous_with_weight(data, x_dict):
        for ntype in x_dict:
            data[ntype].x = x_dict[ntype]
        for _etype, store in data.edge_items():
            num_e = store.edge_index.size(1)
            if _etype == DOSAGE_EDGE_TYPE:
                dose = getattr(store, "edge_weight", None)
                if dose is None:
                    dose = torch.ones(num_e, device=store.edge_index.device)
                store.edge_weight_hom = dose
            else:
                store.edge_weight_hom = torch.ones(num_e, device=store.edge_index.device)
        hom = data.to_homogeneous(edge_attrs=["edge_weight_hom"])
        return hom.x, hom.edge_index, hom.edge_type, hom.edge_weight_hom

    def forward(self, data):
        if self.use_bimodal_film:
            hb = self.herb_bert_lin(data["herb"].x)
            hs = self.herb_struct_lin(self.raw_herb_x[data["herb"].n_id])
            x_herb = self.herb_film(hb, hs, self.herb_fuse(torch.cat([hb, hs], dim=-1)))
            rb = self.reac_bert_lin(data["reaction"].x)
            rs = self.reac_lin(self.raw_reac_x[data["reaction"].n_id])
            x_reac = self.reac_film(rb, rs, self.reac_fuse(torch.cat([rb, rs], dim=-1)))
            tb = self.target_bert_lin(data["target"].x)
            ts = self.target_lin(self.raw_target_x[data["target"].n_id])
            x_target = self.target_film(tb, ts, self.target_fuse(torch.cat([tb, ts], dim=-1)))
            x_dict = {
                "medicine": self.med_ln(self.med_lin(self.raw_med_x[data["medicine"].n_id])),
                "ingredient": self.ingred_ln(self.ingred_lin(data["ingredient"].x)),
                "herb": x_herb,
                "reaction": x_reac,
                "target": x_target,
            }
        else:
            x_dict = {
                "medicine": self.med_lin(self.raw_med_x[data["medicine"].n_id]),
                "ingredient": self.ingred_lin(data["ingredient"].x),
            }
            herb_struct = self.herb_struct_lin(self.raw_herb_x[data["herb"].n_id])
            x_dict["herb"] = self.herb_fuse(torch.cat([self.herb_bert_lin(data["herb"].x), herb_struct], dim=-1))
            reac_struct = self.reac_lin(self.raw_reac_x[data["reaction"].n_id])
            x_dict["reaction"] = self.reac_fuse(torch.cat([self.reac_bert_lin(data["reaction"].x), reac_struct], dim=-1))
            target_struct = self.target_lin(self.raw_target_x[data["target"].n_id])
            x_dict["target"] = self.target_fuse(torch.cat([self.target_bert_lin(data["target"].x), target_struct], dim=-1))

        x, edge_index, edge_type, edge_weight = self._to_homogeneous_with_weight(data, x_dict)
        dosage_mask = torch.zeros_like(edge_type, dtype=torch.bool)
        for rel_id in self.dosage_rel_ids:
            dosage_mask |= edge_type == rel_id
        rel_feat = self.rel_emb(edge_type)

        layer_outs = []
        for i, layer in enumerate(self.layers):
            residual = x
            h = layer(
                x,
                edge_index,
                edge_type,
                edge_weight,
                dosage_mask,
                rel_feat=rel_feat,
                edge_keep_prob=self.edge_keep_prob,
            )
            h = F.elu(h)
            h = self.hsps[i](h)
            h = self.norms[i](h + residual)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h
            layer_outs.append(x)
        if self.jk_proj is not None:
            x = self.jk_proj(torch.cat(layer_outs, dim=-1))

        x_dict = self._homogeneous_x_to_dict(x, data)
        edge_idx = data[TARGET_ETYPE].edge_label_index
        med_h = x_dict["medicine"][edge_idx[0]]
        reac_h = x_dict["reaction"][edge_idx[1]]
        return self.link_head(med_h, reac_h)


PathIngCosMode = Literal["max", "mean_pool"]
SmilesSimMode = Literal["max", "or_pool"]


@dataclass
class BimodalSimilarityScalars:
    """Per-node scalar similarity signals between two H-dimensional vectors."""

    cos_sim: torch.Tensor  # [N, 1]
    bilin_s: torch.Tensor  # [N, 1]

    def as_tensor(self) -> torch.Tensor:
        return torch.cat([self.cos_sim, self.bilin_s], dim=-1)  # [N, 2]


def cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pairwise cosine similarity along the last dimension.

    Args:
        a, b: [..., D]
    Returns:
        [..., 1]
    """
    if normalize:
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
    return (a * b).sum(dim=-1, keepdim=True)


def bilinear_scalar(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Hadamard-weighted inner product: sum_d a_d * w_d * b_d.

    Args:
        a, b: [N, D]
        weights: [D] learnable or fixed; defaults to ones.
    Returns:
        [N, 1]
    """
    if weights is None:
        return (a * b).sum(dim=-1, keepdim=True)
    return (a * weights * b).sum(dim=-1, keepdim=True)


@torch.no_grad()
def bimodal_similarity_scalars(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    *,
    layer_norm: bool = False,
    ln_a: Optional[torch.nn.LayerNorm] = None,
    ln_b: Optional[torch.nn.LayerNorm] = None,
    bilin_weights: Optional[torch.Tensor] = None,
) -> BimodalSimilarityScalars:
    """
    Compute (cosine, bilinear) pair statistics used in v05/v06-style fusion.

    Cosine uses L2-normalized vectors (optionally after LayerNorm).
    Bilinear uses raw (unnormalized) vectors so magnitude differences remain visible.
    """
    ha, hb = h_a, h_b
    if layer_norm:
        if ln_a is None or ln_b is None:
            raise ValueError("layer_norm=True requires ln_a and ln_b")
        ha = ln_a(h_a)
        hb = ln_b(h_b)
    cos = cosine_similarity(ha, hb)
    bilin = bilinear_scalar(h_a, h_b, bilin_weights)
    return BimodalSimilarityScalars(cos_sim=cos, bilin_s=bilin)


@torch.no_grad()
def summarize_similarity(
    cos_sim: torch.Tensor,
    bilin_s: Optional[torch.Tensor] = None,
    name: str = "",
) -> Dict[str, float]:
    """Quick numpy-free stats for logging / sanity checks."""
    c = cos_sim.detach().float().view(-1)
    out = {
        f"{name}cos_mean": float(c.mean().item()),
        f"{name}cos_std": float(c.std(unbiased=False).item()),
        f"{name}cos_p10": float(torch.quantile(c, 0.10).item()),
        f"{name}cos_p50": float(torch.quantile(c, 0.50).item()),
        f"{name}cos_p90": float(torch.quantile(c, 0.90).item()),
    }
    if bilin_s is not None:
        b = bilin_s.detach().float().view(-1)
        out[f"{name}bilin_mean"] = float(b.mean().item())
        out[f"{name}bilin_std"] = float(b.std(unbiased=False).item())
    return out


@torch.no_grad()
def raw_bimodal_similarity_report(
    data: HeteroData,
    med_feat: torch.Tensor,
    reac_feat: torch.Tensor,
    herb_feat: torch.Tensor,
    target_feat: torch.Tensor,
) -> Dict[str, float]:
    """Full offline report with all three bimodal node types."""
    device = med_feat.device
    report: Dict[str, float] = {}
    for tag, bert_x, struct_x in (
        ("herb_", data["herb"].x.to(device), herb_feat),
        ("reac_", data["reaction"].x.to(device), reac_feat),
        ("targ_", data["target"].x.to(device), target_feat),
    ):
        # Align dims for cosine: truncate to min dim if BERT/struct widths differ.
        d = min(bert_x.size(1), struct_x.size(1))
        scalars = bimodal_similarity_scalars(bert_x[:, :d], struct_x[:, :d])
        report.update(summarize_similarity(scalars.cos_sim, scalars.bilin_s, name=tag))
    report.update(summarize_similarity(cosine_similarity(med_feat, med_feat), name="med_self_"))
    return report


def pairwise_link_similarity(med_h: torch.Tensor, reac_h: torch.Tensor) -> torch.Tensor:
    """Cosine similarity on link-pair embeddings [E, H] → [E, 1]."""
    return cosine_similarity(med_h, reac_h)


@torch.no_grad()
def build_reaction_target_pool(target_feat: torch.Tensor, data: HeteroData) -> torch.Tensor:
    """
    Mean-pooled structural/chemical target features per reaction node.

    Uses ``target -> causes -> reaction`` edges and ``target_feat`` (ingredient-aggregated,
    same space as ``med_feat`` from ``prepare_unified_features``).
    Reactions with no linked target receive the global mean target vector.
    """
    if ("target", "causes", "reaction") not in data.edge_types:
        raise KeyError("HeteroData missing edge type ('target', 'causes', 'reaction')")
    t_r = data["target", "causes", "reaction"].edge_index
    num_reac = int(data["reaction"].num_nodes)
    dim = int(target_feat.size(1))
    pool = target_feat.new_zeros((num_reac, dim))
    count = target_feat.new_zeros(num_reac)
    pool.index_add_(0, t_r[1], target_feat[t_r[0]])
    count.index_add_(0, t_r[1], torch.ones(t_r.size(1), device=target_feat.device))
    pool = pool / count.clamp_min(1.0).unsqueeze(-1)
    empty = count == 0
    if empty.any():
        pool[empty] = target_feat.mean(dim=0, keepdim=True)
    return pool


def batch_path_chem_cosine(
    med_chem_feat: torch.Tensor,
    reac_target_pool: torch.Tensor,
    med_global_ids: torch.Tensor,
    reac_global_ids: torch.Tensor,
) -> torch.Tensor:
    """
  Path chemical alignment for medicine–reaction edges.

  ``cos( med_chem_feat[med], reac_target_pool[reac] )`` where:
    - med_chem_feat: dosage-weighted ingredient (ChemBERTa) aggregate per medicine
    - reac_target_pool: mean target_feat over targets that cause this reaction

  Args:
      med_chem_feat: [N_med, D]
      reac_target_pool: [N_reac, D]
      med_global_ids, reac_global_ids: [E] global node indices (from batch ``n_id``)
  Returns:
      [E, 1]
    """
    a = med_chem_feat[med_global_ids]
    b = reac_target_pool[reac_global_ids]
    return cosine_similarity(a, b)


@dataclass
class IngredientPathIndex:
    """
    Ingredient sets on the TCM graph (ChemBERTa / SMILES-derived embeddings).

    - Medicine: ingredients under ``medicine -> herb -> ingredient``
    - Reaction:  ingredients under ``reaction <- target <- ingredient`` (mechanism path)
    """

    med_ingredient_ids: List[torch.Tensor]
    reac_ingredient_ids: List[torch.Tensor]
    ingredient_emb: torch.Tensor  # [N_ing, D] — typically ``data['ingredient'].x`` (ChemBERTa)
    med_mean_emb: Optional[torch.Tensor] = None  # [N_med, D] for fast mean_pool
    reac_mean_emb: Optional[torch.Tensor] = None  # [N_reac, D]
    max_ings_per_side: int = 0  # cap for max mode (random subsample)


@dataclass
class IngredientSmilesIndex:
    """
    Ingredient-level fingerprint path index for SMILES similarity.

    - Uses the same medicine/reaction ingredient paths as IngredientPathIndex.
    - Fingerprints are expected as binary vectors (0/1), shape [N_ing, F].
    """

    med_ingredient_ids: List[torch.Tensor]
    reac_ingredient_ids: List[torch.Tensor]
    ingredient_fp: torch.Tensor  # [N_ing, F], float/bool binary
    med_or_fp: Optional[torch.Tensor] = None  # [N_med, F]
    reac_or_fp: Optional[torch.Tensor] = None  # [N_reac, F]
    max_ings_per_side: int = 0


def _cap_ingredient_ids(ids: torch.Tensor, max_n: int) -> torch.Tensor:
    if max_n <= 0 or ids.numel() <= max_n:
        return ids
    perm = torch.randperm(ids.numel(), device=ids.device)[:max_n]
    return ids[perm]


def _to_binary_fp(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Convert dense vectors/fingerprints to 0/1 float tensor."""
    return (x > threshold).to(dtype=torch.float)


@torch.no_grad()
def _mean_emb_for_sets(emb: torch.Tensor, id_lists: List[torch.Tensor]) -> torch.Tensor:
    dim = emb.size(1)
    out = emb.new_zeros((len(id_lists), dim))
    global_mean = emb.mean(dim=0)
    for i, ids in enumerate(id_lists):
        if ids.numel() == 0:
            out[i] = global_mean
        else:
            out[i] = emb[ids.to(emb.device)].mean(dim=0)
    return out


@torch.no_grad()
def build_ingredient_path_index(
    data: HeteroData,
    ingredient_emb: torch.Tensor,
    *,
    max_ings_per_side: int = 0,
    precompute_mean_pools: bool = True,
) -> IngredientPathIndex:
    """Build per-medicine / per-reaction ingredient ID lists (global ingredient indices)."""
    device = ingredient_emb.device
    num_med = int(data["medicine"].num_nodes)
    num_reac = int(data["reaction"].num_nodes)

    med_sets: List[set] = [set() for _ in range(num_med)]
    reac_sets: List[set] = [set() for _ in range(num_reac)]

    herb_to_ing: Dict[int, set] = defaultdict(set)
    if ("herb", "contains", "ingredient") in data.edge_types:
        h_i = data["herb", "contains", "ingredient"].edge_index
        for h, ing in zip(h_i[0].tolist(), h_i[1].tolist()):
            if int(ing) >= 0:
                herb_to_ing[int(h)].add(int(ing))

    if ("medicine", "contains", "herb") in data.edge_types:
        m_h = data["medicine", "contains", "herb"].edge_index
        for m, h in zip(m_h[0].tolist(), m_h[1].tolist()):
            med_sets[int(m)].update(herb_to_ing.get(int(h), ()))

    target_to_ing: Dict[int, set] = defaultdict(set)
    if ("ingredient", "interacts", "target") in data.edge_types:
        i_t = data["ingredient", "interacts", "target"].edge_index
        for ing, t in zip(i_t[0].tolist(), i_t[1].tolist()):
            if int(ing) >= 0:
                target_to_ing[int(t)].add(int(ing))

    if ("target", "causes", "reaction") in data.edge_types:
        t_r = data["target", "causes", "reaction"].edge_index
        for t, r in zip(t_r[0].tolist(), t_r[1].tolist()):
            reac_sets[int(r)].update(target_to_ing.get(int(t), ()))

    def _to_tensors(sets: List[set]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for s in sets:
            if s:
                out.append(torch.tensor(sorted(s), dtype=torch.long, device=device))
            else:
                out.append(torch.zeros(0, dtype=torch.long, device=device))
        return out

    med_tensors = _to_tensors(med_sets)
    reac_tensors = _to_tensors(reac_sets)
    emb_f = ingredient_emb.detach().float()
    med_mean = reac_mean = None
    if precompute_mean_pools:
        med_mean = _mean_emb_for_sets(emb_f, med_tensors)
        reac_mean = _mean_emb_for_sets(emb_f, reac_tensors)
    return IngredientPathIndex(
        med_ingredient_ids=med_tensors,
        reac_ingredient_ids=reac_tensors,
        ingredient_emb=emb_f,
        med_mean_emb=med_mean,
        reac_mean_emb=reac_mean,
        max_ings_per_side=int(max_ings_per_side),
    )


def _max_cos_between_sets(emb: torch.Tensor, ids_a: torch.Tensor, ids_b: torch.Tensor) -> torch.Tensor:
    """Max cosine between two ingredient sets in ChemBERTa space → scalar [1]."""
    if ids_a.numel() == 0 or ids_b.numel() == 0:
        return emb.new_zeros(1)
    a = F.normalize(emb[ids_a], dim=-1, eps=1e-6)
    b = F.normalize(emb[ids_b], dim=-1, eps=1e-6)
    return (a @ b.T).max().view(1)


def _tanimoto_pair(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Tanimoto similarity for binary-like vectors a,b with shape [N, F].
    Returns [N, 1].
    """
    inter = (a * b).sum(dim=-1, keepdim=True)
    denom = a.sum(dim=-1, keepdim=True) + b.sum(dim=-1, keepdim=True) - inter
    return inter / denom.clamp_min(eps)


def _max_tanimoto_between_sets(fp: torch.Tensor, ids_a: torch.Tensor, ids_b: torch.Tensor) -> torch.Tensor:
    """Max pairwise Tanimoto between ingredient sets -> scalar [1]."""
    if ids_a.numel() == 0 or ids_b.numel() == 0:
        return fp.new_zeros(1)
    a = fp[ids_a]
    b = fp[ids_b]
    inter = a @ b.T
    a_sum = a.sum(dim=-1, keepdim=True)
    b_sum = b.sum(dim=-1, keepdim=True).T
    denom = a_sum + b_sum - inter
    sim = inter / denom.clamp_min(1e-8)
    return sim.max().view(1)


@torch.no_grad()
def _or_pool_fp_for_sets(fp: torch.Tensor, id_lists: List[torch.Tensor]) -> torch.Tensor:
    """
    Bitwise-OR pool per set using float 0/1 tensors.
    Empty sets remain all-zero.
    """
    out = fp.new_zeros((len(id_lists), fp.size(1)))
    for i, ids in enumerate(id_lists):
        if ids.numel() == 0:
            continue
        out[i] = fp[ids.to(fp.device)].amax(dim=0)
    return out


def _mean_pool_cos_between_sets(emb: torch.Tensor, ids_a: torch.Tensor, ids_b: torch.Tensor) -> torch.Tensor:
    """cos( mean(emb[A]), mean(emb[B]) ) → scalar [1]."""
    if ids_a.numel() == 0 or ids_b.numel() == 0:
        return emb.new_zeros(1)
    return cosine_similarity(emb[ids_a].mean(dim=0, keepdim=True), emb[ids_b].mean(dim=0, keepdim=True)).view(1)


@torch.no_grad()
def batch_path_ingredient_chemberta_cos(
    path_index: IngredientPathIndex,
    med_global_ids: torch.Tensor,
    reac_global_ids: torch.Tensor,
    *,
    mode: PathIngCosMode = "mean_pool",
) -> torch.Tensor:
    """
    ChemBERTa-based path similarity (fixed embeddings; link-head feature only).

    - ``mean_pool`` (default): O(1) per edge via precomputed mean vectors — **fast**
    - ``max``: pairwise max cos; slow if sets are huge — set ``max_ings_per_side`` when building index
    """
    if mode == "mean_pool" and path_index.med_mean_emb is not None and path_index.reac_mean_emb is not None:
        dev = med_global_ids.device
        a = path_index.med_mean_emb.to(dev)[med_global_ids]
        b = path_index.reac_mean_emb.to(dev)[reac_global_ids]
        return cosine_similarity(a, b).to(dtype=path_index.ingredient_emb.dtype)

    emb = path_index.ingredient_emb
    device = emb.device
    cap = int(path_index.max_ings_per_side)
    scores = []
    for mi, ri in zip(med_global_ids.tolist(), reac_global_ids.tolist()):
        a_ids = _cap_ingredient_ids(path_index.med_ingredient_ids[int(mi)].to(device), cap)
        b_ids = _cap_ingredient_ids(path_index.reac_ingredient_ids[int(ri)].to(device), cap)
        if mode == "max":
            scores.append(_max_cos_between_sets(emb, a_ids, b_ids))
        else:
            scores.append(_mean_pool_cos_between_sets(emb, a_ids, b_ids))
    out = torch.stack(scores, dim=0)
    return out.to(med_global_ids.device, dtype=emb.dtype).view(-1, 1)


@torch.no_grad()
def build_ingredient_smiles_index(
    data: HeteroData,
    ingredient_fp: torch.Tensor,
    *,
    max_ings_per_side: int = 0,
    precompute_or_pools: bool = True,
) -> IngredientSmilesIndex:
    """
    Build ingredient-path index for SMILES-like binary fingerprints.

    `ingredient_fp` can be real binary fingerprints (recommended), or any dense
    feature that will be binarized with threshold 0.0.
    """
    fp = _to_binary_fp(ingredient_fp.detach().float())
    base = build_ingredient_path_index(
        data,
        ingredient_emb=fp,
        max_ings_per_side=max_ings_per_side,
        precompute_mean_pools=False,
    )
    med_or = reac_or = None
    if precompute_or_pools:
        med_or = _or_pool_fp_for_sets(fp, base.med_ingredient_ids)
        reac_or = _or_pool_fp_for_sets(fp, base.reac_ingredient_ids)
    return IngredientSmilesIndex(
        med_ingredient_ids=base.med_ingredient_ids,
        reac_ingredient_ids=base.reac_ingredient_ids,
        ingredient_fp=fp,
        med_or_fp=med_or,
        reac_or_fp=reac_or,
        max_ings_per_side=int(max_ings_per_side),
    )


@torch.no_grad()
def batch_path_ingredient_smiles_tanimoto(
    smiles_index: IngredientSmilesIndex,
    med_global_ids: torch.Tensor,
    reac_global_ids: torch.Tensor,
    *,
    mode: SmilesSimMode = "or_pool",
) -> torch.Tensor:
    """
    SMILES similarity on ingredient paths -> [E, 1].

    - `or_pool` (default): Tanimoto(OR_pool(med_set), OR_pool(reac_set)) fast.
    - `max`: max pairwise Tanimoto over ingredient pairs, slower.
    """
    if mode == "or_pool" and smiles_index.med_or_fp is not None and smiles_index.reac_or_fp is not None:
        dev = med_global_ids.device
        a = smiles_index.med_or_fp.to(dev)[med_global_ids]
        b = smiles_index.reac_or_fp.to(dev)[reac_global_ids]
        return _tanimoto_pair(a, b).to(dtype=smiles_index.ingredient_fp.dtype)

    fp = smiles_index.ingredient_fp
    device = fp.device
    cap = int(smiles_index.max_ings_per_side)
    scores = []
    for mi, ri in zip(med_global_ids.tolist(), reac_global_ids.tolist()):
        a_ids = _cap_ingredient_ids(smiles_index.med_ingredient_ids[int(mi)].to(device), cap)
        b_ids = _cap_ingredient_ids(smiles_index.reac_ingredient_ids[int(ri)].to(device), cap)
        scores.append(_max_tanimoto_between_sets(fp, a_ids, b_ids))
    out = torch.stack(scores, dim=0)
    return out.to(med_global_ids.device, dtype=fp.dtype).view(-1, 1)


PathReadoutMode = Literal["attn", "attn_cross"]

__all__ = [
    "BimodalFiLM",
    "DoseAwareHSPESARGCNv13",
    "BatchedPathSetAttentionPool",
    "PathIngredientJaccard",
    "TrainOnlyDSANeighbor",
    "FeatureInteractionPathReadoutHead",
    "PathReadoutCrossGate",
    "PathReadoutMode",
    "DoseAwareHSPESARGCNv13PathReadout",
]


def _build_path_index_compat(
    data_obj,
    ingredient_emb: torch.Tensor,
    *,
    path_max_ings: int,
    precompute_mean_pools: bool = True,
) -> IngredientPathIndex:
    """Call ``build_ingredient_path_index`` with only kwargs supported by the local module."""
    params = inspect.signature(build_ingredient_path_index).parameters
    kwargs = {}
    if "precompute_mean_pools" in params:
        kwargs["precompute_mean_pools"] = bool(precompute_mean_pools)
    if "max_ings_per_side" in params:
        kwargs["max_ings_per_side"] = int(path_max_ings)
    path_index = build_ingredient_path_index(data_obj, ingredient_emb, **kwargs)
    cap = int(path_max_ings)
    if cap > 0 and getattr(path_index, "max_ings_per_side", 0) != cap:
        try:
            path_index.max_ings_per_side = cap
        except AttributeError:
            object.__setattr__(path_index, "max_ings_per_side", cap)
    return path_index


def _tensor_ids_to_python_sets(id_lists: List[torch.Tensor]) -> List[Set[int]]:
    out: List[Set[int]] = []
    for ids in id_lists:
        if ids.numel() == 0:
            out.append(set())
            continue
        out.append(set(int(x) for x in ids.detach().view(-1).cpu().tolist()))
    return out


def _jaccard_sets(a: Set[int], b: Set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    if inter <= 0:
        return 0.0
    denom = len(a) + len(b) - inter
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


def _build_train_bipartite_neighbors(
    data_obj,
) -> Tuple[List[Set[int]], List[Set[int]]]:
    num_med = int(data_obj["medicine"].num_nodes)
    num_reac = int(data_obj["reaction"].num_nodes)
    med_to_reac: List[Set[int]] = [set() for _ in range(num_med)]
    reac_to_med: List[Set[int]] = [set() for _ in range(num_reac)]
    ei = data_obj[TARGET_ETYPE].edge_index
    ms = ei[0].detach().view(-1).cpu().tolist()
    rs = ei[1].detach().view(-1).cpu().tolist()
    for m, r in zip(ms, rs):
        mi = int(m)
        ri = int(r)
        if 0 <= mi < num_med and 0 <= ri < num_reac:
            med_to_reac[mi].add(ri)
            reac_to_med[ri].add(mi)
    return med_to_reac, reac_to_med


def _build_padded_id_table(
    id_lists: List[torch.Tensor],
    *,
    max_ings: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(id_lists)
    if max_ings > 0:
        cap = int(max_ings)
    else:
        cap = max((int(ids.numel()) for ids in id_lists), default=0)
    cap = max(1, cap)
    ids_tab = torch.zeros((n, cap), dtype=torch.long)
    mask_tab = torch.zeros((n, cap), dtype=torch.bool)
    for i, ids in enumerate(id_lists):
        if ids.numel() == 0:
            continue
        take = min(cap, int(ids.numel()))
        ids_tab[i, :take] = ids[:take].detach().cpu()
        mask_tab[i, :take] = True
    return ids_tab, mask_tab


def _build_pairwise_jaccard_matrix(neighbor_sets: List[Set[int]]) -> torch.Tensor:
    n = len(neighbor_sets)
    out = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        si = neighbor_sets[i]
        for j in range(i, n):
            v = _jaccard_sets(si, neighbor_sets[j])
            out[i, j] = v
            out[j, i] = v
    return out


def _build_path_jaccard_matrix(path_index: IngredientPathIndex) -> torch.Tensor:
    med_sets = _tensor_ids_to_python_sets(path_index.med_ingredient_ids)
    reac_sets = _tensor_ids_to_python_sets(path_index.reac_ingredient_ids)
    out = torch.zeros((len(med_sets), len(reac_sets)), dtype=torch.float32)
    for m, ms in enumerate(med_sets):
        if not ms:
            continue
        for r, rs in enumerate(reac_sets):
            out[m, r] = _jaccard_sets(ms, rs)
    return out


def _build_dsa_neighbor_matrix(
    med_to_reac: List[Set[int]],
    reac_to_med: List[Set[int]],
) -> torch.Tensor:
    num_med = len(med_to_reac)
    num_reac = len(reac_to_med)
    med_sim = _build_pairwise_jaccard_matrix(med_to_reac)
    reac_sim = _build_pairwise_jaccard_matrix(reac_to_med)
    out = torch.zeros((num_med, num_reac), dtype=torch.float32)
    for m in range(num_med):
        linked_reacs = med_to_reac[m]
        for r in range(num_reac):
            linked_meds = reac_to_med[r]

            rr = [x for x in linked_reacs if x != r]
            mm = [x for x in linked_meds if x != m]

            has_rr = len(rr) > 0
            has_mm = len(mm) > 0

            s_reac = float(reac_sim[rr, r].mean().item()) if has_rr else 0.0
            s_med = float(med_sim[mm, m].mean().item()) if has_mm else 0.0

            if has_rr and has_mm:
                out[m, r] = 0.5 * (s_reac + s_med)
            elif has_rr:
                out[m, r] = s_reac
            elif has_mm:
                out[m, r] = s_med
    return out


class PathIngredientJaccard(torch.nn.Module):
    """Path ingredient set Jaccard precomputed as [num_med, num_reac] matrix."""

    def __init__(self, path_index: IngredientPathIndex):
        super().__init__()
        mat = _build_path_jaccard_matrix(path_index)
        self.register_buffer("score_matrix", mat)

    def batch_scores(
        self,
        med_global_ids: torch.Tensor,
        reac_global_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.score_matrix[med_global_ids, reac_global_ids].unsqueeze(-1).to(dtype=dtype)


class TrainOnlyDSANeighbor(torch.nn.Module):
    """
    Train-only DSA neighbor feature on medicine-reaction bipartite graph.

    Uses only ``data_obj[TARGET_ETYPE].edge_index`` at model init time (per-fold train graph).
    For pair (m, r):
      1) Average similarity between reaction r and reactions already linked to m.
      2) Average similarity between medicine m and medicines already linked to r.
    Similarity is Jaccard over opposite-side neighbor sets.
    """

    def __init__(self, data_obj):
        super().__init__()
        med_to_reac, reac_to_med = _build_train_bipartite_neighbors(data_obj)
        mat = _build_dsa_neighbor_matrix(med_to_reac, reac_to_med)
        self.register_buffer("score_matrix", mat)

    def batch_scores(
        self,
        med_global_ids: torch.Tensor,
        reac_global_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.score_matrix[med_global_ids, reac_global_ids].unsqueeze(-1).to(dtype=dtype)


class BatchedPathSetAttentionPool(torch.nn.Module):
    """Query-conditioned attention pool over variable-size ingredient sets per link."""

    def __init__(self, chem_dim: int, hidden_channels: int):
        super().__init__()
        self.ing_proj = Linear(int(chem_dim), int(hidden_channels))
        self.score = Linear(int(hidden_channels), 1, bias=False)

    def pool_edges(
        self,
        ingredient_emb: torch.Tensor,
        ids_table: torch.Tensor,
        mask_table: torch.Tensor,
        node_global_ids: torch.Tensor,
        query: torch.Tensor,
        *,
        return_weights: bool = False,
    ):
        """Attention-pool ingredient sets.

        When ``return_weights=False`` (default), returns only the pooled vector —
        identical to the original training / eval path.

        When ``return_weights=True``, also returns ``(weights, ingredient_ids, mask)``
        for case-level interpretability export. Softmax is over the full padded set;
        callers should filter with ``mask`` when reporting Top-K.
        """
        ids = ids_table[node_global_ids]
        mask = mask_table[node_global_ids]
        ing = ingredient_emb[ids]
        x = torch.tanh(self.ing_proj(ing) + query.unsqueeze(1))
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        w = F.softmax(logits, dim=1) * mask.to(dtype=logits.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (w.unsqueeze(-1) * x).sum(dim=1)
        if return_weights:
            return pooled, w, ids, mask
        return pooled


class PathReadoutCrossGate(torch.nn.Module):
    """Light cross-gate between path pools before the link head."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        h = int(hidden_channels)
        self.gate = torch.nn.Sequential(Linear(h * 2, h), torch.nn.ReLU(), Linear(h, 1))

    def forward(self, path_med: torch.Tensor, path_reac: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.sigmoid(self.gate(torch.cat([path_med, path_reac], dim=-1)))
        return path_med + g * path_reac, path_reac + g * path_med


class FeatureInteractionPathReadoutHead(torch.nn.Module):
    """v13-style head on GNN pair + path attention pools (4H + 4H -> H)."""

    def __init__(
        self,
        hidden_channels: int,
        extra_dim: int = 0,
        dropout: float = 0.25,
        num_res_blocks: int = 1,
    ):
        super().__init__()
        h = int(hidden_channels)
        self.extra_dim = int(extra_dim)
        self.in_proj = Linear(h * 8 + self.extra_dim, h)
        self.blocks = torch.nn.ModuleList(
            [ResMLPBlock(h, dropout=float(dropout)) for _ in range(max(1, int(num_res_blocks)))]
        )
        self.dropout = torch.nn.Dropout(float(dropout))
        self.out = Linear(h, 1)

    def forward(
        self,
        med_h: torch.Tensor,
        reac_h: torch.Tensor,
        path_med: torch.Tensor,
        path_reac: torch.Tensor,
        extra_scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = [
            med_h,
            reac_h,
            torch.abs(med_h - reac_h),
            med_h * reac_h,
            path_med,
            path_reac,
            torch.abs(path_med - path_reac),
            path_med * path_reac,
        ]
        if self.extra_dim > 0:
            if extra_scalars is None:
                raise ValueError("extra_scalars required when extra_dim > 0")
            if extra_scalars.size(-1) != self.extra_dim:
                raise ValueError(f"expected extra_dim={self.extra_dim}, got {extra_scalars.size(-1)}")
            parts.append(extra_scalars)
        feat = torch.cat(
            [
                *parts,
            ],
            dim=-1,
        )
        h = F.relu(self.in_proj(feat))
        for block in self.blocks:
            h = block(h)
        h = self.dropout(h)
        return self.out(h).view(-1)


class DoseAwareHSPESARGCNv13PathReadout(DoseAwareHSPESARGCNv13):
    """
    v13 message passing + path ingredient attention readout at link prediction.

    Args:
        path_readout_mode: ``attn`` or ``attn_cross`` (adds cross-gate on path pools).
        path_max_ings: cap ingredients per side per edge (speed / memory).
    """

    def __init__(
        self,
        metadata,
        data_obj,
        init_med_x: torch.Tensor,
        init_reac_x: torch.Tensor,
        init_target_x: torch.Tensor,
        init_herb_x: torch.Tensor,
        hidden_channels: int = 64,
        num_conv_layers: int = 2,
        dropout: float = 0.25,
        rel_emb_dim: int = 8,
        edge_drop: float = 0.1,
        ffn_mult: float = 2.0,
        esa_mlp_hidden: int = 16,
        ffn_dropout: Optional[float] = None,
        use_bimodal_film: bool = True,
        head_dropout: Optional[float] = None,
        head_res_blocks: int = 1,
        path_readout_mode: PathReadoutMode = "attn",
        path_max_ings: int = 32,
        ingredient_emb: Optional[torch.Tensor] = None,
        path_ing_cos_mode: PathIngCosMode = "mean_pool",
    ):
        super().__init__(
            metadata=metadata,
            data_obj=data_obj,
            init_med_x=init_med_x,
            init_reac_x=init_reac_x,
            init_target_x=init_target_x,
            init_herb_x=init_herb_x,
            hidden_channels=hidden_channels,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            rel_emb_dim=rel_emb_dim,
            edge_drop=edge_drop,
            ffn_mult=ffn_mult,
            esa_mlp_hidden=esa_mlp_hidden,
            ffn_dropout=ffn_dropout,
            use_bimodal_film=use_bimodal_film,
            head_dropout=head_dropout,
            head_res_blocks=head_res_blocks,
        )
        h = int(hidden_channels)
        chem_dim = int(
            (ingredient_emb if ingredient_emb is not None else data_obj["ingredient"].x).size(-1)
        )
        ing_x = ingredient_emb if ingredient_emb is not None else data_obj["ingredient"].x
        self.path_max_ings = int(path_max_ings)
        self.path_readout_mode: PathReadoutMode = path_readout_mode  # type: ignore[assignment]
        self.path_index = _build_path_index_compat(
            data_obj,
            ing_x.detach().float(),
            path_max_ings=self.path_max_ings,
            precompute_mean_pools=True,
        )
        self.path_ing_cos_mode: PathIngCosMode = path_ing_cos_mode  # type: ignore[assignment]
        self.path_jaccard = PathIngredientJaccard(self.path_index)
        self.dsa_neighbor = TrainOnlyDSANeighbor(data_obj)
        med_tab, med_mask = _build_padded_id_table(self.path_index.med_ingredient_ids, max_ings=self.path_max_ings)
        reac_tab, reac_mask = _build_padded_id_table(self.path_index.reac_ingredient_ids, max_ings=self.path_max_ings)
        self.register_buffer("path_med_ids_tab", med_tab)
        self.register_buffer("path_med_mask_tab", med_mask)
        self.register_buffer("path_reac_ids_tab", reac_tab)
        self.register_buffer("path_reac_mask_tab", reac_mask)
        self.register_buffer("ingredient_emb_fixed", self.path_index.ingredient_emb.detach().float())
        if self.path_index.med_mean_emb is not None and self.path_index.reac_mean_emb is not None:
            self.register_buffer("path_med_mean_emb", self.path_index.med_mean_emb.detach().float())
            self.register_buffer("path_reac_mean_emb", self.path_index.reac_mean_emb.detach().float())
        else:
            # Keep placeholder buffers to simplify device moves when model.to(device) is called.
            self.register_buffer("path_med_mean_emb", torch.zeros((1, chem_dim), dtype=torch.float32))
            self.register_buffer("path_reac_mean_emb", torch.zeros((1, chem_dim), dtype=torch.float32))
        self.path_pool = BatchedPathSetAttentionPool(chem_dim, h)
        self.path_cross = PathReadoutCrossGate(h) if path_readout_mode == "attn_cross" else None
        hd = float(self.dropout if head_dropout is None else head_dropout)
        self.link_head = FeatureInteractionPathReadoutHead(
            hidden_channels=h,
            extra_dim=4,  # embed_cos + path_ing_cos + path_jaccard + dsa_neighbor
            dropout=hd,
            num_res_blocks=head_res_blocks,
        )

    def _path_ing_cos_fast(
        self,
        med_global_ids: torch.Tensor,
        reac_global_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            self.path_index.med_mean_emb is not None
            and self.path_index.reac_mean_emb is not None
            and self.path_med_mean_emb.size(0) > int(med_global_ids.max().item())
            and self.path_reac_mean_emb.size(0) > int(reac_global_ids.max().item())
        ):
            a = self.path_med_mean_emb[med_global_ids]
            b = self.path_reac_mean_emb[reac_global_ids]
            s = F.cosine_similarity(a, b, dim=-1, eps=1e-6).unsqueeze(-1)
            return s.to(dtype=dtype)
        return batch_path_ingredient_chemberta_cos(
            self.path_index,
            med_global_ids,
            reac_global_ids,
            mode=self.path_ing_cos_mode,
        ).to(dtype=dtype)

    def _gather_link_extras(
        self,
        data,
        edge_idx: torch.Tensor,
        med_h: torch.Tensor,
        reac_h: torch.Tensor,
    ) -> torch.Tensor:
        med_gid = data["medicine"].n_id[edge_idx[0]]
        reac_gid = data["reaction"].n_id[edge_idx[1]]
        dt = med_h.dtype
        extras = [
            pairwise_link_similarity(med_h, reac_h).to(dtype=dt),
            self._path_ing_cos_fast(med_gid, reac_gid, dtype=dt),
            self.path_jaccard.batch_scores(med_gid, reac_gid, dtype=dt),
            self.dsa_neighbor.batch_scores(med_gid, reac_gid, dtype=dt),
        ]
        return torch.cat(extras, dim=-1)

    def _encode_message_pass(self, data) -> dict:
        if self.use_bimodal_film:
            hb = self.herb_bert_lin(data["herb"].x)
            hs = self.herb_struct_lin(self.raw_herb_x[data["herb"].n_id])
            x_herb = self.herb_film(hb, hs, self.herb_fuse(torch.cat([hb, hs], dim=-1)))
            rb = self.reac_bert_lin(data["reaction"].x)
            rs = self.reac_lin(self.raw_reac_x[data["reaction"].n_id])
            x_reac = self.reac_film(rb, rs, self.reac_fuse(torch.cat([rb, rs], dim=-1)))
            tb = self.target_bert_lin(data["target"].x)
            ts = self.target_lin(self.raw_target_x[data["target"].n_id])
            x_target = self.target_film(tb, ts, self.target_fuse(torch.cat([tb, ts], dim=-1)))
            x_dict = {
                "medicine": self.med_ln(self.med_lin(self.raw_med_x[data["medicine"].n_id])),
                "ingredient": self.ingred_ln(self.ingred_lin(data["ingredient"].x)),
                "herb": x_herb,
                "reaction": x_reac,
                "target": x_target,
            }
        else:
            x_dict = {
                "medicine": self.med_lin(self.raw_med_x[data["medicine"].n_id]),
                "ingredient": self.ingred_lin(data["ingredient"].x),
            }
            herb_struct = self.herb_struct_lin(self.raw_herb_x[data["herb"].n_id])
            x_dict["herb"] = self.herb_fuse(torch.cat([self.herb_bert_lin(data["herb"].x), herb_struct], dim=-1))
            reac_struct = self.reac_lin(self.raw_reac_x[data["reaction"].n_id])
            x_dict["reaction"] = self.reac_fuse(torch.cat([self.reac_bert_lin(data["reaction"].x), reac_struct], dim=-1))
            target_struct = self.target_lin(self.raw_target_x[data["target"].n_id])
            x_dict["target"] = self.target_fuse(
                torch.cat([self.target_bert_lin(data["target"].x), target_struct], dim=-1)
            )

        x, edge_index, edge_type, edge_weight = self._to_homogeneous_with_weight(data, x_dict)
        dosage_mask = torch.zeros_like(edge_type, dtype=torch.bool)
        for rel_id in self.dosage_rel_ids:
            dosage_mask |= edge_type == rel_id
        rel_feat = self.rel_emb(edge_type)

        layer_outs = []
        for i, layer in enumerate(self.layers):
            residual = x
            h = layer(
                x,
                edge_index,
                edge_type,
                edge_weight,
                dosage_mask,
                rel_feat=rel_feat,
                edge_keep_prob=self.edge_keep_prob,
            )
            h = F.elu(h)
            h = self.hsps[i](h)
            h = self.norms[i](h + residual)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h
            layer_outs.append(x)
        if self.jk_proj is not None:
            x = self.jk_proj(torch.cat(layer_outs, dim=-1))
        return self._homogeneous_x_to_dict(x, data)

    def forward(self, data):
        x_dict = self._encode_message_pass(data)
        edge_idx = data[TARGET_ETYPE].edge_label_index
        med_h = x_dict["medicine"][edge_idx[0]]
        reac_h = x_dict["reaction"][edge_idx[1]]
        med_gid = data["medicine"].n_id[edge_idx[0]]
        reac_gid = data["reaction"].n_id[edge_idx[1]]
        path_med = self.path_pool.pool_edges(
            self.ingredient_emb_fixed,
            self.path_med_ids_tab,
            self.path_med_mask_tab,
            med_gid,
            med_h,
        )
        path_reac = self.path_pool.pool_edges(
            self.ingredient_emb_fixed,
            self.path_reac_ids_tab,
            self.path_reac_mask_tab,
            reac_gid,
            reac_h,
        )
        if self.path_cross is not None:
            path_med, path_reac = self.path_cross(path_med, path_reac)
        extras = self._gather_link_extras(data, edge_idx, med_h, reac_h)
        return self.link_head(med_h, reac_h, path_med, path_reac, extras)

    @torch.no_grad()
    def explain_links(self, data) -> dict:
        """Forward pass plus path-readout ingredient attention (eval only).

        Attention weights are those used inside ``path_pool`` *before* optional
        cross-gate mixing. Prediction logits use the same pooled vectors as
        ``forward`` (including cross-gate when enabled).
        """
        was_training = self.training
        self.eval()
        x_dict = self._encode_message_pass(data)
        edge_idx = data[TARGET_ETYPE].edge_label_index
        med_h = x_dict["medicine"][edge_idx[0]]
        reac_h = x_dict["reaction"][edge_idx[1]]
        med_gid = data["medicine"].n_id[edge_idx[0]]
        reac_gid = data["reaction"].n_id[edge_idx[1]]
        path_med, w_med, ids_med, mask_med = self.path_pool.pool_edges(
            self.ingredient_emb_fixed,
            self.path_med_ids_tab,
            self.path_med_mask_tab,
            med_gid,
            med_h,
            return_weights=True,
        )
        path_reac, w_reac, ids_reac, mask_reac = self.path_pool.pool_edges(
            self.ingredient_emb_fixed,
            self.path_reac_ids_tab,
            self.path_reac_mask_tab,
            reac_gid,
            reac_h,
            return_weights=True,
        )
        path_med_pred, path_reac_pred = path_med, path_reac
        if self.path_cross is not None:
            path_med_pred, path_reac_pred = self.path_cross(path_med, path_reac)
        extras = self._gather_link_extras(data, edge_idx, med_h, reac_h)
        logits = self.link_head(med_h, reac_h, path_med_pred, path_reac_pred, extras)
        if was_training:
            self.train()
        return {
            "logits": logits,
            "med_global_ids": med_gid,
            "reac_global_ids": reac_gid,
            "edge_label": data[TARGET_ETYPE].edge_label,
            "w_med": w_med,
            "ids_med": ids_med,
            "mask_med": mask_med,
            "w_reac": w_reac,
            "ids_reac": ids_reac,
            "mask_reac": mask_reac,
        }
