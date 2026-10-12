"""
Path-readout ablation runner for ``DoseAwareHSPESARGCNv13PathReadout``.

This script is designed for publication-oriented ablations while keeping the
existing RGCN code untouched.

``wo_B`` keeps graph propagation and only removes the richer encoder-side
feature construction before message passing.

  - positive : negative ratio = 1 : 1
  - train negatives are sampled with replacement
  - val / test negatives use unique evaluation negatives

Default ablations cover whole-module ``B/C/D`` removals plus ``v13_full``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import random
import numpy as np
import torch_geometric.transforms as T

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for p in (PROJECT_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

if os.environ.get("GNN02_LINK_LOADER_WORKERS", "").strip():
    os.environ["NUM_WORKERS"] = os.environ.get("GNN02_LINK_LOADER_WORKERS", "").strip()
else:
    os.environ["NUM_WORKERS"] = "0"

from baselines.common.data import (  # noqa: E402
    DEVICE,
    TARGET_ETYPE,
    load_tcm_full_data,
    prepare_unified_features,
    seed_everything,
)
from cpmadr_net.model import DoseAwareHSPESARGCNv13  # noqa: E402
from cpmadr_net.model import (  # noqa: E402
    BatchedPathSetAttentionPool,
    DoseAwareHSPESARGCNv13PathReadout,
    FeatureInteractionPathReadoutHead,
    PathIngredientJaccard,
    PathReadoutCrossGate,
    PathReadoutMode,
    TrainOnlyDSANeighbor,
    _build_padded_id_table,
    _build_path_index_compat,
)
from cpmadr_net.training import (  # noqa: E402
    compute_bce_pos_weight,
    aggregate_seed_summary,
    calc_metrics,
    collect_probs,
    find_best_threshold,
    make_loaders,
    paired_significance_tests,
    print_comparison,
    print_significance_table,
    resolve_pos_weight,
)


@dataclass(frozen=True)
class PathReadoutAblationConfig:
    use_path_pool: bool = True
    use_cross_gate: bool = False
    use_embed_cos: bool = True
    use_path_ing_cos: bool = True
    use_path_jaccard: bool = True
    use_dsa_neighbor: bool = True

    @property
    def extra_dim(self) -> int:
        return sum(
            int(flag)
            for flag in (
                self.use_embed_cos,
                self.use_path_ing_cos,
                self.use_path_jaccard,
                self.use_dsa_neighbor,
            )
        )

    @property
    def active_extra_names(self) -> List[str]:
        names: List[str] = []
        if self.use_embed_cos:
            names.append("embed_cos")
        if self.use_path_ing_cos:
            names.append("path_ing_cos")
        if self.use_path_jaccard:
            names.append("path_jaccard")
        if self.use_dsa_neighbor:
            names.append("dsa_neighbor")
        return names


@dataclass(frozen=True)
class VariantSpec:
    panel: str
    description: str
    path_cfg: Optional[PathReadoutAblationConfig] = None
    remove_encoder: bool = False


VARIANT_ALIASES: Dict[str, str] = {
    "base": "v13_full",
    "v13": "v13_full",
    "full": "full_model",
    "final": "full_model",
    "wo_b": "wo_B",
    "wo_c": "wo_C",
    "wo_d": "wo_D",
}

FULL_PATH_CFG = PathReadoutAblationConfig(use_cross_gate=True)

VARIANT_SPECS: Dict[str, VariantSpec] = {
    "v13_full": VariantSpec(
        panel="Ref",
        description="Backbone-only v13 without path-readout branch.",
    ),
    "path_readout": VariantSpec(
        panel="C",
        description="v13 encoder + path ingredient attention readout.",
        path_cfg=PathReadoutAblationConfig(use_cross_gate=False),
    ),
    "path_cross": VariantSpec(
        panel="C",
        description="Path-readout with cross-gate between medicine/reaction path pools.",
        path_cfg=FULL_PATH_CFG,
    ),
    "full_model": VariantSpec(
        panel="BCD",
        description="Full proposed model: B encoder + C path readout + D auxiliary link features.",
        path_cfg=FULL_PATH_CFG,
    ),
    "wo_B": VariantSpec(
        panel="B",
        description="B ablation with graph propagation kept: raw projected node features are fed into the same propagation stack, removing the richer encoder-side feature construction.",
        path_cfg=FULL_PATH_CFG,
        remove_encoder=True,
    ),
    "wo_C": VariantSpec(
        panel="C",
        description="Remove the whole path readout branch together with auxiliary link scalars (embed cosine, path cosine, Jaccard, DSA neighbor).",
        path_cfg=PathReadoutAblationConfig(
            use_path_pool=False,
            use_cross_gate=False,
            use_embed_cos=False,
            use_path_ing_cos=False,
            use_path_jaccard=False,
            use_dsa_neighbor=False,
        ),
    ),
    "wo_D": VariantSpec(
        panel="D",
        description="Remove the whole auxiliary scalar block (embed cosine, path cosine, Jaccard, DSA neighbor).",
        path_cfg=PathReadoutAblationConfig(
            use_cross_gate=True,
            use_embed_cos=False,
            use_path_ing_cos=False,
            use_path_jaccard=False,
            use_dsa_neighbor=False,
        ),
    ),
}


def _ensure_neighbor_sampler_backend() -> None:
    """Fail early with a clear message before a long experiment starts."""
    try:
        import pyg_lib  # noqa: F401

        return
    except ImportError:
        pass
    try:
        import torch_sparse  # noqa: F401

        return
    except ImportError:
        pass
    raise ImportError(
        "LinkNeighborLoader requires either 'pyg-lib' or 'torch-sparse'. "
        "Please install one of them in the current environment before running this ablation script."
    )


def _sample_neg_with_replacement(num_neg: int, num_m: int, num_r: int, pos_set: set) -> List[List[int]]:
    neg_list: List[List[int]] = []
    while len(neg_list) < num_neg:
        m = random.randint(0, num_m - 1)
        r = random.randint(0, num_r - 1)
        if (m, r) not in pos_set:
            neg_list.append([m, r])
    return neg_list


def _sample_neg_unique(num_neg: int, num_m: int, num_r: int, pos_set: set) -> List[List[int]]:
    total = num_m * num_r
    candidate_ids = [idx for idx in range(total) if ((idx // num_r), (idx % num_r)) not in pos_set]
    if num_neg > len(candidate_ids):
        raise ValueError(
            f"Not enough unique negatives: requested={num_neg}, available={len(candidate_ids)} "
            f"(num_m={num_m}, num_r={num_r}, |pos_set|={len(pos_set)})."
        )
    chosen = random.sample(candidate_ids, num_neg)
    return [[cid // num_r, cid % num_r] for cid in chosen]


def _build_ml_edges(
    pos_ei: torch.Tensor,
    ratio: float,
    num_m: int,
    num_r: int,
    pos_set: set,
    *,
    unique_neg: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_pos = int(pos_ei.size(1))
    num_neg = int(num_pos * float(ratio))
    if num_pos <= 0:
        raise ValueError("Positive edge set is empty for current split.")

    if num_neg > 0:
        if unique_neg:
            neg_list = _sample_neg_unique(num_neg, num_m, num_r, pos_set)
        else:
            neg_list = _sample_neg_with_replacement(num_neg, num_m, num_r, pos_set)
        neg_ei = torch.tensor(neg_list, dtype=torch.long).t().contiguous()
    else:
        neg_ei = torch.empty((2, 0), dtype=torch.long)

    idx = torch.cat([pos_ei, neg_ei], dim=1)
    lab = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)], dim=0)
    return idx, lab


def train_one_cycle_dual_valtest_unique_neg(
    model_name: str,
    model_builder,
    data_full,
    med_feat,
    reac_feat,
    target_feat,
    herb_feat,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    eval_every: int,
    n_splits: int,
    batch_size: int,
    fixed_threshold: float = 0.5,
    use_imbalanced_bce: bool = False,
    compute_pos_weight_fn=None,
    pos_weight_mode: str = "sqrt",
    pos_weight_scale: float = 1.0,
    pos_weight_max: float = 10.0,
) -> pd.DataFrame:
    seed_everything(seed)
    pos_edges = data_full[TARGET_ETYPE].edge_index
    all_pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    num_m = int(data_full["medicine"].num_nodes)
    num_r = int(data_full["reaction"].num_nodes)

    fold_metrics = []
    for fold, (tv_idx, test_idx) in enumerate(kf.split(np.arange(pos_edges.size(1)))):
        test_ei, tv_ei = pos_edges[:, test_idx], pos_edges[:, tv_idx]
        perm = np.random.permutation(tv_ei.size(1))
        tv_ei = tv_ei[:, perm]
        val_size = max(1, int(tv_ei.size(1) * 0.1))
        val_ei, sup_ei = tv_ei[:, :val_size], tv_ei[:, val_size:]

        train_graph = copy.deepcopy(data_full)
        del train_graph[TARGET_ETYPE]
        train_graph[TARGET_ETYPE].edge_index = sup_ei
        train_graph = T.ToUndirected()(train_graph)

        tr_idx, tr_y = _build_ml_edges(sup_ei, 1.0, num_m, num_r, all_pos_set, unique_neg=False)
        val_idx, val_y = _build_ml_edges(val_ei, 1.0, num_m, num_r, all_pos_set, unique_neg=True)
        te_idx, te_y = _build_ml_edges(test_ei, 1.0, num_m, num_r, all_pos_set, unique_neg=True)
        tr_loader, val_loader, te_loader = make_loaders(
            train_graph, tr_idx, tr_y, val_idx, val_y, te_idx, te_y, batch_size=batch_size
        )

        pos_w = None
        if use_imbalanced_bce and compute_pos_weight_fn is not None:
            n_pos_sup = int(sup_ei.size(1))
            base_pos_w = compute_pos_weight_fn(num_m, num_r, n_pos_sup, max_w=pos_weight_max).to(DEVICE)
            pos_w = resolve_pos_weight(base_pos_w, mode=pos_weight_mode, scale=pos_weight_scale)

        model = model_builder(train_graph, med_feat, reac_feat, target_feat, herb_feat).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_v_auc = -1.0
        best_res = {}
        early_stop_counter = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch in tr_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    batch[TARGET_ETYPE].edge_label.float(),
                    pos_weight=pos_w,
                )
                loss.backward()
                optimizer.step()

            if epoch % eval_every != 0:
                continue

            val_true, val_prob = collect_probs(model, val_loader)
            v_auc, _, _, _, _, _, _ = calc_metrics(val_true, val_prob, threshold=fixed_threshold)
            if v_auc > best_v_auc:
                best_v_auc = v_auc
                early_stop_counter = 0
                best_thr = find_best_threshold(val_true, val_prob)
                tr_true, tr_prob = collect_probs(model, tr_loader)
                te_true, te_prob = collect_probs(model, te_loader)

                tr_auc, _, _, _, _, _, _ = calc_metrics(tr_true, tr_prob, threshold=fixed_threshold)
                te_auc, te_pr, _, _, _, _, _ = calc_metrics(te_true, te_prob, threshold=fixed_threshold)
                _, _, te_acc_a, te_f1_a, te_mcc_a, te_prec_a, te_rec_a = calc_metrics(
                    te_true, te_prob, threshold=fixed_threshold
                )
                _, _, te_acc_b, te_f1_b, te_mcc_b, te_prec_b, te_rec_b = calc_metrics(
                    te_true, te_prob, threshold=best_thr
                )

                best_res = {
                    "tr_auc": tr_auc,
                    "v_auc": v_auc,
                    "te_auc": te_auc,
                    "te_pr": te_pr,
                    "te_f1_thr0p5": te_f1_a,
                    "te_acc_thr0p5": te_acc_a,
                    "te_mcc_thr0p5": te_mcc_a,
                    "te_precision_thr0p5": te_prec_a,
                    "te_recall_thr0p5": te_rec_a,
                    "best_thr_valtuned": best_thr,
                    "te_f1_valtuned": te_f1_b,
                    "te_acc_valtuned": te_acc_b,
                    "te_mcc_valtuned": te_mcc_b,
                    "te_precision_valtuned": te_prec_b,
                    "te_recall_valtuned": te_rec_b,
                }
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                break

        if not best_res:
            val_true, val_prob = collect_probs(model, val_loader)
            best_thr = find_best_threshold(val_true, val_prob)
            tr_true, tr_prob = collect_probs(model, tr_loader)
            te_true, te_prob = collect_probs(model, te_loader)
            tr_auc, _, _, _, _, _, _ = calc_metrics(tr_true, tr_prob, threshold=fixed_threshold)
            v_auc, _, _, _, _, _, _ = calc_metrics(val_true, val_prob, threshold=fixed_threshold)
            te_auc, te_pr, _, _, _, _, _ = calc_metrics(te_true, te_prob, threshold=fixed_threshold)
            _, _, te_acc_a, te_f1_a, te_mcc_a, te_prec_a, te_rec_a = calc_metrics(
                te_true, te_prob, threshold=fixed_threshold
            )
            _, _, te_acc_b, te_f1_b, te_mcc_b, te_prec_b, te_rec_b = calc_metrics(
                te_true, te_prob, threshold=best_thr
            )
            best_res = {
                "tr_auc": tr_auc,
                "v_auc": v_auc,
                "te_auc": te_auc,
                "te_pr": te_pr,
                "te_f1_thr0p5": te_f1_a,
                "te_acc_thr0p5": te_acc_a,
                "te_mcc_thr0p5": te_mcc_a,
                "te_precision_thr0p5": te_prec_a,
                "te_recall_thr0p5": te_rec_a,
                "best_thr_valtuned": best_thr,
                "te_f1_valtuned": te_f1_b,
                "te_acc_valtuned": te_acc_b,
                "te_mcc_valtuned": te_mcc_b,
                "te_precision_valtuned": te_prec_b,
                "te_recall_valtuned": te_rec_b,
            }

        fold_metrics.append(best_res)
        print(
            f"[{model_name}][seed={seed}] fold {fold + 1}/{n_splits} "
            f"best_v_auc={best_v_auc:.4f} te_auc={best_res['te_auc']:.4f} | "
            "train_neg=with_replacement val/test_neg=unique"
        )

    return pd.DataFrame(fold_metrics)


class DoseAwareHSPESARGCNPathAblation(DoseAwareHSPESARGCNv13PathReadout):
    """Minimal-ablation wrapper around the original full model."""

    def __init__(
        self,
        *args,
        ablation_cfg: Optional[PathReadoutAblationConfig] = None,
        remove_encoder: bool = False,
        hidden_channels: int = 64,
        head_dropout: Optional[float] = None,
        head_res_blocks: int = 1,
        path_readout_mode: PathReadoutMode = "attn",
        path_max_ings: int = 32,
        ingredient_emb: Optional[torch.Tensor] = None,
        data_obj=None,
        **kwargs,
    ):
        self.path_ablation_cfg = ablation_cfg or PathReadoutAblationConfig()
        self.remove_encoder = bool(remove_encoder)
        super().__init__(
            *args,
            metadata=kwargs.pop("metadata"),
            data_obj=data_obj,
            hidden_channels=hidden_channels,
            head_dropout=head_dropout,
            head_res_blocks=head_res_blocks,
            path_readout_mode=path_readout_mode,
            path_max_ings=path_max_ings,
            ingredient_emb=ingredient_emb,
            **kwargs,
        )

        head_drop = float(self.dropout if head_dropout is None else head_dropout)
        self.link_head = FeatureInteractionPathReadoutHead(
            hidden_channels=int(hidden_channels),
            extra_dim=int(self.path_ablation_cfg.extra_dim),
            dropout=head_drop,
            num_res_blocks=head_res_blocks,
        )
        if not self.path_ablation_cfg.use_cross_gate:
            self.path_cross = None

    def _encode_message_pass(self, data) -> dict:
        if not self.remove_encoder:
            return super()._encode_message_pass(data)

        # wo_B keeps graph propagation but removes the richer encoder-side
        # feature construction before message passing.
        x_dict = {
            "medicine": self.med_lin(self.raw_med_x[data["medicine"].n_id]),
            "ingredient": self.ingred_lin(data["ingredient"].x),
            "herb": self.herb_struct_lin(self.raw_herb_x[data["herb"].n_id]),
            "reaction": self.reac_lin(self.raw_reac_x[data["reaction"].n_id]),
            "target": self.target_lin(self.raw_target_x[data["target"].n_id]),
        }

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

    def _gather_link_extras(
        self,
        data,
        edge_idx: torch.Tensor,
        med_h: torch.Tensor,
        reac_h: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        med_gid = data["medicine"].n_id[edge_idx[0]]
        reac_gid = data["reaction"].n_id[edge_idx[1]]
        dt = med_h.dtype
        extras: List[torch.Tensor] = []
        if self.path_ablation_cfg.use_embed_cos:
            extras.append(
                torch.nn.functional.cosine_similarity(med_h, reac_h, dim=-1, eps=1e-6).unsqueeze(-1).to(dtype=dt)
            )
        if self.path_ablation_cfg.use_path_ing_cos:
            extras.append(self._path_ing_cos_fast(med_gid, reac_gid, dtype=dt))
        if self.path_ablation_cfg.use_path_jaccard:
            extras.append(self.path_jaccard.batch_scores(med_gid, reac_gid, dtype=dt))
        if self.path_ablation_cfg.use_dsa_neighbor:
            extras.append(self.dsa_neighbor.batch_scores(med_gid, reac_gid, dtype=dt))
        if not extras:
            return None
        return torch.cat(extras, dim=-1)

    def forward(self, data):
        x_dict = self._encode_message_pass(data)
        edge_idx = data[TARGET_ETYPE].edge_label_index
        med_h = x_dict["medicine"][edge_idx[0]]
        reac_h = x_dict["reaction"][edge_idx[1]]
        med_gid = data["medicine"].n_id[edge_idx[0]]
        reac_gid = data["reaction"].n_id[edge_idx[1]]

        if self.path_ablation_cfg.use_path_pool:
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
        else:
            path_med = torch.zeros_like(med_h)
            path_reac = torch.zeros_like(reac_h)

        if self.path_cross is not None and self.path_ablation_cfg.use_cross_gate:
            path_med, path_reac = self.path_cross(path_med, path_reac)

        extras = self._gather_link_extras(data, edge_idx, med_h, reac_h)
        return self.link_head(med_h, reac_h, path_med, path_reac, extras)


def parse_args():
    p = argparse.ArgumentParser(
        description="Publication-oriented ablation runner for v13 path-readout variants."
    )
    p.add_argument(
        "--variants",
        type=str,
        default="full_model,wo_B,wo_C,wo_D,v13_full",
        help="Comma-separated variants. Default covers whole-module B/C/D ablations in the figure.",
    )
    p.add_argument("--baseline_variant", type=str, default="full_model")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--dosage_mode", type=str, default="log_share")
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--improve_layers", type=int, default=2)
    p.add_argument("--improve_dropout", type=float, default=0.25)
    p.add_argument("--improve_rel_emb_dim", type=int, default=8)
    p.add_argument("--improve_edge_drop", type=float, default=0.1)
    p.add_argument("--ffn_mult", type=float, default=2.0)
    p.add_argument("--esa_mlp_hidden", type=int, default=16)
    p.add_argument("--ffn_dropout", type=float, default=-1.0)
    p.add_argument("--head_dropout", type=float, default=-1.0)
    p.add_argument("--head_res_blocks", type=int, default=1)
    p.add_argument("--no_bimodal_film", action="store_true")
    p.add_argument("--path_max_ings", type=int, default=32)
    p.add_argument("--fixed_threshold", type=float, default=0.5)
    p.add_argument("--pos_weight_mode", type=str, default="sqrt", choices=["none", "sqrt", "linear"])
    p.add_argument("--pos_weight_scale", type=float, default=1.0)
    p.add_argument("--pos_weight_max", type=float, default=10.0)
    p.add_argument(
        "--out_dir",
        type=str,
        default="results/ablation/path_readout_ablation",
    )
    return p.parse_args()


def _parse_variants(raw: str) -> List[str]:
    out: List[str] = []
    for token in raw.split(","):
        t = token.strip().lower()
        if not t:
            continue
        t = VARIANT_ALIASES.get(t, t)
        if t not in VARIANT_SPECS:
            raise ValueError(f"Unknown variant {token!r}. Choose from: {', '.join(VARIANT_SPECS)}")
        if t not in out:
            out.append(t)
    if not out:
        raise ValueError("No variants selected.")
    return out


def _v13_kwargs(args, ffn_drop: float, head_drop: float, use_film: bool) -> dict:
    return dict(
        hidden_channels=args.hidden_channels,
        num_conv_layers=args.improve_layers,
        dropout=args.improve_dropout,
        rel_emb_dim=args.improve_rel_emb_dim,
        edge_drop=args.improve_edge_drop,
        ffn_mult=args.ffn_mult,
        esa_mlp_hidden=args.esa_mlp_hidden,
        ffn_dropout=ffn_drop,
        use_bimodal_film=use_film,
        head_dropout=head_drop,
        head_res_blocks=args.head_res_blocks,
    )


def _make_v13_builder(kw: dict) -> Callable:
    def builder(train_graph, med_x, reac_x, targ_x, herb_x, _kw=kw):
        return DoseAwareHSPESARGCNv13(
            metadata=train_graph.metadata(),
            data_obj=train_graph,
            init_med_x=med_x,
            init_reac_x=reac_x,
            init_target_x=targ_x,
            init_herb_x=herb_x,
            **_kw,
        )

    return builder


def _make_path_ablation_builder(
    *,
    mode: PathReadoutMode,
    kw: dict,
    path_max_ings: int,
    ing_emb,
    ablation_cfg: PathReadoutAblationConfig,
    remove_encoder: bool,
) -> Callable:
    def builder(
        train_graph,
        med_x,
        reac_x,
        targ_x,
        herb_x,
        _kw=kw,
        _mode=mode,
        _cap=path_max_ings,
        _ing=ing_emb,
        _cfg=ablation_cfg,
        _remove=remove_encoder,
    ):
        return DoseAwareHSPESARGCNPathAblation(
            metadata=train_graph.metadata(),
            data_obj=train_graph,
            init_med_x=med_x,
            init_reac_x=reac_x,
            init_target_x=targ_x,
            init_herb_x=herb_x,
            path_readout_mode=_mode,
            path_max_ings=_cap,
            ingredient_emb=_ing,
            ablation_cfg=_cfg,
            remove_encoder=_remove,
            **_kw,
        )

    return builder


def _summarize_variant_meta(variant: str) -> dict:
    spec = VARIANT_SPECS[variant]
    if variant == "v13_full":
        return {"panel": spec.panel, "model_type": "v13_backbone", "notes": "No path-readout branch."}
    cfg = spec.path_cfg
    assert cfg is not None
    return {
        "panel": spec.panel,
        "model_type": "path_readout_ablation",
        "use_path_pool": cfg.use_path_pool,
        "use_cross_gate": cfg.use_cross_gate,
        "active_extras": cfg.active_extra_names,
        "extra_dim": cfg.extra_dim,
        "remove_encoder": spec.remove_encoder,
    }


def main() -> None:
    args = parse_args()
    _ensure_neighbor_sampler_backend()
    variants = _parse_variants(args.variants)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    baseline_variant = VARIANT_ALIASES.get(args.baseline_variant.strip().lower(), args.baseline_variant.strip().lower())
    if baseline_variant not in variants:
        raise ValueError(f"baseline_variant={baseline_variant!r} must be included in --variants.")

    compute_pos_weight_fn = compute_bce_pos_weight

    ffn_drop = args.improve_dropout if args.ffn_dropout < 0 else args.ffn_dropout
    head_drop = args.improve_dropout if args.head_dropout < 0 else args.head_dropout
    use_film = not args.no_bimodal_film
    kw = _v13_kwargs(args, ffn_drop, head_drop, use_film)

    print("[INFO] Variants:")
    for v in variants:
        print(f"  - {v}: {VARIANT_SPECS[v].description}")
    print(f"[INFO] Baseline for paired tests: {baseline_variant}")
    print(f"[INFO] FiLM={'on' if use_film else 'off'}  path_max_ings={args.path_max_ings}")
    print("[INFO] Positive:negative ratio = 1:1.")
    print("[INFO] Negative sampling policy: train with replacement; val/test unique negatives.")

    print("[INFO] Loading data once...")
    raw_data, _ = load_tcm_full_data(use_dosage=True, dosage_mode=args.dosage_mode)
    med_feat, reac_feat, target_feat, herb_feat = prepare_unified_features(raw_data)
    ing_emb = raw_data["ingredient"].x

    builders: Dict[str, Callable] = {}
    for variant in variants:
        if variant == "v13_full":
            builders[variant] = _make_v13_builder(copy.deepcopy(kw))
            continue

        spec = VARIANT_SPECS[variant]
        assert spec.path_cfg is not None
        variant_kw = copy.deepcopy(kw)
        mode: PathReadoutMode = "attn_cross" if spec.path_cfg.use_cross_gate else "attn"
        builders[variant] = _make_path_ablation_builder(
            mode=mode,
            kw=variant_kw,
            path_max_ings=args.path_max_ings,
            ing_emb=ing_emb,
            ablation_cfg=spec.path_cfg,
            remove_encoder=spec.remove_encoder,
        )

    all_rows = []
    for seed in seeds:
        print(f"\n>>> [Seed] {seed}")
        for label in variants:
            meta = _summarize_variant_meta(label)
            print(f"  - Running {label}... {meta}")
            df_run = train_one_cycle_dual_valtest_unique_neg(
                model_name=label,
                model_builder=builders[label],
                data_full=raw_data,
                med_feat=med_feat,
                reac_feat=reac_feat,
                target_feat=target_feat,
                herb_feat=herb_feat,
                seed=seed,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_epochs=args.max_epochs,
                patience=args.patience,
                eval_every=args.eval_every,
                n_splits=args.n_splits,
                batch_size=args.batch_size,
                fixed_threshold=args.fixed_threshold,
                use_imbalanced_bce=True,
                compute_pos_weight_fn=compute_pos_weight_fn,
                pos_weight_mode=args.pos_weight_mode,
                pos_weight_scale=args.pos_weight_scale,
                pos_weight_max=args.pos_weight_max,
            )
            df_run["seed"] = seed
            df_run["model"] = label
            df_run["variant"] = label
            df_run["variant_desc"] = VARIANT_SPECS[label].description
            df_run["panel"] = VARIANT_SPECS[label].panel
            df_run["baseline_variant"] = baseline_variant
            df_run["path_max_ings"] = args.path_max_ings
            df_run["neg_pos_ratio"] = 1.0
            df_run["negative_sampling_policy"] = "train_with_replacement__val_test_unique"
            if label == "v13_full":
                df_run["path_readout_mode"] = "none"
                df_run["active_extras"] = "none"
                df_run["use_path_pool"] = False
                df_run["use_cross_gate"] = False
                df_run["remove_encoder"] = False
            else:
                cfg = VARIANT_SPECS[label].path_cfg
                assert cfg is not None
                df_run["path_readout_mode"] = "attn_cross" if cfg.use_cross_gate else "attn"
                df_run["active_extras"] = ",".join(cfg.active_extra_names) if cfg.active_extra_names else "none"
                df_run["use_path_pool"] = cfg.use_path_pool
                df_run["use_cross_gate"] = cfg.use_cross_gate
                df_run["remove_encoder"] = VARIANT_SPECS[label].remove_encoder
            all_rows.append(df_run)

    final_df = pd.concat(all_rows, axis=0, ignore_index=True)
    final_df["fold"] = final_df.groupby(["seed", "model"]).cumcount()

    out_root = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_root, exist_ok=True)
    raw_path = os.path.join(out_root, "combined_all_seeds_path_readout_ablation.csv")
    final_df.to_csv(raw_path, index=False)

    metrics_shared = ["tr_auc", "v_auc", "te_auc", "te_pr"]
    metrics_a = metrics_shared + [
        "te_f1_thr0p5",
        "te_acc_thr0p5",
        "te_mcc_thr0p5",
        "te_precision_thr0p5",
        "te_recall_thr0p5",
    ]
    metrics_b = metrics_shared + [
        "te_f1_valtuned",
        "te_acc_valtuned",
        "te_mcc_valtuned",
        "te_precision_valtuned",
        "te_recall_valtuned",
    ]

    summaries_a: Dict[str, dict] = {}
    summaries_b: Dict[str, dict] = {}
    for label in variants:
        sub = final_df[final_df["model"] == label]
        summaries_a[label] = aggregate_seed_summary(sub, seeds, metrics_a)
        summaries_b[label] = aggregate_seed_summary(sub, seeds, metrics_b)

    for label in variants:
        if label == baseline_variant:
            continue
        print_comparison(
            f"PROTOCOL A vs {baseline_variant}: fixed threshold = {args.fixed_threshold} -- {label}",
            baseline_variant,
            summaries_a[baseline_variant],
            label,
            summaries_a[label],
            metrics_a,
        )

    sig_rows = []
    for label in variants:
        if label == baseline_variant:
            continue
        sig_a = paired_significance_tests(
            final_df,
            metrics_a,
            baseline_name=baseline_variant,
            improved_name=label,
            protocol_label=f"fixed_{args.fixed_threshold}",
        )
        sig_b = paired_significance_tests(
            final_df,
            metrics_b,
            baseline_name=baseline_variant,
            improved_name=label,
            protocol_label="val_tuned_mcc",
        )
        print_significance_table(sig_a, f"SIGNIFICANCE {label} vs {baseline_variant} -- Protocol A")
        sig_a["compared_to"] = baseline_variant
        sig_b["compared_to"] = baseline_variant
        sig_rows.append(sig_a)
        sig_rows.append(sig_b)

    sig_combined = pd.concat(sig_rows, axis=0, ignore_index=True) if sig_rows else pd.DataFrame()
    sig_path = os.path.join(out_root, "significance_paired_path_readout_ablation.csv")
    if not sig_combined.empty:
        sig_combined.to_csv(sig_path, index=False)

    summary = {
        "script": "ablation/run_path_readout_ablation.py",
        "baseline_variant": baseline_variant,
        "variants": {v: VARIANT_SPECS[v].description for v in variants},
        "variant_meta": {v: _summarize_variant_meta(v) for v in variants},
        "use_bimodal_film": use_film,
        "path_max_ings": args.path_max_ings,
        "seeds": seeds,
        "pos_neg_ratio": "1:1",
        "neg_pos_ratio": 1.0,
        "negative_sampling": {
            "train": "with_replacement",
            "val": "unique",
            "test": "unique",
            "source_protocol": "custom ablation protocol: 1:1 with unique eval negatives; wo_B keeps graph propagation",
        },
        "summary_protocol_A": summaries_a,
        "summary_protocol_B": summaries_b,
        "significance_paired": sig_combined.to_dict(orient="records") if not sig_combined.empty else [],
    }
    summary_path = os.path.join(out_root, "comparison_summary_path_readout_ablation.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Saved:\n  {raw_path}\n  {summary_path}")
    if not sig_combined.empty:
        print(f"  {sig_path}")


if __name__ == "__main__":
    main()
