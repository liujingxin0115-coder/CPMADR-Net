"""
Export path-readout ingredient attention for one medicine–ADR case.

Reuses the fixed-medicine-split protocol (same as the degree-stratified experiment).
Does NOT overwrite existing result CSVs: writes only under --out_dir.

Example (Niuhuang Jiedu Pian – hepatic function abnormal):
  python scripts/export_path_evidence.py \\
    --seed 42 --fold 1 --medicine_id 36 --adr_id 17 \\
    --expect_prob 0.570666
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from cpmadr_net.model import DoseAwareHSPESARGCNv13PathReadout  # noqa: E402
from cpmadr_net.training import (  # noqa: E402
    compute_bce_pos_weight,
    calc_metrics,
    find_best_threshold,
    resolve_pos_weight,
)
from cpmadr_net.splits import (  # noqa: E402
    build_fold_pos_edges_by_medicine,
    collect_probs,
    load_fixed_medicine_splits,
    make_loaders,
)
from cpmadr_net.splits import (  # noqa: E402
    DEFAULT_MR_PATH,
    _build_ml_edges_for_medicine_set,
)


def make_neighbor_loaders(train_graph, tr_idx, tr_y, val_idx, val_y, te_idx, te_y, batch_size: int):
    """Same LinkNeighborLoader setup as the degree-stratified experiments.

    Requires working ``torch-sparse`` or ``pyg-lib`` (server env). No full-graph fallback.
    """
    try:
        loaders = make_loaders(train_graph, tr_idx, tr_y, val_idx, val_y, te_idx, te_y, batch_size=batch_size)
        _ = next(iter(loaders[0]))  # fail fast if NeighborSampler backend is missing
    except Exception as exc:
        raise RuntimeError(
            "LinkNeighborLoader / NeighborSampler unavailable. "
            "Install a working torch-sparse or pyg-lib in this environment "
            "(use the same env as the stratified experiments). "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    print("[INFO] Using LinkNeighborLoader (num_neighbors=[15, 10]).", flush=True)
    return loaders


DEFAULT_OUT = os.path.join(
    "unbalanced",
    "result",
    "RGCN_fixed_medicine_degree_v13_path_unique_eval_neg",
    "case_explain_niuhuang_hepatic",
)
DEFAULT_FIGURE_INGS = "Realgar,Astragalin,18beta-glycyrrhetinic acid,Acacetin"
MEDICINE_NODE = os.path.join(PROJECT_ROOT, "HerbData", "medicine_herb", "medicine_node1.csv")
ADR_NODE = os.path.join(PROJECT_ROOT, "adr", "adr_node1.csv")
INGREDIENT_NODE = os.path.join(PROJECT_ROOT, "herb_ingredient", "ingredient_node1.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export path-readout attention for one case edge.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=1, help="0-based fold index (CSV fold for this case is 1).")
    p.add_argument("--medicine_id", type=int, default=36)
    p.add_argument("--adr_id", "--reaction_id", dest="adr_id", type=int, default=17)
    p.add_argument("--expect_prob", type=float, default=0.570666, help="Reference y_prob from existing CSV.")
    p.add_argument("--prob_atol", type=float, default=0.02, help="Allowed |prob - expect_prob|.")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument(
        "--figure_ingredients",
        type=str,
        default=DEFAULT_FIGURE_INGS,
        help="Comma-separated ingredient names to highlight (case figure).",
    )
    p.add_argument("--dosage_mode", type=str, default="log_share")
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=5)
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
    p.add_argument("--head_res_blocks", type=int, default=1)
    p.add_argument("--path_readout_mode", type=str, default="attn", choices=["attn", "attn_cross"])
    p.add_argument("--path_max_ings", type=int, default=32)
    p.add_argument("--path_ing_cos_mode", type=str, default="mean_pool", choices=["mean_pool", "max"])
    p.add_argument("--fixed_threshold", type=float, default=0.5)
    p.add_argument("--pos_weight_mode", type=str, default="sqrt", choices=["none", "sqrt", "linear"])
    p.add_argument("--pos_weight_scale", type=float, default=1.0)
    p.add_argument("--pos_weight_max", type=float, default=10.0)
    p.add_argument("--mr_path", type=str, default=DEFAULT_MR_PATH)
    p.add_argument("--split_root", type=str, default="unbalanced/splits")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--train_neg_unique", action="store_true")
    p.add_argument("--allow_dup_val_neg", action="store_true")
    p.add_argument("--allow_dup_test_neg", action="store_true")
    p.add_argument("--ckpt_path", type=str, default="", help="If set, skip training and load this checkpoint.")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    return p.parse_args()


def _load_name_maps() -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str], Dict[str, int]]:
    med = pd.read_csv(MEDICINE_NODE, encoding="utf-8-sig")
    reac = pd.read_csv(ADR_NODE, encoding="utf-8-sig")
    ing = pd.read_csv(INGREDIENT_NODE, encoding="utf-8-sig")
    med_map = {int(r.medicine_id): str(r.medicine_name) for r in med.itertuples(index=False)}
    reac_map = {int(r.adr_id): str(r.adr_name) for r in reac.itertuples(index=False)}
    ing_map = {int(r.ingredient_id): str(r.Ingredient_Name) for r in ing.itertuples(index=False)}
    name_to_id = {str(r.Ingredient_Name).strip().lower(): int(r.ingredient_id) for r in ing.itertuples(index=False)}
    return med_map, reac_map, ing_map, name_to_id


def _weights_to_rows(
    *,
    side: str,
    weights: torch.Tensor,
    ids: torch.Tensor,
    mask: torch.Tensor,
    ing_map: Dict[int, str],
) -> List[Dict]:
    w = weights.detach().cpu().numpy()
    idv = ids.detach().cpu().numpy()
    m = mask.detach().cpu().numpy().astype(bool)
    rows: List[Dict] = []
    for j in range(len(w)):
        if not m[j]:
            continue
        iid = int(idv[j])
        rows.append(
            {
                "side": side,
                "ingredient_id": iid,
                "ingredient_name": ing_map.get(iid, f"id_{iid}"),
                "attn_weight": float(w[j]),
            }
        )
    rows.sort(key=lambda r: r["attn_weight"], reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    return rows


def _pick_figure_rows(
    all_rows: List[Dict],
    figure_names: Sequence[str],
    name_to_id: Dict[str, int],
) -> pd.DataFrame:
    by_id = {int(r["ingredient_id"]): r for r in all_rows}
    by_name_l = {str(r["ingredient_name"]).strip().lower(): r for r in all_rows}
    out: List[Dict] = []
    for name in figure_names:
        key = name.strip()
        key_l = key.lower()
        hit = by_name_l.get(key_l)
        if hit is None:
            iid = name_to_id.get(key_l)
            hit = by_id.get(int(iid)) if iid is not None else None
        if hit is None:
            out.append(
                {
                    "figure_ingredient": key,
                    "side": "",
                    "ingredient_id": -1,
                    "ingredient_name": "",
                    "attn_weight": float("nan"),
                    "rank": -1,
                    "in_path_pool": False,
                }
            )
        else:
            out.append(
                {
                    "figure_ingredient": key,
                    "side": hit["side"],
                    "ingredient_id": hit["ingredient_id"],
                    "ingredient_name": hit["ingredient_name"],
                    "attn_weight": hit["attn_weight"],
                    "rank": hit["rank"],
                    "in_path_pool": True,
                }
            )
    return pd.DataFrame(out)


@torch.no_grad()
def _explain_case_on_loader(
    model: DoseAwareHSPESARGCNv13PathReadout,
    loader,
    medicine_id: int,
    reaction_id: int,
    ing_map: Dict[int, str],
) -> Optional[Tuple[float, int, List[Dict], List[Dict]]]:
    model.eval()
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model.explain_links(batch)
        probs = torch.sigmoid(out["logits"]).detach().cpu().numpy()
        labels = out["edge_label"].detach().cpu().numpy()
        med = out["med_global_ids"].detach().cpu().numpy()
        reac = out["reac_global_ids"].detach().cpu().numpy()
        for i in range(len(probs)):
            if int(med[i]) != int(medicine_id) or int(reac[i]) != int(reaction_id):
                continue
            med_rows = _weights_to_rows(
                side="medicine",
                weights=out["w_med"][i],
                ids=out["ids_med"][i],
                mask=out["mask_med"][i],
                ing_map=ing_map,
            )
            reac_rows = _weights_to_rows(
                side="reaction",
                weights=out["w_reac"][i],
                ids=out["ids_reac"][i],
                mask=out["mask_reac"][i],
                ing_map=ing_map,
            )
            return float(probs[i]), int(labels[i]), med_rows, reac_rows
    return None


def train_one_fold_save_best(
    *,
    model_builder,
    data_full,
    med_feat,
    reac_feat,
    target_feat,
    herb_feat,
    split: Dict[str, List[int]],
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    eval_every: int,
    batch_size: int,
    fixed_threshold: float,
    use_imbalanced_bce: bool,
    compute_pos_weight_fn,
    pos_weight_mode: str,
    pos_weight_scale: float,
    pos_weight_max: float,
    train_neg_unique: bool,
    val_neg_unique: bool,
    test_neg_unique: bool,
) -> Tuple[dict, object, object, Dict[str, float]]:
    """Train a single fixed-medicine fold; return best state_dict + loaders + metrics."""
    seed_everything(seed)
    pos_edges = data_full[TARGET_ETYPE].edge_index
    all_pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))
    n_reac = int(data_full["reaction"].num_nodes)

    train_pos = build_fold_pos_edges_by_medicine(pos_edges, split["train"])
    val_pos = build_fold_pos_edges_by_medicine(pos_edges, split["val"])
    test_pos = build_fold_pos_edges_by_medicine(pos_edges, split["test"])
    if train_pos.size(1) == 0 or val_pos.size(1) == 0 or test_pos.size(1) == 0:
        raise ValueError("Empty positive edges in train/val/test for this fold.")

    train_graph = copy.deepcopy(data_full)
    del train_graph[TARGET_ETYPE]
    train_graph[TARGET_ETYPE].edge_index = train_pos
    train_graph = T.ToUndirected()(train_graph)

    tr_idx, tr_y = _build_ml_edges_for_medicine_set(
        train_pos, 1.0, split["train"], n_reac, all_pos_set, unique_neg=train_neg_unique
    )
    val_idx, val_y = _build_ml_edges_for_medicine_set(
        val_pos, 1.0, split["val"], n_reac, all_pos_set, unique_neg=val_neg_unique
    )
    te_idx, te_y = _build_ml_edges_for_medicine_set(
        test_pos, 1.0, split["test"], n_reac, all_pos_set, unique_neg=test_neg_unique
    )
    tr_loader, val_loader, te_loader = make_neighbor_loaders(
        train_graph, tr_idx, tr_y, val_idx, val_y, te_idx, te_y, batch_size=batch_size
    )

    pos_w = None
    if use_imbalanced_bce and compute_pos_weight_fn is not None:
        n_med_active = len(split["train"])
        n_pos_sup = int(train_pos.size(1))
        base_pos_w = compute_pos_weight_fn(n_med_active, n_reac, n_pos_sup, max_w=pos_weight_max).to(DEVICE)
        pos_w = resolve_pos_weight(base_pos_w, mode=pos_weight_mode, scale=pos_weight_scale)

    model = model_builder(train_graph, med_feat, reac_feat, target_feat, herb_feat).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_v_auc = -1.0
    best_state: Optional[dict] = None
    best_meta: Dict[str, float] = {}
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
            best_state = copy.deepcopy(model.state_dict())
            best_thr = find_best_threshold(val_true, val_prob)
            te_true, te_prob = collect_probs(model, te_loader)
            te_auc, te_pr, _, _, _, _, _ = calc_metrics(te_true, te_prob, threshold=fixed_threshold)
            best_meta = {
                "epoch": float(epoch),
                "v_auc": float(v_auc),
                "te_auc": float(te_auc),
                "te_pr": float(te_pr),
                "best_thr_valtuned": float(best_thr),
                "loader_mode": "neighbor",
            }
            print(
                f"[train] epoch={epoch} best_v_auc={v_auc:.4f} te_auc={te_auc:.4f} loader=neighbor",
                flush=True,
            )
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            break

    if best_state is None:
        raise RuntimeError("No validation improvement; cannot export attention.")
    best_meta["loader_mode"] = "neighbor"
    return best_state, train_graph, te_loader, best_meta


def main() -> None:
    args = parse_args()
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    med_map, reac_map, ing_map, name_to_id = _load_name_maps()
    med_name = med_map.get(int(args.medicine_id), str(args.medicine_id))
    reac_name = reac_map.get(int(args.adr_id), str(args.adr_id))
    figure_names = [x.strip() for x in str(args.figure_ingredients).split(",") if x.strip()]

    split_root_abs = os.path.join(PROJECT_ROOT, args.split_root)
    fixed_splits = load_fixed_medicine_splits(split_root_abs, args.n_splits)
    if not (0 <= int(args.fold) < len(fixed_splits)):
        raise ValueError(f"--fold must be in [0, {len(fixed_splits) - 1}]")
    split = fixed_splits[int(args.fold)]
    if int(args.medicine_id) not in set(int(x) for x in split["test"]):
        raise ValueError(
            f"medicine_id={args.medicine_id} is not in test set of fold={args.fold}. "
            "Check --fold against the stratified CSV."
        )

    print("[INFO] Loading data...", flush=True)
    raw_data, _ = load_tcm_full_data(use_dosage=True, dosage_mode=args.dosage_mode)
    med_feat, reac_feat, target_feat, herb_feat = prepare_unified_features(raw_data)
    ing_emb = raw_data["ingredient"].x

    compute_pos_weight_fn = compute_bce_pos_weight
    train_neg_unique = bool(args.train_neg_unique)
    val_neg_unique = not bool(args.allow_dup_val_neg)
    test_neg_unique = not bool(args.allow_dup_test_neg)

    def model_builder(train_graph, med_x, reac_x, targ_x, herb_x):
        return DoseAwareHSPESARGCNv13PathReadout(
            metadata=train_graph.metadata(),
            data_obj=train_graph,
            init_med_x=med_x,
            init_reac_x=reac_x,
            init_target_x=targ_x,
            init_herb_x=herb_x,
            hidden_channels=args.hidden_channels,
            num_conv_layers=args.improve_layers,
            dropout=args.improve_dropout,
            rel_emb_dim=args.improve_rel_emb_dim,
            edge_drop=args.improve_edge_drop,
            ffn_mult=args.ffn_mult,
            esa_mlp_hidden=args.esa_mlp_hidden,
            ffn_dropout=args.improve_dropout,
            use_bimodal_film=True,
            head_dropout=args.improve_dropout,
            head_res_blocks=args.head_res_blocks,
            path_readout_mode=args.path_readout_mode,
            path_max_ings=args.path_max_ings,
            ingredient_emb=ing_emb,
            path_ing_cos_mode=args.path_ing_cos_mode,
        )

    ckpt_path = os.path.join(out_dir, f"ckpt_fold{args.fold}_seed{args.seed}.pt")
    meta: Dict[str, float] = {}

    if args.ckpt_path:
        ckpt_load = args.ckpt_path if os.path.isabs(args.ckpt_path) else os.path.join(PROJECT_ROOT, args.ckpt_path)
        print(f"[INFO] Loading checkpoint: {ckpt_load}", flush=True)
        payload = torch.load(ckpt_load, map_location="cpu")
        best_state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        # Rebuild train_graph / te_loader for this fold (no training).
        seed_everything(args.seed)
        pos_edges = raw_data[TARGET_ETYPE].edge_index
        all_pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))
        n_reac = int(raw_data["reaction"].num_nodes)
        train_pos = build_fold_pos_edges_by_medicine(pos_edges, split["train"])
        val_pos = build_fold_pos_edges_by_medicine(pos_edges, split["val"])
        test_pos = build_fold_pos_edges_by_medicine(pos_edges, split["test"])
        train_graph = copy.deepcopy(raw_data)
        del train_graph[TARGET_ETYPE]
        train_graph[TARGET_ETYPE].edge_index = train_pos
        train_graph = T.ToUndirected()(train_graph)
        tr_idx, tr_y = _build_ml_edges_for_medicine_set(
            train_pos, 1.0, split["train"], n_reac, all_pos_set, unique_neg=train_neg_unique
        )
        val_idx, val_y = _build_ml_edges_for_medicine_set(
            val_pos, 1.0, split["val"], n_reac, all_pos_set, unique_neg=val_neg_unique
        )
        te_idx, te_y = _build_ml_edges_for_medicine_set(
            test_pos, 1.0, split["test"], n_reac, all_pos_set, unique_neg=test_neg_unique
        )
        _, _, te_loader = make_neighbor_loaders(
            train_graph, tr_idx, tr_y, val_idx, val_y, te_idx, te_y, batch_size=args.batch_size
        )
        meta["loader_mode"] = "neighbor"
        model = model_builder(train_graph, med_feat, reac_feat, target_feat, herb_feat).to(DEVICE)
        model.load_state_dict(best_state)
    else:
        print(
            f"[INFO] Training PathReadout seed={args.seed} fold={args.fold} "
            f"case={med_name}({args.medicine_id})-{reac_name}({args.adr_id})",
            flush=True,
        )
        best_state, train_graph, te_loader, meta = train_one_fold_save_best(
            model_builder=model_builder,
            data_full=raw_data,
            med_feat=med_feat,
            reac_feat=reac_feat,
            target_feat=target_feat,
            herb_feat=herb_feat,
            split=split,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            fixed_threshold=args.fixed_threshold,
            use_imbalanced_bce=True,
            compute_pos_weight_fn=compute_pos_weight_fn,
            pos_weight_mode=args.pos_weight_mode,
            pos_weight_scale=args.pos_weight_scale,
            pos_weight_max=args.pos_weight_max,
            train_neg_unique=train_neg_unique,
            val_neg_unique=val_neg_unique,
            test_neg_unique=test_neg_unique,
        )
        torch.save(
            {
                "state_dict": best_state,
                "seed": args.seed,
                "fold": args.fold,
                "medicine_id": args.medicine_id,
                "adr_id": args.adr_id,
                "meta": meta,
            },
            ckpt_path,
        )
        print(f"[INFO] Saved checkpoint: {ckpt_path}", flush=True)
        model = model_builder(train_graph, med_feat, reac_feat, target_feat, herb_feat).to(DEVICE)
        model.load_state_dict(best_state)

    seed_everything(args.seed)
    hit = _explain_case_on_loader(
        model, te_loader, int(args.medicine_id), int(args.adr_id), ing_map
    )
    if hit is None:
        raise RuntimeError(
            f"Case edge ({args.medicine_id}, {args.adr_id}) not found in test loader. "
            "Check fold / negative sampling settings."
        )
    y_prob, y_true, med_rows, reac_rows = hit
    delta = abs(y_prob - float(args.expect_prob))
    print(
        f"[INFO] case y_true={y_true} y_prob={y_prob:.6f} expect={args.expect_prob:.6f} |delta|={delta:.6f}",
        flush=True,
    )
    if delta > float(args.prob_atol):
        print(
            f"[WARN] |delta| > atol={args.prob_atol}. Neighbor sampling / early-stop epoch may differ; "
            "report attention with this caveat, or re-run.",
            flush=True,
        )

    all_df = pd.DataFrame(med_rows + reac_rows)
    all_path = os.path.join(out_dir, "case_attn_all_ingredients.csv")
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    topk = int(args.topk)
    top_df = pd.concat(
        [
            pd.DataFrame(med_rows).head(topk),
            pd.DataFrame(reac_rows).head(topk),
        ],
        ignore_index=True,
    )
    top_path = os.path.join(out_dir, "case_attn_topk.csv")
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    fig_med = _pick_figure_rows(med_rows, figure_names, name_to_id)
    fig_med.insert(1, "lookup_side", "medicine")
    fig_reac = _pick_figure_rows(reac_rows, figure_names, name_to_id)
    fig_reac.insert(1, "lookup_side", "reaction")
    fig_df = pd.concat([fig_med, fig_reac], ignore_index=True)
    fig_path = os.path.join(out_dir, "case_attn_figure_ingredients.csv")
    fig_df.to_csv(fig_path, index=False, encoding="utf-8-sig")

    summary = {
        "medicine_id": int(args.medicine_id),
        "medicine_name": med_name,
        "adr_id": int(args.adr_id),
        "adr_name": reac_name,
        "seed": int(args.seed),
        "fold": int(args.fold),
        "y_true": int(y_true),
        "y_prob": float(y_prob),
        "expect_prob": float(args.expect_prob),
        "abs_delta_vs_expect": float(delta),
        "prob_ok": bool(delta <= float(args.prob_atol)),
        "path_readout_mode": args.path_readout_mode,
        "path_max_ings": int(args.path_max_ings),
        "train_meta": meta,
        "outputs": {
            "all": all_path,
            "topk": top_path,
            "figure": fig_path,
            "ckpt": ckpt_path if not args.ckpt_path else args.ckpt_path,
        },
        "note": (
            "Attention is query-conditioned softmax over path-readout ingredient pools "
            "(not full multi-hop KG path attention). Figure paths remain post-hoc evidence chains."
        ),
    }
    summary_path = os.path.join(out_dir, "case_attn_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] Wrote:", all_path)
    print("[INFO] Wrote:", top_path)
    print("[INFO] Wrote:", fig_path)
    print("[INFO] Wrote:", summary_path)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
