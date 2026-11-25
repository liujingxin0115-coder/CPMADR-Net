import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# data.py is under baselines/common; walk up to the CPMADR-Net repository root
# so HerbData, BERT, and the graph CSV directories can be resolved.
_auto_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_base_path() -> str:
    """
    Project root containing HerbData/ and BERT/.

    1) CPMADR_NET_ROOT (or legacy GNN02_BASE_PATH) if set and contains HerbData
    2) Auto-detected CPMADR-Net repository root if HerbData exists there
    3) Auto path
    """
    env = (
        os.environ.get("CPMADR_NET_ROOT", "").strip()
        or os.environ.get("GNN02_BASE_PATH", "").strip()
    )
    if env and os.path.isdir(os.path.join(env, "HerbData")):
        return env
    if os.path.isdir(os.path.join(_auto_path, "HerbData")):
        return _auto_path
    return _auto_path


BASE_PATH = _resolve_base_path()


def _resolve_training_device() -> torch.device:
    """
    Training device for GNN / Transformer baselines.

    - GNN02_DEVICE=cpu   : force CPU
    - GNN02_DEVICE=cuda  : use CUDA if available
    - auto: use CUDA when available; only auto-fallback to CPU for Blackwell (sm_100/sm_120)
      when the arch string is missing from get_arch_list() (common wheel gap).
      Ada (sm_89) etc. may be omitted from the list but still run kernels — do not force CPU.
    """
    override = os.environ.get("GNN02_DEVICE", "").strip().lower()
    if override == "cpu":
        return torch.device("cpu")
    if override == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        arch_list = torch.cuda.get_arch_list()
        major, minor = torch.cuda.get_device_capability(0)
        sm_tag = f"sm_{major}{minor}"
        blackwell_like = sm_tag in ("sm_100", "sm_120")
        if arch_list and sm_tag not in arch_list and blackwell_like:
            print(
                f"[WARN] GPU capability {major}.{minor} ({sm_tag}) is not in this PyTorch build's "
                f"CUDA arch list (first few: {arch_list[:12]}). "
                f"Using CPU for training. Install a PyTorch build with Blackwell support, or set "
                f"GNN02_DEVICE=cuda to force GPU anyway."
            )
            return torch.device("cpu")
        if arch_list and sm_tag not in arch_list and not blackwell_like:
            print(
                f"[INFO] GPU {sm_tag} not in torch.cuda.get_arch_list() sample {arch_list[:8]}; "
                f"using CUDA anyway. Set GNN02_DEVICE=cpu if you hit 'no kernel image' errors."
            )
    except Exception:
        pass

    return torch.device("cuda")


DEVICE = _resolve_training_device()
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))

# Hetero edge type constants used across models.
DOSAGE_EDGE_TYPE = ("medicine", "contains", "herb")
TARGET_ETYPE = ("medicine", "has_reaction", "reaction")


def seed_everything(seed: int = 60) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN: must set deterministic=True AND benchmark=False.
    # Previously benchmark was True on CUDA, which let cuDNN pick algorithms
    # by runtime timing (non-reproducible) — root cause of intra-seed drift.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Required for deterministic cuBLAS (matmul / linear). Must be set before
    # the first CUDA context use; setdefault avoids overriding a stricter value.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Enforce determinism on supported ops. warn_only=True keeps scatter_add
    # and similar ops working (they emit a warning instead of erroring out).
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_tcm_full_data(use_dosage: bool = False, dosage_mode: str = "log_share") -> Tuple[HeteroData, dict]:
    """
    Build the hetero graph and node features.

    Notes:
    - Uses CSV edges + precomputed BERT embeddings (.npy) if present.
    - If BERT .npy files are missing, falls back to random features (matches original behavior).
    """
    data = HeteroData()

    med_df = pd.read_csv(os.path.join(BASE_PATH, "HerbData/medicine_herb/medicine_node1.csv"))
    herb_df = pd.read_csv(os.path.join(BASE_PATH, "HerbData/medicine_herb/herb_node1.csv"))
    reac_df = pd.read_csv(os.path.join(BASE_PATH, "adr/adr_node1.csv"))
    target_df = pd.read_csv(os.path.join(BASE_PATH, "ingredient_target/target_node1.csv"))

    reac_map = {old: new for new, old in enumerate(reac_df["adr_id"].unique())}
    target_map = {old: new for new, old in enumerate(target_df["target_id"].unique())}

    data["medicine"].num_nodes = len(med_df)
    data["herb"].num_nodes = len(herb_df)
    data["reaction"].num_nodes = len(reac_df)
    data["target"].num_nodes = len(target_df)

    ing_node_df = pd.read_csv(os.path.join(BASE_PATH, "herb_ingredient/ingredient_node1.csv"))
    num_ingredient = int(ing_node_df["ingredient_id"].max()) + 1

    # ingredient node features (ChemBERTa) — scatter by ingredient_id (same pattern as BiomedBERT)
    chemberta_dir = os.path.join(BASE_PATH, "BERT/ChemBERTa_output")
    chemberta_path = os.path.join(chemberta_dir, "ingredient_chemberta1.npy")
    chemberta_ids_path = os.path.join(chemberta_dir, "ingredient_ids1.csv")

    def load_chemberta_feat(n_nodes: int, emb_path: str, ids_path: str, dim_fallback: int = 128) -> torch.Tensor:
        try:
            emb = np.load(emb_path)
            dim = emb.shape[1]
            feat = np.zeros((n_nodes, dim), dtype=np.float32)
            ids_df = pd.read_csv(ids_path)
            for row, ing_id in enumerate(ids_df["ingredient_id"].astype(int)):
                ing_id = int(ing_id)
                if 0 <= ing_id < n_nodes:
                    feat[ing_id] = emb[row]
            return torch.from_numpy(feat).to(torch.float)
        except (FileNotFoundError, ValueError, OSError) as e:
            print(
                f"[WARN] ChemBERTa embeddings unavailable: {emb_path}\n"
                f"  ({e})\n"
                f"  Using random ingredient features (same as missing file)."
            )
            return torch.from_numpy(np.random.randn(n_nodes, dim_fallback)).to(torch.float)

    data["ingredient"].x = load_chemberta_feat(num_ingredient, chemberta_path, chemberta_ids_path)
    data["ingredient"].num_nodes = num_ingredient

    # herb/target/reaction node features (BiomedBERT)
    biomed_dir = os.path.join(BASE_PATH, "BERT/BiomedBERT_output")

    def load_biomed_feat(
        n_nodes: int,
        id_series: pd.Series,
        id_col_name: str,
        emb_name: str,
        ids_name: str,
        dim_fallback: int = 128,
    ) -> torch.Tensor:
        feat = torch.zeros((n_nodes, dim_fallback), dtype=torch.float)
        try:
            emb = np.load(os.path.join(biomed_dir, emb_name))
            dim = emb.shape[1]
            feat = torch.zeros((n_nodes, dim), dtype=torch.float)
            ids_df = pd.read_csv(os.path.join(biomed_dir, ids_name))
            id2row = {str(i): k for k, i in enumerate(ids_df[id_col_name].astype(str))}
            for idx, nid in enumerate(id_series.astype(str)):
                if nid in id2row:
                    feat[idx] = torch.from_numpy(emb[id2row[nid]]).to(torch.float)
        except FileNotFoundError:
            feat = torch.randn(n_nodes, dim_fallback)
        return feat

    data["herb"].x = load_biomed_feat(
        len(herb_df), herb_df["herb_id"], "herb_id", "herb_emb_biomedbert1.npy", "herb_ids1.csv"
    )
    data["target"].x = load_biomed_feat(
        len(target_df),
        target_df["target_id"],
        "target_id",
        "target_emb_biomedbert1.npy",
        "target_ids1.csv",
    )
    data["reaction"].x = load_biomed_feat(
        len(reac_df),
        reac_df["adr_id"],
        "adr_id",
        "adr_emb_biomedbert1.npy",
        "adr_ids1.csv",
    )

    # medicine node features are aggregated later in prepare_unified_features
    dim_med = data["herb"].x.size(1)
    data["medicine"].x = torch.randn(len(med_df), dim_med)

    # medicine - contains - herb (dosage)
    mh_df = pd.read_csv(os.path.join(BASE_PATH, "HerbData/medicine_herb/edge_medicine_herb1.csv"))
    data["medicine", "contains", "herb"].edge_index = torch.from_numpy(
        np.stack([mh_df["medicine_id"].values, mh_df["herb_id"].values], axis=0)
    ).long()
    if "dosage" in mh_df.columns and use_dosage:
        # dosage_mode options:
        # - raw_log      : log1p(dosage)
        # - log_share    : log1p(dosage) normalized within each medicine
        # - linear_share : dosage normalized within each medicine
        # - none         : all ones
        dose_mode = (dosage_mode or "log_share").strip().lower()
        dose_raw = pd.to_numeric(mh_df["dosage"], errors="coerce").fillna(0.0).clip(lower=0.0)

        if dose_mode == "none":
            data["medicine", "contains", "herb"].edge_weight = torch.ones(mh_df.shape[0])
        elif dose_mode == "raw_log":
            dose_log = np.log1p(dose_raw.to_numpy(dtype=np.float32))
            data["medicine", "contains", "herb"].edge_weight = torch.from_numpy(dose_log).float()
        elif dose_mode in ("log_share", "linear_share"):
            if dose_mode == "log_share":
                base = np.log1p(dose_raw.to_numpy(dtype=np.float32))
            else:
                base = dose_raw.to_numpy(dtype=np.float32)

            tmp = pd.DataFrame({"medicine_id": mh_df["medicine_id"].values, "base": base})
            med_sum = tmp.groupby("medicine_id")["base"].transform("sum").to_numpy(dtype=np.float32)
            med_edge_count = tmp.groupby("medicine_id")["base"].transform("size").to_numpy(dtype=np.float32)
            normalized = np.divide(
                base,
                med_sum,
                out=np.zeros_like(base, dtype=np.float32),
                where=med_sum > 0,
            )
            zero_sum_mask = med_sum <= 0
            if np.any(zero_sum_mask):
                normalized[zero_sum_mask] = 1.0 / np.maximum(med_edge_count[zero_sum_mask], 1.0)
            data["medicine", "contains", "herb"].edge_weight = torch.from_numpy(normalized).float()
        else:
            raise ValueError(
                f"Unsupported dosage_mode={dosage_mode!r}. "
                "Use one of: 'none', 'raw_log', 'log_share', 'linear_share'."
            )
    else:
        data["medicine", "contains", "herb"].edge_weight = torch.ones(mh_df.shape[0])

    # herb - contains - ingredient
    hi_df = pd.read_csv(os.path.join(BASE_PATH, "herb_ingredient/edge_herb_ingredient1.csv"))
    hi_df = hi_df[(hi_df["ingredient_id"] >= 0) & (hi_df["ingredient_id"] < num_ingredient)]
    data["herb", "contains", "ingredient"].edge_index = torch.from_numpy(
        np.stack([hi_df["herb_id"].values, hi_df["ingredient_id"].values], axis=0)
    ).long()

    # ingredient - interacts - target
    it_df = pd.read_csv(os.path.join(BASE_PATH, "ingredient_target/edge_ingredient_target1.csv"))
    it_df = it_df[(it_df["ingredient_id"] >= 0) & (it_df["ingredient_id"] < num_ingredient)]
    it_df["target_id"] = it_df["target_id"].map(target_map)
    data["ingredient", "interacts", "target"].edge_index = torch.from_numpy(
        np.stack([it_df["ingredient_id"].values, it_df["target_id"].values], axis=0)
    ).long()

    # target - causes - reaction
    tr_df = pd.read_csv(os.path.join(BASE_PATH, "adr/edge_target_adr1.csv"))
    tr_df["target_id"] = tr_df["target_id"].map(target_map)
    tr_df["adr_id"] = tr_df["adr_id"].map(reac_map)
    data["target", "causes", "reaction"].edge_index = torch.from_numpy(
        np.stack([tr_df["target_id"].values, tr_df["adr_id"].values], axis=0)
    ).long()

    # target link prediction edge: medicine - has_reaction - reaction
    mr_df = pd.read_csv(os.path.join(BASE_PATH, "adr/edge_medicine_adr1.csv"))
    mr_df["adr_id"] = mr_df["adr_id"].map(reac_map)
    data["medicine", "has_reaction", "reaction"].edge_index = torch.stack(
        [
            torch.from_numpy(mr_df["medicine_id"].values.copy()),
            torch.from_numpy(mr_df["adr_id"].values.copy()),
        ],
        dim=0,
    ).long()

    def _emb_files_ok(emb_name: str, ids_name: str) -> bool:
        return os.path.isfile(os.path.join(biomed_dir, emb_name)) and os.path.isfile(
            os.path.join(biomed_dir, ids_name)
        )

    chem_loaded = os.path.isfile(chemberta_path) and os.path.isfile(chemberta_ids_path)
    chem_extra = ""
    if chem_loaded:
        z = int((data["ingredient"].x.abs().sum(dim=1) == 0).sum().item())
        chem_extra = f" | all-zero rows (no SMILES / id not in emb index) {z}/{num_ingredient}"
    print(
        f"[load_tcm_full_data] BASE_PATH={BASE_PATH}\n"
        f"  ChemBERTa ingredient: {'file' if chem_loaded else 'MISSING -> random'} "
        f"| x shape {tuple(data['ingredient'].x.shape)}{chem_extra}"
    )
    for nodetype, emb_f, ids_f in (
        ("herb", "herb_emb_biomedbert1.npy", "herb_ids1.csv"),
        ("target", "target_emb_biomedbert1.npy", "target_ids1.csv"),
        ("reaction", "adr_emb_biomedbert1.npy", "adr_ids1.csv"),
    ):
        ok = _emb_files_ok(emb_f, ids_f)
        extra = ""
        if ok:
            z = int((data[nodetype].x.abs().sum(dim=1) == 0).sum().item())
            extra = f" | all-zero rows (id not in emb index) {z}/{data[nodetype].num_nodes}"
        print(
            f"  BiomedBERT {nodetype}: {'file' if ok else 'MISSING -> random'} "
            f"| x shape {tuple(data[nodetype].x.shape)}{extra}"
        )

    return data, reac_map


@torch.no_grad()
def prepare_unified_features(data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare aggregated node features used by all models:
    - med_feat: dosage-weighted aggregation from (medicine -> contains -> herb -> ingredient)
    - reac_feat: aggregation from (reaction <- causes <- target <- interacts <- ingredient)
    - target_feat: aggregation from (target <- interacts <- ingredient)
    - herb_feat: mean-pooled ingredient (ChemBERTa) embeddings per herb via (herb <- contains <- ingredient)

    This matches the original project's data logic; herb_feat is exposed for BiomedBERT + structure fusion on herbs.
    """
    # ingredient aggregation to herb space
    h_i = data["herb", "contains", "ingredient"].edge_index
    h_feat = torch.zeros((data["herb"].num_nodes, data["ingredient"].x.size(1)))
    h_feat.index_add_(0, h_i[0], data["ingredient"].x[h_i[1]])
    h_count = torch.zeros(data["herb"].num_nodes).index_add_(0, h_i[0], torch.ones(h_i.size(1)))
    h_feat = h_feat / (h_count.view(-1, 1) + 1e-6)

    # dosage-weighted aggregation from herb to medicine
    m_h = data["medicine", "contains", "herb"].edge_index
    dose = data["medicine", "contains", "herb"].edge_weight
    m_feat = torch.zeros((data["medicine"].num_nodes, h_feat.size(1)))
    m_weight_sum = torch.zeros(data["medicine"].num_nodes).index_add_(0, m_h[0], dose)
    m_feat.index_add_(0, m_h[0], h_feat[m_h[1]] * dose.view(-1, 1))
    med_feat = m_feat / (m_weight_sum.view(-1, 1) + 1e-6)

    # ingredient -> target
    i_t = data["ingredient", "interacts", "target"].edge_index
    t_feat = torch.zeros((data["target"].num_nodes, data["ingredient"].x.size(1)))
    t_feat.index_add_(0, i_t[1], data["ingredient"].x[i_t[0]])
    t_count = torch.zeros(data["target"].num_nodes).index_add_(0, i_t[1], torch.ones(i_t.size(1)))
    t_feat = t_feat / (t_count.view(-1, 1) + 1e-6)

    # target -> reaction
    t_r = data["target", "causes", "reaction"].edge_index
    r_feat = torch.zeros((data["reaction"].num_nodes, t_feat.size(1)))
    r_feat.index_add_(0, t_r[1], t_feat[t_r[0]])
    r_count = torch.zeros(data["reaction"].num_nodes).index_add_(0, t_r[1], torch.ones(t_r.size(1)))
    reac_feat = r_feat / (r_count.view(-1, 1) + 1e-6)

    # if some reactions have zero count, fall back to ones (matches original).
    mask = r_count == 0
    if mask.any():
        reac_feat[mask] = torch.ones(reac_feat.size(1))

    # h_feat: per-herb mean-pooled ingredient (ChemBERTa) embeddings
    return med_feat, reac_feat, t_feat, h_feat

