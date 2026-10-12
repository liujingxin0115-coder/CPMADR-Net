from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GINEConv,
    GraphConv,
    HGTConv,
    HeteroConv,
    Linear,
    RGCNConv,
    SAGEConv,
)

from .data import DOSAGE_EDGE_TYPE


class UniversalBaselineAllEdge(torch.nn.Module):
    """
    Universal model that reproduces the project's baseline architectures:
    - LR / MLP: use concatenated (medicine, reaction) node embeddings + classifier
    - GCN / GraphEdgeDosage: HeteroConv + GraphConv per edge type (PyG does not support Kipf GCNConv on bipartite
      hetero edges; GraphConv is the documented alternative). Optional edge_weight for GraphEdgeDosage.
    - RGCN: RGCNConv on `data.to_homogeneous()` (relation-specific transforms; Schlichtkrull-style), then split
      embeddings back to node types. Does not use dosage edge weights.
    - SAGE: HeteroConv + SAGEConv per edge type
    - GIN / RGAT: use GINEConv / GATConv with edge_dim=1 using dosage edge_attr for dosage edges
    - HGT: use HGTConv layers
    - Transformer: edge-wise transformer over [med_emb, reac_emb]

    Feature usage (top-paper common setting):
    - medicine: structural dosage-weighted feature (med_feat) as there is no explicit BERT medicine embedding in this project
    - herb / reaction / target: fuse
        (1) node-level BiomedBERT embedding from `data_obj[node_type].x` (herb/reaction/target)
        (2) structural aggregation from ingredient metapaths (`herb_feat`, `reac_feat`, `target_feat`) via `init_*_x`
    """

    def __init__(
        self,
        model_type: str,
        hidden_channels: int,
        metadata,
        data_obj,
        init_med_x,
        init_reac_x,
        init_target_x,
        init_herb_x,
        num_conv_layers: int = 2,
        transformer_num_layers: int = 2,
        transformer_nhead: int = 4,
    ):
        super().__init__()
        self.model_type = model_type
        self.num_conv_layers = int(num_conv_layers)

        self.med_lin = Linear(init_med_x.size(1), hidden_channels)
        # init_reac_x / init_target_x are structural aggregated features (from ingredient metapaths)
        self.reac_lin = Linear(init_reac_x.size(1), hidden_channels)
        self.target_lin = Linear(init_target_x.size(1), hidden_channels)

        # data_obj["reaction"].x / data_obj["target"].x are node-level BiomedBERT embeddings
        self.reac_bert_lin = Linear(data_obj["reaction"].x.size(1), hidden_channels)
        self.target_bert_lin = Linear(data_obj["target"].x.size(1), hidden_channels)

        # fuse BERT and structural features into a single hidden representation
        self.reac_fuse = Linear(hidden_channels * 2, hidden_channels)
        self.target_fuse = Linear(hidden_channels * 2, hidden_channels)
        self.herb_fuse = Linear(hidden_channels * 2, hidden_channels)

        # store raw features for index-based access in heterogeneous batches
        self.register_buffer("raw_med_x", init_med_x)
        self.register_buffer("raw_reac_x", init_reac_x)
        self.register_buffer("raw_target_x", init_target_x)
        self.register_buffer("raw_herb_x", init_herb_x)

        # herb: BiomedBERT(node) + mean-pooled ingredient (ChemBERTa) on herb–ingredient edges
        self.herb_bert_lin = Linear(data_obj["herb"].x.size(1), hidden_channels)
        self.herb_struct_lin = Linear(init_herb_x.size(1), hidden_channels)
        self.ingred_lin = Linear(data_obj["ingredient"].x.size(-1), hidden_channels)

        self.convs = torch.nn.ModuleList()
        if model_type not in ["LR", "MLP", "Transformer"]:
            for _ in range(self.num_conv_layers):
                if model_type == "HGT":
                    # Some PyG versions restrict edge_attr in HGTConv; keep as original behavior.
                    self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, heads=4))
                elif model_type == "RGCN":
                    num_rel = len(metadata[1])
                    self.convs.append(
                        RGCNConv(
                            hidden_channels,
                            hidden_channels,
                            num_relations=num_rel,
                            aggr="sum",
                        )
                    )
                else:
                    conv_dict = {}
                    for etype in metadata[1]:
                        if model_type == "GIN":
                            gin_nn = torch.nn.Sequential(
                                Linear(hidden_channels, hidden_channels),
                                torch.nn.ReLU(),
                                Linear(hidden_channels, hidden_channels),
                            )
                            conv_dict[etype] = GINEConv(gin_nn, edge_dim=1)
                        elif model_type == "GCN":
                            conv_dict[etype] = GraphConv(hidden_channels, hidden_channels)
                        elif model_type == "GraphEdgeDosage":
                            conv_dict[etype] = GraphConv(hidden_channels, hidden_channels)
                        elif model_type == "SAGE":
                            conv_dict[etype] = SAGEConv(hidden_channels, hidden_channels)
                        elif model_type == "RGAT":
                            conv_dict[etype] = GATConv(
                                hidden_channels,
                                hidden_channels,
                                heads=4,
                                concat=False,
                                add_self_loops=False,
                                edge_dim=1,
                            )
                    self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Transformer encoder operates on each (medicine, reaction) edge pair.
        if model_type == "Transformer":
            if hidden_channels % transformer_nhead != 0:
                raise ValueError(
                    f"hidden_channels ({hidden_channels}) must be divisible by transformer_nhead ({transformer_nhead})"
                )
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_channels, nhead=transformer_nhead, batch_first=True
            )
            self.trans_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
            self.classifier_pair = Linear(hidden_channels, 1)

        self.classifier = (
            Linear(hidden_channels * 2, 1)
            if model_type == "LR"
            else torch.nn.Sequential(
                Linear(hidden_channels * 2, hidden_channels),
                torch.nn.ReLU(),
                Linear(hidden_channels, 1),
            )
        )

    def _get_edge_dicts(self, data) -> Tuple[Dict, Dict]:
        """
        Build edge_weight_dict / edge_attr_dict:
        - dosage edge uses its true edge_weight, shaped as (E, ) and (E,1)
        - all other relations use ones placeholders
        """
        edge_weight_dict, edge_attr_dict = {}, {}
        for etype, store in data.edge_items():
            num_e = store.edge_index.size(1)
            if etype == DOSAGE_EDGE_TYPE:
                dose = getattr(store, "edge_weight", None)
                if dose is None:
                    dose = torch.ones(num_e, device=store.edge_index.device)
                edge_weight_dict[etype] = dose
                edge_attr_dict[etype] = dose.view(-1, 1)
            else:
                ones = torch.ones(num_e, device=store.edge_index.device)
                edge_weight_dict[etype] = ones
                edge_attr_dict[etype] = ones.view(-1, 1)
        return edge_weight_dict, edge_attr_dict

    @staticmethod
    def _hetero_x_dict_to_homogeneous(data, x_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write projected features into the batch, then merge for RGCNConv."""
        for ntype in x_dict.keys():
            data[ntype].x = x_dict[ntype]
        hom = data.to_homogeneous()
        return hom.x, hom.edge_index, hom.edge_type

    @staticmethod
    def _homogeneous_x_to_dict(x: torch.Tensor, data) -> Dict[str, torch.Tensor]:
        off = 0
        out: Dict[str, torch.Tensor] = {}
        for ntype in data.node_types:
            n = data[ntype].num_nodes
            out[ntype] = x[off : off + n]
            off += n
        return out

    def forward(self, data):
        # Build initial node features for heterogeneous message passing.
        # - medicine: structural med_feat (dosage-weighted)
        # - herb / reaction / target: fuse node-level BiomedBERT with structural (ingredient-based) aggregation
        x_dict = {
            "medicine": self.med_lin(self.raw_med_x[data["medicine"].n_id]),
            "ingredient": self.ingred_lin(data["ingredient"].x),
        }

        herb_struct = self.herb_struct_lin(self.raw_herb_x[data["herb"].n_id])
        herb_bert = self.herb_bert_lin(data["herb"].x)
        x_dict["herb"] = self.herb_fuse(torch.cat([herb_bert, herb_struct], dim=-1))

        reac_struct = self.reac_lin(self.raw_reac_x[data["reaction"].n_id])
        reac_bert = self.reac_bert_lin(data["reaction"].x)
        x_dict["reaction"] = self.reac_fuse(torch.cat([reac_bert, reac_struct], dim=-1))

        target_struct = self.target_lin(self.raw_target_x[data["target"].n_id])
        target_bert = self.target_bert_lin(data["target"].x)
        x_dict["target"] = self.target_fuse(torch.cat([target_bert, target_struct], dim=-1))

        # Transformer-only path
        if self.model_type == "Transformer":
            edge_idx = data["medicine", "has_reaction", "reaction"].edge_label_index
            pair_seq = torch.stack(
                [x_dict["medicine"][edge_idx[0]], x_dict["reaction"][edge_idx[1]]],
                dim=1,  # (E, 2, H)
            )
            enc = self.trans_encoder(pair_seq)  # (E, 2, H)
            pooled = enc.mean(dim=1)  # (E, H)
            return self.classifier_pair(pooled).view(-1)

        if self.model_type not in ["MLP", "LR"]:
            if self.model_type == "RGCN":
                x, edge_index, edge_type = self._hetero_x_dict_to_homogeneous(data, x_dict)
                for conv in self.convs:
                    x = conv(x, edge_index, edge_type)
                    x = F.elu(x)
                x_dict = self._homogeneous_x_to_dict(x, data)
            else:
                edge_weight_dict, edge_attr_dict = self._get_edge_dicts(data)
                for conv in self.convs:
                    if self.model_type == "GraphEdgeDosage":
                        x_dict = conv(
                            x_dict,
                            data.edge_index_dict,
                            edge_weight_dict=edge_weight_dict,
                        )
                    elif self.model_type in ("GIN", "RGAT"):
                        x_dict = conv(
                            x_dict,
                            data.edge_index_dict,
                            edge_attr_dict=edge_attr_dict,
                        )
                    else:
                        x_dict = conv(x_dict, data.edge_index_dict)
                    x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        edge_idx = data["medicine", "has_reaction", "reaction"].edge_label_index
        return self.classifier(
            torch.cat([x_dict["medicine"][edge_idx[0]], x_dict["reaction"][edge_idx[1]]], dim=-1)
        ).view(-1)

