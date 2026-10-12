from baselines.common.models import UniversalBaselineAllEdge

MODEL_TYPE = "RGAT"


def build_model(hidden_channels, metadata, data_obj, med_feat, reac_feat, target_feat, herb_feat):
    return UniversalBaselineAllEdge(
        MODEL_TYPE,
        hidden_channels,
        metadata,
        data_obj,
        med_feat,
        reac_feat,
        target_feat,
        herb_feat,
    )

