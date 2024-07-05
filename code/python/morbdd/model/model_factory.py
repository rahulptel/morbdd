import torch
from morbdd import ResourcePaths as path


def get_model(cfg, model_path=None, mode=None):
    model = None
    best_model = None if model_path is None else torch.load(path.resource / "checkpoint" / model_path,
                                                            map_location="cpu")

    if cfg.prob.name == "indepset" and cfg.model_version == 1:
        from .pytorch_v1 import ParetoStatePredictorMIS

        model = ParetoStatePredictorMIS(encoder_type="transformer",
                                        n_node_feat=cfg.n_node_feat,
                                        n_edge_type=cfg.n_edge_type,
                                        d_emb=cfg.d_emb,
                                        top_k=cfg.top_k,
                                        n_blocks=cfg.n_blocks,
                                        n_heads=cfg.n_heads,
                                        dropout_token=cfg.dropout_token,
                                        bias_mha=cfg.bias_mha,
                                        dropout=cfg.dropout,
                                        bias_mlp=cfg.bias_mlp,
                                        h2i_ratio=cfg.h2i_ratio)

    elif cfg.prob.name == "indepset" and cfg.model_version == 2:
        from .pytorch_v2 import ParetoStatePredictorMIS

        model = ParetoStatePredictorMIS(encoder_type="transformer",
                                        n_node_feat=cfg.n_node_feat,
                                        n_edge_type=cfg.n_edge_type,
                                        d_emb=cfg.d_emb,
                                        top_k=cfg.top_k,
                                        n_blocks=cfg.n_blocks,
                                        n_heads=cfg.n_heads,
                                        dropout_token=cfg.dropout_token,
                                        bias_mha=cfg.bias_mha,
                                        dropout=cfg.dropout,
                                        bias_mlp=cfg.bias_mlp,
                                        h2i_ratio=cfg.h2i_ratio)

    elif cfg.prob.name == "indepset" and cfg.model_version == 3:
        from .pytorch_v3 import ParetoStatePredictorMIS

        model = ParetoStatePredictorMIS(encoder_type=cfg.encoder_type,
                                        n_node_feat=cfg.n_node_feat,
                                        n_edge_type=cfg.n_edge_type,
                                        d_emb=cfg.d_emb,
                                        top_k=cfg.top_k,
                                        n_blocks=cfg.n_blocks,
                                        n_heads=cfg.n_heads,
                                        dropout_token=cfg.dropout_token,
                                        dropout_attn=cfg.dropout_attn,
                                        dropout_proj=cfg.dropout_proj,
                                        dropout_mlp=cfg.dropout_mlp,
                                        bias_mha=cfg.bias_mha,
                                        bias_mlp=cfg.bias_mlp,
                                        h2i_ratio=cfg.h2i_ratio)

    assert model is not None

    if best_model:
        model.load_state_dict(best_model["model"])
    if mode == "eval":
        model.eval()

    return model
