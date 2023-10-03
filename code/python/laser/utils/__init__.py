import numpy as np
import torch

from laser import resource_path


def get_context_features(layer_idxs, inst_feat, num_objs, num_vars):
    max_lidx = np.max(layer_idxs)
    context = []
    for inst_idx, lidx in enumerate(layer_idxs):
        _inst_feat = inst_feat[inst_idx, :lidx, :]

        ranks = (torch.arange(lidx).reshape(-1, 1) + 1) / num_vars
        _context = torch.concat((_inst_feat, ranks), axis=1)

        ranks_pad = torch.zeros(max_lidx - _inst_feat.shape[0], num_objs + 2)
        _context = torch.concat((_context, ranks_pad), axis=0)

        context.append(_context)
    context = torch.stack(context)

    return context
def update_scores(scores_df, scores, tp, fp, tn, fn):
    tp += scores[:, 1].sum()
    fp += scores[:, 2].sum()
    tn += scores[:, 3].sum()
    fn += scores[:, 4].sum()
    for i in range(scores.shape[0]):
        layer = int(scores[i][0])
        scores_df.loc[layer, "TP"] += scores[i][1]
        scores_df.loc[layer, "FP"] += scores[i][2]
        scores_df.loc[layer, "TN"] += scores[i][3]
        scores_df.loc[layer, "FN"] += scores[i][4]
        scores_df.loc[layer, "Support"] += scores[i][5]

    return scores_df, tp, fp, tn, fn


def calculate_accuracy(tp, fp, tn, fn):
    correct = tp + tn
    total = tp + fp + tn + fn
    return correct / total, correct, total


def print_result(epoch,
                 split,
                 pid=None,
                 acc=None,
                 correct=None,
                 total=None,
                 inst_per_step=None,
                 is_best=None):
    is_best_str = " -- BEST ACC" if is_best else ""
    if pid is not None and inst_per_step is not None:
        print(f"\tEpoch: {epoch}, Inst: {pid}-{pid + inst_per_step}, "
              f"{split} acc: {acc:.2f}, {correct}, {total}")
    else:
        print(f"\tEpoch: {epoch}, {split} acc: {acc:.2f}, {correct}, {total} {is_best_str}")


def get_log_dir_name(cfg):
    checkpoint_str = f"{cfg.prob.name}-{cfg.prob.size}/"

    if cfg.train.flag_layer_penalty:
        checkpoint_str += f"{cfg.train.layer_penalty}-"
    if cfg.train.flag_label_penalty:
        checkpoint_str += f"{int(cfg.train.label_penalty * 10)}-"

    checkpoint_str += f"{cfg.train.neg_pos_ratio}-"
    if cfg.val.neg_pos_ratio < 0:
        checkpoint_str += f"n{int(-1 * cfg.val.neg_pos_ratio)}-"
    else:
        checkpoint_str += f"{cfg.val.neg_pos_ratio}-"

    checkpoint_str += f"{cfg.prob.order}-{cfg.prob.layer_norm_const}-{cfg.prob.state_norm_const}/"

    return checkpoint_str


def checkpoint(cfg, split, epoch=None, model=None, scores_df=None, is_best=None):
    checkpoint_dir = resource_path / "experiments/"
    checkpoint_str = get_log_dir_name(cfg)
    checkpoint_str += f"{cfg[split].log_dir}"
    checkpoint_dir /= checkpoint_str
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    if model is not None:
        model_name = f"model_{epoch}.ckpt"
        model_path = checkpoint_dir / model_name
        torch.save(model.state_dict(), model_path)

        if is_best:
            model_name = f"model_best.ckpt"
            model_path = checkpoint_dir / model_name
            torch.save(model.state_dict(), model_path)

    if scores_df is not None:
        score_name = f"scores_{epoch}.csv"
        scores_df_name = checkpoint_dir / score_name
        scores_df.to_csv(scores_df_name, index=False)
        if is_best:
            scores_df_name = checkpoint_dir / "scores_best.csv"
            scores_df.to_csv(scores_df_name, index=False)

        # Normalize scores
        scores_df["NSupport"] = (scores_df["TP"] + scores_df["FP"] +
                                 scores_df["TN"] + scores_df["FN"]) - scores_df["Support"]
        scores_df["TP"] /= scores_df["Support"]
        scores_df["FP"] /= scores_df["NSupport"]
        scores_df["TN"] /= scores_df["NSupport"]
        scores_df["FN"] /= scores_df["Support"]

        score_name = f"scores_norm_{epoch}.csv"
        scores_df_name = checkpoint_dir / score_name
        scores_df.to_csv(scores_df_name, index=False)
        if is_best:
            scores_df_name = checkpoint_dir / "scores_norm_best.csv"
            scores_df.to_csv(scores_df_name, index=False)


def checkpoint_test(cfg, scores_df):
    checkpoint_dir = resource_path / "experiments/"
    checkpoint_str = get_log_dir_name(cfg)
    checkpoint_str += f"{cfg.test.log_dir}"
    checkpoint_dir /= checkpoint_str
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    scores_df_name = checkpoint_dir / "scores.csv"
    scores_df.to_csv(scores_df_name, index=False)

    # Normalize scores
    scores_df["NSupport"] = (scores_df["TP"] + scores_df["FP"] +
                             scores_df["TN"] + scores_df["FN"]) - scores_df["Support"]
    scores_df["TP"] /= scores_df["Support"]
    scores_df["FP"] /= scores_df["NSupport"]
    scores_df["TN"] /= scores_df["NSupport"]
    scores_df["FN"] /= scores_df["Support"]
    score_name = f"scores_norm.csv"
    scores_df_name = checkpoint_dir / score_name
    scores_df.to_csv(scores_df_name, index=False)
