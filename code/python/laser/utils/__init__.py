from laser import resource_path
import torch


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


def checkpoint(dir, epoch=None, split=None, model=None, scores_df=None):
    checkpoint_dir = resource_path / dir
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    if model is not None:
        model_name = f"model_{epoch}.ckpt" if split == "train" else "model_best.ckpt"
        model_path = checkpoint_dir / model_name
        torch.save(model.state_dict(), model_path)

    if scores_df is not None:
        score_name = f"scores_{epoch}.csv" if split == "train" else "scores_best.csv"
        scores_df_name = checkpoint_dir / score_name
        scores_df.to_csv(scores_df_name, index=False)

        # Normalize scores
        scores_df["NSupport"] = (scores_df["TP"] + scores_df["FP"] +
                                 scores_df["TN"] + scores_df["FN"]) - scores_df["Support"]
        scores_df["TP"] /= scores_df["Support"]
        scores_df["FP"] /= scores_df["NSupport"]
        scores_df["TN"] /= scores_df["NSupport"]
        scores_df["FN"] /= scores_df["Support"]

        score_name = f"scores_norm_{epoch}.csv" if split == "train" else "scores_norm_best.csv"
        scores_df_name = checkpoint_dir / score_name
        scores_df.to_csv(scores_df_name, index=False)
