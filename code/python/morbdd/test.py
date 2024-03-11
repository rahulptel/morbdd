import sys
import time

import hydra
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import StatScores

from laser import resource_path
from laser.model import model_factory
from laser.utils import calculate_accuracy
from laser.utils import checkpoint_test
from laser.utils import get_context_features
from laser.utils import get_log_dir_name
from laser.utils import get_split_datasets
from laser.utils import set_device
from laser.utils import update_scores

statscores = StatScores(task='binary', multidim_average='samplewise')

dataset_dict = {}


def initialize_model(cfg, model):
    # Only one of the two can be true at a time
    assert (cfg.test.epoch and cfg.test.best) is False

    experiment_dir = resource_path / "experiments"
    checkpoint_dir_name = get_log_dir_name(cfg.prob.name,
                                           cfg.prob.size,
                                           cfg.train.flag_layer_penalty,
                                           cfg.train.layer_penalty,
                                           cfg.train.flag_label_penalty,
                                           cfg.train.label_penalty,
                                           cfg.train.neg_pos_ratio,
                                           cfg.prob.order,
                                           cfg.prob.layer_norm_const,
                                           cfg.prob.state_norm_const)
    if cfg.test.best:
        model_path = experiment_dir / checkpoint_dir_name / "val_log/model_best.ckpt"
    elif cfg.test.epoch:
        model_path = experiment_dir / checkpoint_dir_name / f"val_log/model_{cfg.test.epoch_idx}.ckpt"
    else:
        raise ValueError("Invalid model initialization params!")

    model.load_state_dict(torch.load(model_path))


def test_step(model, dataloader, num_objs, num_vars, device, layer_norm_const=100):
    model.eval()
    scores = None
    num_batches = len(dataloader)
    with torch.no_grad():
        for bid, batch in enumerate(dataloader):
            sys.stdout.write("\r")
            sys.stdout.write(f"\t\tProgress: {((bid + 1) / num_batches) * 100:.2f}%")
            sys.stdout.flush()

            nf, pf, inst_feat, label = batch['nf'], batch['pf'], batch['if'], batch['label']

            # Get layer ids of the nodes in the current batch
            lidxs_t = torch.round(nf[:, 1] * layer_norm_const)
            lidxs = list(map(int, lidxs_t.cpu().numpy()))
            context_feat = get_context_features(lidxs, inst_feat, num_objs, num_vars, device)

            preds = model(inst_feat, context_feat, nf, pf)

            score = statscores(preds.clone().detach(), label)
            score = score if len(score.shape) > 1 else score.unsqueeze(0)
            layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
            scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)
    print()

    return scores.cpu().numpy()


def test_loop(cfg, model, device):
    global dataset_dict
    print("\tTest loop")
    tp, fp, tn, fn = 0, 0, 0, 0
    scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
                             np.zeros((cfg.prob.num_vars, 5))), axis=1)
    scores_df = pd.DataFrame(scores, columns=["layer", "TP", "FP", "TN", "FN", "Support"])
    scores_df["layer"] = list(map(int, scores_df["layer"]))

    for idx, pid in enumerate(range(cfg.test.from_pid,
                                    cfg.test.to_pid,
                                    cfg.test.inst_per_step)):
        _to_pid = pid + cfg.test.inst_per_step

        # test_dataloader = dataloader.get("test", pid, _to_pid)
        _pids = list(range(pid, _to_pid))
        datasets, dataset_dict = get_split_datasets(_pids,
                                                    cfg.prob.name, cfg.prob.size, "test",
                                                    cfg.test.neg_pos_ratio,
                                                    cfg.test.min_samples,
                                                    dataset_dict,
                                                    device)
        if len(datasets) == 0:
            continue
        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False)
        print(f"\t\tDataset size: {len(dataset)}")
        print(f"\t\tNumber of batches: {len(dataloader)}")

        scores = test_step(model,
                           dataloader,
                           cfg.prob.num_objs,
                           cfg.prob.num_vars,
                           device,
                           layer_norm_const=cfg.prob.layer_norm_const)
        scores_df, tp, fp, tn, fn = update_scores(scores_df, scores, tp, fp, tn, fn)

    return scores_df, tp, fp, tn, fn


@hydra.main(version_base="1.2", config_path="./configs", config_name="cfg.yaml")
def main(cfg):  # Set device
    device = set_device(cfg.device)

    # Get model, optimizer and loss function
    model_cls = model_factory.get("ParetoStatePredictor")
    model = model_cls(cfg.mdl)
    model.to(device)
    initialize_model(cfg, model)

    start = time.time()
    test_result = test_loop(cfg, model, device)
    end = time.time()
    print("\tTest time ", end - start)
    scores_df, tp, fp, tn, fn = test_result
    acc, correct, total = calculate_accuracy(tp, fp, tn, fn)

    # Log training accuracy metrics
    print(f"\tTest acc: {acc:.2f}, {correct}, {total}")
    checkpoint_test(cfg, scores_df)


if __name__ == "__main__":
    main()
