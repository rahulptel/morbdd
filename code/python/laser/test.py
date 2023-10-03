import hydra
import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import StatScores

from laser import resource_path
from laser.data import dataloader_factory
from laser.model import model_factory
from laser.utils import calculate_accuracy
from laser.utils import checkpoint_test
from laser.utils import get_context_features
from laser.utils import get_log_dir_name
from laser.utils import update_scores

statscores = StatScores(task='binary', multidim_average='samplewise')


def initialize_model(cfg, model):
    # Only one of the two can be true at a time
    assert (cfg.test.epoch and cfg.test.best) is False

    experiment_dir = resource_path / "experiments"
    checkpoint_dir_name = get_log_dir_name(cfg)
    if cfg.test.best:
        model_path = experiment_dir / checkpoint_dir_name / "val_log/model_best.ckpt"
    elif cfg.test.epoch:
        model_path = experiment_dir / checkpoint_dir_name / f"val_log/model_{cfg.test.epoch_idx}.ckpt"
    else:
        raise ValueError("Invalid model initialization params!")

    model.load_state_dict(torch.load(model_path))


def test_step(model, dataloader, num_objs, num_vars, layer_norm_const=100):
    model.eval()
    scores = None

    with torch.no_grad():
        # Train over this instance
        for bid, batch in enumerate(dataloader):
            nf, pf, inst_feat, label = batch['nf'], batch['pf'], batch['if'], batch['label']

            # Get layer ids of the nodes in the current batch
            lidxs_t = nf[:, 1] * layer_norm_const
            lidxs = list(map(int, lidxs_t.numpy()))
            context_feat = get_context_features(lidxs, inst_feat, num_objs, num_vars)

            preds = model(inst_feat, context_feat, nf, pf)

            score = statscores(preds.clone().detach(), label)
            layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
            scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)

    return scores.numpy()


def test_loop(cfg, model, dataloader):
    tp, fp, tn, fn = 0, 0, 0, 0
    scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
                             np.zeros((cfg.prob.num_vars, 5))), axis=1)
    scores_df = pd.DataFrame(scores, columns=["layer", "TP", "FP", "TN", "FN", "Support"])
    for idx, pid in enumerate(range(cfg.val.from_pid,
                                    cfg.val.to_pid,
                                    cfg.val.inst_per_step)):
        _to_pid = pid + cfg.val.inst_per_step
        test_dataloader = dataloader.get("test", pid, _to_pid)
        scores = test_step(model,
                           test_dataloader,
                           cfg.prob.num_objs,
                           cfg.prob.num_vars,
                           layer_norm_const=cfg.prob.layer_norm_const)
        scores_df, tp, fp, tn, fn = update_scores(scores_df, scores, tp, fp, tn, fn)

    return scores_df, tp, fp, tn, fn


@hydra.main(version_base="1.2", config_path="./configs", config_name="cfg.yaml")
def main(cfg):
    # Get model, optimizer and loss function
    model_cls = model_factory.get("ParetoStatePredictor")
    model = model_cls(cfg.mdl)
    initialize_model(cfg, model)

    dataloader_cls = dataloader_factory.get(cfg.prob.name)
    dataloader = dataloader_cls(cfg)

    test_result = test_loop(cfg, model, dataloader)
    scores_df, tp, fp, tn, fn = test_result
    acc, correct, total = calculate_accuracy(tp, fp, tn, fn)

    # Log training accuracy metrics
    print(f"\tTest acc: {acc:.2f}, {correct}, {total}")
    checkpoint_test(cfg, scores_df)


if __name__ == "__main__":
    main()
