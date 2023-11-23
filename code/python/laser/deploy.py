import json
import time
import zipfile

import hydra
import torch
from torch.utils.data import DataLoader

from laser import resource_path
from laser.model import model_factory
from laser.utils import get_context_features
from laser.utils import get_dataset
from laser.utils import get_log_dir_name
from laser.utils import set_device
import os

dataset_dict = {}

CONNECTED = 0
NOT_CONNECTED = 1
NOT_CONNECTED_EMPTY_LAYER = 2

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def initialize_model(cfg, model):
    # Only one of the two can be true at a time
    assert (cfg.deploy.epoch and cfg.deploy.best) is False

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


def get_preds(model, dataloader, num_objs, num_vars, device, layer_norm_const=100):
    all_preds = None
    model.eval()
    with torch.no_grad():
        for bid, batch in enumerate(dataloader):
            nf, pf, inst_feat, label = batch['nf'], batch['pf'], batch['if'], batch['label']

            # Get layer ids of the nodes in the current batch
            lidxs_t = torch.round(nf[:, 1] * layer_norm_const)
            lidxs = list(map(int, lidxs_t.cpu().numpy()))
            context_feat = get_context_features(lidxs, inst_feat, num_objs, num_vars, device)
            preds = model(inst_feat, context_feat, nf, pf)

            score = statscores(preds.clone().detach(), label)
            if all_preds is None:
                all_preds = preds.clone().detach()
            else:
                all_preds = torch.cat((all_preds, preds), axis=0)

    all_preds = all_preds.squeeze(1)
    return all_preds.cpu().numpy()


def get_predicted_pareto_states(problem, size, split, pid, preds, threshold):
    zf = zipfile.ZipFile(resource_path / f"bdds/{problem}/{size}.zip")
    bdd = json.load(zf.open(f"{size}/{split}/{pid}.json"))

    idx = 0
    pareto_states = []
    for lidx, layer in enumerate(bdd):
        _pareto_states = []
        is_connected = False

        for node in layer:
            if preds[idx] >= threshold:
                _pareto_states.append(node["s"][0])

                if lidx > 0 and not is_connected:
                    for p in node["op"]:
                        parent_state = bdd[lidx - 1][p]["s"][0]
                        if parent_state in pareto_states[-1]:
                            is_connected = True

                    for p in node["zp"]:
                        parent_state = bdd[lidx - 1][p]["s"][0]
                        if parent_state in pareto_states[-1]:
                            is_connected = True

        pareto_states.append(_pareto_states)

        # None of the nodes is predicted to be a Pareto State
        if len(_pareto_states) == 0:
            return NOT_CONNECTED_EMPTY_LAYER, pareto_states
        if lidx > 0 and not is_connected:
            return NOT_CONNECTED, pareto_states

    return CONNECTED, pareto_states


def get_partial_pareto_frontier(problem, size, split, pid, pareto_states):
    pass


def test_runtime_loop(cfg, model, device):
    global dataset_dict
    print("\tDeploy loop")

    for idx, pid in enumerate(range(cfg.deploy.from_pid, cfg.deploy.to_pid)):
        if pid not in dataset_dict:
            dataset = get_dataset(cfg.prob.name,
                                  cfg.prob.size,
                                  cfg.deploy.split,
                                  pid,
                                  cfg.train.neg_pos_ratio,
                                  cfg.train.min_samples,
                                  device)
            dataset_dict[pid] = dataset

        dataloader = DataLoader(dataset_dict[pid],
                                batch_size=cfg.deploy.batch_size,
                                shuffle=False)
        preds = get_preds(model,
                          dataloader,
                          cfg.prob.num_objs,
                          cfg.prob.num_vars,
                          device,
                          layer_norm_const=cfg.prob.layer_norm_const)

        status, pareto_states = get_predicted_pareto_states(cfg.prob.name,
                                                            cfg.prob.size,
                                                            cfg.deploy.split,
                                                            pid,
                                                            preds,
                                                            cfg.deploy.threshold)
        # if status == CONNECTED:
        #     result = get_partial_pareto_frontier(cfg.prob, cfg.size, cfg.split, pid, pareto_states)

        print(status, len(pareto_states))


@hydra.main(version_base="1.2", config_path="./configs", config_name="cfg.yaml")
def main(cfg):  # Set device
    device = set_device(cfg.device)

    # Get model, optimizer and loss function
    model_cls = model_factory.get("ParetoStatePredictor")
    model = model_cls(cfg.mdl)
    model.to(device)
    initialize_model(cfg, model)

    start = time.time()
    test_result = test_runtime_loop(cfg, model, device)
    end = time.time()


if __name__ == "__main__":
    main()
