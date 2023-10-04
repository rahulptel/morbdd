import json

import hydra
import libbddenvv1
import numpy as np

from laser import resource_path
from laser.utils import get_instance_data
from laser.utils import get_order
from laser.utils import read_from_zip


def get_pareto_states_per_layer(weight, x):
    paretoStates = []
    for i in range(1, x.shape[1]):
        x_partial = x[:, :i].reshape(-1, i)
        w_partial = weight[:i].reshape(i, 1)
        wt_dist = np.dot(x_partial, w_partial)
        paretoStates.append(np.unique(wt_dist))

    return paretoStates


def label_bdd(bdd, pareto_states):
    assert len(pareto_states) == len(bdd)

    for l in range(len(bdd)):
        for n in bdd[l]:
            if n['s'] in pareto_states[l]:
                n['l'] = 1
            else:
                n['l'] = 0

    return bdd


@hydra.main(version_base="1.2", config_path="./configs", config_name="generate_bdd_dataset.yaml")
def main(cfg):
    env = libbddenvv1.BDDEnv()

    for pid in range(cfg.from_pid, cfg.to_pid):
        # Read instance
        data = get_instance_data(cfg.prob, cfg.size, cfg.split, pid)
        order = get_order(cfg.prob, cfg.order_type, data)

        archive = resource_path / f"sols/{cfg.prob}/{cfg.size}.zip"
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        sol = read_from_zip(archive, file, format="json")

        # Extract BDD before reduction
        env.set_knapsack_inst(cfg.num_vars,
                              cfg.num_objs,
                              data['value'],
                              data['weight'],
                              data['capacity'])
        bdd = env.get_bdd(cfg.problem_type, order)

        # Label BDD
        weight = np.array(data['weight'])[order]
        pareto_states = get_pareto_states_per_layer(weight, np.array(sol["x"]))
        bdd = label_bdd(bdd, pareto_states)

        # Save
        file_path = resource_path / f"bdds/{cfg.prob}/{cfg.size}/{cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.json"
        with open(file_path, "w") as fp:
            json.dump(bdd, fp)


if __name__ == '__main__':
    main()
