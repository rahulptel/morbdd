import json
import multiprocessing as mp

import hydra
import libbddenvv1
import numpy as np

from morbdd import resource_path
from morbdd.utils import get_instance_data
from morbdd.utils import get_order
from morbdd.utils import read_from_zip


def get_pareto_states_per_layer(weight, x):
    pareto_state_scores = []
    for i in range(1, x.shape[1]):
        x_partial = x[:, :i].reshape(-1, i)
        w_partial = weight[:i].reshape(i, 1)
        wt_dist = np.dot(x_partial, w_partial)
        pareto_state, pareto_score = np.unique(wt_dist, return_counts=True)
        pareto_score = pareto_score / pareto_score.sum()
        pareto_state_scores.append((pareto_state, pareto_score))

    return pareto_state_scores


def tag_bdd_states(bdd, pareto_state_scores):
    assert len(pareto_state_scores) == len(bdd)

    for l in range(len(bdd)):
        pareto_states, pareto_scores = pareto_state_scores[l]

        for n in bdd[l]:
            node_state = n["s"][0]
            index = np.where(pareto_states == node_state)[0]
            if len(index):
                n["pareto"] = 1
                n["score"] = pareto_scores[index[0]]
            else:
                n["pareto"] = 0
                n["score"] = 0

    return bdd


def worker(rank, cfg):
    env = libbddenvv1.BDDEnv()

    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.num_processes):
        # print(f"Processing pid {pid}...")
        # Read instance
        data = get_instance_data(cfg.prob, cfg.size, cfg.split, pid)
        order = get_order(cfg.prob, cfg.order_type, data)

        # print("\tReading sol...")
        archive = resource_path / f"sols/{cfg.prob}/{cfg.size}.zip"
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        sol = read_from_zip(archive, file, format="json")
        # Ignore instances not solved within time limit
        if sol is None:
            continue

        # Extract BDD before reduction
        # print("\tExtracting non-reduced BDD...")
        env.set_knapsack_inst(cfg.num_vars,
                              cfg.num_objs,
                              data['value'],
                              data['weight'],
                              data['capacity'])
        bdd = env.get_bdd(cfg.problem_type, order)

        # Label BDD
        # print("\tLabelling BDD...")
        weight = np.array(data['weight'])[order]
        pareto_state_scores = get_pareto_states_per_layer(weight, np.array(sol["x"]))
        bdd = tag_bdd_states(bdd, pareto_state_scores)

        # Save
        print(f"Saving BDD {pid}...")
        file_path = resource_path / f"bdds/{cfg.prob}/{cfg.size}/{cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.json"
        with open(file_path, "w") as fp:
            json.dump(bdd, fp)


@hydra.main(version_base="1.2", config_path="./configs", config_name="bdd_dataset.yaml")
def main(cfg):
    pool = mp.Pool(processes=cfg.num_processes)
    results = []

    for rank in range(cfg.num_processes):
        results.append(pool.apply_async(worker, args=(rank, cfg)))

    for r in results:
        r.get()


if __name__ == '__main__':
    main()
