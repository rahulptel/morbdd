import json
import signal

import hydra
import libbddenvv1
import pandas as pd

from morbdd import resource_path
from morbdd.utils import get_instance_data
from morbdd.utils import get_static_order
from morbdd.utils import handle_timeout

import multiprocessing as mp


def worker(rank, cfg):
    env = libbddenvv1.BDDEnv()
    signal.signal(signal.SIGALRM, handle_timeout)

    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.num_processes):
        data = get_instance_data(cfg.prob, cfg.size, cfg.split, pid)
        order = get_static_order(cfg.prob, cfg.order_type, data)

        env.set_knapsack_inst(cfg.num_vars,
                              cfg.num_objs,
                              data['value'],
                              data['weight'],
                              data['capacity'])
        env.initialize_run(cfg.problem_type,
                           cfg.preprocess,
                           cfg.bdd_type,
                           cfg.maxwidth,
                           order)

        sol = None
        try:
            signal.alarm(1800)
            env.compute_pareto_frontier()
            sol = {"x": env.x_sol,
                   "z": env.z_sol,
                   "ot": cfg.order_type}
        except TimeoutError as exc:
            print(f"PF not computed within 1800s for pid {pid}")
        else:
            print(f"PF computed successfully for pid {pid}")
        signal.alarm(0)

        if sol is not None:
            file_path = resource_path / f"sols/{cfg.prob}/{cfg.size}/{cfg.split}"
            file_path.mkdir(parents=True, exist_ok=True)
            file_path /= f"{pid}.json"
            with open(file_path, "w") as fp:
                json.dump(sol, fp)

            df = pd.DataFrame([[cfg.size, pid, cfg.split,
                                env.nnds,
                                env.initial_node_count,
                                env.reduced_node_count,
                                env.initial_arcs_count,
                                env.reduced_arcs_count,
                                env.num_comparisons,
                                env.time_result["compilation"],
                                env.time_result["reduction"],
                                env.time_result["pareto"]
                                ]], columns=["size", "pid", "split", "nnds", "inc", "rnc", "iac", "rac", "Comp.",
                                             "compilation", "reduction", "pareto"])
            df.to_csv(file_path.parent / f"{pid}.csv", index=False)


@hydra.main(version_base="1.2", config_path="./configs", config_name="bdd_dataset.yaml")
def main(cfg):
    # pool = mp.Pool(processes=cfg.num_processes)
    # results = []
    # for rank in range(cfg.num_processes):
    #     results.append(pool.apply_async(worker, args=(rank, cfg)))
    # results = [r.get() for r in results]

    worker(0, cfg)


if __name__ == '__main__':
    main()
