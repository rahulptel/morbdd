import json
import multiprocessing as mp
import signal
import time
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd

from morbdd import Const as CONST
from morbdd import ResourcePaths as path
from morbdd.utils import get_instance_data
from morbdd.utils import get_static_order
from morbdd.utils import handle_timeout


def get_size(cfg):
    if cfg.problem_type == 1:
        return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
    elif cfg.problem_type == 2:
        if cfg.graph_type == "stidsen":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
        elif cfg.graph_type == "ba":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}-{cfg.prob.attach}"


def get_lib(bin, n_objs=3):
    libname = 'libbddenv'
    if bin == 'multiobj':
        libname += 'v1'
    elif bin == 'network':
        libname += f'v2o{n_objs}'

    print("Importing lib: ", libname)
    lib = __import__(libname)

    return lib


def set_instance(bin, env, problem_type, data, graph_type="stidsen"):
    if bin == 'multiobj':
        if problem_type == 1:
            env.set_inst(data['n_vars'],
                         data['n_objs'],
                         data['obj_coeffs'],
                         data['cons_coeffs'][0])
        else:
            print(f"Problem type {problem_type} not yet supported by {bin}")

    elif bin == 'network':
        if problem_type == 1 or problem_type == 3 or (problem_type == 2 and graph_type == "stidsen"):
            env.set_inst(data['n_vars'],
                         data['n_cons'],
                         data['n_objs'],
                         data['obj_coeffs'],
                         data['cons_coeffs'],
                         data['rhs'])
        elif problem_type == 2 and graph_type == "ba":
            # obj_coeffs = list(data['obj_coeffs'])
            # cons_coeffs = list(data['edges'])
            # n_objs = len(obj_coeffs)
            # n_vars = int(data['n_vars'])
            # n_cons = 0
            # rhs = []
            # env.set_inst(n_vars, n_cons, n_objs, obj_coeffs, cons_coeffs, rhs)
            env.set_inst_indepset(data["n_vars"],
                                  data["n_objs"],
                                  data["obj_coeffs"],
                                  data["edges"])
        # elif problem_type == 4:
        #     env.set_inst_absval(data["n_vars"],
        #                         data["n_objs"],
        #                         data["card"],
        #                         data["cons_coeffs"],
        #                         data["rhs"])
        # elif problem_type == 5:
        #     env.set_inst_tsp(data["n_vars"], data["n_objs"], data["obj_coeffs"])
        else:

            print(f"Problem type {problem_type} not yet supported by {bin}")


def initialize_run(bin, env, problem_type, preprocess, method, bdd_type, maxwidth, order, maximization=None,
                   dominance=None):
    if bin == 'multiobj':
        env.initialize(
            problem_type,
            preprocess,
            bdd_type,
            maxwidth,
            order
        )
    elif bin == 'network':
        env.reset(problem_type,
                  preprocess,
                  method,
                  maximization,
                  dominance,
                  bdd_type,
                  maxwidth,
                  order)


def preprocess_inst(bin, env, problem_type):
    if bin == "network":
        if problem_type == 1:
            env.preprocess_inst()


def initialize_dd_constructor(bin, env):
    if bin == "network":
        env.initialize_dd_constructor()


def get_dynamic_order(bin, env, problem_type, order_type, order):
    if bin == "network" and problem_type == 2:
        if order_type == "min_state":
            order = env.get_var_layer()

    return order


def get_pareto_states_per_layer_knapsack(weight, x):
    pareto_state_scores = []
    for i in range(1, x.shape[1]):
        x_partial = x[:, :i].reshape(-1, i)
        w_partial = weight[:i].reshape(i, 1)
        wt_dist = np.dot(x_partial, w_partial)
        pareto_state, pareto_score = np.unique(wt_dist, return_counts=True)
        pareto_score = pareto_score / pareto_score.sum()
        pareto_state_scores.append((pareto_state, pareto_score))

    return pareto_state_scores


def get_pareto_states_per_layer_indepset(order, x_sol, adj_list_comp):
    x_sol = np.array(x_sol)
    pareto_state_scores = []

    total = x_sol.shape[0]
    # print(set(list(x_sol[:, 0])))
    for i in range(1, x_sol.shape[1]):
        # Get partial sols upto level i in BDD
        partial_sols = x_sol[:, :i]
        uniques, counts = np.unique(partial_sols, axis=0, return_counts=True)

        pareto_states = []
        pareto_counts = []

        # Compute pareto state for each unique sol
        for unique_sol, count in zip(uniques, counts):
            _pareto_state = np.ones(x_sol.shape[1])
            for var_idx, is_active in enumerate(list(unique_sol)):
                var = order[var_idx]
                _pareto_state[var] = 0
                # print(var_idx, var, is_active)
                if is_active:
                    # print(adj_list_comp[var])
                    _pareto_state = np.logical_and(_pareto_state, adj_list_comp[var]).astype(int)
                    # print(_pareto_state)

            is_found = False
            for j, ps in enumerate(pareto_states):
                # print(_pareto_state, ps, np.array_equal(_pareto_state))
                if np.array_equal(_pareto_state, ps):
                    pareto_counts[j] += count
                    is_found = True
                    break
            if not is_found:
                # print(_pareto_state, "not found")
                pareto_states.append(_pareto_state)
                pareto_counts.append(count)

        for k in range(len(pareto_counts)):
            pareto_counts[k] /= total

        pareto_state_scores.append((pareto_states, pareto_counts))

    return pareto_state_scores


def get_pareto_state_scores_per_layer(problem_type, data, x_sol, order=None, graph_type=None):
    pareto_state_scores = None
    if problem_type == 1:
        pareto_state_scores = get_pareto_states_per_layer_knapsack(data["cons_coeffs"][0], x_sol)
    elif problem_type == 2:
        if graph_type == "stidsen":
            pareto_state_scores = get_pareto_states_per_layer_indepset(order, x_sol, data["adj_list_comp"])
        elif graph_type == "ba":
            pareto_state_scores = get_pareto_states_per_layer_indepset(order, x_sol, data["adj_list_comp"])

    return pareto_state_scores


def tag_dd_nodes(bdd, pareto_state_scores):
    assert len(pareto_state_scores) == len(bdd)
    items = np.array(range(0, 100))

    for l in range(len(bdd)):
        pareto_states, pareto_scores = pareto_state_scores[l]
        for pareto_state, score in zip(pareto_states, pareto_scores):
            _pareto_state = items[pareto_state.astype(bool)]
            is_found = False
            for n in bdd[l]:
                if np.array_equal(n["s"], _pareto_state):
                    n["pareto"] = 1
                    n["score"] = score
                    is_found = True
                    break

            assert is_found

        for n in bdd[l]:
            if "pareto" not in n:
                n["pareto"] = 0
                n["score"] = 0

    return bdd


def worker(rank, cfg):
    libv2 = get_lib(cfg.bin, n_objs=cfg.prob.n_objs)
    env = libv2.BDDEnv()
    signal.signal(signal.SIGALRM, handle_timeout)

    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.n_processes):
        print(f"{rank}/1/10: Fetching instance data and order...")
        data = get_instance_data(cfg.prob.name, cfg.size, cfg.split, pid)
        order = get_static_order(cfg.prob.name, cfg.order_type, data)

        print(f"{rank}/2/10: Resetting env...")
        initialize_run(cfg.bin,
                       env,
                       cfg.problem_type,
                       cfg.preprocess,
                       cfg.pf_enum_method,
                       cfg.bdd_type,
                       cfg.maxwidth,
                       order,
                       cfg.maximization,
                       cfg.dominance)

        print(f"{rank}/3/10: Initializing instance...")
        set_instance(cfg.bin,
                     env,
                     cfg.problem_type,
                     data,
                     graph_type=cfg.graph_type)

        print(f"{rank}/4/10: Preprocessing instance...")
        preprocess_inst(cfg.bin,
                        env,
                        cfg.problem_type)

        print(f"{rank}/5/10: Generating decision diagram...")
        initialize_dd_constructor(cfg.bin, env)
        env.generate_dd()
        time_compile = env.get_time(CONST.TIME_COMPILE)

        print(f"{rank}/6/10: Fetching decision diagram...")
        start = time.time()
        dd = env.get_dd()
        time_fetch = time.time() - start

        exact_size = []
        for i, layer in enumerate(dd):
            exact_size.append(len(layer))
        dynamic_order = get_dynamic_order(cfg.bin, env, cfg.problem_type, cfg.order_type, order)

        print(f"{rank}/7/10: Computing Pareto Frontier...")
        try:
            signal.alarm(cfg.time_limit)
            env.compute_pareto_frontier()
        except TimeoutError as exc:
            is_pf_computed = False
            print(f"PF not computed within {cfg.time_limit} for pid {pid}")
        else:
            is_pf_computed = True
            print(f"PF computed successfully for pid {pid}")
        signal.alarm(0)

        if not is_pf_computed:
            continue
        time_pareto = env.get_time(CONST.TIME_PARETO)

        print(f"{rank}/8/10: Fetching Pareto Frontier...")
        frontier = env.get_frontier()

        print(f"{rank}/9/10: Marking Pareto nodes...")
        pareto_state_scores = get_pareto_state_scores_per_layer(cfg.problem_type, data, frontier["x"],
                                                                order=dynamic_order,
                                                                graph_type=cfg.graph_type)
        dd = tag_dd_nodes(dd, pareto_state_scores)

        print(f"{rank}/10/10: Saving data...")
        # Save order
        file_path = path.order / f"{cfg.prob.name}/{cfg.size}/{cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.dat"
        with open(file_path, "w") as fp:
            fp.write(" ".join(map(str, dynamic_order)))

        # Save BDD
        file_path = path.bdd / f"{cfg.prob.name}/{cfg.size}/{cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.json"
        with open(file_path, "w") as fp:
            json.dump(dd, fp)

        # Save Solution
        file_path = path.sol / f"{cfg.prob.name}/{cfg.size}/{cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.json"
        with open(file_path, "w") as fp:
            frontier["order"] = dynamic_order
            json.dump(frontier, fp)

        # Save stats
        df = pd.DataFrame([
            [cfg.size,
             cfg.split,
             pid,
             len(frontier["z"]),
             env.initial_node_count,
             env.initial_arcs_count,
             -1,
             time_fetch,
             time_compile,
             time_pareto]], columns=["size", "split", "pid", "nnds", "inc", "iac", "Comp.",
                                     "fetch", "compilation", "pareto"])
        df.to_csv(file_path.parent / f"{pid}.csv", index=False)


@hydra.main(config_path="./configs", config_name="raw_data.yaml", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)

    if cfg.n_processes == 1:
        worker(0, cfg)
    else:
        pool = mp.Pool(processes=cfg.n_processes)
        results = []

        for rank in range(cfg.n_processes):
            results.append(pool.apply_async(worker, args=(rank, cfg)))

        for r in results:
            r.get()


if __name__ == "__main__":
    main()
