from collections import defaultdict

import hydra
import numpy as np

from morbdd.utils import get_instance_data
import time
import multiprocessing as mp
from morbdd.utils import get_static_order


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
    pareto_states = []
    counter = 0

    for i in range(1, x_sol.shape[1]):
        # Get partial sols upto level i in BDD
        partial_sols = x_sol[:, :i]
        # Find unique partial sols
        uniques = defaultdict(int)
        for ps in partial_sols:
            if ps not in uniques:
                uniques[ps] = 1
            else:
                uniques[ps] += 1
        total = np.sum([v for v in uniques.values()])

        _pareto_states = []
        # Compute pareto state for each unique sol
        for unique_sol, count in uniques.items():
            pareto_state = np.ones(x_sol.shape[1])
            for var_idx, is_active in enumerate(list(unique_sol)):
                var = order[var_idx]
                pareto_state[var] = 0
                if is_active:
                    pareto_state = np.logical_and(pareto_state, adj_list_comp[var]).astype(int)

            is_found = False
            for ps in _pareto_states:
                if np.array_equal(pareto_state, ps):
                    is_found = True
                    break
            if not is_found:
                _pareto_states.append(pareto_state)
                counter += 1

    pareto_states.append(_pareto_states)

    return pareto_states, counter


def get_pareto_states_per_layer(problem_type, data, x_sol, graph_type):
    pareto_sols_per_layer = None
    if problem_type == 1:
        pareto_sols_per_layer = get_pareto_states_per_layer_knapsack(data["cons_coeffs"][0], x_sol)
    elif problem_type == 2:
        if graph_type == "stidsen":
            pareto_sols_per_layer = get_pareto_states_per_layer_indepset(data, x_sol)
        elif graph_type == "ba":
            pareto_sols_per_layer = get_pareto_states_per_layer_indepset(data, x_sol)

    return pareto_sols_per_layer


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
    libv2 = get_lib(cfg.bin, n_objs=cfg.prob.n_objs)
    env = libv2.BDDEnv()
    stats = []

    for pid in range(rank, cfg.to_pid, cfg.n_processes):
        _stats = []

        print("1/10: Fetching instance data and order...")
        data = get_instance_data(cfg.prob.name, cfg.size, cfg.split, pid)
        order = get_static_order(cfg.prob.name, cfg.order_type, data)

        print("2/10: Resetting env...")
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

        print("3/10: Initializing instance...")
        set_instance(cfg.bin,
                     env,
                     cfg.problem_type,
                     data,
                     graph_type=cfg.graph_type)

        print("4/10: Preprocessing instance...")
        preprocess_inst(cfg.bin,
                        env,
                        cfg.problem_type)

        print("5/10: Generating decision diagram...")
        initialize_dd_constructor(cfg.bin, env)
        start = time.time()
        env.generate_dd()
        end = time.time() - start
        _stats.append(end)

        print("6/10: Fetching decision diagram...")
        start = time.time()
        dd = env.get_dd()
        end = time.time() - start
        _stats.append(end)

        exact_size = []
        for layer in enumerate(dd):
            exact_size.append(len(layer))
        print(np.mean(exact_size), np.max(exact_size))
        order = get_dynamic_order(cfg.bin, env, cfg.problem_type, cfg.order_type, order)

        print("7/10: Computing Pareto Frontier...")
        env.compute_pareto_frontier()
        _stats.append(env.get_construction_time())
        _stats.append(env.get_pareto_time())

        print("8/10: Fetching Pareto Frontier...")
        frontier = env.get_frontier()

        print("9/10: Marking Pareto nodes...")

        print("10/10: Saving data...")


@hydra.main(config_path="./configs", config_name="raw_data.yaml", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)

    pool = mp.Pool(processes=cfg.n_processes)
    results = []

    for rank in range(cfg.n_processes):
        results.append(pool.apply_async(worker, args=(rank, cfg)))

    for r in results:
        r.get()


if __name__ == "__main__":
    main()
