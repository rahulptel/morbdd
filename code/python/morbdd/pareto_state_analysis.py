import hydra
from morbdd.utils import get_instance_data
import numpy as np
from collections import defaultdict


def get_pareto_states_per_layer_knapsack(data, x_sol):
    return []


def get_pareto_states_per_layer_indepset(data, order, x_sol, adj_list_comp):
    x_sol = np.array(x_sol)
    pareto_states = []
    counter = 0

    for i in range(1, x_sol.shape[1]):
        # Get partial sols upto level i + 1 in BDD
        partial_sols = x_sol[:, :i]
        # Find unique partial sols
        uniques = defaultdict(int)
        for ps in partial_sols:
            if ps not in uniques:
                uniques[ps] = 1
            else:
                uniques[ps] += 1

        _pareto_states = []
        # Compute pareto state for each unique sol
        for unique_sol in uniques.keys():
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


def get_pareto_states_per_layer(problem_type, data, x_sol):
    pareto_sols_per_layer = None
    if problem_type == 1:
        pareto_sols_per_layer = get_pareto_states_per_layer_knapsack(data, x_sol)
    elif problem_type == 2:
        pareto_sols_per_layer = get_pareto_states_per_layer_indepset(data, x_sol)

    return pareto_sols_per_layer


def get_lib(bin, n_objs=3):
    libname = 'libbddenv'
    if bin == 'multiobj':
        libname += 'v1'
    elif bin == 'network':
        libname += f'v2o{n_objs}'

    lib = __import__(libname)

    return lib


def set_instance(bin, env, problem_type, n_vars, n_objs, n_cons=None, obj_coeffs=None, edges=None, cons_coeffs=None,
                 rhs=None,
                 card=None):
    if bin == 'multiobj':
        if problem_type == 1:
            env.set_inst(n_vars, n_objs, obj_coeffs, cons_coeffs[0], rhs)
        elif problem_type == 2 or problem_type == 3:
            env.set_inst(n_vars, n_objs, obj_coeffs, cons_coeffs, rhs)

    elif bin == 'network':
        if problem_type == 1 or problem_type == 3:
            env.set_inst(n_vars, n_cons, n_objs, obj_coeffs, cons_coeffs, rhs)
        elif problem_type == 2:
            env.set_inst_indepset(n_vars, n_objs, obj_coeffs, edges)
        elif problem_type == 4:
            env.set_inst_absval(n_vars, n_objs, card, cons_coeffs, rhs)
        elif problem_type == 5:
            env.set_inst_tsp(n_vars, n_objs, obj_coeffs)


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


@hydra.main(config_path="./config", config_name="pareto_state_analysis.yaml", version_base="1.2")
def main(cfg):
    libv2 = get_lib('network', n_objs=cfg.problem.n_obj)
    env = libv2.BDDEnv()
    order = []

    for pid in range(cfg.from_pid, cfg.to_pid):
        data = get_instance_data(cfg.prob, cfg.size, cfg.split, pid)

        initialize_run('network',
                       env,
                       cfg.problem_type,
                       cfg.preprocess,
                       cfg.method,
                       cfg.maximization,
                       cfg.dominance,
                       cfg.bdd_type,
                       cfg.maxwidth,
                       order)
        set_instance('network',
                     env,
                     cfg.problem_type,
                     data.get('n_vars'),
                     data.get('n_objs'),
                     data.get('n_cons'),
                     data.get('obj_coeffs'),
                     data.get('cons_coeffs'),
                     data.get('edges'),
                     data.get('rhs'),
                     data.get('card'))
        env.preprocess_inst()
        env.initialize_dd_constructor()
        env.generate_dd()
        dd = env.get_dd()
        frontier = env.get_frontier()


if __name__ == "__main__":
    main()
