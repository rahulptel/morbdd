import hydra


def get_pareto_states_per_layer_knapsack(data, x_sol):
    return []


def get_pareto_states_per_layer_indepset(data, x_sol):
    return []


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


def set_instance(bin, env, problem_type, n_vars, n_objs, n_cons=None, obj_coeffs=None, cons_coeffs=None, rhs=None,
                 card=None):
    if bin == 'multiobj':
        if problem_type == 1:
            env.set_inst(n_vars, n_objs, obj_coeffs, cons_coeffs[0], rhs)
        elif problem_type == 2 or problem_type == 3:
            env.set_inst(n_vars, n_objs, obj_coeffs, cons_coeffs, rhs)

    elif bin == 'network':
        if problem_type >= 1 and problem_type <= 3:
            env.set_inst(n_vars, n_cons, n_objs, obj_coeffs, cons_coeffs, rhs)
        elif problem_type == 4:
            env.set_inst_absval(n_vars, n_objs, card, cons_coeffs, rhs)
        elif problem_type == 5:
            env.set_inst_tsp(n_vars, n_objs, obj_coeffs)


def initialize_run(bin, env, problem_type, preprocess, bdd_type, maxwidth, order, maximization=None, dominance=None):
    if bin == 'multiobj':
        env.initialize_run(
            problem_type,
            preprocess,
            bdd_type,
            maxwidth,
            order
        )
    elif bin == 'network':
        env.initialize_run(
            problem_type,
            preprocess,
            maximization,
            dominance,
            bdd_type,
            maxwidth,
            order
        )


def main(cfg):
    libv1 = get_lib('multiobj')
    libv2 = get_lib('network', n_objs=cfg.problem.n_obj)

    # v1 is used to access the x_sol
    env1 = libv1.BDDEnv()
    # v2 is used to access the BDD
    env2 = libv2.BDDEnv()

    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.num_processes):
        data = get_instance_data(cfg.prob, cfg.size, cfg.split, pid)

        env1.initialize_run(cfg.bin, env1, cfg.problem_type, cfg.preprocess, cfg.bdd_type, cfg.maxwidth, cfg.order)
        env1.set_instance('multiobj',
                          env1,
                          cfg.problem_type,
                          data.get('n_vars'),
                          data.get('n_objs'),
                          data.get('n_cons'),
                          data.get('obj_coeffs'),
                          data.get('cons_coeffs'),
                          data.get('rhs'),
                          data.get('card'))
        env1.compute_pareto_frontier()
        x = env1.x_sol
        z = env1.z_sol
        dd = env1.get_bdd(problem_type)
        pareto_states = get_pareto_states_per_layer(cfg.problem_type, data, x_sol)

        env2.initialize_run(cfg.bin, env1, cfg.problem_type, cfg.preprocess, cfg.bdd_type, cfg.maxwidth, cfg.order,
                            maximization=cfg.maximization, dominance=cfg.dominance)
        env2.set_instance('network',
                          env2,
                          cfg.problem_type,
                          data.get('n_vars'),
                          data.get('n_objs'),
                          data.get('n_cons'),
                          data.get('obj_coeffs'),
                          data.get('cons_coeffs'),
                          data.get('rhs'),
                          data.get('card'))
        env2.preprocess_inst()
        env2.initialize_dd_constructor()
        env2.generate_dd()
        dd = env2.get_dd()


if __name__ == "__main__":
    main()
