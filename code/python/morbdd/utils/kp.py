from morbdd import ResourcePaths as path


def get_instance_path(seed, n_objs, n_vars, split, pid, name="knapsack"):
    return path.inst / f'{name}/{n_objs}_{n_vars}/{split}/kp_{seed}_{n_objs}_{n_vars}_{pid}.dat'
