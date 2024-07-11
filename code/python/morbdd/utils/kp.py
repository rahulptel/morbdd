from morbdd import ResourcePaths as path
from morbdd.utils import read_from_zip


def read_instance(archive, inst):
    data = {'value': [], 'n_vars': 0, 'n_cons': 1, 'n_objs': 3}
    data['weight'], data['capacity'] = [], 0

    raw_data = read_from_zip(archive, inst, format="raw")
    data['n_vars'] = int(raw_data.readline())
    data['n_objs'] = int(raw_data.readline())
    for _ in range(data['n_objs']):
        data['value'].append(list(map(int, raw_data.readline().split())))
    data['weight'].extend(list(map(int, raw_data.readline().split())))
    data['capacity'] = int(raw_data.readline().split()[0])

    return data


def get_instance_path(seed, n_objs, n_vars, split, pid, name="knapsack"):
    return path.inst / f'{name}/{n_objs}_{n_vars}/{split}/kp_{seed}_{n_objs}_{n_vars}_{pid}.dat'


def get_instance_data(size, split, pid):
    prefix = "kp"
    archive = path.inst / f"knapsack/{size}.zip"
    suffix = "dat"
    inst = f'{size}/{split}/{prefix}_{size}_{pid}.{suffix}'
    data = read_instance(archive, inst)

    return data
