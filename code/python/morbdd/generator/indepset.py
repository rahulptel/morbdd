from pathlib import Path

import hydra
import networkx as nx
import numpy as np

from morbdd import ResourcePaths as path

PROB_NAME = "indepset"
PROB_PREFIX = "ind"


def generate_instance_ba(rng, cfg):
    data = {'n_vars': cfg.n_vars,
            'n_objs': cfg.n_objs,
            'attach': cfg.attach,
            'obj_coeffs': [],
            'edges': []}

    # Value
    for _ in range(cfg.n_objs):
        data['obj_coeffs'].append(rng.randint(cfg.obj_lb, cfg.obj_ub, cfg.n_vars))

    graph = nx.barabasi_albert_graph(cfg.n_vars, cfg.attach, seed=np.random)
    data['edges'] = np.array(nx.edges(graph), dtype=int)

    return data


def generate_instance_stidsen(rng, cfg, should_print=False):
    data = {'n_vars': cfg.n_vars, 'n_objs': cfg.n_objs, 'n_cons': int(cfg.n_vars / 5), 'obj_coeffs': [],
            'cons_coeffs': []}
    items = list(range(1, cfg.n_vars + 1))

    # Value
    for _ in range(cfg.n_objs):
        data['obj_coeffs'].append(list(rng.randint(cfg.obj_lb, cfg.obj_ub + 1, cfg.n_vars)))

    # Constraints
    for _ in range(data['n_cons']):
        vars_in_con = rng.randint(2, (2 * cfg.vars_per_con) + 1)
        data['cons_coeffs'].append(list(rng.choice(items, vars_in_con, replace=False)))

    # Ensure no variable is missed
    var_count = []
    for con in data['cons_coeffs']:
        var_count.extend(con)
    missing_vars = list(set(range(1, cfg.n_vars + 1)).difference(set(var_count)))
    for v in missing_vars:
        cons_id = rng.randint(data['n_cons'])
        data['cons_coeffs'][cons_id].append(v)

    return data


def generate_instance(rng, cfg):
    data = None
    if cfg.graph_type == "stidsen":
        data = generate_instance_stidsen(rng, cfg)
    elif cfg.graph_type == "ba":
        data = generate_instance_ba(rng, cfg)

    return data


def write_to_file_ba(inst_path, data):
    inst_path = Path(str(inst_path) + ".npz")
    np.savez(inst_path,
             n_vars=data['n_vars'],
             n_objs=data['n_objs'],
             attach=data['attach'],
             obj_coeffs=data['obj_coeffs'],
             edges=data['edges'])


def write_to_file_stidsen(inst_path, data):
    dat = f"{data['n_vars']} {data['n_cons']}\n"
    dat += f"{len(data['obj_coeffs'])}\n"
    for coeffs in data['obj_coeffs']:
        dat += " ".join(list(map(str, coeffs))) + "\n"

    for coeffs in data["cons_coeffs"]:
        dat += f"{len(coeffs)}\n"
        dat += " ".join(list(map(str, coeffs))) + "\n"

    inst_path = Path(str(inst_path) + ".dat")
    inst_path.write_text(dat)


def write_to_file(graph_type, inst_path, data):
    if graph_type == "stidsen":
        write_to_file_stidsen(inst_path, data)
    elif graph_type == "ba":
        write_to_file_ba(inst_path, data)


@hydra.main(config_path="../configs", config_name="generator.yaml", version_base="1.2")
def main(cfg):
    rng = np.random.RandomState(cfg.seed)
    cfg.size = f"{cfg.n_objs}-{cfg.n_vars}"
    if cfg.graph_type == "ba":
        cfg.size += f"{cfg.attach}"

    train_path = path.inst / f'{PROB_NAME}/{cfg.size}/train/'
    train_path.mkdir(parents=True, exist_ok=True)
    print(train_path)
    for id in range(cfg.n_train):
        data = generate_instance(rng, cfg)
        inst_path = train_path / f'{PROB_PREFIX}_{cfg.seed}_{cfg.size}_{id}'
        write_to_file(cfg.graph_type, inst_path, data)

    val_path = path.inst / f'{PROB_NAME}/{cfg.size}/val'
    val_path.mkdir(parents=True, exist_ok=True)
    start = cfg.n_train
    end = start + cfg.n_val
    for id in range(start, end):
        data = generate_instance(rng, cfg)
        inst_path = val_path / f'{PROB_PREFIX}_{cfg.seed}_{cfg.size}_{id}'
        write_to_file(cfg.graph_type, inst_path, data)

    test_path = path.inst / f'{PROB_NAME}/{cfg.size}/test'
    test_path.mkdir(parents=True, exist_ok=True)
    start = cfg.n_train + cfg.n_val
    end = start + cfg.n_test
    for id in range(start, end):
        data = generate_instance(rng, cfg)
        inst_path = test_path / f'{PROB_PREFIX}_{cfg.seed}_{cfg.size}_{id}'
        write_to_file(cfg.graph_type, inst_path, data)


if __name__ == '__main__':
    main()
