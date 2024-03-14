import hydra
import networkx as nx
import numpy as np

from morbdd import ResourcePaths as path


def generate_instance(rng, cfg):
    data = {'obj_coeffs': [], 'edges': [], 'attach': cfg.attach, 'num_vars': cfg.num_vars}

    # Value
    for _ in range(cfg.num_objs):
        data['obj_coeffs'].append(rng.randint(cfg.obj_lb, cfg.obj_ub, cfg.num_vars))

    graph = nx.barabasi_albert_graph(cfg.num_vars, cfg.attach, seed=np.random)
    data['edges'] = np.array(nx.edges(graph), dtype=int)

    return data


def write_to_file(inst_path, data):
    print(inst_path)
    np.savez(inst_path,
             obj_coeff=data['obj_coeffs'],
             edges=data['edges'],
             attach=data['attach'],
             num_vars=data['num_vars'])


@hydra.main(config_path="../configs", config_name="generator.yaml", version_base="1.2")
def main(cfg):
    rng = np.random.RandomState(cfg.seed)
    cfg.size = f"{cfg.num_objs}-{cfg.num_vars}-{cfg.attach}"

    train_path = path.inst / f'ind/{cfg.size}/train/'
    train_path.mkdir(parents=True, exist_ok=True)
    print(train_path)
    for id in range(cfg.num_train):
        data = generate_instance(rng, cfg)
        inst_path = train_path / f'ind_{cfg.seed}_{cfg.size}_{id}.npz'
        write_to_file(inst_path, data)

    val_path = path.inst / f'ind/{cfg.size}/val'
    val_path.mkdir(parents=True, exist_ok=True)
    start = cfg.num_train
    end = start + cfg.num_val
    for id in range(start, end):
        data = generate_instance(rng, cfg)
        inst_path = val_path / f'ind_{cfg.seed}_{cfg.size}_{id}.npz'
        write_to_file(inst_path, data)

    test_path = path.inst / f'ind/{cfg.size}/test'
    test_path.mkdir(parents=True, exist_ok=True)
    start = cfg.num_train + cfg.num_val
    end = start + cfg.num_test
    for id in range(start, end):
        data = generate_instance(rng, cfg)
        inst_path = test_path / f'ind_{cfg.seed}_{cfg.size}_{id}.npz'
        write_to_file(inst_path, data)


if __name__ == '__main__':
    main()
