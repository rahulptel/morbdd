import hydra
import pandas as pd
from pymoo.indicators.hv import HV
import pygmo as pg
import numpy as np

from morbdd import resource_path
import json
from morbdd.utils import read_from_zip

from pymoo.indicators.igd import IGD

seeds = ["1234", "279183", "3432", "876"]


def compute_cardinality(true_pf=None, pred_pf=None):
    z, z_pred = np.array(true_pf), np.array(pred_pf)
    assert z.shape[1] == z_pred.shape[1]

    if pred_pf.shape[0] == 0:
        return 0
    else:
        # Defining a data type
        rows, cols = z.shape
        dt_z = {'names': ['f{}'.format(i) for i in range(cols)],
                'formats': cols * [z.dtype]}

        rows, cols = z_pred.shape
        dt_z_pred = {'names': ['f{}'.format(i) for i in range(cols)],
                     'formats': cols * [z_pred.dtype]}

        # Finding intersection
        found_ndps = np.intersect1d(z.view(dt_z), z_pred.view(dt_z_pred))

        return found_ndps.shape[0]


@hydra.main(version_base="1.2", config_path="./configs", config_name="metric_eval.yaml")
def main(cfg):
    if cfg.deploy.eval == "hv":
        archive = resource_path / f"sols/{cfg.prob.name}/{cfg.prob.size}.zip"
        ref_point = np.zeros(cfg.prob.num_objs)
        hv_algo = pg.bf_fpras(eps=0.1, delta=0.1, seed=1)

        for pid in range(cfg.prob.from_pid, cfg.prob.to_pid):
            file = f"{cfg.prob.size}/{cfg.prob.split}/{pid}.json"
            sol = read_from_zip(archive, file, format="json")
            # Ignore instances not solved within time limit
            if sol is None:
                continue
            sol_np = -np.array(sol["z"])
            norm = np.abs(np.min(sol_np, axis=0))
            sol_np = sol_np / norm

            if cfg.deploy.algorithm == "pf":
                hv = pg.hypervolume(sol_np)
                hv_approx = hv.compute(ref_point, hv_algo=hv_algo)
                df = pd.DataFrame([[cfg.prob.size, pid, cfg.prob.split, "pf", "hv", hv_approx]],
                                  columns=["size", "pid", "split", "algorithm", "metric", "value"])
                path = archive.parent / f"{cfg.prob.size}/{cfg.prob.split}"
                path.mkdir(exist_ok=True, parents=True)
                df.to_csv(path / f"{pid}_hv.csv", index=False)

            if cfg.deploy.algorithm == "nsga2":
                for seed in seeds:
                    fname = f"{pid}_{seed}"
                    approx_pf_path = resource_path / f"ea/nsga2/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}/{fname}.npz"
                    if not approx_pf_path.exists():
                        continue

                    approx_pf = np.load(approx_pf_path)
                    solC = approx_pf["F"]
                    solC = solC / norm

                    hv = pg.hypervolume(solC)
                    hv_approx = hv.compute(ref_point, hv_algo=hv_algo)

                    df = pd.DataFrame([[cfg.prob.size, pid, cfg.prob.split, "nsga2", "hv", hv_approx]],
                                      columns=["size", "pid", "split", "algorithm", "metric", "value"])
                    df.to_csv(approx_pf_path.parent / f"{fname}_hv.csv", index=False)

            elif cfg.deploy.algorithm == "restricted":
                for mw in [20, 40, 60]:
                    approx_pf_path = resource_path / (
                        f"restricted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                        f"/{mw}/{pid}.json")
                    if not approx_pf_path.exists():
                        continue
                    approx_pf = json.load(open(approx_pf_path, "r"))
                    solC = -np.array(approx_pf["z"])
                    solC = solC / norm

                    hv = pg.hypervolume(solC)
                    hv_approx = hv.compute(ref_point, hv_algo=hv_algo)

                    df = pd.DataFrame(
                        [[cfg.prob.size, pid, cfg.prob.split, f"restricted-{mw}", "hv", hv_approx]],
                        columns=["size", "pid", "split", "algorithm", "metric", "value"])
                    df.to_csv(approx_pf_path.parent / f"{pid}_hv.csv", index=False)

            elif cfg.deploy.algorithm == "sparse":
                approx_pf_path = resource_path / (f"predicted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                                                  f"/sols_pred/sol_{pid}.json")
                if approx_pf_path.exists():
                    approx_pf = json.load(open(approx_pf_path, "r"))
                    solC = -np.array(approx_pf["z"])
                    solC = solC / norm

                    hv = pg.hypervolume(solC)
                    hv_approx = hv.compute(ref_point, hv_algo=hv_algo)

                    df = pd.DataFrame(
                        [[cfg.prob.size, pid, cfg.prob.split, f"sparse", "hv", hv_approx]],
                        columns=["size", "pid", "split", "algorithm", "metric", "value"])
                    df.to_csv(approx_pf_path.parent / f"{pid}_hv.csv", index=False)

                else:
                    # Min resistance
                    approx_pf_path = resource_path / (f"predicted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                                                      f"/0-mrh2-sols_pred/sol_{pid}.json")
                    if approx_pf_path.exists():
                        approx_pf = json.load(open(approx_pf_path, "r"))
                        solC = -np.array(approx_pf["z"])
                        solC = solC / norm

                        hv = pg.hypervolume(solC)
                        hv_approx = hv.compute(ref_point, hv_algo=hv_algo)

                        df = pd.DataFrame(
                            [[cfg.prob.size, pid, cfg.prob.split, f"sparse-0-mrh2", "hv", hv_approx]],
                            columns=["size", "pid", "split", "algorithm", "metric", "value"])
                        df.to_csv(approx_pf_path.parent / f"{pid}_hv.csv", index=False)

                    # MIP sol
                    approx_pf_path = resource_path / (f"predicted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                                                      f"/0-mip-sols_pred/sol_{pid}.json")
                    if approx_pf_path.exists():
                        approx_pf = json.load(open(approx_pf_path, "r"))
                        solC = -np.array(approx_pf["z"])
                        solC = solC / norm

                        hv = pg.hypervolume(solC)
                        hv_approx = hv.compute(ref_point, hv_algo=hv_algo)

                        df = pd.DataFrame(
                            [[cfg.prob.size, pid, cfg.prob.split, f"sparse-0-mip", "hv", hv_approx]],
                            columns=["size", "pid", "split", "algorithm", "metric", "value"])
                        df.to_csv(approx_pf_path.parent / f"{pid}_hv.csv", index=False)

    elif cfg.deploy.eval == "igd":
        archive = resource_path / f"sols/{cfg.prob.name}/{cfg.prob.size}.zip"

        for pid in range(cfg.prob.from_pid, cfg.prob.to_pid):
            file = f"{cfg.prob.size}/{cfg.prob.split}/{pid}.json"
            sol = read_from_zip(archive, file, format="json")
            # Ignore instances not solved within time limit
            if sol is None:
                continue
            sol_np = -np.array(sol["z"])
            norm = np.min(sol_np, axis=0)
            sol_np = sol_np / norm

            if cfg.deploy.algorithm == "nsga2":
                approx_pf_path = resource_path / f"ea/nsga2/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}/{pid}.npz"
                if not approx_pf_path.exists():
                    continue
                approx_pf = np.load(approx_pf_path)
                solC = approx_pf["F"]
                solC = solC / norm

                ind = IGD(sol_np)
                igd = ind(solC)
                df = pd.DataFrame([[cfg.prob.size, pid, cfg.prob.split, "nsga2", "igd", igd]],
                                  columns=["size", "pid", "split", "algorithm", "metric", "value"])
                df.to_csv(approx_pf_path.parent / f"{pid}_igd.csv", index=False)

            elif cfg.deploy.algorithm == "restricted":
                for mw in [20, 40, 60]:
                    approx_pf_path = resource_path / (
                        f"restricted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                        f"/{mw}/{pid}.json")
                    if not approx_pf_path.exists():
                        continue
                    approx_pf = json.load(open(approx_pf_path, "r"))
                    solC = -np.array(approx_pf["z"])
                    solC = solC / norm

                    ind = IGD(sol_np)
                    igd = ind(solC)
                    df = pd.DataFrame(
                        [[cfg.prob.size, pid, cfg.prob.split, f"restricted-{mw}", "igd", igd]],
                        columns=["size", "pid", "split", "algorithm", "metric", "value"])
                    df.to_csv(approx_pf_path.parent / f"{pid}_igd.csv", index=False)

            elif cfg.deploy.algorithm == "sparse":
                approx_pf_path = resource_path / (f"predicted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                                                  f"/sols_pred/sol_{pid}.json")
                if approx_pf_path.exists():
                    approx_pf = json.load(open(approx_pf_path, "r"))
                    solC = -np.array(approx_pf["z"])
                    solC = solC / norm

                    ind = IGD(sol_np)
                    igd = ind(solC)

                    df = pd.DataFrame(
                        [[cfg.prob.size, pid, cfg.prob.split, f"sparse", "igd", igd]],
                        columns=["size", "pid", "split", "algorithm", "metric", "value"])
                    df.to_csv(approx_pf_path.parent / f"{pid}_igd.csv", index=False)

                else:
                    # Min resistance
                    approx_pf_path = resource_path / (f"predicted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                                                      f"/0-mrh2-sols_pred/sol_{pid}.json")
                    if approx_pf_path.exists():
                        approx_pf = json.load(open(approx_pf_path, "r"))
                        solC = -np.array(approx_pf["z"])
                        solC = solC / norm

                        ind = IGD(sol_np)
                        igd = ind(solC)

                        df = pd.DataFrame(
                            [[cfg.prob.size, pid, cfg.prob.split, f"sparse-0-mrh2", "igd", igd]],
                            columns=["size", "pid", "split", "algorithm", "metric", "value"])
                        df.to_csv(approx_pf_path.parent / f"{pid}_igd.csv", index=False)

                    # MIP sol
                    approx_pf_path = resource_path / (f"predicted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                                                      f"/0-mip-sols_pred/sol_{pid}.json")
                    if approx_pf_path.exists():
                        approx_pf = json.load(open(approx_pf_path, "r"))
                        solC = -np.array(approx_pf["z"])
                        solC = solC / norm

                        ind = IGD(sol_np)
                        igd = ind(solC)

                        df = pd.DataFrame(
                            [[cfg.prob.size, pid, cfg.prob.split, f"sparse-0-mip", "igd", igd]],
                            columns=["size", "pid", "split", "algorithm", "metric", "value"])
                        df.to_csv(approx_pf_path.parent / f"{pid}_igd.csv", index=False)

    elif cfg.deploy.eval == "card":
        archive = resource_path / f"sols/{cfg.prob.name}/{cfg.prob.size}.zip"

        for pid in range(cfg.prob.from_pid, cfg.prob.to_pid):
            file = f"{cfg.prob.size}/{cfg.prob.split}/{pid}.json"
            sol = read_from_zip(archive, file, format="json")
            # Ignore instances not solved within time limit
            if sol is None or len(sol) == 0:
                continue
            sol_np = np.array(sol["z"])

            if cfg.deploy.algorithm == "nsga2":
                for seed in seeds:
                    fname = f"{pid}_{seed}"
                    approx_pf_path = resource_path / f"ea/nsga2/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}/{fname}.npz"
                    if not approx_pf_path.exists():
                        continue

                    approx_pf = np.load(approx_pf_path)
                    solC = -approx_pf["F"]
                    cardinality = compute_cardinality(true_pf=sol_np, pred_pf=solC)

                    df = pd.DataFrame([[cfg.prob.size, pid, cfg.prob.split, "nsga2", "card", cardinality]],
                                      columns=["size", "pid", "split", "algorithm", "metric", "value"])
                    df.to_csv(approx_pf_path.parent / f"{fname}_card.csv", index=False)

            elif cfg.deploy.algorithm == "restricted":
                for mw in [20, 40, 60]:
                    approx_pf_path = resource_path / (
                        f"restricted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                        f"/{mw}/{pid}.json")
                    if not approx_pf_path.exists():
                        continue
                    approx_pf = json.load(open(approx_pf_path, "r"))
                    solC = np.array(approx_pf["z"])
                    cardinality = compute_cardinality(true_pf=sol_np, pred_pf=solC)

                    df = pd.DataFrame(
                        [[cfg.prob.size, pid, cfg.prob.split, f"restricted-{mw}", "card", cardinality]],
                        columns=["size", "pid", "split", "algorithm", "metric", "value"])
                    df.to_csv(approx_pf_path.parent / f"{pid}_card.csv", index=False)

            elif cfg.deploy.algorithm == "sparse":
                approx_pf_path = resource_path / (f"predicted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                                                  f"/sols_pred/sol_{pid}.json")
                if approx_pf_path.exists():
                    approx_pf = json.load(open(approx_pf_path, "r"))
                    solC = np.array(approx_pf["z"])
                    cardinality = compute_cardinality(true_pf=sol_np, pred_pf=solC)

                    df = pd.DataFrame(
                        [[cfg.prob.size, pid, cfg.prob.split, f"sparse", "card", cardinality]],
                        columns=["size", "pid", "split", "algorithm", "metric", "value"])
                    df.to_csv(approx_pf_path.parent / f"{pid}_card.csv", index=False)

                else:
                    # Min resistance
                    approx_pf_path = resource_path / (f"predicted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                                                      f"/0-mrh2-sols_pred/sol_{pid}.json")
                    if approx_pf_path.exists():
                        approx_pf = json.load(open(approx_pf_path, "r"))
                        solC = np.array(approx_pf["z"])
                        cardinality = compute_cardinality(true_pf=sol_np, pred_pf=solC)

                        df = pd.DataFrame(
                            [[cfg.prob.size, pid, cfg.prob.split, f"sparse-0-mrh2", "card", cardinality]],
                            columns=["size", "pid", "split", "algorithm", "metric", "value"])
                        df.to_csv(approx_pf_path.parent / f"{pid}_card.csv", index=False)

                    # MIP sol
                    approx_pf_path = resource_path / (f"predicted_sols/{cfg.prob.name}/{cfg.prob.size}/{cfg.prob.split}"
                                                      f"/0-mip-sols_pred/sol_{pid}.json")
                    if approx_pf_path.exists():
                        approx_pf = json.load(open(approx_pf_path, "r"))
                        solC = np.array(approx_pf["z"])
                        cardinality = compute_cardinality(true_pf=sol_np, pred_pf=solC)

                        df = pd.DataFrame(
                            [[cfg.prob.size, pid, cfg.prob.split, f"sparse-0-mip", "card", cardinality]],
                            columns=["size", "pid", "split", "algorithm", "metric", "value"])
                        df.to_csv(approx_pf_path.parent / f"{pid}_card.csv", index=False)


if __name__ == "__main__":
    main()
