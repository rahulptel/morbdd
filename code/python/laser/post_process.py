from pathlib import Path
import json
import pandas as pd
from laser import resource_path
import zipfile
import numpy as np
from laser.utils import get_instance_data
from laser.utils import get_order
import copy
import multiprocessing as mp

sols_pred_path = resource_path / "sols_pred/knapsack/7_40/val"


def count_ndps(true_pf, pred_pf):
    z, z_pred = np.array(true_pf), np.array(pred_pf)

    assert z.shape[1] == z_pred.shape[1]
    num_objs = z.shape[1]

    z = z[np.lexsort([z[:, num_objs - i - 1] for i in range(num_objs)])]
    z_pred = z_pred[np.lexsort([z_pred[:, num_objs - i - 1] for i in range(num_objs)])]

    j_prev = 0
    counter = 0
    found_ndps = []
    for i in range(z.shape[0]):
        # print(i, counter)
        j = copy.deepcopy(j_prev)
        while j < z_pred.shape[0]:
            if np.array_equal(z[i], z_pred[j]):
                counter += 1
                j_prev = copy.deepcopy(j + 1)
                found_ndps.append(z[i])
                break

            j += 1

    # found_ndps = np.array(found_ndps)
    # with open("found_ndps.npy", "wb") as fp:
    #     np.save(fp, found_ndps)
    #
    # with open("true_ndps.npy", "wb") as fp:
    #     np.save(fp, z)
    #
    # with open("pred_ndps.npy", "wb") as fp:
    #     np.save(fp, z_pred)
    return counter


def worker(i):
    # Load predicted solution
    sol_path = sols_pred_path / f"sol_{i}.json"
    if sol_path.exists():
        with open(sols_pred_path / f"sol_{i}.json", "r") as fp:
            sol_pred = json.load(fp)

        zfp = zipfile.Path(resource_path / f"sols/knapsack/7_40.zip")
        if zfp.joinpath(f"7_40/val/{i}.json"):
            zf = zipfile.ZipFile(resource_path / f"sols/knapsack/7_40.zip")
            with zf.open(f"7_40/val/{i}.json") as fp:
                sol = json.load(fp)

            print("Working with pid", i)
            ndps_in_pred = count_ndps(sol["z"], sol_pred["z"])
            print(f"Result for inst {i}: ", ndps_in_pred, ndps_in_pred / len(sol_pred["z"]),
                  ndps_in_pred / len(sol["z"]))
            return i, ndps_in_pred

    return i, -1


def main():
    pool = mp.Pool(processes=16)

    results = [pool.apply_async(worker, args=(i,)) for i in range(1000, 1100)]
    results = [r.get() for r in results]
    results = [r for r in results if r[1] != -1]
    results = np.array(results)

    df = pd.DataFrame(results, columns=["pid", "pareto_pred"])
    df.to_csv("count_pareto.csv", index=False)


if __name__ == '__main__':
    main()
