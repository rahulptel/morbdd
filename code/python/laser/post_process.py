from pathlib import Path
import json
import pandas as pd
from laser import resource_path
import zipfile
import numpy as np
from laser.utils import get_instance_data
from laser.utils import get_order

sols_pred_path = resource_path / "sols_pred/knapsack/7_40/val"
pareto_count = []

for i in range(1000, 1100):
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

            z_pred = np.array(sol_pred["z"])
            x_pred = np.array(sol_pred["x"])
            z = np.array(sol["z"])
            x = np.array(sol["x"])

            # z_pred = z_pred[np.lexsort([z_pred[:, z_pred.shape[1] - i - 1] for i in range(z.shape[1])])]
            # z = z[np.lexsort([z[:, z.shape[1] - i - 1] for i in range(z.shape[1])])]

            inst_data = get_instance_data("knapsack", "7_40", "val", i)
            order = get_order("knapsack", "MinWt", inst_data)
            inst_data["weight"] = np.array(inst_data["weight"])[order]

            idxs = list(map(int, np.arange(x_pred.shape[0])))
            for j in range(1, x.shape[1]):

                weight_pareto = np.dot(x[:, :j], inst_data["weight"][:j].reshape(-1, 1))
                weight_pareto = np.unique(weight_pareto)

                x_pred = x_pred[idxs]
                # print(x_pred.shape[0])
                weight_pareto_pred = np.dot(x_pred[:, :j], inst_data["weight"][:j].reshape(-1, 1))

                idxs = []
                for idx, wt in enumerate(weight_pareto_pred):
                    if wt in weight_pareto:
                        idxs.append(idx)

            # count_pareto = 0
            # print(z.shape[0], z_pred.shape[0])
            # for j in range(z.shape[0]):
            #     for k in range(z_pred.shape[0]):
            #         print(j, k, count_pareto)
            #         if np.array_equal(z[j], z_pred[k]):
            #             count_pareto += 1
            #             break
            # print("Pareto count", count_pareto)
            pareto_count.append([i, x_pred.shape[0]])

df = pd.DataFrame(pareto_count, columns=["pid", "pareto_pred"])
df.to_csv("count_pareto.csv", index=False)
