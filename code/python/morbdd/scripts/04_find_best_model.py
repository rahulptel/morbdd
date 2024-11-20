import pickle as pkl
import shutil

import hydra
import numpy as np
from omegaconf import OmegaConf

from morbdd import ResourcePaths as path


@hydra.main(
    config_path="../configs", config_name="04_find_best_model.yaml", version_base="1.2"
)
def main(cfg):
    ckpt_dir = path.checkpoint / cfg.prob.name / cfg.prob.size
    best_f1, best_result = 0, None
    for p in ckpt_dir.rglob(f"{cfg.model.type}*"):
        if not p.is_dir():
            continue
        print(p)
        cfg = OmegaConf.load(p / "config.yaml")

        result = []
        for result_path in p.rglob("result*.pkl"):
            _, ep, iter = result_path.stem.split("_")
            d = pkl.load(open(result_path, "rb"))
            result.append(
                [
                    d["val_result"]["f1"],
                    d["val_result"]["precision"],
                    d["val_result"]["recall"],
                    d["val_result"]["loss"],
                    int(ep),
                    int(iter),
                ]
            )

        result = np.array(result)
        best_idx_ = np.argmax(result[:, 0])
        print(result[best_idx_])
        f1, _, _, _, ep, iter = result[best_idx_]
        if f1 > best_f1:
            best_f1 = f1
            best_result = p / f"ckpt_{int(ep)}_{int(iter)}.pt"

    print(best_result.parent / f"{cfg.model.type}_best_model.pt")

    shutil.copy(
        best_result, best_result.parent.parent / f"{cfg.model.type}_best_model.pt"
    )


if __name__ == "__main__":
    main()
