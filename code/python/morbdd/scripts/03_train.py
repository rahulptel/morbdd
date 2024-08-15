import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import hydra
from morbdd.trainer import trainer_factory


@hydra.main(config_path="../configs", config_name="03_train.yaml", version_base="1.2")
def main(cfg):
    trainer = trainer_factory(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
