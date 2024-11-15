import hydra
from morbdd.dm import dm_factory


@hydra.main(config_path="../configs", config_name="04_eval_learning_metrics.yaml", version_base="1.2")
def main(cfg):
    trainer = trainer_factory(cfg)
    trainer.eval_learning_metrics()


if __name__ == "__main__":
    main()
