import hydra
from morbdd.dm import dm_factory


@hydra.main(config_path="../configs", config_name="05_deploy.yaml", version_base="1.2")
def main(cfg):
    trainer = trainer_factory(cfg)
    trainer.predict()


if __name__ == "__main__":
    main()
