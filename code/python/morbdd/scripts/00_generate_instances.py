import hydra
from morbdd.dm import dm_factory


@hydra.main(config_path="../configs", config_name="00_generate_instances.yaml", version_base="1.2")
def main(cfg):
    dm = dm_factory(cfg)
    dm.generate_instances()


if __name__ == "__main__":
    main()
