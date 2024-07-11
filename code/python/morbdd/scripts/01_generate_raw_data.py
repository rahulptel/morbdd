import hydra
from morbdd.dm import dm_factory


@hydra.main(config_path="../configs", config_name="01_generate_raw_data.yaml", version_base="1.2")
def main(cfg):
    dm = dm_factory(cfg)
    dm.generate_raw_data()


if __name__ == "__main__":
    main()
