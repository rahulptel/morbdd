import hydra
from morbdd.baseline import baseline_factory


@hydra.main(config_path="../configs", config_name="06_baseline.yaml", version_base="1.2")
def main(cfg):
    baseline = baseline_factory(cfg)
    baseline.run()


if __name__ == "__main__":
    main()
