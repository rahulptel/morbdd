import hydra
from morbdd.deployer import deployer_factory


@hydra.main(config_path="../configs", config_name="05_deploy.yaml", version_base="1.2")
def main(cfg):
    deployer = deployer_factory(cfg)
    deployer.deploy()


if __name__ == "__main__":
    main()
