import hydra

from test_bin.tester import tester_factory


@hydra.main(version_base="1.2", config_path="./configs", config_name="test_bin.yaml")
def main(cfg):
    tester = tester_factory(cfg)
    tester.run_test()


if __name__ == "__main__":
    main()
