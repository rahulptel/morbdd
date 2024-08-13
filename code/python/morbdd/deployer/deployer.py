import multiprocessing as mp
from abc import abstractmethod, ABC


class Deployer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def worker(self, rank):
        pass

    def get_env(self):
        libbddenv = __import__("libbddenvv2o" + str(self.cfg.prob.n_objs))
        env = libbddenv.BDDEnv()

        return env

    def deploy(self):
        if self.cfg.deploy.n_processes == 1:
            self.worker(0)
        else:
            pool = mp.Pool(processes=self.cfg.deployer.n_processes)
            results = []

            for rank in range(self.cfg.deployer.n_processes):
                results.append(pool.apply_async(self.worker, args=(rank,)))

            for r in results:
                r.get()
