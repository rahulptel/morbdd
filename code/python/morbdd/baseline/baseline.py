from morbdd.utils import MetricCalculator


class Baseline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hv_approx = None
        self.cardinality = None
        self.precision = None
        self.times = None
        self.problem = None
        self.inst_data = None
        self.reset()
        self.metric_calculator = MetricCalculator(self.cfg.prob.n_objs)

    def reset(self):
        self.hv_approx = []
        self.cardinality = []
        self.precision = []
        self.times = []
        self.problem = None
        self.inst_data = None

    def save_final_result(self):
        raise NotImplementedError

    def worker(self, pid):
        raise NotImplementedError

    def run(self):
        for pid in range(self.cfg.from_pid, self.cfg.to_pid):
            self.worker(pid)
