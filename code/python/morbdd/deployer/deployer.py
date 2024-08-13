import json
from abc import ABC

from morbdd import ResourcePaths as path
from morbdd.utils import LayerNodeSelector
from morbdd.utils import compute_cardinality
from morbdd.utils import read_from_zip


class LayerStitcher:
    def __init__(self, strategy="select_all"):
        self.strategy = strategy

    def __call__(self, scores):
        if self.strategy == "select_all":
            return []


class Deployer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha_lid = 0
        self.beta_lid = 1000
        self.trainer = None
        self.run_path = None
        self.size_ratio = None
        self.pred_pf = None
        self.cardinality_raw = None
        self.cardinality = None
        self.rest_size = None
        self.orig_size = None

        self.node_selector = LayerNodeSelector(self.cfg.deploy.node_select.strategy,
                                               tau=self.cfg.deploy.node_select.tau)
        self.layer_stitcher = LayerStitcher()

    def get_env(self):
        libbddenv = __import__("libbddenvv2o" + str(self.cfg.prob.n_objs))
        env = libbddenv.BDDEnv()

        return env

    def set_alpha_beta_lid(self):
        alpha_lid, beta_lid = (self.cfg.deploy.node_select.alpha / 100), (self.cfg.deploy.node_select.beta / 100)
        self.alpha_lid = int(alpha_lid * self.cfg.prob.n_vars)
        self.beta_lid = self.cfg.prob.n_vars - int(beta_lid * self.cfg.prob.n_vars)
        print(f"Restrict layers: {self.alpha_lid} < l < {self.beta_lid}")

    def set_trainer(self):
        pass

    def load_orig_dd(self, pid):
        size = self.cfg.prob.size + ".zip"
        archive_bdds = path.bdd / self.cfg.prob.name / size
        file = f"{self.cfg.prob.size}/{self.cfg.deploy.split}/{pid}.json"
        orig_dd = read_from_zip(archive_bdds, file, format="json")

        return orig_dd

    @staticmethod
    def compute_dd_size(dd):
        s = 0
        for l in dd:
            s += len(l)

        return s

    def load_pf(self, pid):
        pid = str(pid) + ".json"
        sol_path = path.sol / self.cfg.prob.name / self.cfg.prob.size / self.cfg.deploy.split / pid
        if sol_path.exists():
            print("Reading sol: ", sol_path)
            with open(sol_path, "r") as fp:
                sol = json.load(fp)
                return sol["z"]

        print("Sol path not found!")

    def get_run_path(self, model_name):
        method = str(self.cfg.deploy.node_select.alpha) + "-" + str(self.cfg.deploy.node_select.beta) + "-"
        method += self.cfg.deploy.node_select.strategy + "-"
        if self.cfg.deploy.node_select.strategy == "threshold":
            method += str(self.cfg.deploy.node_select.tau)

        run_path = path.resource / "sols_pred" / self.cfg.prob.name / self.cfg.prob.size / self.cfg.deploy.split / model_name / method
        return run_path

    def post_process(self, env, pid):
        orig_dd = self.load_orig_dd(pid)
        restricted_dd = env.get_dd()
        self.orig_size = self.compute_dd_size(orig_dd)
        self.rest_size = self.compute_dd_size(restricted_dd)
        self.size_ratio = self.rest_size / self.orig_size
        true_pf = self.load_pf(pid)

        self.cardinality_raw, self.cardinality = -1, -1
        if true_pf is not None and self.pred_pf is not None:
            self.cardinality_raw = compute_cardinality(true_pf=true_pf, pred_pf=self.pred_pf)
            self.cardinality = self.cardinality_raw / len(true_pf)

    def save_result(self, *args):
        pass

    def deploy(self):
        pass
