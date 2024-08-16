import signal
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from morbdd import CONST
from morbdd import ResourcePaths as path
from morbdd.utils import FeaturizerConfig
from morbdd.utils import get_featurizer
from morbdd.utils import get_instance_data
from morbdd.utils import get_static_order
from morbdd.utils import handle_timeout
from morbdd.utils import read_from_zip
# from morbdd.utils.kp import get_bdd_node_features
from morbdd.utils.kp import get_bdd_node_features_gbt_rank
from .deployer import Deployer


class BDDLayerToXGBConverter:
    def __init__(self, with_parent=False):
        featurizer_conf = FeaturizerConfig()
        self.featurizer = get_featurizer("knapsack", featurizer_conf)
        self.with_parent = with_parent
        self.inst_data = None
        self.order = None
        self.features = None
        self.inst_features = None
        self.var_features = None
        self.num_var_features = 0
        self.layer_norm_const = 100
        self.state_norm_const = 1000

    def set_features(self, inst_data, order):
        self.inst_data = inst_data
        self.order = order

        self.features = self.featurizer.get(inst_data)
        # Instance features
        self.inst_features = self.features["inst"][0]
        # Variable features. Reordered features based on ordering
        self.var_features = self.features["var"][order]
        self.num_var_features = self.var_features.shape[1]

    def convert_bdd_layer(self, lidx, layer):
        """lidx starts from 0 for layer 1"""
        features_lst = []

        # Parent variable features
        var_feat = self.var_features[lidx]
        for node in layer:
            node_feat = get_bdd_node_features_gbt_rank(lidx, node, self.inst_data["capacity"],
                                                       self.layer_norm_const, self.state_norm_const)

            features_lst.append(np.concatenate((self.inst_features,
                                                var_feat,
                                                node_feat)))

        return np.array(features_lst)


class KnapsackDeployer(Deployer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trainer = None
        self.converter = None

        if "gbt" in self.cfg.model.type:
            self.converter = BDDLayerToXGBConverter()
        assert self.converter is not None

    def set_trainer(self):
        if self.cfg.model.type == "gbt":
            from morbdd.trainer.xgb import XGBTrainer
            self.trainer = XGBTrainer(self.cfg)
            self.trainer.setup_predict()
        if self.cfg.model.type == "gbt_rank":
            from morbdd.trainer.xgb import XGBRankTrainer
            self.trainer = XGBRankTrainer(self.cfg)
            self.trainer.setup_predict()
        else:
            raise ValueError("Invalid model name!")

    def save_result(self, pid, build_time, pareto_time):
        total_time = build_time + pareto_time
        df = pd.DataFrame([[self.cfg.prob.size, self.cfg.deploy.split, pid, total_time, self.size_ratio, self.orig_size,
                            self.rest_size, self.orig_width, self.rest_width, self.cardinality, self.cardinality_raw,
                            self.precision, len(self.pred_pf), build_time, pareto_time]],
                          columns=["size", "split", "pid", "total_time", "size_ratio", "orig_size", "rest_size",
                                   "orig_width", "rest_width", "cardinality", "cardinality_raw", "pred_precision",
                                   "n_pred_pf", "build_time", "pareto_time"])
        print(df)
        pid = str(pid) + ".csv"

        print(self.trainer.mdl_name)
        # self.run_path = self.get_run_path(self.trainer.ckpt_path.stem)
        # self.run_path.mkdir(parents=True, exist_ok=True)
        # result_path = self.run_path / pid
        # df.to_csv(result_path)

    def build_dd(self, env):
        lid = 0

        # Restrict and build
        while lid < self.cfg.prob.n_vars - 1:
            env.generate_next_layer()
            lid += 1

            layer = env.get_layer(lid)
            print(lid, len(layer))
            if len(layer) > self.cfg.deploy.node_select.width:
                print("Restricting...")
                features = self.converter.convert_bdd_layer(lid, layer)

                scores = self.trainer.predict(xgb.DMatrix(np.array(features)))

                _, _, removed_idx = self.node_selector(scores)
                # Restrict if necessary
                if len(removed_idx):
                    env.approximate_layer(lid, CONST.RESTRICT, 1, removed_idx)
                print(len(env.get_layer(lid)))
        # Generate terminal layer
        env.generate_next_layer()

    def deploy(self):
        self.set_trainer()
        # self.set_alpha_beta_lid()

        for pid in range(self.cfg.deploy.from_pid, self.cfg.deploy.to_pid):
            archive = path.bdd / f"{self.cfg.prob.name}/{self.cfg.prob.size}.zip"
            file = f"{self.cfg.prob.size}/{self.cfg.deploy.split}/{pid}.json"
            bdd = read_from_zip(archive, file, format="json")
            if bdd is not None:
                # Read instance
                inst_data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.deploy.split, pid)
                order = get_static_order(self.cfg.prob.name, self.cfg.deploy.order_type, inst_data)
                print("Static Order: ", order)

                env = self.get_env()
                self.converter.set_features(inst_data, order)

                signal.signal(signal.SIGALRM, handle_timeout)
                env.reset(self.cfg.prob.problem_type,
                          self.cfg.prob.preprocess,
                          self.cfg.prob.pf_enum_method,
                          self.cfg.prob.maximization,
                          self.cfg.prob.dominance,
                          self.cfg.prob.bdd_type,
                          self.cfg.prob.maxwidth,
                          order)

                env.set_inst(inst_data['n_vars'],
                             inst_data['n_cons'],
                             inst_data['n_objs'],
                             list(np.array(inst_data['value']).T),
                             [inst_data['weight']],
                             [inst_data['capacity']])
                env.preprocess_inst()
                env.initialize_dd_constructor()

                start = time.time()
                self.build_dd(env)
                build_time = time.time() - start

                start = time.time()
                env.reduce_dd()
                reduce_time = time.time() - start
                build_time += reduce_time

                print(f"/7/10: Computing Pareto Frontier...")
                try:
                    signal.alarm(self.cfg.prob.time_limit)
                    env.compute_pareto_frontier()
                    self.pred_pf = env.get_frontier()["z"]
                except TimeoutError:
                    is_pf_computed = False
                    print(f"PF not computed within {self.cfg.prob.time_limit} for pid {pid}")
                else:
                    is_pf_computed = True
                    print(f"PF computed successfully for pid {pid}")
                signal.alarm(0)
                if not is_pf_computed:
                    continue
                time_pareto = env.get_time(CONST.TIME_PARETO)

                self.post_process(env, pid)
                self.save_result(pid, build_time, time_pareto)
                # frontier = env.get_frontier()
                # print(f"{pid}: |Z| = {len(frontier['z'])}")


class KnapsackRankDeployer(KnapsackDeployer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def set_alpha_beta_lid(self):
        pass
