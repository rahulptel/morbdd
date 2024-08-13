import json
import signal
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from morbdd import CONST
from morbdd import resource_path
from morbdd.utils import FeaturizerConfig
from morbdd.utils import extract_node_features
from morbdd.utils import get_featurizer
from morbdd.utils import get_instance_data
from morbdd.utils import get_static_order
from morbdd.utils import handle_timeout
from morbdd.utils import statscore
from .deployer import Deployer
import json
import sys


def get_run_data_from_env(env, order_type, was_disconnected):
    sol = {"x": env.x_sol,
           "z": env.z_sol,
           "ot": order_type}

    data = [was_disconnected,
            env.initial_node_count,
            env.reduced_node_count,
            env.initial_arcs_count,
            env.reduced_arcs_count,
            env.num_comparisons,
            sol,
            env.time_result]

    return data


def get_prediction_stats(bdd, pred_stats_per_layer, threshold=0.5, round_upto=1):
    for lidx, layer in enumerate(bdd):
        labels = np.array([node["l"] for node in layer])
        assert np.max(labels) >= threshold

        preds = np.array([node["pred"] for node in layer])
        score = statscore(preds=preds, labels=labels, threshold=threshold, round_upto=round_upto,
                          is_type="numpy")
        tp, fp, tn, fn = np.sum(score[:, 0]), np.sum(score[:, 1]), np.sum(score[:, 2]), np.sum(score[:, 3])
        pred_stats_per_layer[lidx + 1][1] += tp
        pred_stats_per_layer[lidx + 1][2] += fp
        pred_stats_per_layer[lidx + 1][3] += tn
        pred_stats_per_layer[lidx + 1][4] += fn

    return pred_stats_per_layer


def save_stats_per_layer(cfg, pred_stats_per_layer, mdl_hex):
    df = pd.DataFrame(pred_stats_per_layer,
                      columns=["layer", "tp", "fp", "tn", "fn"])
    name = resource_path / f"predictions/{cfg.deploy.mdl}/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}/{mdl_hex}"
    # if cfg.deploy.stitching_heuristic == "min_resistance":
    #     name /= f"{cfg.deploy.select_all_upto}-mrh{cfg.deploy.lookahead}-spl.csv"
    # elif cfg.deploy.stitching_heuristic:
    #     name /= f"{cfg.deploy.select_all_upto}-sph-spl.csv"
    # else:
    #     raise ValueError("Invalid heuristic!")
    name /= "spl.csv"
    df.to_csv(name, index=False)


def save_bdd_data(cfg, pids, bdd_data, mdl_hex):
    out_path = resource_path / f"predictions/{cfg.deploy.mdl}/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}/{mdl_hex}"
    if cfg.deploy.stitching_heuristic == "min_resistance":
        disconnected_prefix = f"{cfg.deploy.select_all_upto}-mrh{cfg.deploy.lookahead}"
    elif cfg.deploy.stitching_heuristic == "shortest_path":
        disconnected_prefix = f"{cfg.deploy.select_all_upto}-sph"
    elif cfg.deploy.stitching_heuristic == "mip":
        disconnected_prefix = f"{cfg.deploy.select_all_upto}-mip"
    else:
        raise ValueError("Invalid heuristic!")

    bdd_stats = []
    bdd_stats_disconnected = []
    for pid, data in zip(pids, bdd_data):
        time_stitching, time_mip, count_stitching, was_disconnected, inc, rnc, iac, rac, num_comparisons, sol, _time = data

        sol_pred_path = out_path / f"{disconnected_prefix}-sols_pred" \
            if was_disconnected else out_path / "sols_pred"
        sol_pred_path.mkdir(exist_ok=True, parents=True)
        sol_path = sol_pred_path / f"sol_{pid}.json"
        with open(sol_path, "w") as fp:
            json.dump(sol, fp)
        time_path = sol_pred_path / f"time_{pid}.json"
        with open(time_path, "w") as fp:
            json.dump(_time, fp)

        if was_disconnected:
            bdd_stats_disconnected.append([cfg.prob.size,
                                           pid,
                                           cfg.deploy.split,
                                           1,
                                           count_stitching,
                                           time_stitching,
                                           time_mip,
                                           cfg.deploy.stitching_heuristic,
                                           cfg.deploy.lookahead,
                                           cfg.deploy.select_all_upto,
                                           len(sol["x"]),
                                           inc,
                                           rnc,
                                           iac,
                                           rac,
                                           num_comparisons,
                                           _time["compilation"],
                                           _time["reduction"],
                                           _time["pareto"]])
        else:
            bdd_stats.append([cfg.prob.size,
                              pid,
                              cfg.deploy.split,
                              0,
                              0,
                              0,
                              0,
                              "",
                              "",
                              cfg.deploy.select_all_upto,
                              len(sol["x"]),
                              inc,
                              rnc,
                              iac,
                              rac,
                              num_comparisons,
                              _time["compilation"],
                              _time["reduction"],
                              _time["pareto"]])

    columns = ["size",
               "pid",
               "split",
               "was_disconnected",
               "count_stitching",
               "time_stitching",
               "time_mip",
               "stitching_heuristic",
               "lookahead",
               "select_all_upto",
               "pred_nnds",
               "pred_inc",
               "pred_rnc",
               "pred_iac",
               "pred_rac",
               "pred_num_comparisons",
               "pred_compile",
               "pred_reduce",
               "pred_pareto"]
    if len(bdd_stats):
        df = pd.DataFrame(bdd_stats, columns=columns)
        df.to_csv(out_path / f"pred_result_{pids[0]}.csv", index=False)

    if len(bdd_stats_disconnected):
        df = pd.DataFrame(bdd_stats_disconnected, columns=columns)
        df.to_csv(out_path / f"{disconnected_prefix}-pred_result_{pids[0]}.csv", index=False)


class BDDLayerToXGBConverter:
    def __init__(self):
        featurizer_conf = FeaturizerConfig()
        self.featurizer = get_featurizer("knapsack", featurizer_conf)
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

    def convert_bdd_layer(self, lidx, layer, prev_layer):
        """lidx starts from 0 for layer 1"""
        features_lst = []

        # Parent variable features
        _parent_var_feat = -1 * np.ones(self.num_var_features) \
            if lidx == 0 \
            else self.var_features[lidx - 1]
        _var_feat = self.var_features[lidx]

        # prev_layer = bdd[lidx - 1]
        for node in layer:
            _node_feat, _parent_node_feat = extract_node_features("knapsack",
                                                                  lidx,
                                                                  node,
                                                                  prev_layer,
                                                                  self.inst_data,
                                                                  layer_norm_const=self.layer_norm_const,
                                                                  state_norm_const=self.state_norm_const)

            # Features
            features_lst.append(np.concatenate((self.inst_features,
                                                _parent_var_feat,
                                                _parent_node_feat,
                                                _var_feat,
                                                _node_feat)))
            # labels_lst.append(node["l"])

        return np.array(features_lst)


class KnapsackDeployer(Deployer):
    def __init__(self, cfg):
        Deployer.__init__(self, cfg)
        self.trainer = None
        self.env = None
        self.converter = BDDLayerToXGBConverter() if self.cfg.model.type == "gbt" else None

    def set_trainer(self):
        if self.cfg.model.type == "gbt":
            from morbdd.trainer.xgb import XGBTrainer
            self.trainer = XGBTrainer(self.cfg)
            self.trainer.setup_predict()
        else:
            raise ValueError("Invalid model name!")

    @staticmethod
    def node_selector(scores, round_upto=1, tau=0.5):
        idx_score = [(i, s) for i, s in enumerate(scores)]
        selection = [0] * len(scores)

        selected_idx = []
        for i in idx_score:
            if np.round(i[1], round_upto) >= tau:
                selection[i[0]] = 1
                selected_idx.append(i[0])

        removed_idx = list(set(np.arange(len(scores))).difference(set(selected_idx)))

        return selection, selected_idx, removed_idx

    def worker(self, rank):
        self.set_trainer()
        alpha_lid, beta_lid = (self.cfg.deploy.node_select.alpha / 100), (self.cfg.deploy.node_select.beta / 100)
        alpha_lid = int(alpha_lid * self.cfg.prob.n_vars)
        beta_lid = self.cfg.prob.n_vars - int(beta_lid * self.cfg.prob.n_vars)
        print(alpha_lid, beta_lid)
        pred_sol = json.load(open('1100.json', 'r'))
        pred_bdd = json.load(open('1100_bdd.json', 'r'))
        for pid in range(self.cfg.deploy.from_pid + rank, self.cfg.deploy.to_pid, self.cfg.deploy.n_processes):
            env = self.get_env()

            start = time.time()
            # Read instance
            inst_data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.deploy.split, pid)
            order = get_static_order(self.cfg.prob.name, self.cfg.deploy.order_type, inst_data)
            print(order)
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
            lid = 0

            # Restrict and build
            while lid < inst_data["n_vars"] - 1:
                env.generate_next_layer()
                prev_layer = env.get_layer(lid)
                layer = env.get_layer(lid + 1)
                features = self.converter.convert_bdd_layer(lid, layer, prev_layer)
                # print(features.shape)
                # print(features[:2])
                dmatrix = xgb.DMatrix(np.array(features))
                scores = self.trainer.predict(dmatrix)

                # countA, countB = 0, 0
                # for m in range(len(list(scores))):
                #     countA += int(np.round(scores[m], 1) >= 0.5)
                #     countB += int(np.round(pred_sol[lid][m]['pred'], 1) >= 0.5)
                #
                # scoresA = set(scores)
                # scoresB = set([pred_sol[lid][m]['pred'] for m in range(len(layer))])
                # diff = scoresA.difference(scoresB)
                # print(lid, diff)

                # for m in range(len(list(scores))):
                #     if scores[m] != pred_sol[lid][m]['pred']:
                #         print(lid, m, scores[m], )
                #         sys.exit()
                lid += 1
                # print(lid, len(scores), np.mean(scores), np.median(scores), np.min(scores), np.max(scores))

                if alpha_lid < lid < beta_lid:
                    _, _, removed_idx = self.node_selector(scores, tau=self.cfg.deploy.node_select.tau)
                    # Stitch in case of a disconnected BDD
                    if len(removed_idx) == len(scores):
                        removed_idx = []
                        # env.stitch(lid, self.cfg.deploy.lookback)
                        print("Disconnected at layer: ", {lid})
                    # Restrict if necessary
                    if len(removed_idx):
                        env.approximate_layer(lid, CONST.RESTRICT, 1, removed_idx)
                # if lid == 7:
                #     sys.exit()
            # Generate terminal layer
            env.generate_next_layer()
            build_time = time.time() - start

            # env.reduce_dd()
            # time_compile = env.get_time(CONST.TIME_COMPILE)
            #
            # print(f"{rank}/6/10: Fetching decision diagram...")
            # start = time.time()
            # dd = env.get_dd()
            # time_fetch = time.time() - start
            #
            # print(f"{rank}/7/10: Computing Pareto Frontier...")
            # try:
            #     signal.alarm(self.cfg.prob.time_limit)
            #     env.compute_pareto_frontier()
            # except TimeoutError:
            #     is_pf_computed = False
            #     print(f"PF not computed within {self.cfg.prob.time_limit} for pid {pid}")
            # else:
            #     is_pf_computed = True
            #     print(f"PF computed successfully for pid {pid}")
            # signal.alarm(0)
            # if not is_pf_computed:
            #     continue
            # time_pareto = env.get_time(CONST.TIME_PARETO)
            #
            # frontier = env.get_frontier()
            # print(f"{pid}: |Z| = {len(frontier['z'])}")
            #
            # # Save dd, solution and stats
            # # self._save_dd(pid, dd)
            # # self._save_solution(pid, frontier, sol_var_order)
            # # self._save_dm_stats(pid, frontier, env, time_fetch, time_compile, time_pareto)
