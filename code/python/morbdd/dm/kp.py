import random

import numpy as np

from morbdd.featurizer.knapsack import KnapsackFeaturizer
from morbdd.utils import FeaturizerConfig
from morbdd.utils import get_dataset_path
from morbdd.utils.kp import get_bdd_node_features
from morbdd.utils.kp import get_instance_data
from morbdd.utils.kp import get_instance_path
from morbdd.utils.kp import get_static_order
from .dm import DataManager


class KnapsackDataManager(DataManager):
    def _get_instance_path(self, seed, n_objs, n_vars, split, pid):
        return get_instance_path(seed, n_objs, n_vars, split, pid, name=self.cfg.prob.name, prefix=self.cfg.prob.prefix)

    def _generate_instance(self, rng, n_vars, n_objs):
        data = {"value": [], "weight": [], "capacity": 0}

        # Cost
        data["weight"] = rng.randint(self.cfg.prob.cons_lb, self.cfg.prob.cons_ub + 1, n_vars)
        # Value
        if self.cfg.prob.inst_type == "uncorr":
            for _ in range(n_objs):
                data["value"].append(rng.randint(self.cfg.prob.obj_lb, self.cfg.prob.obj_ub + 1, n_vars))
        else:
            for _ in range(n_objs):
                _objective = []
                for i in range(n_vars):
                    lb = max(self.cfg.prob.obj_lb, data["weight"][i] - (self.cfg.prob.obj_ub / 10))
                    ub = data["weight"][i] + (self.cfg.prob.obj_ub / 10)
                    _objective.append(rng.randint(lb, ub))
                data["value"].append(_objective)

        # Capacity
        data["capacity"] = np.ceil(0.5 * (np.sum(data["weight"])))

        return data

    def _save_instance(self, inst_path, data):
        inst_path.parent.mkdir(parents=True, exist_ok=True)

        n_vars = len(list(data["weight"]))
        n_objs = len(data["value"])

        text = f"{n_vars}\n{n_objs}\n"
        for i in range(n_objs):
            string = " ".join([str(v) for v in data["value"][i]])
            text += string + "\n"
        string = " ".join([str(w) for w in data["weight"]])
        text += string + "\n"
        text += str(int(data["capacity"]))

        inst_path.open("w").write(text)

    def _get_instance_data(self, pid):
        return get_instance_data(self.cfg.prob.name, self.cfg.prob.prefix, self.cfg.seed, self.cfg.prob.size,
                                 self.cfg.split, pid)

    @staticmethod
    def _set_inst(env, data):
        env.set_inst(data['n_vars'],
                     data['n_cons'],
                     data['n_objs'],
                     data['value'],
                     [data['weight']],
                     [data['capacity']])

    def _get_static_order(self, data):
        return get_static_order(self.cfg.prob.order_type, data)

    def _get_dynamic_order(self, env):
        return []

    def _get_pareto_state_scores(self, data, x, order=None):
        weight = data["cons_coeffs"][0]
        pareto_state_scores = []
        for i in range(1, x.shape[1]):
            x_partial = x[:, :i].reshape(-1, i)
            w_partial = weight[:i].reshape(i, 1)
            wt_dist = np.dot(x_partial, w_partial)
            pareto_state, pareto_score = np.unique(wt_dist, return_counts=True)
            pareto_score = pareto_score / pareto_score.sum()
            pareto_state_scores.append((pareto_state, pareto_score))

        return pareto_state_scores

    def _get_bdd_node_dataset_tf(self, pid, inst_data, order, bdd, rng):
        features_lst, labels_lst, weights_lst = [], [], []
        for lidx, layer in enumerate(bdd):
            # _features_lst, _labels_lst, _weights_lst = [], [], []
            pos_ids = [node_id for node_id, node in enumerate(layer) if node["pareto"] == 1]
            neg_ids = list(set(range(len(layer))).difference(set(pos_ids)))

            # Subsample negative samples
            num_pos_samples = len(pos_ids)
            if self.cfg.neg_to_pos_ratio < 1:
                num_neg_samples = len(neg_ids)
            else:
                num_neg_samples = int(self.cfg.neg_pos_ratio * num_pos_samples)
                num_neg_samples = np.min([num_neg_samples, len(neg_ids)])
                rng.shuffle(neg_ids)
            neg_ids = neg_ids[:num_neg_samples]

            prev_layer = bdd[lidx - 1] if self.cfg.with_parent and lidx > 0 else None
            node_ids = pos_ids[:]
            node_ids.extend(neg_ids)
            for i, node_id in enumerate(node_ids):
                node = layer[node_id]
                _node_feat = get_bdd_node_features(lidx, node, prev_layer, inst_data["capacity"],
                                                   layer_norm_const=self.cfg.prob.layer_norm_const,
                                                   state_norm_const=self.cfg.prob.state_norm_const,
                                                   with_parent=self.cfg.with_parent)

                features_lst.append(np.concatenate((_node_feat, [order[lidx], node["score"]])))

        return np.array(features_lst)

    def _get_bdd_node_dataset_xgboost(self, inst_data, order, bdd, rng):
        # Extract instance and variable features
        featurizer = KnapsackFeaturizer(FeaturizerConfig(norm_const=self.cfg.prob.state_norm_const,
                                                         raw=False,
                                                         context=True))
        features = featurizer.get(inst_data)
        # Instance features
        inst_features = features["inst"][0]
        # Variable features. Reordered features based on ordering
        var_features = features["var"][order]
        num_var_features = features["var"].shape[1]

        features_lst, labels_lst, weights_lst = [], [], []
        for lidx, layer in enumerate(bdd):
            # _features_lst, _labels_lst, _weights_lst = [], [], []
            pos_ids = [node_id for node_id, node in enumerate(layer) if node["pareto"] == 1]
            neg_ids = list(set(range(len(layer))).difference(set(pos_ids)))

            # Subsample negative samples
            num_pos_samples = len(pos_ids)
            if self.cfg.neg_to_pos_ratio < 1:
                num_neg_samples = len(neg_ids)
            else:
                num_neg_samples = int(self.cfg.neg_pos_ratio * num_pos_samples)
                num_neg_samples = np.min([num_neg_samples, len(neg_ids)])
                rng.shuffle(neg_ids)
            neg_ids = neg_ids[:num_neg_samples]

            _var_feat = var_features[lidx]
            _parent_var_feat = None
            if self.cfg.with_parent:
                # Variable features: Parent and current layer
                _parent_var_feat = -1 * np.ones(num_var_features) if lidx == 0 else var_features[lidx - 1]

            prev_layer = bdd[lidx - 1] if self.cfg.with_parent and lidx > 0 else None
            node_ids = pos_ids[:]
            node_ids.extend(neg_ids)
            for i, node_id in enumerate(node_ids):
                node = layer[node_id]
                _node_feat = get_bdd_node_features(lidx, node, prev_layer, inst_data["capacity"],
                                                   layer_norm_const=self.cfg.prob.layer_norm_const,
                                                   state_norm_const=self.cfg.prob.state_norm_const,
                                                   with_parent=self.cfg.with_parent)
                if self.cfg.with_parent:
                    features_lst.append(np.concatenate((inst_features,
                                                        _parent_var_feat,
                                                        _var_feat,
                                                        _node_feat,
                                                        node["score"])))
                else:
                    features_lst.append(np.concatenate((inst_features,
                                                        _var_feat,
                                                        _node_feat,
                                                        node["score"])))

        return np.array(features_lst)

    def _get_bdd_node_dataset(self, pid, data, order, bdd, dataset_path):
        rng = random.Random(self.cfg.seed_dataset)
        dataset = None
        if self.cfg.for_model == "tf":
            dataset = self._get_bdd_node_dataset_tf(pid, data, order, bdd, rng)
        elif self.cfg.for_model == "xgboost":
            dataset = self._get_bdd_node_dataset_xgboost(data, order, bdd, rng)

        if dataset is not None:
            np.save(dataset_path / f"{pid}.npy", dataset)
