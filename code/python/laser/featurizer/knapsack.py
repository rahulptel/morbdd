from enum import Enum
import numpy as np

from operator import itemgetter


class KnapsackStaticOrderings(Enum):
    max_weight = 0
    min_weight = 1
    max_avg_value = 2
    min_avg_value = 3
    max_max_value = 4
    min_max_value = 5
    max_min_value = 6
    min_min_value = 7
    max_avg_value_by_weight = 8
    max_max_value_by_weight = 9


feat_names = {'inst': ['n_objs',
                       'n_vars',
                       'capacity',
                       'weight.mean',
                       'weight.min',
                       'weight.max',
                       'weight.std',
                       '<value.mean_per_obj>.mean',
                       '<value.mean_per_obj>.min',
                       '<value.mean_per_obj>.max',
                       '<value.mean_per_obj>.std',
                       '<value.min_per_obj>.mean',
                       '<value.min_per_obj>.min',
                       '<value.min_per_obj>.max',
                       '<value.min_per_obj>.std',
                       '<value.max_per_obj>.mean',
                       '<value.max_per_obj>.min',
                       '<value.max_per_obj>.max',
                       '<value.max_per_obj>.std'],
              'var': ['weight',
                      'value.mean',
                      'value.min',
                      'value.max',
                      'value.std',
                      'value.mean/wt',
                      'value.max/wt',
                      'value.min/wt'],
              'vrank': ['rk_des_weight',
                        'rk_asc_weight',
                        'rk_des_value.mean',
                        'rk_asc_value.mean',
                        'rk_des_value.max',
                        'rk_asc_value.max',
                        'rk_des_value.min',
                        'rk_asc_value.min',
                        'rk_des_value.mean/wt',
                        'rk_des_value.max/wt']}


class KnapsackFeaturizer:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.norm_const = self.cfg.norm_const

    def _set_data(self, data):
        self.data = data
        # Get features
        self.norm_value = (1 / self.norm_const) * np.asarray(self.data['value'])
        self.norm_weight = (1 / self.norm_const) * np.asarray(self.data['weight'])
        self.n_objs, self.n_vars = self.norm_value.shape

    def _get_instance_features(self):
        inst_feat = [self.n_objs / 7,
                     self.n_vars / 100,
                     (np.ceil(self.norm_weight.sum()) / 2) / self.n_vars,  # Normalized capacity
                     self.norm_weight.mean(), self.norm_weight.min(), self.norm_weight.max(),
                     self.norm_weight.std()]  # Weight aggregate stats

        # Value double-aggregate stats
        value_mean = self.norm_value.mean(axis=1)
        value_min = self.norm_value.min(axis=1)
        value_max = self.norm_value.max(axis=1)

        inst_feat.extend([value_mean.mean(), value_mean.min(), value_mean.max(), value_mean.std(),
                          value_min.mean(), value_min.min(), value_min.max(), value_min.std(),
                          value_max.mean(), value_max.min(), value_max.max(), value_max.std()])

        inst_feat = np.asarray(inst_feat)

        return inst_feat

    def _get_heuristic_variable_rank_features(self):
        self.n_vars = len(self.data['weight'])

        ranks = []
        for o in KnapsackStaticOrderings:
            if o.name == 'max_weight':
                idx_weight = [(i, w) for i, w in enumerate(self.data['weight'])]
                idx_weight.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_weight)

            elif o.name == 'min_weight':
                idx_weight = [(i, w) for i, w in enumerate(self.data['weight'])]
                idx_weight.sort(key=itemgetter(1))
                idx_rank = self._get_rank(idx_weight)

            elif o.name == 'max_avg_value':
                mean_profit = np.mean(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
                idx_profit.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'min_avg_value':
                mean_profit = np.mean(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
                idx_profit.sort(key=itemgetter(1))
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'max_max_value':
                max_profit = np.max(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
                idx_profit.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'min_max_value':
                max_profit = np.max(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
                idx_profit.sort(key=itemgetter(1))
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'max_min_value':
                min_profit = np.min(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
                idx_profit.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'min_min_value':
                min_profit = np.min(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
                idx_profit.sort(key=itemgetter(1))
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'max_avg_value_by_weight':
                mean_profit = np.mean(self.data['value'], 0)
                profit_by_weight = [v / w for v, w in zip(mean_profit, self.data['weight'])]
                idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
                idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_profit_by_weight)

            elif o.name == 'max_max_value_by_weight':
                max_profit = np.max(self.data['value'], 0)
                profit_by_weight = [v / w for v, w in zip(max_profit, self.data['weight'])]
                idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
                idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_profit_by_weight)

            ranks.append([idx_rank[i] for i in range(self.n_vars)])

        ranks = (1 / self.n_vars) * np.asarray(ranks)
        return ranks

    def _get_variable_features(self):
        return np.vstack([self.norm_weight,
                          self.norm_value.mean(axis=0),
                          self.norm_value.min(axis=0),
                          self.norm_value.max(axis=0),
                          self.norm_value.std(axis=0),
                          self.norm_value.mean(axis=0) / self.norm_weight,
                          self.norm_value.max(axis=0) / self.norm_weight,
                          self.norm_value.min(axis=0) / self.norm_weight])

        # return variable_features
        #     item_features = np.vstack([item_features,
        #                                idx_rank_array])

    @staticmethod
    def _get_rank(sorted_data):
        idx_rank = {}
        for rank, item in enumerate(sorted_data):
            idx_rank[item[0]] = rank

        return idx_rank

    def get(self, data=None):
        self._set_data(data)

        # Calculate instance features
        feat = {'raw': None, 'inst': None, 'var': None, 'vrank': None}

        if self.cfg.raw:
            raw_feat = np.vstack((self.norm_value,
                                  self.norm_weight,
                                  np.repeat(self.data['capacity'] / self.norm_const, self.n_vars).reshape(1, -1)))
            feat['raw'] = raw_feat.T
            assert feat['raw'].shape[1] == self.data['n_objs'] + 2

        if self.cfg.context:
            inst_feat = self._get_instance_features()
            inst_feat = inst_feat.reshape((1, -1))
            inst_feat = np.repeat(inst_feat, self.n_vars, axis=0)
            feat['inst'] = inst_feat
            assert feat['inst'].shape[1] == len(feat_names['inst'])

        # Calculate item features
        var_feat = self._get_variable_features()
        feat['var'] = var_feat.T
        # print(item_feat.shape)
        assert feat['var'].shape[1] == len(feat_names['var'])

        var_rank_feat = self._get_heuristic_variable_rank_features()
        feat['vrank'] = var_rank_feat.T
        assert feat['vrank'].shape[1] == len(feat_names['vrank'])

        return feat
