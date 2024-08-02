import random

import numpy as np

from morbdd.utils.mis import get_instance_data
from morbdd.utils.mis import get_instance_path
from .dm import DataManager


class IndepsetDataManager(DataManager):
    def _get_instance_path(self, seed, n_objs, n_vars, split, pid):
        if self.cfg.prob.inst_type == "stidsen":
            return get_instance_path(seed, n_objs, n_vars, split, pid, attach=None, name=self.cfg.prob.name,
                                     prefix=self.cfg.prob.prefix)
        elif self.cfg.prob.inst_type == "ba":
            return get_instance_path(seed, n_objs, n_vars, split, pid, attach=self.cfg.prob.attach,
                                     name=self.cfg.prob.name, prefix=self.cfg.prob.prefix)

    def _generate_instance_ba(self, rng, n_vars, n_objs):
        import networkx as nx

        data = {'n_vars': n_vars,
                'n_objs': n_objs,
                'attach': self.cfg.prob.attach,
                'obj_coeffs': [],
                'edges': []}

        # Value
        for _ in range(n_objs):
            data['obj_coeffs'].append(rng.randint(self.cfg.prob.obj_lb, self.cfg.prob.obj_ub, n_vars))

        graph = nx.barabasi_albert_graph(n_vars, self.cfg.attach, seed=np.random)
        data['edges'] = np.array(nx.edges(graph), dtype=int)

        return data

    def _generate_instance_stidsen(self, rng, n_vars, n_objs):
        data = {'n_vars': n_vars, 'n_objs': n_objs, 'n_cons': int(n_vars / 5), 'obj_coeffs': [], 'cons_coeffs': []}
        items = list(range(1, self.cfg.prob.n_vars + 1))

        # Value
        for _ in range(self.cfg.prob.n_objs):
            data['obj_coeffs'].append(
                list(rng.randint(self.cfg.prob.obj_lb, self.cfg.prob.obj_ub + 1, self.cfg.prob.n_vars)))

        # Constraints
        for _ in range(data['n_cons']):
            vars_in_con = rng.randint(2, (2 * self.cfg.prob.vars_per_con) + 1)
            data['cons_coeffs'].append(list(rng.choice(items, vars_in_con, replace=False)))

        # Ensure no variable is missed
        var_count = []
        for con in data['cons_coeffs']:
            var_count.extend(con)
        missing_vars = list(set(range(1, self.cfg.prob.n_vars + 1)).difference(set(var_count)))
        for v in missing_vars:
            cons_id = rng.randint(data['n_cons'])
            data['cons_coeffs'][cons_id].append(v)

        return data

    def _save_ba(self, inst_path, data):
        # inst_path = Path(str(inst_path) + ".npz")
        np.savez(inst_path,
                 n_vars=data['n_vars'],
                 n_objs=data['n_objs'],
                 attach=data['attach'],
                 obj_coeffs=data['obj_coeffs'],
                 edges=data['edges'])

    def _save_stidsen(self, inst_path, data):
        dat = f"{data['n_vars']} {data['n_cons']}\n"
        dat += f"{len(data['obj_coeffs'])}\n"
        for coeffs in data['obj_coeffs']:
            dat += " ".join(list(map(str, coeffs))) + "\n"

        for coeffs in data["cons_coeffs"]:
            dat += f"{len(coeffs)}\n"
            dat += " ".join(list(map(str, coeffs))) + "\n"

        # inst_path = Path(str(inst_path) + ".dat")
        inst_path.write_text(dat)

    def _generate_instance(self, rng, n_vars, n_objs):
        data = None
        if self.cfg.prob.inst_type == "stidsen":
            data = self._generate_instance_stidsen(rng, n_vars, n_objs)
        elif self.cfg.prob.inst_type == "ba":
            data = self._generate_instance_ba(rng, n_vars, n_objs)

        return data

    def _save_instance(self, inst_path, data):
        if self.cfg.prob.inst_type == "stidsen":
            self._save_stidsen(inst_path, data)
        elif self.cfg.prob.inst_type == "ba":
            self._save_ba(inst_path, data)

    def _get_instance_data(self, pid):
        return get_instance_data(self.cfg.prob.size, self.cfg.split, pid)

    def _get_static_order(self, data):
        return []

    def _get_dynamic_order(self, env):
        return env.get_var_layer()

    @staticmethod
    def _set_inst(env, data):
        env.set_inst(data['n_vars'],
                     data['n_cons'],
                     data['n_objs'],
                     data['obj_coeffs'],
                     data['cons_coeffs'],
                     data['rhs'])

    def _get_pareto_state_scores(self, data, x, order=None):
        x_sol, adj_list_comp = np.array(x), data["adj_list_comp"]
        pareto_state_scores = []

        total = x_sol.shape[0]
        # print(set(list(x_sol[:, 0])))
        for i in range(1, x_sol.shape[1]):
            # Get partial sols upto level i in BDD
            partial_sols = x_sol[:, :i]
            uniques, counts = np.unique(partial_sols, axis=0, return_counts=True)

            pareto_states = []
            pareto_counts = []

            # Compute pareto state for each unique sol
            for unique_sol, count in zip(uniques, counts):
                _pareto_state = np.ones(x_sol.shape[1])
                for var_idx, is_active in enumerate(list(unique_sol)):
                    var = order[var_idx]
                    _pareto_state[var] = 0
                    # print(var_idx, var, is_active)
                    if is_active:
                        # print(adj_list_comp[var])
                        _pareto_state = np.logical_and(_pareto_state, adj_list_comp[var]).astype(int)
                        # print(_pareto_state)

                is_found = False
                for j, ps in enumerate(pareto_states):
                    # print(_pareto_state, ps, np.array_equal(_pareto_state))
                    if np.array_equal(_pareto_state, ps):
                        pareto_counts[j] += count
                        is_found = True
                        break
                if not is_found:
                    # print(_pareto_state, "not found")
                    pareto_states.append(_pareto_state)
                    pareto_counts.append(count)

            for k in range(len(pareto_counts)):
                pareto_counts[k] /= total

            pareto_state_scores.append((pareto_states, pareto_counts))

        return pareto_state_scores

    def _get_bdd_node_dataset(self, pid, inst_data, order, bdd, dataset_path):
        rng = random.Random(self.cfg.seed_dataset)
        dataset = None

        for lid, layer in enumerate(bdd):
            neg_data_lst = []
            pos_data_lst = []
            for nid, node in enumerate(layer):
                # Binary state of the current node
                state = np.zeros(self.cfg.prob.n_vars)
                state[node['s']] = 1
                # Binary state of the parent state
                if self.cfg.with_parent:
                    n_parents_op, n_parents_zp = 0, 0
                    zp_lst, op_lst = np.array([]), np.array([])
                    if lid > 0:
                        n_parents_zp = len(node['zp'])
                        n_parents_op = len(node['op'])

                        for zp in node['zp']:
                            zp_state = np.zeros(self.cfg.prob.n_vars)
                            zp_state[bdd[lid - 1][zp]['s']] = 1
                            zp_lst = np.concatenate((zp_lst, zp_state))

                        op_lst = np.array([])
                        for op in node['op']:
                            op_state = np.zeros(self.cfg.prob.n_vars)
                            op_state[bdd[lid - 1][op]['s']] = 1
                            op_lst = np.concatenate((op_lst, op_state))

                    node_data = np.concatenate(([lid, order[lid + 1]],
                                                state,
                                                [n_parents_zp],
                                                zp_lst,
                                                [n_parents_op],
                                                op_lst))
                else:
                    node_data = np.concatenate(([lid, order[lid + 1]],
                                                state))

                if node['score'] > 0:
                    pos_data_lst.append(node_data)
                else:
                    neg_data_lst.append(node_data)

            # Get label
            pos_data = np.stack(pos_data_lst)
            n_pos = pos_data.shape[0]
            pos_data = np.concatenate((pos_data, np.array([1] * n_pos).reshape(-1, 1)), axis=-1)
            # data_lst.extend(pos_data_lst)
            # Undersample negative class
            n_neg = min(len(neg_data_lst), int(self.cfg.neg_to_pos_ratio * n_pos))
            if n_neg > 0:
                # labels += [0] * n_neg
                rng.shuffle(neg_data_lst)
                neg_data_lst = neg_data_lst[:n_neg]

                neg_data = np.stack(neg_data_lst)
                neg_data = np.concatenate((neg_data, np.array([0] * n_neg).reshape(-1, 1)), axis=-1)

                data = np.concatenate((pos_data, neg_data), axis=0)
            else:
                data = pos_data

            if data.shape[0]:
                np.save(dataset_path / f"{pid}.npy", data)
