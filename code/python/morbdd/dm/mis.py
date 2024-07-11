from .dm import DataManager
import numpy as np
from morbdd.utils.mis import get_instance_data
from morbdd import ResourcePaths as path
from morbdd.utils import read_from_zip


class MISDataManager(DataManager):
    def generate_instance(self):
        pass

    def save_instance(self, inst_path, data):
        dat = f"{data['n_vars']} {data['n_cons']}\n"
        dat += f"{len(data['obj_coeffs'])}\n"
        for coeffs in data['obj_coeffs']:
            dat += " ".join(list(map(str, coeffs))) + "\n"

        for coeffs in data["cons_coeffs"]:
            dat += f"{len(coeffs)}\n"
            dat += " ".join(list(map(str, coeffs))) + "\n"

        inst_path.write_text(dat)

    def get_pareto_state_score_per_layer(self, *args):
        pass

    def get_dynamic_order(self, env):
        if self.cfg.prob.order_type == "min_state":
            return env.get_var_layer

    def get_pareto_state_score(self, data, x, order=None):
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

    def generate_dataset_worker(self, rank):
        archive_bdds = path.bdd / f"{self.cfg.prob.name}/{self.cfg.size}.zip"
        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            # Read instance data
            data = get_instance_data(self.cfg.prob.name, self.cfg.size, self.cfg.split, pid)
            file = f"{self.cfg.size}/{self.cfg.split}/{pid}.json"
            bdd = read_from_zip(archive_bdds, file, format="json")
            # Read order
            order = path.order.joinpath(f"{self.cfg.prob.name}/{self.cfg.size}/{self.cfg.split}/{pid}.dat").read_text()
            order = np.array(list(map(int, order.strip().split())))
            # Get node data
            # obj_coeffs = torch.from_numpy(np.array(data["obj_coeffs"]))
            # adj = torch.from_numpy(np.array(data["adj_list"]))
            # X, Y = get_node_data(cfg, order, bdd)
            dataset = self.get_node_data(order, bdd)
            dataset = np.concatenate((np.array([pid] * dataset.shape[0]).reshape(-1, 1),
                                      dataset), axis=1)
            print(dataset.shape)
            # X, Y, order = torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(order)

            dataset = dataset.astype(np.ushort)

            # Save data
            file_path = path.dataset / f"{self.cfg.prob.name}/{self.cfg.size}/{self.cfg.split}"
            # prefix = f"{cfg.layer_weight}-{cfg.neg_to_pos_ratio}"
            # file_path /= f"{prefix}-parent" if cfg.with_parent else f"{prefix}-no-parent"
            file_path.mkdir(exist_ok=True, parents=True)
            # obj = {"x": X, "y": Y, "order": order, "obj_coeffs": obj_coeffs, "adj": adj}
            # torch.save(obj, file_path / f"{pid}.pt")
            np.save(file_path / f"{pid}.npy", dataset)

    def get_node_data(cfg, order, bdd):
        data_lst = []
        labels_lst = []
        counts = []
        n_total = 0
        dataset = None
        for lid, layer in enumerate(bdd):
            neg_data_lst = []
            pos_data_lst = []
            for nid, node in enumerate(layer):
                # Binary state of the current node
                state = np.zeros(cfg.prob.n_vars)
                state[node['s']] = 1
                # Binary state of the parent state
                if cfg.with_parent:
                    n_parents_op, n_parents_zp = 0, 0
                    zp_lst, op_lst = np.array([]), np.array([])
                    if lid > 0:
                        n_parents_zp = len(node['zp'])
                        n_parents_op = len(node['op'])

                        for zp in node['zp']:
                            zp_state = np.zeros(cfg.prob.n_vars)
                            zp_state[bdd[lid - 1][zp]['s']] = 1
                            zp_lst = np.concatenate((zp_lst, zp_state))

                        op_lst = np.array([])
                        for op in node['op']:
                            op_state = np.zeros(cfg.prob.n_vars)
                            op_state[bdd[lid - 1][op]['s']] = 1
                            op_lst = np.concatenate((op_lst, op_state))

                    node_data = np.concatenate(([lid],
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
            n_neg = min(len(neg_data_lst), int(cfg.neg_to_pos_ratio * n_pos))
            if n_neg > 0:
                # labels += [0] * n_neg
                random.shuffle(neg_data_lst)
                neg_data_lst = neg_data_lst[:n_neg]

                neg_data = np.stack(neg_data_lst)
                neg_data = np.concatenate((neg_data, np.array([0] * n_neg).reshape(-1, 1)), axis=-1)

                data = np.concatenate((pos_data, neg_data), axis=0)
            else:
                data = pos_data

            if dataset is None:
                dataset = data
            else:
                dataset = np.concatenate((dataset, data), axis=0)

            # data_lst.extend(neg_data_lst)
            # labels_lst.extend(labels)
            #
            # # Update class counts
            # n_total += n_pos + n_neg
            # counts.append((n_neg, n_pos))

        # Pad features
        # max_feat = np.max([n.shape[0] for n in data_lst])
        # padded = [np.concatenate((n, np.ones(max_feat - n.shape[0]) * -1)) for n in data_lst]

        # Sample weights
        # layer_weights = get_layer_weights(True, cfg.layer_weight, cfg.prob.n_vars)
        # weights = []
        # for nid, n in enumerate(data_lst):
        #     label, lid = labels_lst[nid], int(n[0])
        #     weight = 1 - (counts[lid][label] / n_total)
        #     weight *= layer_weights[lid]
        #     weights.append(weight)
        #
        # padded, weights, labels = np.stack(padded), np.array(weights), np.array(labels_lst)
        # padded = np.hstack((weights.reshape(-1, 1), padded))

        # return padded, labels
        return dataset

    def get_instance_data(self, size, split, pid):
        return get_instance_data(size, split, pid)
