from .dm import DataManager
import numpy as np
from morbdd.utils.mis import get_instance_data


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

    def generate_dataset(self):
        pass

    def get_instance_data(self, size, split, pid):
        return get_instance_data(size, split, pid)
