from .dm import DataManager


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

    def generate_dataset(self):
        pass

    def save_order(self):
        pass

    def save_dd(self):
        pass

    def save_solution(self):
        pass

    def save_dm_stats(self):
        pass
