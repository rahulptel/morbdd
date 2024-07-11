from abc import ABC, abstractmethod


class DataManager(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def generate_instances(self):
        pass

    @abstractmethod
    def generate_instance(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_instance(self, inst_path, data):
        pass

    @abstractmethod
    def get_pareto_state_score_per_layer(self, *args):
        pass

    @abstractmethod
    def generate_dataset(self):
        pass

    @abstractmethod
    def save_order(self):
        pass

    @abstractmethod
    def save_dd(self):
        pass

    @abstractmethod
    def save_solution(self):
        pass

    @abstractmethod
    def save_dm_stats(self):
        pass

    def _tag_dd_nodes(self):
        pass
