from abc import abstractmethod, ABC
from morbdd import ResourcePaths as path
import time


class Trainer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.ckpt_path = None

    def set_checkpoint_path(self):
        exp = self.get_exp_name()
        self.ckpt_path = path.resource / "checkpoint" / exp

    def get_exp_name(self):
        exp = self.get_trainer_str() + self.get_dataset_str()
        if self.cfg.with_timestamp:
            exp += "-" + str(float(time.time()))

        return exp

    def get_trainer_str(self):
        return ""

    def get_dataset_str(self):
        dstr = ""
        if not self.cfg.validate_on_master:
            dstr += "-nvm"

        dstr += "-t"
        if self.cfg.dataset.train.from_pid != 0:
            dstr += f"-f-{self.cfg.dataset.train.from_pid}"
        if self.cfg.dataset.train.to_pid != 1000:
            dstr += f"-t-{self.cfg.dataset.train.to_pid}"

        dstr += f"-v"
        if self.cfg.dataset.val.from_pid != 1000:
            dstr += f"-f-{self.cfg.dataset.val.from_pid}"
        if self.cfg.dataset.val.to_pid != 1100:
            dstr += f"-t-{self.cfg.dataset.val.to_pid}"

        return dstr

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train(self):
        pass
