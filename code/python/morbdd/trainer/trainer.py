from abc import abstractmethod, ABC


class Trainer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.exp_name = None
        self.ckpt_path = None

        self.train_dataset = None
        self.train_loader = None
        self.val_dataset = None
        self.val_loader = None

        self.warmup_steps = None
        self.global_step = 0
        self.best_step = -1
        self.best_val_metric = -1
        self.train_results = []
        self.val_results = []
        self.lrs = []

        self.model = None
        self.optimizer = None
        self.loss_fn = None

    @abstractmethod
    def set_checkpoint_path(self):
        pass

    @abstractmethod
    def setup_train(self):
        pass

    @abstractmethod
    def train(self):
        pass
