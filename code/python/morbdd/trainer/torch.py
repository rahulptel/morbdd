import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from morbdd import ResourcePaths as path
from morbdd.utils import get_device
from morbdd.utils import reduce_epoch_time
from morbdd.utils import set_seed
from morbdd.utils.kp import get_instance_data as get_instance_data_kp
from morbdd.utils.mis import get_instance_data as get_instance_data_ind
from .trainer import Trainer


class IndepsetBDDNodeDataset(Dataset):
    def __init__(self, n_vars, n_objs, bdd_node_dataset, obj, adj, top_k=5, norm_const=100, max_objs=10):
        super(IndepsetBDDNodeDataset, self).__init__()
        self.n_vars = n_vars
        self.n_objs = n_objs
        self.norm_const = norm_const
        self.max_objs = max_objs

        self.nodes = torch.from_numpy(bdd_node_dataset.astype('int16'))
        perm = torch.randperm(self.nodes.shape[0])
        self.nodes = self.nodes[perm]

        self.top_k = top_k
        self.obj, self.adj = torch.from_numpy(obj) / self.norm_const, torch.from_numpy(adj)
        self.append_obj_id()
        self.pos = self.precompute_pos_enc(top_k, self.adj)

    def __getitem__(self, item):
        pid = self.nodes[item, 0]

        return (self.obj[pid],
                self.adj[pid],
                self.pos[pid] if self.top_k > 0 else None,
                self.nodes[item, 0],
                self.nodes[item, 1],
                self.nodes[item, 2],
                self.nodes[item, 3:3 + self.n_vars],
                self.nodes[item, 3 + self.n_vars])

    def __len__(self):
        return self.nodes.shape[0]

    @staticmethod
    def precompute_pos_enc(top_k, adj):
        p = None
        if top_k > 0:
            # Calculate position encoding
            U, S, Vh = torch.linalg.svd(adj)
            U = U[:, :, :top_k]
            S = (torch.diag_embed(S)[:, :top_k, :top_k]) ** (1 / 2)
            Vh = Vh[:, :top_k, :]

            L, R = U @ S, S @ Vh
            R = R.permute(0, 2, 1)
            p = torch.cat((L, R), dim=-1)  # B x n_vars x (2*top_k)

        return p

    def append_obj_id(self):
        n_items, n_objs, n_vars = self.obj.shape
        obj_id = torch.arange(1, n_objs + 1) / self.max_objs
        obj_id = obj_id.repeat((n_vars, 1))
        obj_id = obj_id.repeat((n_items, 1, 1))
        # n_items x n_objs x n_vars x 2
        self.obj = torch.cat((self.obj.transpose(1, 2).unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1)


class KnapsackBDDNodeDataset(Dataset):
    pass


class TorchTrainer(Trainer):
    def __init__(self, cfg):
        Trainer.__init__(self, cfg)
        self.master = False
        self.device_str = "cpu"
        self.device = torch.device(self.device_str)
        self.pin_memory = False
        self.device_id = 0
        self.world_size = 1

        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None
        self.train_dataloader = None
        self.val_dataloader = None

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.start_epoch = 0
        self.best_f1 = 0
        self.writer = None
        self.train_stats = []
        self.val_stats = []

    def get_trainer_str(self):
        trainer_str = self.get_model_str() + self.get_opt_str()
        return trainer_str

    def get_model_str(self):
        model_str = ""
        if self.cfg.model.name == "gtf":
            model_str += "tf-v" + str(self.cfg.model_version) + "-"
            if self.cfg.d_emb != 64:
                model_str += f"-emb-{self.cfg.d_emb}"
            if self.cfg.top_k != 5:
                model_str += f"-k-{self.cfg.top_k}"
            if self.cfg.n_blocks != 2:
                model_str += f"-l-{self.cfg.n_blocks}"
            if self.cfg.n_heads != 8:
                model_str += f"-h-{self.cfg.n_heads}"
            if self.cfg.dropout_token != 0.0:
                model_str += f"-dptk-{self.cfg.dropout_token}"
            if self.cfg.model_version == 3:
                if self.cfg.dropout_attn != 0.2:
                    model_str += f"-dpa-{self.cfg.dropout_attn}"
                if self.cfg.dropout_proj != 0.2:
                    model_str += f"-dpp-{self.cfg.dropout_proj}"
                if self.cfg.dropout_mlp != 0.2:
                    model_str += f"-dpm-{self.cfg.dropout_mlp}"
            else:
                if self.cfg.dropout != 0.2:
                    model_str += f"-dp-{self.cfg.dropout}"
            if self.cfg.bias_mha:
                model_str += f"-ba-{self.cfg.bias_mha}"
            if self.cfg.bias_mha:
                model_str += f"-bm-{self.cfg.bias_mlp}"
            if self.cfg.h2i_ratio != 2:
                model_str += f"-h2i-{self.cfg.h2i_ratio}"

        elif self.cfg.encoder_type == "gat":
            model_str += "gat"
            if self.cfg.d_emb != 64:
                model_str += f"-emb-{self.cfg.d_emb}"

        return model_str

    def get_opt_str(self):
        ostr = ""
        if self.cfg.opt.name != "Adam":
            ostr += "-" + self.cfg.opt.name
        if self.cfg.opt.lr != 1e-3:
            ostr += f"-lr-{self.cfg.opt.lr}"
        if self.cfg.opt.wd != 1e-4:
            ostr += f"-wd-{self.cfg.opt.wd}"
        if self.cfg.clip_grad != 1.0:
            ostr += f"-clip-{self.cfg.clip_grad}"
        if self.cfg.norm_type != 2.0:
            ostr += f"-norm-{self.cfg.norm_type}"

        return ostr

    def set_model(self):
        if self.cfg.model.name == "gtf":
            from morbdd.model.psp import GTFParetoStatePredictor
            self.model = GTFParetoStatePredictor()
        elif self.cfg.model.name == "tf":
            from morbdd.model.psp import TFParetoStatePredictor
            self.model = TFParetoStatePredictor()
        assert self.model is not None, "Invalid model name"

    def set_optimizer(self):
        opt_cls = getattr(optim, self.cfg.opt.name)
        optimizer = opt_cls(self.model.parameters(), lr=self.cfg.opt.lr, weight_decay=self.cfg.opt.wd)

        return optimizer

    def get_instance_data(self, split, pid):
        data = None
        if self.cfg.prob.name == "knapsack":
            data = get_instance_data_kp(self.cfg.prob.name, self.cfg.prob.prefix, self.cfg.prob.seed,
                                        self.cfg.prob.size, split, pid)
            data["inst_node"] = np.vstack((np.array(data["value"]),
                                           np.array(data["weight"]).reshape(1, -1),
                                           np.array(data["capacity"] * self.cfg.prob.n_vars).reshape(1, -1)))
            data["inst_node"] = data["inst_node"].T
            data["inst_edge"] = None

        elif self.cfg.prob.name == "indepset":
            data = get_instance_data_ind(self.cfg.prob.size, split, pid)
            data["inst_node"] = data.pop("obj_coeffs")
            data["inst_node"] = np.array(data["inst_node"]).T

            data["inst_edge"] = data.pop("adj_list")
            data["inst_edge"] = np.array(data["inst_edge"])

        assert data is not None

        return data

    def set_dataset(self, split):
        bdd_node_dataset = np.load(str(path.dataset) + f"/{self.cfg.prob.name}/{self.cfg.size}/{split}.npy")
        from_pid, to_pid = self.cfg.dataset[split].from_pid, self.cfg.dataset[split].to_pid
        valid_rows = (from_pid <= bdd_node_dataset[:, 0])
        valid_rows &= (bdd_node_dataset[:, 0] < to_pid)

        bdd_node_dataset = bdd_node_dataset[valid_rows]
        if split == "val":
            bdd_node_dataset[:, 0] -= 1000
        if split == "test":
            bdd_node_dataset[:, 0] -= 1100

        inst_node, inst_edge = [], []
        for pid in range(from_pid, to_pid):
            data = self.get_instance_data(split, pid)
            if data["inst_node"] is not None:
                inst_node.append(data["inst_node"])
            if data["inst_edge"] is not None:
                inst_edge.append(data["inst_edge"])
        inst_node_dataset, inst_edge_dataset = np.array(inst_node), np.stack(inst_edge)

        dataset = None
        if self.cfg.prob.name == "knapsack":
            pass
        elif self.cfg.prob.name == "indepset":
            dataset = IndepsetBDDNodeDataset(bdd_node_dataset, inst_node_dataset, inst_edge_dataset,
                                             top_k=self.cfg.top_k)
        assert dataset is not None

        setattr(self, f"{split}_dataset", dataset)

    def setup(self):
        set_seed(self.cfg.seed)

        # Set-up device
        device_data = get_device(distributed=self.cfg.distributed,
                                 init_method=self.cfg.init_method,
                                 dist_backend=self.cfg.dist_backend)
        (self.device, self.device_str, self.pin_memory, self.master, self.device_id, self.world_size) = device_data
        print("Device :", self.device)

        self.set_model()
        self.set_optimizer()

        # Load model if restarting
        self.set_checkpoint_path()
        self.ckpt_path.mkdir(exist_ok=True, parents=True)
        print("Checkpoint path: {}".format(self.ckpt_path))
        self.writer = SummaryWriter(self.ckpt_path) if self.cfg.log_every else None
        if self.cfg.training_from == "last_checkpoint":
            ckpt = torch.load(self.ckpt_path / "model.pt", map_location="cpu")
            self.model.load_state_dict(ckpt["state_dict"])
            self.optimizer.load_state_dict(ckpt["opt_dict"])

            stats = torch.load(self.ckpt_path / "stats.pt", map_location=torch.device("cpu"))
            for v in stats["val"]:
                if v["f1"] < self.best_f1:
                    self.best_f1 = v["f1"]
            self.start_epoch = int(ckpt["epoch"])

        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.device_id]) if self.cfg.distributed else self.model

        # Initialize dataloaders
        print("N worker dataloader: ", self.cfg.n_worker_dataloader)
        self.set_dataset("train")
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.cfg.distributed else None
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.cfg.batch_size,
                                           shuffle=(self.train_sampler is None),
                                           sampler=self.train_sampler,
                                           num_workers=self.cfg.n_worker_dataloader,
                                           pin_memory=self.pin_memory)

        self.set_dataset("val")
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False) \
            if self.cfg.distributed and not self.cfg.validate_on_master \
            else None
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=self.cfg.batch_size,
                                         sampler=self.val_sampler,
                                         shuffle=False,
                                         num_workers=self.cfg.n_worker_dataloader,
                                         pin_memory=self.pin_memory)
        # print_validation_machine(master, val_sampler, device_str, self.cfg.distributed)
        # debug(cfg, model, train_dataset)

    def train_step(self):
        return {}, {}

    def validate(self):
        return {}

    def train(self):
        if self.master:
            print("Train samples: {}, Val samples {}".format(len(self.train_dataset), len(self.val_dataset)))
            print("Train loader: {}, Val loader {}".format(len(self.train_dataloader), len(self.val_dataloader)))

        for epoch in range(self.start_epoch, self.cfg.epochs):
            if self.cfg.distributed:
                self.train_sampler.set_epoch(epoch)

            # Train
            start_time = time.time()
            stats, result = self.train_step()
            epoch_time = time.time() - start_time
            epoch_time = reduce_epoch_time(epoch_time, self.device) if self.cfg.distributed else epoch_time

            if self.master:
                # stats = dict2cpu(stats) if "cpu" not in str(device) else stats
                epoch_time = float(epoch_time.cpu().numpy()) if self.cfg.distributed else epoch_time
                stats.update({"epoch_time": epoch_time, "epoch": epoch + 1})
                # stats = helper.compute_meta_stats_and_print("train", stats)
                self.train_stats.append(stats)
                print("Result shape: ", result.shape)
                helper.print_stats("train", stats)

            # Validate
            if (epoch + 1) % self.cfg.validate_every == 0:
                stats = {"epoch": epoch + 1}
                for split in self.cfg.validate_on_split:
                    start_time = time.time()
                    new_stats, result = self.validate("val")
                    epoch_time = time.time() - start_time
                    epoch_time = reduce_epoch_time(epoch_time, self.device) \
                        if self.cfg.distributed and not self.cfg.validate_on_master \
                        else epoch_time

                    if self.master:
                        # new_stats = dict2cpu(new_stats) if "cpu" not in str(device) else new_stats
                        epoch_time = float(
                            epoch_time.cpu().numpy()) if self.cfg.distributed and not self.cfg.validate_on_master \
                            else epoch_time
                        new_stats.update({"epoch_time": epoch_time})

                        prefix = "tr_" if split == "train" else ""
                        new_stats = {prefix + k: v for k, v in new_stats.items()} if split == "train" else new_stats
                        stats.update(new_stats)

                        print("Result shape: ", result.shape)
                        helper.print_stats("val", stats, prefix=prefix)
                        if split == "val" and stats["f1"] > self.best_f1:
                            best_f1 = stats["f1"]
                            helper.save_model_and_opt(epoch, self.ckpt_path, best_model=True,
                                                      model=(self.model.module.state_dict() if self.cfg.distributed else
                                                             self.model.state_dict()),
                                                      optimizer=self.optimizer.state_dict())
                            helper.save_stats(self.ckpt_path)
                            print("* Best F1: {}".format(best_f1))

                helper.val_stats.append(stats)

            if self.master and (epoch + 1) % cfg.save_every == 0:
                helper.save_model_and_opt(epoch, self.ckpt_path, best_model=False,
                                          model=(self.model.module.state_dict() if self.cfg.distributed else
                                                 self.model.state_dict()),
                                          optimizer=self.optimizer.state_dict())
                helper.save_stats(self.ckpt_path)

            print() if self.master else None
