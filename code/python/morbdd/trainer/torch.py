import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from morbdd import ResourcePaths as path
from morbdd.utils import get_device
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

    def get_model(self):
        if self.cfg.model.name == "gtf":
            from morbdd.model.psp import GTFParetoStatePredictor
            model = GTFParetoStatePredictor()
            return model

        print("Invalid model name")

    def get_optimizer(self, model):
        opt_cls = getattr(optim, self.cfg.opt.name)
        optimizer = opt_cls(model.parameters(), lr=self.cfg.opt.lr, weight_decay=self.cfg.opt.wd)

        return optimizer

    def training_loop(self):
        pass

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

    def get_dataset(self, split):
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

        return dataset

    def training_loop(self, master, start_epoch, end_epoch, train_dataset, train_sampler, train_dataloader,
                      val_dataset, val_sampler, val_dataloader, model, optimizer, device, pin_memory, ckpt_path,
                      world_size, writer):
        pass

    def train(self):
        set_seed(self.cfg.seed)

        # Set-up device
        device_data = get_device(distributed=self.cfg.distributed,
                                 init_method=self.cfg.init_method,
                                 dist_backend=self.cfg.dist_backend)
        device, device_str, pin_memory, master, device_id, world_size = device_data
        print("Device :", device)

        model = self.get_model()
        optimizer = self.get_optimizer(model)

        start_epoch, end_epoch, best_f1 = 0, self.cfg.epochs, 0

        # Load model if restarting
        ckpt_path = self.get_checkpoint_path()
        ckpt_path.mkdir(exist_ok=True, parents=True)
        print("Checkpoint path: {}".format(ckpt_path))
        writer = SummaryWriter(ckpt_path) if self.cfg.log_every else None
        if self.cfg.training_from == "last_checkpoint":
            ckpt = torch.load(ckpt_path / "model.pt", map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["opt_dict"])

            stats = torch.load(ckpt_path / "stats.pt", map_location=torch.device("cpu"))
            for v in stats["val"]:
                if v["f1"] < best_f1:
                    best_f1 = v["f1"]
            start_epoch = int(ckpt["epoch"])

        model.to(device)
        model = DDP(model, device_ids=[device_id]) if self.cfg.distributed else model

        # Initialize dataloaders
        print("N worker dataloader: ", self.cfg.n_worker_dataloader)
        train_dataset = self.get_dataset("train")
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.cfg.distributed else None
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.cfg.batch_size,
                                      shuffle=(train_sampler is None),
                                      sampler=train_sampler,
                                      num_workers=self.cfg.n_worker_dataloader)

        val_dataset = self.get_dataset("val")
        val_sampler = DistributedSampler(val_dataset, shuffle=False) \
            if self.cfg.distributed and not self.cfg.validate_on_master \
            else None
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=self.cfg.batch_size,
                                    sampler=val_sampler,
                                    shuffle=False,
                                    num_workers=self.cfg.n_worker_dataloader,
                                    pin_memory=pin_memory)
        # print_validation_machine(master, val_sampler, device_str, self.cfg.distributed)

        # debug(cfg, model, train_dataset)
        self.training_loop(master, start_epoch, end_epoch, train_dataset, train_sampler, train_dataloader,
                           val_dataset, val_sampler, val_dataloader, model, optimizer, device, pin_memory, ckpt_path,
                           world_size, writer)
