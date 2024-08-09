import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from morbdd import ResourcePaths as path
from morbdd.utils import Meter
from morbdd.utils import get_dataset_prefix
from morbdd.utils import get_device
from morbdd.utils import reduce_epoch_time
from morbdd.utils import set_seed
from morbdd.utils.kp import get_instance_data as get_instance_data_kp
from morbdd.utils.mis import get_instance_data as get_instance_data_ind
from .trainer import Trainer

LABEL = 0
LID = 1
LGT0 = 2
LGT1 = 3
SCORE = 4
PREDICTION = 5
NBINS = 10


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


def get_stats(losses, data_time, batch_time, result):
    l0c, l0b = np.histogram(result[:, LGT0], NBINS)
    l1c, l1b = np.histogram(result[:, LGT1], NBINS)
    s0c, s0b = np.histogram(result[result[:, LABEL] == 0][:, SCORE], NBINS)
    s1c, s1b = np.histogram(result[result[:, LABEL] == 1][:, SCORE], NBINS)
    stats = {
        "loss": losses.avg.cpu().numpy(),
        "data_time": data_time.avg.cpu().numpy(),
        "batch_time": batch_time.avg.cpu().numpy(),
        "epoch_time": batch_time.sum.cpu().numpy(),
        "f1": f1_score(result[:, LABEL], result[:, PREDICTION]),
        "recall": recall_score(result[:, LABEL], result[:, PREDICTION]),
        "precision": precision_score(result[:, LABEL], result[:, PREDICTION]),
        "acc": accuracy_score(result[:, LABEL], result[:, PREDICTION]),
        "specificity": (result[result[:, LABEL] == 0][:, PREDICTION] == 0).sum() / result.shape[0],
        "lgt0-mean": np.mean(result[:, LGT0]),
        "lgt0-std": np.std(result[:, LGT0]),
        "lgt0-med": np.median(result[:, LGT0]),
        "lgt0-min": np.min(result[:, LGT0]),
        "lgt0-max": np.max(result[:, LGT0]),
        "lgt0-count": l0c,
        "lgt0-bins": l0b,
        "score0-count": s0c,
        "score0-bins": s0b,
        "lgt1-mean": np.mean(result[:, LGT1]),
        "lgt1-std": np.std(result[:, LGT1]),
        "lgt1-med": np.median(result[:, LGT1]),
        "lgt1-min": np.min(result[:, LGT1]),
        "lgt1-max": np.max(result[:, LGT1]),
        "lgt1-count": l1c,
        "lgt1-bins": l1b,
        "score1-count": s1c,
        "score1-bins": s1b
    }

    return stats


def aggregate_distributed_stats(master, losses=None, data_time=None, batch_time=None, result=None):
    if losses is not None:
        dist.all_reduce(losses.sum, dist.ReduceOp.SUM)
        losses.count = torch.tensor(losses.count).to(losses.sum.device)
        dist.all_reduce(losses.count, dist.ReduceOp.SUM)
        losses.avg = losses.sum / losses.count
    if data_time is not None:
        dist.all_reduce(data_time.avg, dist.ReduceOp.AVG)
    if batch_time is not None:
        dist.all_reduce(batch_time.avg, dist.ReduceOp.AVG)
        dist.all_reduce(batch_time.sum, dist.ReduceOp.AVG)
    if result is not None:
        result_lst = [torch.zeros_like(result) for _ in range(dist.get_world_size())] if master else None
        dist.gather(result, gather_list=result_lst, dst=0)
        result = torch.cat(result_lst) if master else result

    return result


class ANNTrainer(Trainer):
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
        self.batch_processor = None

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
        raise NotImplementedError

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
        raise NotImplementedError

    def set_optimizer(self):
        opt_cls = getattr(optim, self.cfg.opt.name)
        self.optimizer = opt_cls(self.model.parameters(), lr=self.cfg.opt.lr, weight_decay=self.cfg.opt.wd)

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
            data["inst_node"] = np.array(data["inst_node"])

            data["inst_edge"] = data.pop("adj_list")
            data["inst_edge"] = np.array(data["inst_edge"])

        assert data is not None

        return data

    def set_dataset(self, split):
        self.cfg.split = split
        dataset_path = path.dataset / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        prefix = get_dataset_prefix(with_parent=self.cfg.bdd_data.with_parent,
                                    layer_weight=self.cfg.layer_weight,
                                    neg_to_pos_ratio=self.cfg.bdd_data.neg_to_pos_ratio)
        dataset_path = dataset_path / f"{prefix}-{split}.npy"
        print(f"Dataset {split} path: ", dataset_path)
        bdd_node_dataset = np.load(dataset_path)

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
            dataset = IndepsetBDDNodeDataset(self.cfg.prob.n_vars, self.cfg.prob.n_objs, bdd_node_dataset,
                                             inst_node_dataset, inst_edge_dataset, top_k=self.cfg.model.top_k)
        assert dataset is not None

        setattr(self, f"{split}_dataset", dataset)

    @staticmethod
    def print_stats(split, stats, prefix=""):
        epoch = stats["epoch"]
        ept, bt, dt, = stats[prefix + "epoch_time"], stats[prefix + "batch_time"], stats[prefix + "data_time"]

        print_str = ("{}:{}: F1: {:4f}, Acc: {:.4f}, Loss {:.4f}, Recall: {:.4f}, Precision: {:.4f}, "
                     "Specificity: {:.4f}, Epoch Time: {:.4f}, Batch Time: {:.4f}, Data Time: {:.4f}")
        print(print_str.format(epoch, prefix + split, stats[prefix + "f1"], stats[prefix + "acc"],
                               stats[prefix + "loss"], stats[prefix + "recall"], stats[prefix + "precision"],
                               stats[prefix + "specificity"], ept, bt, dt))

        print("Logit distribution:")
        print_str = "{}: Mean: {:.4f}, Std: {:.4f}, Min: {:.4f}, Max: {:.4f},"
        for i in ["0", "1"]:
            print(print_str.format("lgt" + i,
                                   stats[prefix + "lgt" + i + "-mean"],
                                   stats[prefix + "lgt" + i + "-std"],
                                   stats[prefix + "lgt" + i + "-min"],
                                   stats[prefix + "lgt" + i + "-max"]))
        print()

    @staticmethod
    def save_model_and_opt(epoch, save_path, best_model=False, model=None, optimizer=None):
        # print(epoch)
        # print("Is best: {}".format(best_model))
        if best_model:
            model_path = save_path / "best_model.pt"
        else:
            model_path = save_path / "model.pt"
        # print("Saving model to: {}".format(model_path))

        model_obj = {"epoch": epoch + 1, "model": model, "optimizer": optimizer}
        torch.save(model_obj, model_path)

    def save_stats(self, save_path):
        stats_path = save_path / f"stats.pt"
        # print("Saving stats to: {}".format(stats_path))
        stats_obj = {"train": self.train_stats, "val": self.val_stats}
        torch.save(stats_obj, stats_path)

    def print_validation_machine(self):
        if self.master:
            if self.val_sampler is None:
                if "cuda" in self.device_str:
                    if self.cfg.distributed:
                        print("Training: Multi-GPU, Validation: Master/Single GPU")
                    else:
                        print("Training/Validation: Single GPU")
                else:
                    print("Training/Validation: CPU")
            else:
                print("Training/Validation: Multi-GPU")

    def get_grad_norm(self, norm=2):
        grads = torch.empty(0).to(next(self.model.parameters()).device)
        for p in self.model.parameters():
            if p.grad is not None:
                grads = torch.cat((grads, p.grad.detach().data.view(-1)))

        return torch.norm(grads, p=norm)

    def setup(self):
        print("Training set-up in progress...")
        set_seed(self.cfg.seed)

        # Set-up device
        device_data = get_device(distributed=self.cfg.distributed,
                                 init_method=self.cfg.init_method,
                                 dist_backend=self.cfg.dist_backend)
        (self.device, self.device_str, self.pin_memory, self.master, self.device_id, self.world_size) = device_data
        print("Device: ", self.device)

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
        self.print_validation_machine()

    @torch.no_grad()
    def validate(self, epoch, split, dataloader):
        stats = {}
        result = None
        if (not self.cfg.distributed) or (self.cfg.distributed and self.cfg.validate_on_master and self.master) or (
                self.cfg.distributed and not self.cfg.validate_on_master):
            data_time, batch_time, losses = Meter('DataTime'), Meter('BatchTime'), Meter('Loss')
            result = torch.empty((6, 0)).to(self.device)
            max_batches = len(dataloader)

            self.model.eval()
            start_time = time.time()
            for batch_id, batch in enumerate(dataloader):
                if self.pin_memory:
                    batch = [item.to(self.device, non_blocking=True) for item in batch]
                data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=self.device))
                log = self.master and self.cfg.log_every > 0 and batch_id % self.cfg.log_every == 0
                curr_iter = (epoch * max_batches) + batch_id
                loss, batch_result = self.process_batch(batch, curr_iter=curr_iter, log=log, split="val-" + split)

                result = torch.cat((result, batch_result), dim=1)
                losses.update(loss.detach(), batch_result.shape[0])
                batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=self.device))
                start_time = time.time()

            result = result.T
            if self.cfg.distributed and not self.cfg.validate_on_master:
                result = aggregate_distributed_stats(self.master, losses=losses, data_time=data_time,
                                                     batch_time=batch_time,
                                                     result=result)
            result = result.cpu().numpy()
            stats = get_stats(losses, data_time, batch_time, result)

        return stats, result

    def process_batch(self, batch, curr_iter=0, log=False, split="train"):
        raise NotImplementedError

    def train_step(self, epoch):
        data_time, batch_time, losses = Meter('DataTime'), Meter('BatchTime'), Meter('Loss')
        result = torch.empty((6, 0)).to(self.device)
        max_batches = len(self.train_dataloader)

        self.model.train()
        start_time = time.time()
        for batch_id, batch in enumerate(self.train_dataloader):
            if self.pin_memory:
                batch = [item.to(self.device, non_blocking=True) for item in batch]
            data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=self.device))
            log = self.master and self.cfg.log_every > 0 and batch_id % self.cfg.log_every == 0
            curr_iter = (epoch * max_batches) + batch_id

            # Get logits and compute loss
            output = self.process_batch(batch, curr_iter=curr_iter, log=log, split="train")
            loss, batch_result = output
            # Learn
            self.optimizer.zero_grad()
            loss.backward()
            if log:
                norm = self.get_grad_norm()
                self.writer.add_scalar("grad_norm", norm, curr_iter)
            if self.cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad,
                                               norm_type=self.cfg.norm_type)
                if log:
                    norm = self.get_grad_norm(self.model)
                    self.writer.add_scalar("grad_norm_clipped", norm, curr_iter)
            self.optimizer.step()
            result = torch.cat((result, batch_result), dim=1)
            losses.update(loss.detach(), batch_result.shape[0])
            batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=self.device))
            start_time = time.time()

        result = result.T
        if self.cfg.distributed:
            result = aggregate_distributed_stats(self.master, losses=losses, data_time=data_time,
                                                 batch_time=batch_time, result=result)
        result = result.cpu().numpy()
        stats = get_stats(losses, data_time, batch_time, result)

        return stats, result

    def train(self):
        print()
        print("Training in progress...")
        if self.master:
            print("Train samples: {}, Val samples {}".format(len(self.train_dataset), len(self.val_dataset)))
            print("Train loader: {}, Val loader {}".format(len(self.train_dataloader), len(self.val_dataloader)))

        for epoch in range(self.start_epoch, self.cfg.epochs):
            if self.cfg.distributed:
                self.train_sampler.set_epoch(epoch)

            # Train
            start_time = time.time()
            stats, result = self.train_step(epoch)
            epoch_time = time.time() - start_time
            epoch_time = reduce_epoch_time(epoch_time, self.device) if self.cfg.distributed else epoch_time

            if self.master:
                epoch_time = float(epoch_time.cpu().numpy()) if self.cfg.distributed else epoch_time
                stats.update({"epoch_time": epoch_time, "epoch": epoch + 1})
                self.train_stats.append(stats)
                # print("Result shape: ", result.shape)
                self.print_stats("train", stats)

            # Validate
            if (epoch + 1) % self.cfg.validate_every == 0:
                stats = {"epoch": epoch + 1}
                for split in self.cfg.validate_on_split:
                    start_time = time.time()
                    new_stats, result = self.validate(epoch, split, getattr(self, f"{split}_dataloader"))
                    epoch_time = time.time() - start_time
                    epoch_time = reduce_epoch_time(epoch_time, self.device) \
                        if self.cfg.distributed and not self.cfg.validate_on_master \
                        else epoch_time

                    if self.master:
                        # new_stats = dict2cpu(new_stats) if "cpu" not in str(device) else new_stats
                        epoch_time = float(epoch_time.cpu().numpy()) \
                            if self.cfg.distributed and not self.cfg.validate_on_master \
                            else epoch_time
                        new_stats.update({"epoch_time": epoch_time})

                        prefix = "tr_" if split == "train" else ""
                        new_stats = {prefix + k: v for k, v in new_stats.items()} if split == "train" else new_stats
                        stats.update(new_stats)

                        # print("Result shape: ", result.shape)
                        self.print_stats("val", stats, prefix=prefix)
                        if split == "val" and stats["f1"] > self.best_f1:
                            self.best_f1 = stats["f1"]
                            self.save_model_and_opt(epoch, self.ckpt_path, best_model=True,
                                                    model=(self.model.module.state_dict() if self.cfg.distributed else
                                                           self.model.state_dict()),
                                                    optimizer=self.optimizer.state_dict())
                            self.save_stats(self.ckpt_path)
                            print("** Best F1: {} **".format(self.best_f1))

                self.val_stats.append(stats)

            if self.master and (epoch + 1) % self.cfg.save_every == 0:
                self.save_model_and_opt(epoch, self.ckpt_path, best_model=False,
                                        model=(self.model.module.state_dict() if self.cfg.distributed else
                                               self.model.state_dict()),
                                        optimizer=self.optimizer.state_dict())
                self.save_stats(self.ckpt_path)

            print("--------------------------\n") if self.master else None


class TransformerTrainer(ANNTrainer):
    def __init__(self, cfg):
        super(TransformerTrainer, self).__init__(cfg)
        self.loss_fn = F.cross_entropy
        if self.cfg.prob.name == "knapsack":
            self.batch_processor = self.process_batch_knapsack
        elif self.cfg.prob.name == "indepset":
            self.batch_processor = self.process_batch_indepset
        else:
            print("Invalid problem type!")
            sys.exit()

    def get_model_str(self):
        model_str = f"{self.cfg.model.type}-v{self.cfg.model.version}-"
        if self.cfg.model.d_emb != 64:
            model_str += f"-emb-{self.cfg.model.d_emb}"
        if self.cfg.model.n_layers != 2:
            model_str += f"-l-{self.cfg.model.n_layers}"
        if self.cfg.model.n_heads != 8:
            model_str += f"-h-{self.cfg.model.n_heads}"
        if self.cfg.model.dropout_token != 0.0:
            model_str += f"-dptk-{self.cfg.model.dropout_token}"
        if self.cfg.model.dropout_attn != 0.0:
            model_str += f"-dpa-{self.cfg.model.dropout_attn}"
        if self.cfg.model.dropout_proj != 0.0:
            model_str += f"-dpp-{self.cfg.model.dropout_proj}"
        if self.cfg.model.dropout_mlp != 0.0:
            model_str += f"-dpm-{self.cfg.model.dropout_mlp}"
        if self.cfg.model.bias_mha:
            model_str += f"-ba-{self.cfg.model.bias_mha}"
        if self.cfg.model.bias_mha:
            model_str += f"-bm-{self.cfg.model.bias_mlp}"
        if self.cfg.model.h2i_ratio != 2:
            model_str += f"-h2i-{self.cfg.model.h2i_ratio}"
        # Graph transformer specific
        if self.cfg.model.type == "gtf" and self.cfg.model.top_k != 5:
            model_str += f"-k-{self.cfg.model.top_k}"

        return model_str

    def set_model(self):
        if self.cfg.model.type == "gtf":
            from morbdd.model.psp import GTFParetoStatePredictor
            self.model = GTFParetoStatePredictor()
        elif self.cfg.model.type == "tf":
            from morbdd.model.psp import TFParetoStatePredictor
            self.model = TFParetoStatePredictor()
        assert self.model is not None, "Invalid model type"

    def process_batch_indepset(self, batch, curr_iter=0, log=False, split="train"):
        objs, adjs, pos, _, lids, vids, states, labels = batch
        if log and self.writer is not None:
            model_ = self.model.module if self.cfg.distributed else self.model
            # ----------------------------------------------------------------------
            n_emb, e_emb = model_.token_emb(objs, adjs.int(), pos.float())
            self.writer.add_histogram("token_emb/n/" + split, n_emb, curr_iter)
            self.writer.add_histogram("token_emb/e/" + split, e_emb, curr_iter)

            n_emb = model_.node_encoder(n_emb, e_emb)
            self.writer.add_histogram("node_emb/" + split, n_emb, curr_iter)

            # Instance embedding
            # B x d_emb
            inst_emb = model_.graph_encoder(n_emb.sum(1))
            self.writer.add_histogram("inst_emb/" + split, inst_emb, curr_iter)

            # Layer-index embedding
            # B x d_emb
            li_emb = model_.layer_index_encoder(lids.reshape(-1, 1).float())
            self.writer.add_histogram("lindex_emb/" + split, li_emb, curr_iter)

            # Layer-variable embedding
            # B x d_emb
            lv_emb = n_emb[torch.arange(vids.shape[0]), vids.int()]
            self.writer.add_histogram("lvar_emb/" + split, lv_emb, curr_iter)

            # State embedding
            state_emb = torch.einsum("ijk,ij->ik", [n_emb, states.float()])
            state_emb = model_.aggregator(state_emb)
            state_emb = state_emb + inst_emb + li_emb + lv_emb
            self.writer.add_histogram("state/" + split, state_emb, curr_iter)

            # Pareto-state predictor
            logits = model_.predictor(model_.ln(state_emb))
            # ----------------------------------------------------------------------
        else:
            logits = self.model(objs, adjs, pos, lids, vids, states)

        logits, labels = logits.reshape(-1, 2), labels.long().reshape(-1)
        loss = self.loss_fn(logits, labels)

        # Get predictions
        logits = logits.detach()
        scores = F.softmax(logits, dim=-1)
        preds = torch.argmax(scores, dim=-1)
        batch_result = torch.stack((labels, lids, logits[:, 0], logits[:, 1], scores[:, 1], preds))

        return loss, batch_result

    def process_batch_knapsack(self, batch, curr_iter=0, log=False, split="train"):
        pass

    def process_batch(self, batch, curr_iter=0, log=False, split="train"):
        return self.batch_processor(batch, curr_iter, log=log, split=split)
