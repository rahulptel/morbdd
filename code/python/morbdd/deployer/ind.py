import time

import numpy as np
import pandas as pd
import torch

from morbdd.deployer.deployer import Deployer
from morbdd.trainer.ann import TransformerTrainer
from morbdd.utils import get_instance_data
from morbdd import CONST
import torch.nn.functional as F


def get_state_tensor(layer, n_vars):
    states = torch.zeros((len(layer), n_vars))
    for node_id, node in enumerate(layer):
        states[node_id][torch.tensor(node['s']).int()] = 1
    states = states.float()

    return states


@torch.no_grad()
def get_var_embedding(model, obj, adj, pos):
    n_emb, e_emb = model.token_emb(obj.float(), adj.int(), pos.float())
    # Encode: B x n_vars x d_emb
    n_emb = model.node_encoder(n_emb, e_emb)

    return n_emb


@torch.no_grad()
def get_inst_embedding(model, n_emb):
    # B x d_emb
    inst_emb = model.graph_encoder(n_emb.sum(1))
    return inst_emb


@torch.no_grad()
def get_node_scores(model, v_emb, inst_emb, lid, vid, states):
    # Layer-index embedding
    # B x d_emb
    n_items, _ = states.shape
    inst_emb = inst_emb.repeat((n_items, 1))

    li_emb = model.layer_index_encoder(torch.tensor(lid).reshape(-1, 1).float())
    li_emb = li_emb.repeat((n_items, 1))

    # Layer-variable embedding
    # B x d_emb
    lv_emb = v_emb[0, vid].unsqueeze(0)
    lv_emb = lv_emb.repeat((n_items, 1))

    # State embedding
    state_emb = torch.einsum("ijk,ij->ik", [v_emb.repeat((n_items, 1, 1)), states])
    state_emb = model.aggregator(state_emb)
    state_emb = state_emb + inst_emb + li_emb + lv_emb

    # Pareto-state predictor
    logits = model.predictor(model.ln(state_emb))

    return F.softmax(logits, dim=-1)[:, 1].cpu().numpy()


def append_obj_id(obj, n_items, n_objs, n_vars, max_objs):
    obj_id = torch.arange(1, n_objs + 1) / max_objs
    obj_id = obj_id.repeat((n_vars, 1))
    obj_id = obj_id.repeat((n_items, 1, 1))
    # n_items x n_objs x n_vars x 2
    obj = torch.cat((obj.transpose(1, 2).unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1)

    return obj


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


def preprocess_data(data, obj_norm_const=100, max_objs=10, top_k=5):
    # Prepare instance data to be consumed by model
    obj = np.array(data["obj_coeffs"]) / obj_norm_const
    obj = torch.from_numpy(obj).unsqueeze(0)
    obj = append_obj_id(obj, 1, data["n_objs"], data["n_vars"], max_objs)

    adj = torch.from_numpy(np.array(data["adj_list"])).unsqueeze(0)
    pos = precompute_pos_enc(top_k, adj)

    return obj, adj, pos


class IndepsetDeployer(Deployer):
    def __init__(self, cfg):
        Deployer.__init__(self, cfg)
        self.trainer = None

    def set_trainer(self):
        self.trainer = TransformerTrainer(self.cfg)
        self.trainer.setup_predict()
        print(self.trainer.ckpt_path)

    def save_result(self, pid, data_preprocess_time, node_emb_time, inst_emb_time, build_time, pareto_time):
        total_time = data_preprocess_time + node_emb_time + inst_emb_time + build_time + pareto_time
        n_pred_pf, pred_precision = -1, -1
        if self.pred_pf is not None:
            n_pred_pf = len(self.pred_pf)
            if n_pred_pf > 0:
                pred_precision = self.cardinality_raw / n_pred_pf

        df = pd.DataFrame([[self.cfg.prob.size, self.cfg.deploy.split, pid, total_time, self.size_ratio, self.orig_size,
                            self.rest_size, self.cardinality, self.cardinality_raw, pred_precision, n_pred_pf,
                            data_preprocess_time, node_emb_time, inst_emb_time, build_time, pareto_time]],
                          columns=["size", "split", "pid", "total_time", "size", "orig_size", "rest_size",
                                   "cardinality", "cardinality_raw", "pred_precision", "n_pred_pf",
                                   "data_preprocess_time", "node_emb_time",
                                   "inst_emb_time", "build_time", "pareto_time"])
        print(df)

        pid = str(pid) + ".csv"
        self.run_path = self.get_run_path(self.trainer.ckpt_path.stem)
        self.run_path.mkdir(parents=True, exist_ok=True)
        result_path = self.run_path / pid
        df.to_csv(result_path)

    def deploy(self):
        self.set_trainer()
        self.set_alpha_beta_lid()
        print(self.trainer.model.training)
        for pid in range(self.cfg.deploy.from_pid, self.cfg.deploy.to_pid, self.cfg.deploy.n_processes):

            # Load instance data
            data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.deploy.split, pid)

            start = time.time()
            # Preprocess data for ML model
            obj, adj, pos = preprocess_data(data, top_k=self.cfg.model.top_k)
            data_preprocess_time = time.time() - start

            # Obtain variable and instance embedding
            # Same for all the nodes of the BDD
            start = time.time()
            v_emb = get_var_embedding(self.trainer.model, obj, adj, pos)
            node_emb_time = time.time() - start

            start = time.time()
            inst_emb = get_inst_embedding(self.trainer.model, v_emb)
            inst_emb_time = time.time() - start

            # Set BDD Manager
            order = []
            env = self.get_env()
            env.reset(self.cfg.prob.problem_type,
                      self.cfg.prob.preprocess,
                      self.cfg.prob.pf_enum_method,
                      self.cfg.prob.maximization,
                      self.cfg.prob.dominance,
                      self.cfg.prob.bdd_type,
                      self.cfg.prob.maxwidth,
                      order)
            env.set_inst(self.cfg.prob.n_vars, data["n_cons"], self.cfg.prob.n_objs, data["obj_coeffs"],
                         data["cons_coeffs"], data["rhs"])

            # Initializes BDD with the root node
            env.initialize_dd_constructor()
            start = time.time()

            # Set the variable used to generate the next layer
            env.set_var_layer(-1)
            lid = 0

            # Restrict and build
            while lid < data["n_vars"] - 1:
                # print(lid)
                env.generate_next_layer()
                env.set_var_layer(-1)

                layer = env.get_layer(lid + 1)
                states = get_state_tensor(layer, self.cfg.prob.n_vars)
                vid = torch.tensor(env.get_var_layer()[lid + 1]).int()

                scores = get_node_scores(self.trainer.model, v_emb, inst_emb, lid, vid, states)
                lid += 1

                if self.alpha_lid < lid < self.beta_lid:
                    selection, selected_idx, removed_idx = self.node_selector(scores)
                    # Stitch in case of a disconnected BDD
                    if len(removed_idx) == len(scores):
                        removed_idx = self.layer_stitcher(scores)
                        print("Disconnected at layer: ", {lid})
                    # Restrict if necessary
                    if len(removed_idx):
                        env.approximate_layer(lid, CONST.RESTRICT, 1, removed_idx)

            # Generate terminal layer
            env.generate_next_layer()
            build_time = time.time() - start

            start = time.time()
            # Compute pareto frontier
            env.compute_pareto_frontier()
            pareto_time = time.time() - start
            try:
                self.pred_pf = env.get_frontier()["z"]
            except:
                self.pred_pf = None

            self.post_process(env, pid)
            self.save_result(pid, data_preprocess_time, node_emb_time, inst_emb_time, build_time, pareto_time)
