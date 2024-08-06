from abc import abstractmethod, ABC
from morbdd import ResourcePaths as path
import time


class Trainer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_checkpoint_path(self):
        exp = self.get_exp_name()
        ckpt_path = path.resource / "checkpoint" / exp

        return ckpt_path

    def get_exp_name(self):
        exp = self.get_model_str() + self.get_opt_str() + self.get_dataset_str()
        if self.cfg.with_timestamp:
            exp += "-" + str(float(time.time()))

        return exp

    def get_model_str(self):
        model_str = ""
        if self.cfg.encoder_type == "transformer":
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
    def train(self):
        raise NotImplementedError
