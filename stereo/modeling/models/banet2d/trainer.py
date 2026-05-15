import os

import torch

from stereo.modeling.trainer_template import TrainerTemplate
from .banet2d import BANet2D

__all__ = {
    "BANet2D": BANet2D,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

    def build_model(self, model):
        """SceneFlow 等原版权重键为 module.*；封装后需写入 banet.*。"""
        pretrained_path = getattr(self.cfgs.MODEL, "PRETRAINED_MODEL", None)
        pretrained_path = (pretrained_path or "").strip()
        if pretrained_path:
            self.cfgs.MODEL.PRETRAINED_MODEL = ""
        model = super().build_model(model)
        if pretrained_path:
            self.cfgs.MODEL.PRETRAINED_MODEL = pretrained_path
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(pretrained_path)
            self.logger.info("Loading pretrained weights into BANet2D: %s" % pretrained_path)
            self._load_sceneflow_into_banet2d(model, pretrained_path)
        return model

    def _load_sceneflow_into_banet2d(self, model, filename):
        device = "cuda:%d" % self.local_rank
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            pretrained_state_dict = checkpoint["model_state"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            pretrained_state_dict = checkpoint["model"]
        else:
            pretrained_state_dict = checkpoint

        keys = list(pretrained_state_dict.keys())
        if any(k.startswith("banet.") for k in keys):
            # OpenStereo 训练保存的 model_state（已是 banet.*）
            pretrained_state_dict = {
                k.replace("module.", "", 1): v for k, v in pretrained_state_dict.items()
            }
        elif any(k.startswith("module.banet.") for k in keys):
            pretrained_state_dict = {
                k.replace("module.", "", 1): v for k, v in pretrained_state_dict.items()
            }
        elif any(k.startswith("module.") for k in keys):
            # 原版 SceneFlow：module.fnet.* → banet.fnet.*
            pretrained_state_dict = {
                k.replace("module.", "", 1): v for k, v in pretrained_state_dict.items()
            }
            pretrained_state_dict = {
                "banet." + k: v for k, v in pretrained_state_dict.items()
            }
        else:
            pretrained_state_dict = {
                "banet." + k: v for k, v in pretrained_state_dict.items()
            }

        tmp_model = model.module if self.args.dist_mode and hasattr(model, "module") else model
        state_dict = tmp_model.state_dict()
        update_state_dict = {}
        unused = []
        for key, val in pretrained_state_dict.items():
            if key in state_dict and state_dict[key].shape == val.shape:
                update_state_dict[key] = val
            else:
                unused.append(key)
        state_dict.update(update_state_dict)
        tmp_model.load_state_dict(state_dict)

        self.logger.info(
            "BANet2D pretrained: matched %d / %d tensors from checkpoint."
            % (len(update_state_dict), len(pretrained_state_dict))
        )
        if unused:
            self.logger.info("Unmatched checkpoint keys (first 12): %s" % (unused[:12],))
        not_updated = [k for k in tmp_model.state_dict() if k not in update_state_dict]
        if not_updated:
            self.logger.info("Params not filled from checkpoint (first 12): %s" % (not_updated[:12],))
