# BANet-2d wrapped for OpenStereo (same data dict / loss interface as LightStereo).
import torch.nn as nn
import torch.nn.functional as F

from .banet_core import BANet as BANetCore


class BANet2D(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.banet = BANetCore(None)

    def forward(self, data):
        left = data["left"]
        right = data["right"]
        out = self.banet(left, right, self.max_disp)
        if self.training:
            return {"disp_pred": out[0], "disp_coarse": out[1]}
        return {"disp_pred": out}

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]
        disp_gt = disp_gt.unsqueeze(1)
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)

        disp_pred = model_pred["disp_pred"]
        disp_coarse = model_pred["disp_coarse"]
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction="mean")
        loss += 0.3 * F.smooth_l1_loss(disp_coarse[mask], disp_gt[mask], reduction="mean")

        loss_info = {"scalar/train/loss_disp": loss.item()}
        return loss, loss_info
