# BANet-2d wrapped for OpenStereo (same data dict / loss interface as LightStereo).
import torch
import torch.nn as nn
import torch.nn.functional as F

from .banet_core import BANet as BANetCore


class BANet2D(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.banet = BANetCore(None)
        # 蒸馏（方案 1）：边界处用 teacher 软分布做 KL。lambda<=0 或无 teacher_prob 时退化为纯 smooth_l1
        distill_cfg = cfgs.get("DISTILL", None)
        if distill_cfg:
            self.lambda_kl = float(distill_cfg.get("LAMBDA_KL", 0.0))
            # KL 逐像素加权：none=均匀（首/二版，被 flat 稀释失败）；entropy=按 teacher 熵加权，
            # 把信号集中到边界（edge 处 teacher init prob 多峰→熵高）。pow 控制集中强度。
            self.kl_weight = str(distill_cfg.get("KL_WEIGHT", "none")).lower()
            self.kl_weight_pow = float(distill_cfg.get("KL_WEIGHT_POW", 1.0))
        else:
            self.lambda_kl = 0.0
            self.kl_weight = "none"
            self.kl_weight_pow = 1.0

    def forward(self, data):
        left = data["left"]
        right = data["right"]
        out = self.banet(left, right, self.max_disp)
        if self.training:
            return {"disp_pred": out[0], "disp_coarse": out[1], "disp_volume": out[2]}
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

        if self.lambda_kl > 0 and "teacher_prob" in input_data:
            teacher_prob = input_data["teacher_prob"]            # [B, bins, H/4, W/4]
            student_logits = model_pred["disp_volume"]           # [B, bins, H/4, W/4]
            student_logp = F.log_softmax(student_logits, dim=1)
            # full-res valid mask 最近邻降到 1/4 分辨率
            mask_q = F.interpolate(mask.float(), size=student_logits.shape[-2:], mode="nearest") > 0.5
            mask_q = mask_q[:, 0]                                # [B, H/4, W/4]
            eps = 1e-8
            kl = (teacher_prob * (torch.log(teacher_prob + eps) - student_logp)).sum(dim=1)  # [B, H/4, W/4]
            if not mask_q.any():
                kl_val = kl.sum() * 0.0
            elif self.kl_weight == "entropy":
                # teacher 熵作为边界权重：edge 处熵高→权重大，flat 处熵低→权重小。
                # 加权均值 sum(w*kl)/sum(w)，对 w 的整体尺度不变，pow 调集中强度。
                ent = -(teacher_prob * torch.log(teacher_prob + eps)).sum(dim=1)  # [B, H/4, W/4]
                w = ent.clamp_min(0).pow(self.kl_weight_pow)
                wm = w[mask_q]
                kl_val = (wm * kl[mask_q]).sum() / wm.sum().clamp_min(eps)
            else:
                kl_val = kl[mask_q].mean()
            loss = loss + self.lambda_kl * kl_val
            loss_info["scalar/train/loss_kl"] = kl_val.item()

        return loss, loss_info
