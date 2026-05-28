import sys
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from stereo.modeling.models.lightstereo.lightstereo import LightStereo

class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
    def __setattr__(self, key, value):
        self[key] = value

def to_attr_dict(value):
    if isinstance(value, dict):
        return AttrDict({k: to_attr_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_attr_dict(v) for v in value]
    return value

class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, left_img, right_img):
        return self.model({"left": left_img, "right": right_img})["disp_pred"]

cfg_path = "output/OrbbecDataset/LightStereo/lightstereo_m_orin/run_5ds_260519/lightstereo_m_orin.yaml"
weights_path = "output/OrbbecDataset/LightStereo/lightstereo_m_orin/run_5ds_260519/ckpt/checkpoint_epoch_199.pth"
output_path = "/data2/shendu/app/lightstereo/models/lightstereo_m_orin_480x640_official.onnx"

with open(cfg_path, "r") as f:
    cfg = to_attr_dict(yaml.safe_load(f))

model = LightStereo(cfg.MODEL)
ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)
model.eval()

wrapper = ONNXWrapper(model).eval()
left = torch.zeros(1, 3, 480, 640, dtype=torch.float32)
right = torch.zeros(1, 3, 480, 640, dtype=torch.float32)

torch.onnx.export(
    wrapper,
    (left, right),
    output_path,
    input_names=["left_img", "right_img"],
    output_names=["disp_pred"],
    opset_version=11,
    do_constant_folding=True,
)
print(f"✅ ONNX exported: {output_path}")

# Simplify
import onnx
import onnxsim
model_onnx = onnx.load(output_path)
model_opt, check = onnxsim.simplify(model_onnx)
assert check, 'simplify check failed'
onnx.save(model_opt, output_path)
print(f"✅ ONNX simplified: {output_path}")
