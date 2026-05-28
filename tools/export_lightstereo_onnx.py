import argparse
import sys
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = to_attr_dict(yaml.safe_load(f))

    model = LightStereo(cfg.MODEL)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    wrapper = ONNXWrapper(model).eval()
    left = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    right = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (left, right),
        str(output),
        input_names=["left_img", "right_img"],
        output_names=["disp_pred"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"✅ ONNX exported: {output}")

if __name__ == "__main__":
    main()
