import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

from models import model as model_module


def main():
    parser = argparse.ArgumentParser(description="Profile S2ENet parameters and supported FLOPs.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # The weights do not affect model size or operation count; avoid hub downloads.
    original_resnet18 = model_module.resnet18

    def resnet18_without_pretrained(*model_args, **model_kwargs):
        model_kwargs.pop("pretrained", None)
        return original_resnet18(*model_args, pretrained=False, **model_kwargs)

    model_module.resnet18 = resnet18_without_pretrained
    device = torch.device(args.device)
    model = model_module.MYNet_2().eval().to(device)
    inputs = tuple(torch.randn(1, 3, args.height, args.width, device=device) for _ in range(2))

    flop_error = None
    unsupported = {}
    try:
        analysis = FlopCountAnalysis(model, inputs)
        flops = analysis.total()
        unsupported = dict(Counter(analysis.unsupported_ops()))
    except Exception as exc:
        flops = None
        flop_error = f"{type(exc).__name__}: {exc}"

    result = {
        "model": "S2ENet-MYNet_2",
        "input": "two RGB images, batch=1",
        "image_size": [args.height, args.width],
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "fvcore_flops_per_pair": flops,
        "fvcore_convention": "one multiply-add is counted as one FLOP",
        "unsupported_ops": unsupported,
        "fvcore_error": flop_error,
        "note": "Custom wavelet operations may be unsupported by fvcore and excluded from the operation total.",
    }
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
