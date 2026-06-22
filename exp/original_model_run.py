#!/usr/bin/env python3
"""
Run inference through the ONNX runtime on the full test dataset (Python baseline).

Prints per-sample predictions and overall accuracy. With --save-file, saves
intermediate feature maps to a .pb file for comparison with the C simulator via
exp/compare-model-output.py (requires cmake -D USE_PROTOBUF=ON).

Usage: python exp/original_model_run.py CONFIG [--limit N] [--save-file FILE.pb]
"""

import argparse
import pathlib
import sys

TOPDIR = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(TOPDIR / "dnn-models"))

from configs import configs
from utils import load_model, run_model_single, run_model_batched
from onnx_utils import get_sample_size


def main():
    parser = argparse.ArgumentParser(
        description="Run inference through the ONNX runtime on the full test dataset (Python baseline)."
    )
    parser.add_argument(
        "config",
        choices=configs.keys(),
        help="Model config name (cifar10-dnp, har-dnp, kws-dnp)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N samples (0 = all samples)",
    )
    parser.add_argument(
        "--save-file",
        help="Save intermediate feature maps to a .pb file for compare-model-output.py (requires cmake -D USE_PROTOBUF=ON -D MY_DEBUG=2)",
    )
    args = parser.parse_args()

    if args.limit == 0:
        args.limit = None

    config = configs[args.config]
    models = load_model(config)
    model = models["batched"]
    model_data = config["data_loader"](train=False, target_size=get_sample_size(model))
    if args.limit == 1:
        run_model_single(
            model, model_data, verbose=not args.save_file, save_file=args.save_file
        )
    else:
        run_model_batched(model, model_data)


if __name__ == "__main__":
    main()
