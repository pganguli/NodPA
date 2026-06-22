#!/usr/bin/env python3
"""
Compare per-layer outputs between the ONNX runtime (Python) and the C simulator.

Both sides must have saved their intermediate feature maps to .pb files using
exp/original_model_run.py --save-file and ./build/intermittent-cnn -s respectively
(the simulator requires cmake -D USE_PROTOBUF=ON -D MY_DEBUG=2).

Usage: python exp/compare-model-output.py --baseline PY.pb --target CPP.pb --topk N
"""

import argparse
import pathlib
import sys

import numpy as np

TOPDIR = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(TOPDIR / "dnn-models"))

from utils import import_model_output_pb2


def get_tensor(layer_out) -> np.ndarray:
    arr = np.array(layer_out.value)
    if not len(arr):
        return np.array([])
    dims = np.array(layer_out.dims)
    try:
        return np.reshape(arr, dims[dims != 0])
    except ValueError:
        # Mismatched dimensions, which happen when T_n != N - ignoring for now
        return np.array([])


def find_layer_output(model_output, name):
    for layer_out in model_output.layer_out:
        if layer_out.name == name:
            return layer_out


def main():
    parser = argparse.ArgumentParser(
        description="Compare per-layer outputs between the ONNX runtime (Python) and the C simulator."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the .pb file saved by exp/original_model_run.py --save-file",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to the .pb file saved by the C simulator with -s",
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=True,
        help="Number of top relative-error values to print per layer",
    )
    args = parser.parse_args()

    np.seterr(divide="raise")

    baseline_data: dict[str, np.ndarray] = {}
    model_output = import_model_output_pb2().ModelOutput()
    with open(args.baseline, "rb") as f:
        model_output.ParseFromString(f.read())
    for layer_out in model_output.layer_out:
        baseline_data[layer_out.name] = get_tensor(layer_out)

    with open(args.target, "rb") as f:
        model_output.ParseFromString(f.read())

    for layer_out in model_output.layer_out:
        name = layer_out.name
        op_type = layer_out.op_type

        print(f"Layer output {name}, op_type = {op_type}")
        if name not in baseline_data:
            continue
        cur_baseline_data = baseline_data[name]
        cur_target_data = get_tensor(layer_out)

        if cur_baseline_data.shape != cur_target_data.shape:
            print(
                f"ERROR: baseline data shape {cur_baseline_data.shape} != target data shape {cur_target_data.shape}"
            )
            continue

        if op_type == "ConvStage2":
            mask_layer_out = find_layer_output(model_output, name + ":mask")
            if mask_layer_out:
                channel_masks = get_tensor(mask_layer_out)
                pruning_threshold = get_tensor(
                    find_layer_output(model_output, name + ":thres")
                )[0]
                cur_baseline_data[channel_masks < pruning_threshold] = 0
                cur_target_data[channel_masks < pruning_threshold] = 0

        max_num = np.max(np.abs(baseline_data[name]))
        if max_num:
            errors = np.abs(cur_baseline_data - cur_target_data) / max_num
        else:
            errors = np.abs(cur_baseline_data - cur_target_data)

        # Sort on negative values to get indices for decreasing values
        # https://www.kite.com/python/answers/how-to-use-numpy-argsort-in-descending-order-in-python
        error_indices = np.unravel_index(np.argsort(-errors, axis=None), errors.shape)
        top_error_indices = np.array(error_indices)[:, : args.topk]
        for index_idx in range(top_error_indices.shape[1]):
            # indices should be a tuple
            value_idx = tuple(top_error_indices[:, index_idx])
            value_idx_str = "(" + ", ".join(f"{idx:3d}" for idx in value_idx) + ")"
            # :e => scientific notation
            print(
                ", ".join(
                    [
                        f"index={value_idx_str}",
                        f"baseline={cur_baseline_data[value_idx]:+e}",
                        f"target={cur_target_data[value_idx]:+e}",
                        f"error={errors[value_idx]:e}",
                    ]
                )
            )


if __name__ == "__main__":
    main()
