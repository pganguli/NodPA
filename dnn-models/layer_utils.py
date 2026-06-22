"""
Tile-size determination for Conv and Gemm layers given MCU memory constraints.

Computes the maximum input/output channel tile counts that fit in LEA SRAM
(lea_buffer) after accounting for filter storage, pState, and DMA alignment.
Called by transform.py after NVM layout is fixed.
"""

from __future__ import annotations

import logging
import math
import os

import onnx

from utils import (
    DMA_Q15_LIMIT,
    PRUNING_OUTPUT_CHANNELS,
    find_initializer,
    find_tensor_value_info,
)
from configs import (
    ARM_PSTATE_LEN,
    OUTPUT_LEN,
    vm_size,
    ConfigType,
)

logger = logging.getLogger("intermittent-cnn.layer_utils")


def extend_for_footprints(batch_size, n):
    return n + n // batch_size


def determine_conv_tile_c(
    onnx_model: onnx.ModelProto,
    config: ConfigType,
    intermediate_values_size,
    target,
    node,
    model_variant,
):
    logger.debug("Determine tile size for Conv node %s", node.name)

    output_value_info = find_tensor_value_info(onnx_model, node.output[0])
    filter_info = find_initializer(onnx_model, node.input[1])
    assert filter_info is not None

    shape = output_value_info.type.tensor_type.shape
    OUTPUT_CHANNEL = shape.dim[1].dim_value
    OUTPUT_H = shape.dim[2].dim_value
    OUTPUT_W = shape.dim[3].dim_value
    CHANNEL = filter_info.dims[1]
    kH = filter_info.dims[2]
    kW = filter_info.dims[3]

    conv_flags = node.flags.conv

    if (
        not conv_flags.pruning_threshold
        or conv_flags.pruning_target == PRUNING_OUTPUT_CHANNELS
    ):
        conv_flags.input_tile_c = CHANNEL
    else:
        if model_variant == "static":
            # NVM is too small to store all output features with input_tile_c = 1
            conv_flags.input_tile_c = 2
        else:
            conv_flags.input_tile_c = 1

    logger.debug("Initial input_tile_c=%d", conv_flags.input_tile_c)

    def get_tile_input_usage(output_tile_c, filter_len):
        real_output_tile_c = output_tile_c
        ret = ((real_output_tile_c + 1) + 1) * filter_len
        return ret

    def get_pstate_usage(output_tile_c, filter_len):
        if target != "msp432":
            return 0

        n_filters = output_tile_c
        return filter_len * n_filters

    while True:
        input_tile_too_large = False
        # inner +1 for biases
        filter_len = ((conv_flags.input_tile_c * kW + 1) + 1) // 2 * 2 * kH

        if (
            conv_flags.pruning_threshold
            and conv_flags.pruning_target == PRUNING_OUTPUT_CHANNELS
        ):
            # Not using 1, as LEA requires even dimensions
            output_tile_c = 2
        else:
            output_tile_c = min(OUTPUT_CHANNEL, config.get("max_output_channel", 32))

        while True:
            tile_input_usage = get_tile_input_usage(output_tile_c, filter_len)
            pState_usage = get_pstate_usage(output_tile_c, filter_len)
            total_vm_usage = tile_input_usage + pState_usage
            logger.debug(
                "Checking output_tile_c=%d, filter_len=%d, tile_input_usage=%d, pState_usage=%d, total_vm_usage=%d",
                output_tile_c,
                filter_len,
                tile_input_usage,
                pState_usage,
                total_vm_usage,
            )
            if ARM_PSTATE_LEN is not None and target == "msp432":
                if (
                    tile_input_usage <= vm_size[target] - OUTPUT_LEN - ARM_PSTATE_LEN
                    and pState_usage <= ARM_PSTATE_LEN
                ):
                    break
            else:
                if total_vm_usage <= vm_size[target] - OUTPUT_LEN:
                    break
            logger.debug("output_tile_c=%d", output_tile_c)
            output_tile_c //= 2
            if output_tile_c % 2 or output_tile_c < config["op_filters"]:
                # current input_tile_c is too large such that no even output_tile_c fits
                input_tile_too_large = True
                logger.debug("Input too large!")
                break

        if not input_tile_too_large:
            params_len = (
                math.ceil(CHANNEL / conv_flags.input_tile_c)
                * OUTPUT_CHANNEL
                * OUTPUT_H
                * OUTPUT_W
                * 2
            )
            if params_len <= intermediate_values_size:
                break
            logger.debug(f"params_len={params_len}, too high!")
        assert conv_flags.input_tile_c // 2 * 2 == conv_flags.input_tile_c
        conv_flags.input_tile_c //= 2
        logger.debug("input_tile_c=%d", conv_flags.input_tile_c)
    conv_flags.output_tile_c = output_tile_c

    reduce_output_ratio = float(os.getenv("TILE_SIZE_RATIO") or 1)
    conv_flags.output_tile_c = round(conv_flags.output_tile_c * reduce_output_ratio)
    conv_flags.output_tile_c = max(2, conv_flags.output_tile_c // 2 * 2)
    conv_flags.pState_len = pState_usage

    return output_tile_c


def get_gemm_pState_usage(tile_channel, tile_b_cols, target):
    if target == "msp432":
        return (tile_channel + 2) * (tile_b_cols * 2)
    return 0


def check_gemm_vm_usage(A, tile_channel, tile_a_rows, tile_b_cols, batch_size, target):
    A_shape = A.type.tensor_type.shape
    input_dims = len(A_shape.dim)
    A_cols = A_shape.dim[input_dims - 1].dim_value

    full_tile_b_cols = (extend_for_footprints(batch_size, tile_b_cols) + 1) / 2 * 2
    tile_input_usage = (
        (tile_a_rows * A_cols + 2)
        + (tile_channel + 2) * full_tile_b_cols
        + tile_a_rows * full_tile_b_cols
    )
    pState_usage = get_gemm_pState_usage(tile_channel, tile_b_cols, target)
    total_vm_usage = tile_input_usage + pState_usage

    ret = False
    if ARM_PSTATE_LEN is not None and target == "msp432":
        if (
            tile_input_usage <= vm_size[target] - ARM_PSTATE_LEN
            and pState_usage <= ARM_PSTATE_LEN
        ):
            ret = True
    else:
        if total_vm_usage <= vm_size[target]:
            ret = True
    logger.debug(
        "tile_channel=%d, tile_a_rows=%d, tile_b_cols=%d, tile_input_usage=%d, pState_usage=%d, total_vm_usage=%d => %s",
        tile_channel,
        tile_a_rows,
        tile_b_cols,
        tile_input_usage,
        pState_usage,
        total_vm_usage,
        "OK" if ret else "not OK",
    )

    return ret


def determine_gemm_tile_sizes(
    onnx_model: onnx.ModelProto, config: ConfigType, batch_size, target, node
):
    logger.debug("Determine tile size for Gemm node %s", node.name)

    A = find_tensor_value_info(onnx_model, node.input[0])
    A_shape = A.type.tensor_type.shape
    input_dims = len(A_shape.dim)
    A_rows = A_shape.dim[input_dims - 2].dim_value

    B = find_initializer(onnx_model, node.input[1])
    if B is not None:
        weight_dims = len(B.dims)
        B_rows = B.dims[weight_dims - 2]
        B_cols = B.dims[weight_dims - 1]
    else:
        B = find_tensor_value_info(onnx_model, node.input[1])
        B_shape = B.type.tensor_type.shape
        weight_dims = len(B_shape.dim)
        B_rows = B_shape.dim[weight_dims - 2].dim_value
        B_cols = B_shape.dim[weight_dims - 1].dim_value

    B_rows += B_rows % 2
    B_cols += B_cols % 2

    # writing a batch at a time is simpler and faster
    tile_size_unit = config["op_filters"]

    gemm_flags = node.flags.gemm
    gemm_flags.tile_a_rows = 1
    gemm_flags.tile_b_cols = tile_size_unit

    # LEA wants addresses to be 4 byte-aligned, or 2 Q15-aligned
    gemm_flags.tile_channel = (
        min([B_rows, (config["gemm_tile_length"] or float("inf")), DMA_Q15_LIMIT])
        // tile_size_unit
        * tile_size_unit
    )
    while True:
        if check_gemm_vm_usage(
            A,
            gemm_flags.tile_channel,
            gemm_flags.tile_a_rows,
            gemm_flags.tile_b_cols,
            batch_size,
            target,
        ):
            break
        assert gemm_flags.tile_channel > gemm_flags.tile_b_cols
        gemm_flags.tile_channel -= gemm_flags.tile_b_cols

    assert gemm_flags.tile_b_cols % tile_size_unit == 0

    while True:
        new_tile_b_cols = gemm_flags.tile_b_cols + tile_size_unit
        if new_tile_b_cols > B_cols:
            break
        if not check_gemm_vm_usage(
            A,
            gemm_flags.tile_channel,
            gemm_flags.tile_a_rows,
            new_tile_b_cols,
            batch_size,
            target,
        ):
            break
        gemm_flags.tile_b_cols = new_tile_b_cols

    while True:
        new_tile_a_rows = gemm_flags.tile_a_rows + 1
        if new_tile_a_rows > A_rows:
            break
        if not check_gemm_vm_usage(
            A,
            gemm_flags.tile_channel,
            new_tile_a_rows,
            gemm_flags.tile_b_cols,
            batch_size,
            target,
        ):
            break
        gemm_flags.tile_a_rows = new_tile_a_rows

    gemm_flags.pState_len = get_gemm_pState_usage(
        gemm_flags.tile_channel, gemm_flags.tile_b_cols, target
    )

    return gemm_flags.tile_b_cols
