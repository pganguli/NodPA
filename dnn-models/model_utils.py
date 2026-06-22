"""
Dynamic channel pruning application and slot-allocation simulation.

apply_dynamic_channel_pruning() modifies the ONNX graph to encode the pruning
decision-map tensors as extra Conv inputs with threshold flags.
find_min_range() simulates the NVM slot allocator to determine the minimum gap
between slot sizes, used to size INTERMEDIATE_VALUES_SIZE.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnxoptimizer

from utils import (
    INPLACE_UPDATE_OPS,
    MULTIPLE_STAGE_OPS,
    PRUNING_INPUT_CHANNELS,
    PRUNING_OUTPUT_CHANNELS,
    find_tensor_value_info,
    find_node_by_input,
    find_node_by_output,
)
from onnx_utils import (
    dims_from_value_info,
    find_initializer,
)
from configs import ConfigType

logger = logging.getLogger("intermittent-cnn.model_utils")


@dataclasses.dataclass
class SlotInfo:
    user: int


def get_next_slot(
    onnx_model: onnx.ModelProto,
    slots: list[SlotInfo],
    nodes: list,
    config: dict[str, Any],
    layer_idx: int,
):
    # pick a unused slot
    next_slot_id = 0
    cycle_count = 0
    while True:
        next_slot_id += 1
        # Fail if the loop has run a cycle
        if next_slot_id >= config["num_slots"]:
            next_slot_id = 0
            cycle_count += 1
            assert cycle_count <= 1
        slot_user_id = slots[next_slot_id].user
        if slot_user_id < 0:
            break
        slot_user = nodes[slot_user_id]
        if slot_user.max_output_id < layer_idx:
            break

    slots[next_slot_id].user = layer_idx
    return next_slot_id


def find_min_range(
    onnx_model: onnx.ModelProto, nodes: list, config: ConfigType, N_INPUT: int
):
    slots = [SlotInfo(user=-1) for _ in range(config["num_slots"])]
    layer_slots = [None] * len(nodes)
    ofm_sizes = [[] for _ in range(config["num_slots"])]

    for layer_idx, node in enumerate(nodes):
        if node.op_type in INPLACE_UPDATE_OPS:
            if node.inputs[0] < N_INPUT:
                continue
            cur_slot_id = layer_slots[node.inputs[0] - N_INPUT]
            if cur_slot_id is not None:
                slots[cur_slot_id].user = layer_idx
                layer_slots[layer_idx] = cur_slot_id
            continue

        next_slot_id = get_next_slot(onnx_model, slots, nodes, config, layer_idx)
        layer_slots[layer_idx] = next_slot_id
        logger.debug(
            "next_slot_id for layer %d (%s) %d", layer_idx, node.op_type, next_slot_id
        )

        output_value_info = find_tensor_value_info(onnx_model, node.output[0])
        output_dims = dims_from_value_info(output_value_info)
        output_len = np.prod(output_dims)

        if node.op_type in MULTIPLE_STAGE_OPS.keys():
            # find IFM channels
            input_value_info = find_tensor_value_info(onnx_model, node.input[0])
            input_dims = dims_from_value_info(input_value_info)

            # find tile input channel
            wrapped_node = find_node_by_output(nodes, node.output[0])

            if node.op_type == "Conv":
                tile_channel = wrapped_node.flags.conv.input_tile_c
            elif node.op_type in ("Gemm", "MatMul"):
                tile_channel = wrapped_node.flags.gemm.tile_channel

            n_tiles = math.ceil(input_dims[0] / tile_channel)
            ofm_sizes[next_slot_id].append(n_tiles * output_len)
        else:
            ofm_sizes[next_slot_id].append(output_len)

    min_range = 65536
    for idx, sizes in enumerate(ofm_sizes):
        sizes = sorted(set(sizes))
        logger.debug("sizes for slot %d: %r", idx, sizes)
        for idx in range(len(sizes) - 1):
            min_range = min(min_range, sizes[idx + 1] - sizes[idx])

    return min_range


def remove_tensorflow_input_transpose(onnx_model: onnx.ModelProto):
    # Make input NCHW
    graph = onnx_model.graph
    input_transpose_node = find_node_by_input(
        nodes=graph.node, input_name=graph.input[0].name
    )
    node_after_transpose = find_node_by_input(
        nodes=graph.node, input_name=input_transpose_node.output[0]
    )
    node_after_transpose.input[0] = graph.input[0].name
    input_dims = graph.input[0].type.tensor_type.shape.dim
    # Swap C and W
    input_dims[3].dim_value, input_dims[1].dim_value = (
        input_dims[1].dim_value,
        input_dims[3].dim_value,
    )

    # Remove the now unused Transpose node
    onnx_model = onnxoptimizer.optimize(onnx_model, ["eliminate_deadend"])

    return onnx_model


def apply_dynamic_channel_pruning(
    onnx_model: onnx.ModelProto, nodes: list[onnx.NodeProto], config: ConfigType
):
    pruning_threshold = config.get("pruning_threshold", 0)

    if not pruning_threshold:
        return

    logger.info("pruning_threshold: %f", pruning_threshold)

    for node in nodes:
        if node.op_type != "Mul":
            continue

        decision_head_unsqueeze_node = find_node_by_output(nodes, node.input[0])
        decision_head_gather_node = find_node_by_output(
            nodes, decision_head_unsqueeze_node.input[0]
        )

        assert decision_head_gather_node.op_type == "Gather"

        decision_map = decision_head_gather_node.output[0]

        decision_map_candidates = find_initializer(
            onnx_model, decision_head_gather_node.input[0]
        )
        decision_map_candidates_arr = np.copy(
            onnx.numpy_helper.to_array(decision_map_candidates)
        )
        decision_map_candidates_arr[
            decision_map_candidates_arr < config["pruning_threshold"]
        ] = 0
        print(decision_map_candidates_arr.shape)
        print(decision_map_candidates_arr)
        print(np.count_nonzero(decision_map_candidates_arr, axis=1))
        decision_map_candidates.CopyFrom(
            onnx.helper.make_tensor(
                name=decision_map_candidates.name,
                data_type=decision_map_candidates.data_type,
                dims=decision_map_candidates.dims,
                vals=decision_map_candidates_arr,
            )
        )

        conv_stage2_node = find_node_by_output(nodes, node.input[1])
        conv_node = find_node_by_output(nodes, conv_stage2_node.input[0])

        relu_node = find_node_by_input(nodes, node.output[0])
        conv_node_after = find_node_by_input(nodes, relu_node.output[0])

        pruning_threshold_q15 = int(pruning_threshold * 2**15)

        conv_node.input.append(decision_map)
        conv_flags = conv_node.flags.conv
        conv_flags.pruning_target = PRUNING_OUTPUT_CHANNELS
        conv_flags.pruning_threshold = pruning_threshold_q15
        logger.debug(
            "Conv node %s, pruning target = PRUNING_OUTPUT_CHANNELS", conv_node.name
        )

        if conv_node_after.op_type == "Conv":
            conv_node_after.input.append(decision_map)
            conv_after_flags = conv_node_after.flags.conv
            conv_after_flags.pruning_target = PRUNING_INPUT_CHANNELS
            conv_after_flags.pruning_threshold = pruning_threshold_q15
            logger.debug(
                "Conv node %s, pruning target = PRUNING_INPUT_CHANNELS",
                conv_node_after.name,
            )

        logger.debug(
            "Conv node %s, pruning_threshold=%d",
            conv_node.name,
            conv_node.flags.conv.pruning_threshold,
        )
