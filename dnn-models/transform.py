"""
ONNX-to-C model transformer for intermittent inference firmware.

Reads a pre-trained ONNX model and the test dataset, quantizes weights to Q15
fixed-point, computes tiling parameters, lays out NVM, and writes build/data.h
and build/data.cpp which are compiled into the C/C++ firmware.

Usage: python dnn-models/transform.py --target {msp430,msp432} --hawaii CONFIG
       CONFIGS: cifar10-dnp, har-dnp, kws-dnp
       Run from the repository root. Output goes to --data-output-dir (default: build/).
"""

import argparse
import dataclasses
import enum
import io
import itertools
import logging
import os.path
import pathlib
import textwrap
from typing import Optional
import warnings

import cffi
import onnx
import onnx.defs
import onnx.helper
import onnx.numpy_helper
import numpy as np

from configs import (
    configs,
    vm_size,
    OUTPUT_LEN,
    ConfigType,
)
from utils import (
    DataLayout,
    INPLACE_UPDATE_OPS,
    THIS_DIR,
    PRUNING_INPUT_CHANNELS,
    PRUNING_OUTPUT_CHANNELS,
    add_multi_stage_nodes,
    ensure_non_negative_axis,
    get_attr,
    get_parameter_dims,
    find_kernel_shape,
    find_initializer,
    find_node_by_input,
    find_node_by_output,
    get_model_ops,
    infer_auto_pad,
    load_model,
    op_name,
    run_model_batched,
    run_model_single,
    to_bytes,
    sort_nodes,
)
from onnx_utils import (
    compute_parameter_scales,
    find_tensor_annotation,
    get_sample_size,
    list_tensors_for_annotations,
)
from model_utils import (
    find_min_range,
    apply_dynamic_channel_pruning,
)
from layer_utils import (
    determine_conv_tile_c,
    determine_gemm_tile_sizes,
)

logging.basicConfig()
logger = logging.getLogger("intermittent-cnn.transform")

# Map ONNX tensor names to integer indices.
# Indices 0..N_INPUT-1 are model inputs/initializers; N_INPUT+ are node outputs.


class DYNAMIC_DNN(enum.IntEnum):
    COARSE_GRAINED = 1
    FINE_GRAINED = 2
    ONE_INDICATOR = 3
    MULTIPLE_INDICATORS_BASIC = 4
    MULTIPLE_INDICATORS = 5


def enum_to_approach(enum_item):
    return enum_item.name.lower().replace("_", "-")


def approach_to_enum(approach):
    return getattr(DYNAMIC_DNN, approach.upper().replace("-", "_"))


DYNAMIC_DNN_APPROACH_NAMES = [enum_to_approach(enum_item) for enum_item in DYNAMIC_DNN]


class Constants:
    SLOT_PARAMETERS = 0xFE
    SLOT_TEST_SET = 0xFF
    NODE_NAME_LEN = 0  # will be filled later
    TURNING_POINTS_LEN = 8
    MODEL_NODES_LEN = 0
    INPUTS_DATA_LEN = 0
    NUM_INPUTS = 0  # will be filled during parsing
    N_INPUT = 0
    # Match the size of external FRAM
    NVM_SIZE = 2048 * 1024
    ORIG_NVM_SIZE = NVM_SIZE
    INTERMEDIATE_VALUES_SIZE = 0  # will be filled by nvm_layout()
    N_SAMPLES = 1
    LEA_BUFFER_SIZE = 0
    OUTPUT_LEN = OUTPUT_LEN
    USE_ARM_CMSIS = 0
    CONFIG: Optional[str] = None
    FORCE_STATIC_NETWORKS = 0

    BATCH_SIZE = 1
    HAWAII = 0
    INTERMITTENT = 0
    METHOD = "Baseline"
    FIRST_SAMPLE_OUTPUTS: list[float] = []
    USE_STATES_ARRAY = 0
    ENABLE_PER_LAYER_COUNTERS = 0
    COUNTERS_LEN = 0

    FP32_ACCURACY: float = 0.0  # will be filled

    DYNAMIC_DNN_APPROACH = DYNAMIC_DNN.FINE_GRAINED
    PRUNING_INPUT_CHANNELS = PRUNING_INPUT_CHANNELS
    PRUNING_OUTPUT_CHANNELS = PRUNING_OUTPUT_CHANNELS


other_flags = [
    # parameter flags
    "CHANNEL_LAST",
    "TRANSPOSED",
    "INTEGER",  # For integer types like INT64; no conversion like Q15 involved
]
other_node_flags = [
    "OFM_DUMPED",
    "INPUT_1_SCALE",
]


def op_flag(flag):
    return 2 ** other_flags.index(flag)


def _Q15(arr, name):
    """Transform a floating point number to TI's fixed point _q15 format"""

    # flatten data here, as the overflow check below assumes a flattened array
    arr = arr.flatten()

    # See DSPLib_1_30_00_02/include/DSPLib_support.h

    lower = -1
    upper = 32767.0 / 32768.0

    overflowed_indices = np.concatenate(
        (
            np.flatnonzero(np.asarray(arr < lower)),
            np.flatnonzero(np.asarray(arr > upper)),
        )
    )
    for idx in overflowed_indices:
        warnings.warn(
            f"{name} value {arr[idx]} goes beyond the range of _q15 ({lower}, {upper})"
        )

    arr = np.minimum(np.maximum(arr, lower), upper)

    return (arr * 2**15).astype(int)


def init_cffi():
    ffi = cffi.FFI()

    c_sources = ""
    in_static_assert = False
    with open(THIS_DIR.parent / "common" / "data_structures.h") as f:
        for line in f:
            if line.startswith("#include"):
                continue
            if line.startswith("static_assert"):
                in_static_assert = True
            if in_static_assert:
                # static_assert may span multiple lines; skip until the closing ");"
                if ");" in line:
                    in_static_assert = False
                continue
            c_sources += line
    ffi.cdef(c_sources)
    return ffi


ffi = init_cffi()


class ONNXNodeWrapper:
    def __init__(self, orig_node: onnx.NodeProto):
        self.orig_node = orig_node
        self.max_output_id = 0
        self.name = orig_node.name or orig_node.output[0] or orig_node.op_type
        self.inputs: list[int] = []
        self.parameters_by_importance = [-1, -1]
        self.flags = ffi.new("struct NodeFlags*")
        self.flags.canary = 0x55

    def __getattr__(self, name):
        return getattr(self.orig_node, name)


def get_prev_node(n):
    return nodes[names[n.input[0]] - Constants.N_INPUT]


parser = argparse.ArgumentParser()
parser.add_argument("config", choices=configs.keys())
parser.add_argument("--all-samples", action="store_true")
parser.add_argument("--write-images", action="store_true")
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--target", choices=("msp430", "msp432"), required=True)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--data-output-dir", metavar="DIR", default="build")
parser.add_argument("--model-variant", type=str, default="")
# Use type=str instead of converting it to an enum, as the latter is discouraged https://docs.python.org/3/library/argparse.html
parser.add_argument(
    "--dynamic-dnn-approach",
    type=str,
    choices=DYNAMIC_DNN_APPROACH_NAMES,
    default="multiple-indicators",
)
intermittent_methodology = parser.add_mutually_exclusive_group(required=True)
intermittent_methodology.add_argument("--ideal", action="store_true")
intermittent_methodology.add_argument("--hawaii", action="store_true")
args = parser.parse_args()
if args.debug:
    logging.getLogger("intermittent-cnn").setLevel(logging.DEBUG)
else:
    logging.getLogger("intermittent-cnn").setLevel(logging.INFO)

config: ConfigType = configs[args.config]
onnx_models = load_model(config)
onnx_model = onnx_models["single"]
onnx_model_batched = onnx_models["batched"]

sample_size = get_sample_size(onnx_model)

config["total_sample_size"] = int(np.prod(sample_size))
if "gemm_tile_length" not in config:
    config["gemm_tile_length"] = 0
Constants.CONFIG = args.config
if args.all_samples:
    Constants.N_SAMPLES = config["n_all_samples"]
    Constants.NVM_SIZE += (
        config["n_all_samples"] * 2 * config["total_sample_size"]
    )  # multiply by 2 for Q15
model_data = config["data_loader"](train=False, target_size=sample_size)
images, labels = next(iter(model_data.data_loader(limit=Constants.N_SAMPLES)))
images = images.numpy()

Constants.FIRST_SAMPLE_OUTPUTS = list(
    run_model_single(onnx_model_batched, model_data, verbose=False)[0]
)
Constants.FP32_ACCURACY = run_model_batched(
    onnx_model_batched, model_data, verbose=False
)
add_multi_stage_nodes(onnx_model)

Constants.BATCH_SIZE = args.batch_size
if args.hawaii:
    Constants.HAWAII = 1
    Constants.METHOD = "HAWAII"
Constants.INTERMITTENT = Constants.HAWAII
if args.target == "msp432":
    Constants.USE_ARM_CMSIS = 1
Constants.LEA_BUFFER_SIZE = vm_size[args.target]

if args.model_variant == "static" or not args.hawaii:
    # XXX: Currently, dynamic networks are supported only with HAWAII
    Constants.FORCE_STATIC_NETWORKS = 1
Constants.DYNAMIC_DNN_APPROACH = approach_to_enum(args.dynamic_dnn_approach)

names = {}

# Remove Squeeze and Reshape nodes with constants as the input
replaced_nodes_map: dict[str, str] = {}


def replace_squeeze(node, inp):
    # Since opset 13, axes is an input instead of an attribute
    try:
        axes_name = node.input[1]
        axes = find_initializer(onnx_model, axes_name).int64_data
    except IndexError:
        axes = get_attr(node, "axes")
    new_dims = [dim for dim_idx, dim in enumerate(inp.dims) if dim_idx not in axes]
    # Repeated fields cannot be assigned directly
    # https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-fields
    inp.dims[:] = new_dims


def replace_reshape(node, inp):
    dims_name = node.input[1]
    new_dims = find_initializer(onnx_model, dims_name).int64_data
    assert new_dims
    inp.dims[:] = new_dims


replace_handlers = {
    "Squeeze": replace_squeeze,
    "Reshape": replace_reshape,
}


def replace_nodes():
    for n in onnx_model.graph.node:
        if n.op_type not in ("Squeeze", "Reshape"):
            continue
        inp = find_initializer(onnx_model, n.input[0])
        if inp:
            replace_handlers[n.op_type](n, inp)
            replaced_nodes_map[n.output[0]] = n.input[0]


def transpose_gemm(onnx_model: onnx.ModelProto):
    for node in onnx_model.graph.node:
        # Only Gemm has transB, MatMul doesn't
        if node.op_type != "Gemm":
            continue
        transB = get_attr(node, "transB")
        B = find_initializer(onnx_model, node.input[1])
        if transB != 1 or B is None:
            continue
        data = onnx.numpy_helper.to_array(B)
        data = np.transpose(data)
        B.CopyFrom(
            onnx.helper.make_tensor(
                B.name, B.data_type, (B.dims[1], B.dims[0]), np.concatenate(data)
            )
        )
        for idx, attr in enumerate(node.attribute):
            if attr.name == "transB":
                del node.attribute[idx]
                break


replace_nodes()
transpose_gemm(onnx_model)

new_nodes = [
    n for n in onnx_model.graph.node if n.output[0] not in replaced_nodes_map.keys()
]
for n in new_nodes:
    for idx, inp in enumerate(n.input):
        n.input[idx] = replaced_nodes_map.get(inp, inp)

del onnx_model.graph.node[:]
onnx_model.graph.node.extend(new_nodes)

nodes = [ONNXNodeWrapper(n) for n in onnx_model.graph.node]

conv_param_names = set()

apply_dynamic_channel_pruning(onnx_model, nodes, config)

# apply_dynamic_channel_pruning may add more dependencies to some nodes - sort nodes again
nodes = sort_nodes(nodes)

for idx, inp in enumerate(onnx_model.graph.input):
    names[inp.name] = idx

# For some ONNX models, inputs
# do not include initializers. Merge them here.
inputs_len = len(names.keys())
for idx, initializer in enumerate(onnx_model.graph.initializer):
    if initializer.name not in names:
        names[initializer.name] = idx + inputs_len

compute_parameter_scales(onnx_model)

Constants.N_INPUT = len(names.keys())
logger.info("Constants.N_INPUT = %d", Constants.N_INPUT)

for idx, n in enumerate(nodes):
    if n.op_type == "Dropout":
        output = n.output[:1]  # we don't care the second output `mask`
    else:
        output = n.output
    if n.op_type == "Conv":
        conv_param_names.add(n.input[1])
        n.flags.conv.pads = infer_auto_pad(onnx_model, n)
        n.flags.conv.group = get_attr(n, "group") or 1
    if n.op_type in ("Conv", "MaxPool"):
        extra_flags = getattr(n.flags, op_name(n.op_type))
        kernel_shape = find_kernel_shape(onnx_model, n)
        extra_flags.kernel_shape = kernel_shape
        strides = get_attr(n, "strides")
        if strides is not None:
            extra_flags.strides = strides
        else:
            # "If not present, the stride defaults to 1 along each spatial axis."
            # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
            # https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool
            extra_flags.strides = (1, 1)
    if n.op_type == "MaxPool":
        ceil_mode = get_attr(n, "ceil_mode")
        if ceil_mode:
            n.flags.max_pool.ceil = 1
    if n.op_type == "Reshape":
        prev_node = n
        while prev_node and prev_node.op_type in INPLACE_UPDATE_OPS:
            prev_node = find_node_by_output(nodes, prev_node.input[0])
        if prev_node and prev_node.op_type == "MaxPool":
            prev_node.flags.max_pool.nhwc2nchw = 1
    if n.op_type in ("Squeeze", "Unsqueeze"):
        # axes is an input fo Squeeze and Unsqueeze since opset 13
        # https://onnx.ai/onnx/operators/onnx__Squeeze.html
        # https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
        axes_initializer = find_initializer(onnx_model, n.input[1])
        assert axes_initializer is not None
        axes = list(onnx.numpy_helper.to_array(axes_initializer))
        if n.op_type == "Unsqueeze":
            # Semantically, axes is required only for Unsqueeze
            assert axes is not None
        else:
            # For Squeeze, axes is optional. Here I use an empty list to indicate the default (squeeze all the single dimensions)
            axes = axes or []
        n.flags.squeeze.axes = 0
        for axis in axes:
            n.flags.squeeze.axes |= 1 << ensure_non_negative_axis(onnx_model, n, axis)

    if n.op_type in ("Gemm", "MatMul"):
        n.flags.gemm.input_dims = len(get_parameter_dims(onnx_model, n.input[0]))
        n.flags.gemm.weight_dims = len(get_parameter_dims(onnx_model, n.input[1]))
        logger.debug(
            "%s: input_dims=%d weight_dims=%d",
            n.name,
            n.flags.gemm.input_dims,
            n.flags.gemm.weight_dims,
        )
    if n.op_type in ("GemmStage2", "MatMulStage2"):
        n.flags.gemm_stage2.input_dims = len(get_parameter_dims(onnx_model, n.input[0]))
        n.flags.gemm_stage2.tile_length = config["gemm_tile_length"]

    if n.op_type == "Concat":
        # https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
        axis = get_attr(n, "axis")
        assert axis
        n.flags.concat.axis = ensure_non_negative_axis(onnx_model, n, axis)

    if n.op_type == "Transpose":
        perm = get_attr(n, "perm")
        assert len(perm) <= 4
        inverse_mapping = {
            mapped_index: original_index
            for original_index, mapped_index in enumerate(perm)
        }
        inverse_perm = [
            inverse_mapping[mapped_index] for mapped_index in range(len(perm))
        ]
        n.flags.transpose.perm = perm + [-1] * (4 - len(perm))
        n.flags.transpose.inverse_perm = inverse_perm + [-1] * (4 - len(perm))
    if n.op_type == "Softmax":
        # The default value for 'axis' is -1 since opset 13
        # https://onnx.ai/onnx/operators/onnx__Softmax.html
        axis = get_attr(n, "axis") or -1
        axis = ensure_non_negative_axis(onnx_model, n, axis)
        n.flags.softmax.axis = axis
        stage2_node = find_node_by_input(nodes, n.output[0])
        stage2_node.flags.softmax.axis = axis  # stage 2

    if n.op_type == "Add":
        # Make sure constants are in the second input is one of inputs contain constants
        # In some cases, no inputs are constants. Swapping inputs may be needed to avoid exploded
        # quantization scales, which happen when the first input feature map comes after many layers.
        if (
            config.get("swap_add", False)
            or len(n.input) == 2
            and find_initializer(onnx_model, n.input[0])
        ):
            n.input = [n.input[1], n.input[0]]

    if n.op_type == "ArgMax":
        # https://onnx.ai/onnx/operators/onnx__ArgMax.html#argmax-13
        axis = get_attr(n, "axis") or 0
        n.flags.arg_max.axis = ensure_non_negative_axis(onnx_model, n, axis)
        n.flags.arg_max.keepdims = get_attr(n, "keepdims")

    if n.op_type == "Gather":
        # https://onnx.ai/onnx/operators/onnx__Gather.html#gather-13
        axis = get_attr(n, "axis") or 0
        n.flags.gather.axis = ensure_non_negative_axis(onnx_model, n, axis)

    if n.op_type == "AveragePool":
        # https://onnx.ai/onnx/operators/onnx__AveragePool.html#averagepool-11
        kernel_shape = get_attr(n, "kernel_shape")
        assert kernel_shape
        auto_pad = get_attr(n, "auto_pad") or "NOTSET"
        pads = get_attr(n, "pads")

        input_dims = get_parameter_dims(onnx_model, n.input[0])
        if kernel_shape == input_dims[2:] and (
            auto_pad == "NOTSET" and set(pads) == {0} or auto_pad == "VALID"
        ):
            n.orig_node.op_type = "GlobalAveragePool"

    force_scale = config.get("force_scales", {}).get(n.name, "")
    if force_scale == "INPUT_1_SCALE":
        n.flags.general_flags |= 1 << other_node_flags.index(force_scale)
        print(n.flags.general_flags)
    for output_ in output:
        names[output_] = idx + Constants.N_INPUT

for idx, node in enumerate(nodes):
    node.inputs = [names[i] for i in node.input]
    for inp in node.inputs:
        if inp < Constants.N_INPUT:
            continue
        used_node = nodes[inp - Constants.N_INPUT]
        used_node.max_output_id = max([idx, used_node.max_output_id])

for node in nodes:
    if node.op_type not in INPLACE_UPDATE_OPS:
        continue

    inp = node.inputs[0]

    if inp < Constants.N_INPUT:
        continue

    node.max_output_id = max(
        [node.max_output_id, nodes[inp - Constants.N_INPUT].max_output_id]
    )

parameters = [None for _ in range(Constants.N_INPUT)]

tensors_referenced_in_annotations = list_tensors_for_annotations(onnx_model)
for params in onnx_model.graph.initializer:
    if params.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64):
        raise Exception("unsupported data type {}".format(params.data_type))
    if params.name in tensors_referenced_in_annotations:
        continue

    assert parameters[names[params.name]] is None
    parameters[names[params.name]] = params


def nchw2nhwc(arr, dims):
    arr = np.reshape(arr, dims)  # Change flattened to 4-D
    arr = np.transpose(arr, axes=(0, 2, 3, 1))  # NCHW -> NHWC
    return arr


outputs: dict[str, io.BytesIO] = {
    "parameters": io.BytesIO(),
    "samples": io.BytesIO(),
    "model": io.BytesIO(),
    "nodes": io.BytesIO(),
    "model_parameters_info": io.BytesIO(),
    "intermediate_parameters_info": io.BytesIO(),
    "labels": io.BytesIO(),
    "counters": io.BytesIO(),
}

ffi_objects: dict[str, list] = {
    "node_flags": [],
    "node_orig_flags": [],
    "footprints": [],
    "inference_stats": [],
    "inference_results": [],
}

Constants.MODEL_NODES_LEN = len(nodes)

model = outputs["model"]
model.write(to_bytes(0))  # Model.running
model.write(to_bytes(0))  # Model.run_counter
model.write(to_bytes(0))  # Model.layer_idx
for _ in range(config["num_slots"]):  # Model.slots_info
    model.write(to_bytes(-1))  # SlotInfo.user
model.write(to_bytes(0, size=8))  # Model.dummy
model.write(to_bytes(0, size=8))  # Model.version


@dataclasses.dataclass
class ParametersSlot:
    offset: int
    target: io.BytesIO
    slot_id: int


parameters_slot = ParametersSlot(
    offset=0, target=outputs["parameters"], slot_id=Constants.SLOT_PARAMETERS
)

output_nodes = outputs["nodes"]
for node in nodes:
    Constants.NUM_INPUTS = max(Constants.NUM_INPUTS, len(node.inputs))
logger.info("Maximum number of inputs = %d", Constants.NUM_INPUTS)

ops = get_model_ops(onnx_model)

for node in nodes:
    Constants.NODE_NAME_LEN = max(
        Constants.NODE_NAME_LEN, len(node.name), len(node.output[0])
    )


def write_str(buffer: io.BytesIO, data: str):
    buffer.write(data.encode("ascii") + b"\0" * (Constants.NODE_NAME_LEN - len(data)))


for node in nodes:
    write_str(output_nodes, node.name)
    write_str(output_nodes, node.output[0])
    output_nodes.write(to_bytes(len(node.inputs)))
    for inp in node.inputs:
        output_nodes.write(to_bytes(inp))
    for _ in range(Constants.NUM_INPUTS - len(node.inputs)):
        output_nodes.write(to_bytes(0))
    output_nodes.write(to_bytes(node.max_output_id))
    output_nodes.write(to_bytes(ops.index(node.op_type)))
    for parameter_idx in node.parameters_by_importance:
        output_nodes.write(to_bytes(parameter_idx))

if Constants.DYNAMIC_DNN_APPROACH in (
    DYNAMIC_DNN.MULTIPLE_INDICATORS_BASIC,
    DYNAMIC_DNN.MULTIPLE_INDICATORS,
):
    footprint_struct = "struct _ExtendedFootprint[]"
else:
    footprint_struct = "struct _Footprint[]"
footprints_arr = ffi.new(footprint_struct, 2 * len(nodes))
ffi_objects["footprints"].append(footprints_arr)

inference_stats_arr = ffi.new("struct InferenceStats[]", 2 * 2)
ffi_objects["inference_stats"].append(inference_stats_arr)

inference_results_arr = ffi.new("struct InferenceResults[]", 2)
ffi_objects["inference_results"].append(inference_results_arr)

# Allocate memory for NodeFlags but not fill in flags, as some flags (ex: tile sizes) depend on remaining NVM space.
# Actual flags will be filled later
node_flags = ffi.new("struct NodeFlags[]", len(nodes))

ffi_objects["node_orig_flags"].append(node_flags)

# Two copies for shadowing
for _ in range(2):
    ffi_objects["node_flags"].append(node_flags)

parameter_info_idx = 0


def write_scale(dest, scale):
    shift = 0
    while scale >= 1:
        shift += 1
        scale /= 2
    dest.write(to_bytes(int(scale * 2**15)))  # scale.fract
    dest.write(to_bytes(shift, size=8))  # scale.shift
    dest.write(to_bytes(0, size=8))  # scale.dummy


model_parameters_info = outputs["model_parameters_info"]
total_params = 0
for params in parameters:
    param_flags = 0

    if params is None:  # input
        if model_data.data_layout != DataLayout.NEUTRAL:
            param_flags |= 1 << other_flags.index("CHANNEL_LAST")

        # Actual data for test samples are added last
        dims = images[0].shape
        model_parameters_info.write(
            to_bytes(parameters_slot.offset, size=32)
        )  # params_offset
        model_parameters_info.write(
            to_bytes(np.prod(dims) * 2, size=32)
        )  # A _q15 is 16-bit
        model_parameters_info.write(to_bytes(Constants.SLOT_TEST_SET, size=8))  # slot
        model_parameters_info.write(to_bytes(param_flags, size=8))  # param_flags
        # extend_dims
        model_parameters_info.write(to_bytes(1))
        for dim in dims:
            model_parameters_info.write(to_bytes(dim))
        for _ in range(3 - len(dims)):
            model_parameters_info.write(to_bytes(0))
        write_scale(model_parameters_info, config["input_scale"])
    else:
        assert len(params.dims) <= 4
        params_data = onnx.numpy_helper.to_array(params)
        model_parameters_info.write(
            to_bytes(parameters_slot.offset, size=32)
        )  # params_offset
        if params.data_type == onnx.TensorProto.FLOAT:
            param_size = 2
            if params.name in conv_param_names:
                logger.info("Reorder conv param %s", params.name)
                params_data = nchw2nhwc(params_data, params.dims)
            used_node = find_node_by_input(onnx_model.graph.node, params.name)

            if used_node.op_type in ("Gemm", "MatMul"):
                params_data = np.reshape(params_data, params.dims)
                params_data = np.transpose(params_data)
                param_flags |= 1 << other_flags.index("TRANSPOSED")

            param_scale = (
                find_tensor_annotation(
                    onnx_model, key="Q15_SCALE_TENSOR", tensor_name=used_node.output[0]
                )
                or config["scale"]
            )
            logger.debug(
                "param_scale for %s (used for output %s): %f",
                params.name,
                used_node.output[0],
                param_scale,
            )
            parameters_slot.target.write(
                to_bytes(_Q15(params_data / param_scale, "Parameter"))
            )
        elif params.data_type == onnx.TensorProto.INT64:
            param_size = 8
            for param in params_data:
                parameters_slot.target.write(to_bytes(param, size=64))
            param_scale = 1
        else:
            assert False
        # params.dims is an empty list for scalars. In this case, np.prod returns a floating-point value.
        data_len = int(np.prod(params.dims))
        parameters_slot.offset += data_len * param_size
        model_parameters_info.write(to_bytes(data_len * param_size, size=32))
        model_parameters_info.write(to_bytes(parameters_slot.slot_id, size=8))  # slot
        model_parameters_info.write(to_bytes(param_flags, size=8))  # param_flags
        if len(params.dims) == 4:
            channels = params.dims[1]
        else:
            channels = 0
        logger.info("dims = %r, length = %d", params.dims, data_len)
        for dim in params.dims:
            model_parameters_info.write(to_bytes(dim))
        # dims are always 4 uint16_t's in C++
        for _ in range(4 - len(params.dims)):
            model_parameters_info.write(to_bytes(0))
        write_scale(model_parameters_info, param_scale)
        total_params += data_len

    # common to input and non-inputs
    model_parameters_info.write(to_bytes(parameter_info_idx))  # parameter_info_idx
    parameter_info_idx += 1

logger.info("Total params: %d", total_params)

# Placeholder for ParameterInfo of intermediate values
intermediate_parameters_info = outputs["intermediate_parameters_info"]
for idx, n in enumerate(nodes):
    intermediate_parameters_info.write(to_bytes(0, size=32))  # params_offset
    intermediate_parameters_info.write(to_bytes(0, size=32))  # params_len
    intermediate_parameters_info.write(to_bytes(0, size=8))  # slot
    intermediate_parameters_info.write(to_bytes(0, size=8))  # param_flags
    for _ in range(4):  # dims[4]
        intermediate_parameters_info.write(to_bytes(0))
    intermediate_parameters_info.write(to_bytes(0, size=32))  # scale
    intermediate_parameters_info.write(
        to_bytes(parameter_info_idx)
    )  # parameter_info_idx
    parameter_info_idx += 1


def ensure_channel_last(images, data_layout):
    if data_layout in (DataLayout.NEUTRAL, DataLayout.NHWC, DataLayout.NWC):
        return images
    elif data_layout == DataLayout.NCW:
        return np.transpose(images, axes=(0, 2, 1))  # NCW => NWC
    elif data_layout == DataLayout.NCHW:
        return np.transpose(images, axes=(0, 2, 3, 1))  # NCHW => NHWC
    else:
        raise NotImplementedError


images = ensure_channel_last(images, model_data.data_layout)
for idx in range(images.shape[0]):
    im = images[idx, :]
    outputs["samples"].write(
        to_bytes(_Q15(im.flatten(order="C") / config["input_scale"], "Input"))
    )
    if args.write_images:
        import cv2

        os.makedirs("images", exist_ok=True)
        # Restore conanical image format (H, W, C)
        im = np.squeeze(im * 256)
        cv2.imwrite(f"images/test{idx:02d}.png", im)

for label in labels:
    outputs["labels"].write(to_bytes(label, size=16))

if args.write_images:
    with open("images/ans.txt", "w") as f:
        f.write(" ".join(map(lambda label: str(label.item()), labels)))
    if model_data.label_names:
        with open("images/ans-texts.txt", "w") as f:
            f.write(
                " ".join(
                    map(lambda label: model_data.label_names[label.item()], labels)
                )
            )

if Constants.ENABLE_PER_LAYER_COUNTERS:
    Constants.COUNTERS_LEN = Constants.MODEL_NODES_LEN + 1
else:
    Constants.COUNTERS_LEN = 1
outputs["counters"].write(
    b"\0" * ffi.sizeof("struct Counters") * Constants.COUNTERS_LEN
)


def nvm_layout():
    # See common/platform.h; some items are duplicated for double buffering
    nvm_data_names = [
        "inference_stats",
        "model",
        "model",
        "intermediate_parameters_info",
        "node_flags",
        "nodes",
        "footprints",
        "counters",
        "inference_results",
        "parameters",
    ]
    remaining_size = Constants.ORIG_NVM_SIZE - 256
    for data_name in nvm_data_names:
        if data_name in outputs:
            cur_data_size = outputs[data_name].tell()
        else:
            cur_data_size = sum(ffi.sizeof(item) for item in ffi_objects[data_name])
        logger.debug("Data size for %s: %d", data_name, cur_data_size)
        remaining_size -= cur_data_size
    # Size for samples are different for plat-pc and plat-mcu
    remaining_size -= 2 * config["total_sample_size"]
    # intermediate_values_size should < 65536, or TI's compiler gets confused
    Constants.INTERMEDIATE_VALUES_SIZE = (
        int((remaining_size / config["num_slots"]) / 16) * 16
    )
    logger.debug("INTERMEDIATE_VALUES_SIZE=%d", Constants.INTERMEDIATE_VALUES_SIZE)


nvm_layout()

max_output_tile_size = 0
for idx, n in enumerate(nodes):
    if n.op_type == "Conv":
        cur_output_tile_c = determine_conv_tile_c(
            onnx_model,
            config,
            Constants.INTERMEDIATE_VALUES_SIZE,
            args.target,
            n,
            args.model_variant,
        )
        max_output_tile_size = max(max_output_tile_size, cur_output_tile_c)
    if n.op_type in ("Gemm", "MatMul"):
        cur_output_tile_c = determine_gemm_tile_sizes(
            onnx_model, config, Constants.BATCH_SIZE, args.target, n
        )
        max_output_tile_size = max(max_output_tile_size, cur_output_tile_c)

# Fill in actual flags
for node_idx, node in enumerate(nodes):
    # Dereferencing the pointer to flags via `node.flags[0]` as something like `*node.flags` is not valid Python
    # https://cffi.readthedocs.io/en/latest/using.html
    ffi_objects["node_orig_flags"][0][node_idx] = node.flags[0]
    for node_flags_copy_idx in range(2):
        ffi_objects["node_flags"][node_flags_copy_idx][node_idx] = node.flags[0]

pathlib.Path(args.data_output_dir).mkdir(exist_ok=True)


def enum_item_to_c_macro(enum_item):
    return enum_item.__class__.__name__ + "_" + enum_item.name


def enum_to_c_defines(enum_cls):
    ret = ""
    for enum_item in enum_cls:
        ret += f"#define {enum_item_to_c_macro(enum_item)} {enum_item.value}\n"
    return ret


with (
    open(f"{args.data_output_dir}/data.cpp", "w") as output_c,
    open(f"{args.data_output_dir}/data.h", "w") as output_h,
):
    output_h.write("""
#pragma once

#include <stdint.h>
#include "config.h"
#include "layer-defs.h"

struct ParameterInfo;
struct Model;
struct Node;
struct NodeFlags;

""")
    for item in itertools.chain(dir(Constants), config.keys()):
        if hasattr(Constants, item):
            if item.startswith("__"):
                continue
            val = getattr(Constants, item)
        else:
            val = config[item]
            # Somehow for integers, numpy.array uses int64 on Linux and int32 on Windows
            if not isinstance(val, (int, float, np.int64, np.int32)):
                continue

        if isinstance(val, enum.Enum):
            output_h.write(enum_to_c_defines(val.__class__))

        # Making it long to avoid overflow for expressions like
        # INTERMEDIATE_VALUES_SIZE * NUM_SLOTS on 16-bit systems
        suffix = "l" if item == "INTERMEDIATE_VALUES_SIZE" else ""
        output_h.write(f"#define {item.upper()} ")
        if isinstance(val, str):
            output_h.write(f'"{val}"')
        elif isinstance(val, list):
            output_h.write("{" + ", ".join(map(str, val)) + "}")
        elif isinstance(val, enum.Enum):
            output_h.write(enum_item_to_c_macro(val))
        else:
            output_h.write(f"{val}")
        output_h.write(f"{suffix}\n")

    output_c.write("""
#include "data.h"
#include "cnn_common.h"
#include "platform.h"
""")

    func_params = ",".join(
        (
            "struct Model *model",
            "const struct ParameterInfo *input[]",
            "struct ParameterInfo *output",
            "const struct Node* node",
            "CurNodeFlags* node_flags",
            "const struct NodeFlags* orig_node_flags",
        )
    )
    # ops
    output_h.write("\n")
    for idx, op in enumerate(ops):
        output_h.write(f"#define Op{op} {idx}\n")

    for op in ops:
        output_h.write("void alloc_{}({});\n".format(op_name(op), func_params))
        output_h.write("void handle_{}({});\n".format(op_name(op), func_params))
    output_c.write("const handler handlers[] = {\n")
    for op in ops:
        output_c.write(f"    handle_{op_name(op)},\n")
    output_c.write("};\n")
    output_c.write("const allocator allocators[] = {\n")
    for op in ops:
        output_c.write(f"    alloc_{op_name(op)},\n")
    output_c.write("};\n")

    output_h.write("extern const uint8_t INPLACE_UPDATE_OPS_MAP[];\n")
    output_c.write("const uint8_t INPLACE_UPDATE_OPS_MAP[] = {\n")
    for op in ops:
        output_c.write("{}, /* {} */".format(int(op in INPLACE_UPDATE_OPS), op))
    output_c.write("};\n")

    for op in ops:
        if op in INPLACE_UPDATE_OPS:
            output_c.write(
                textwrap.dedent(f"""
                void alloc_{op_name(op)}({func_params}) {{
                    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
                    if (cur_slot_info) {{
                        cur_slot_info->user = model->layer_idx;
                    }}
                }}
            """)
            )
        else:
            output_c.write(
                textwrap.dedent(f"""
                #if defined(FALLBACK_HANDLERS) && (defined(__GNUC__) || defined(__clang__))
                void __attribute__((weak)) alloc_{op_name(op)}({func_params}) {{
                    ERROR_OCCURRED();
                }}
                #endif
            """)
            )
        output_c.write(
            textwrap.dedent(f"""
            #if defined(FALLBACK_HANDLERS) && (defined(__GNUC__) || defined(__clang__))
            void __attribute__((weak)) handle_{op_name(op)}({func_params}) {{
                ERROR_OCCURRED();
            }}
            #endif
        """)
        )

    # data
    for flag_list in (other_flags, other_node_flags):
        for idx, name in enumerate(flag_list):
            output_h.write(f"#define {name} {2**idx}\n")

    def hex_str(arr):
        return "  " + ", ".join([f"0x{num:02x}" for num in arr]) + ",\n"

    def define_var(var_name, data):
        output_h.write(f"""
extern const uint8_t * const {var_name};
#define {var_name.upper()}_LEN {len(data)}
""")
        # #define with _Pragma seems to be broken :/
        output_c.write(f"""
const uint8_t _{var_name}[{len(data)}] = {{
""")
        n_pieces, remaining = divmod(len(data), 16)
        for idx in range(n_pieces):
            output_c.write(hex_str(data[idx * 16 : (idx + 1) * 16]))
        if remaining:
            output_c.write(hex_str(data[len(data) - remaining : len(data)]))
        output_c.write(f"""}};
const uint8_t * const {var_name} = _{var_name};
""")

    for var_name, data_obj in outputs.items():
        full_var_name = var_name + "_data"
        data_obj.seek(0)
        if full_var_name == "samples_data":
            data = data_obj.read(2 * config["total_sample_size"])
        else:
            data = data_obj.read()
        define_var(full_var_name, data)

    for var_name, ffi_data in ffi_objects.items():
        full_var_name = var_name + "_data"
        data = b""
        for item in ffi_data:
            data += ffi.buffer(item)
        define_var(full_var_name, data)

with open("samples.bin", "wb") as sample_file:
    samples = outputs["samples"]
    samples.seek(0)
    sample_file.write(samples.read())
