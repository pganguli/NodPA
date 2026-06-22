"""
ONNX graph utilities and data-format helpers for the transform pipeline.

Covers: model loading/optimisation, topological sorting, multi-stage node insertion
(Conv/Gemm/Softmax splitting), Q15 byte serialization, dataset download helpers,
and inference runners via onnxruntime.
"""

from __future__ import annotations

import enum
import graphlib
import itertools
import logging
import os.path
import pathlib
import re
import struct
import sys
import tarfile
import zipfile
from typing import Callable, Iterator, NamedTuple, Optional, Union
from urllib.request import urlretrieve

import filelock
import numpy as np
import onnx
import onnx.numpy_helper
import onnxoptimizer
import onnxruntime
import onnxruntime.backend as backend
import platformdirs
from torch.utils.data import Dataset, DataLoader

from onnx_utils import (
    find_initializer,
)

logger = logging.getLogger("intermittent-cnn.utils")


class OperatorProperties(NamedTuple):
    num_stages: int
    include_all_inputs: bool = False


INPLACE_UPDATE_OPS = ["Dropout", "Reshape", "Squeeze", "Unsqueeze"]
MULTIPLE_STAGE_OPS: dict[str, OperatorProperties] = {
    # Split Conv/Gemm/MatMul for merging OFMs from channel tiling
    "Conv": OperatorProperties(num_stages=2, include_all_inputs=True),
    "Gemm": OperatorProperties(num_stages=2),
    "MatMul": OperatorProperties(num_stages=2),
    # Split Softmax for normalization over exponential results
    "Softmax": OperatorProperties(num_stages=2),
    "GlobalAveragePool": OperatorProperties(num_stages=2),
}

THIS_DIR = pathlib.Path(__file__).absolute().parent

# The opset version this repo supports. Models with older opset versions will be converted to this version.
ONNX_OPSET_VERSION = 15

audio_ops = ["DecodeWav", "AudioSpectrogram", "Mfcc"]

# MSP432 DMA controller only allows 1024 transfers for a DMA command. For external FRAM,
# 1024 transfers = 1024 bytes = 512 Q-15 values
DMA_Q15_LIMIT = 512

PRUNING_INPUT_CHANNELS = 1
PRUNING_OUTPUT_CHANNELS = 2


class DataLayout(enum.Enum):
    NEUTRAL = 0
    NCW = 1
    NWC = 2
    NCHW = 3
    NHWC = 4


class ModelData(NamedTuple):
    dataset: Dataset
    data_layout: DataLayout
    label_names: Optional[list[str]] = None

    def data_loader(self, limit):
        return DataLoader(self.dataset, batch_size=(limit or len(self.dataset)))


def extract_archive(archive_path: pathlib.Path, subdir: str):
    archive_dir = archive_path.with_name(subdir)
    if not archive_dir.exists():
        if ".tar" in str(archive_path):
            with tarfile.open(archive_path) as tar:
                tar_members = [
                    member
                    for member in tar.getmembers()
                    if member.name.startswith(subdir)
                ]
                tar.extractall(archive_path.parent, members=tar_members)
        elif str(archive_path).endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zip_f:
                zip_members = [
                    member for member in zip_f.namelist() if member.startswith(subdir)
                ]
                zip_f.extractall(archive_path.parent, members=zip_members)
    return archive_dir


def kws_dnn_model():
    return download_file(
        "https://github.com/ARM-software/ML-KWS-for-MCU/raw/master/Pretrained_models/DNN/DNN_S.pb",
        "KWS-DNN_S.pb",
    )


def download_file(
    url: str, filename: str, post_processor: Optional[Callable] = None
) -> os.PathLike:
    xdg_cache_home = platformdirs.user_cache_path()

    lock_path = xdg_cache_home / f"{filename}.lock"

    # Inspired by https://stackoverflow.com/a/53643011
    class ProgressHandler:
        def __init__(self):
            self.last_reported = 0

        def __call__(self, block_num, block_size, total_size):
            progress = int(block_num * block_size / total_size * 100)
            if progress > self.last_reported + 5:
                logger.info("Downloaded: %d%%", progress)
                self.last_reported = progress

    with filelock.FileLock(lock_path):
        local_path = xdg_cache_home / filename
        if not local_path.exists():
            urlretrieve(url, local_path, ProgressHandler())

        ret = local_path
        if post_processor:
            ret = post_processor(local_path)

    return ret


def find_tensor_value_info(
    onnx_model: onnx.ModelProto, name: str
) -> onnx.ValueInfoProto:
    # Assume all stages have the same feature map dimensions
    name = re.sub(":stage[0-9]+$", "", name)

    g = onnx_model.graph
    for value_info in itertools.chain(g.value_info, g.input, g.output):
        if value_info.name == name:
            return value_info
    raise ValueError(f"No value_info found for {name}")


def find_node_and_idx_by_output(
    nodes: list[onnx.NodeProto], output_name: str
) -> tuple[int, Optional[onnx.NodeProto]]:
    for idx, node in enumerate(nodes):
        for output in node.output:
            if output == output_name:
                return idx, node
    return -1, None


def find_node_by_output(
    nodes: list[onnx.NodeProto], output_name: str
) -> Optional[onnx.NodeProto]:
    _, node = find_node_and_idx_by_output(nodes, output_name)
    return node


def find_node_and_idx_by_input(
    nodes: list[onnx.NodeProto], input_name: str
) -> tuple[int, Optional[onnx.NodeProto]]:
    for idx, node in enumerate(nodes):
        for input_ in node.input:
            if input_ == input_name:
                return idx, node

    return -1, None


def find_node_by_input(
    nodes: list[onnx.NodeProto], input_name: str
) -> Optional[onnx.NodeProto]:
    _, node = find_node_and_idx_by_input(nodes, input_name)
    return node


def get_attr(node, attr_name):
    for attr in node.attribute:
        if attr.name != attr_name:
            continue
        return onnx.helper.get_attribute_value(attr)

    # Not found
    return None


def find_kernel_shape(model, node):
    kernel_shape = get_attr(node, "kernel_shape")
    if not kernel_shape:
        if node.op_type == "MaxPool":  # this field is required for maxpool
            raise Exception("kernel_shape is required for MaxPool")
        weights = node.input[1]
        w = find_initializer(model, weights)
        kernel_shape = w.dims[2:]
    assert len(kernel_shape) == 2
    return kernel_shape


def infer_auto_pad(model, node):
    # https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv
    auto_pad = get_attr(node, "auto_pad")
    pads = get_attr(node, "pads") or [0] * 4
    assert len(pads) <= 4
    if auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
        kernel_shape = find_kernel_shape(model, node)
        pads[0] = pads[2] = kernel_shape[0] // 2
        pads[1] = pads[3] = kernel_shape[1] // 2
        # In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        # https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11
        for dim_idx in (0, 1):
            if kernel_shape[dim_idx] % 2 == 0:
                if auto_pad == b"SAME_UPPER":
                    pads[dim_idx] -= 1
                else:
                    pads[dim_idx + 2] -= 1
    logger.debug("Inferred pads: %r", pads)
    return pads


def numpy_type_to_onnx_elem_type(numpy_type):
    if numpy_type == np.float32:
        return onnx.TensorProto.FLOAT
    if numpy_type == np.int64:
        return onnx.TensorProto.INT64
    if numpy_type == np.bool_:
        return onnx.TensorProto.BOOL
    raise Exception(f"Unsupported type {numpy_type}")


def ensure_non_negative_axis(onnx_model: onnx.ModelProto, node: onnx.NodeProto, axis):
    # In many ONNX operators, a negative axis means counting back from the last dimension

    if axis < 0:
        axis += len(get_parameter_dims(onnx_model, node.input[0]))

    return axis


def get_model_ops(onnx_model):
    # Retrieving information for operators. Inspired by the script for generating
    # https://github.com/onnx/onnx/blob/v1.10.2/docs/Operators.md [1,2]
    # [1] https://github.com/onnx/onnx/blob/v1.10.2/onnx/defs/gen_doc.py
    # [2] https://github.com/onnx/onnx/blob/v1.10.2/onnx/onnx_cpp2py_export/defs.pyi
    ops = set()
    for schema in onnx.defs.get_all_schemas():
        ops.add(schema.name)

    ops = ops.intersection(node.op_type for node in onnx_model.graph.node)
    for op in MULTIPLE_STAGE_OPS.keys():
        if op in ops:
            for op_stage_idx in range(2, MULTIPLE_STAGE_OPS[op].num_stages + 1):
                ops.add(f"{op}Stage{op_stage_idx}")
    ops = sorted(ops)

    return ops


def load_model_from_file(model_name):
    # https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
    onnx_model = onnx.load_model(THIS_DIR / f"{model_name}.onnx")
    onnx_model = onnx.version_converter.convert_version(
        onnx_model, target_version=ONNX_OPSET_VERSION
    )
    return onnx_model


def load_model(config):
    onnx_model_batched = load_model_from_file(config["onnx_model"])

    # Get a single ONNX model (first dimension of input data is 1) besides the original batched
    # model (first dimension of input data is a placeholder). The batched model is needed for
    # running inference with the whole dataset (ex: for getting accuracy), and the single model is
    # simpler and suitable for on-device inference.
    if "onnx_model_single" in config:
        onnx_model_single = load_model_from_file(config["onnx_model_single"])
    else:
        onnx_model_single = change_batch_size(onnx_model_batched)
    onnx_model_single = onnx.shape_inference.infer_shapes(onnx_model_single)

    # Use the single model as onnxoptimizer requires known dimensions.
    # https://github.com/onnx/optimizer/blob/v0.2.6/onnxoptimizer/passes/fuse_matmul_add_bias_into_gemm.h#L60
    # https://zhuanlan.zhihu.com/p/41255090
    onnx_model_single = onnxoptimizer.optimize(
        onnx_model_single,
        [
            # Not using eliminate_nop_dropout, as it is not designed to work with opset >= 12
            # https://github.com/onnx/onnx/commit/8b3f7e2e7a0f2aba0e629e23d89f07c7fc0e6a5e#diff-43bf5856a3ddfa250094a7762f89327b0093049dd3c31df04e47f4801297f6d4
            "extract_constant_to_initializer",
            "fuse_add_bias_into_conv",
            "fuse_matmul_add_bias_into_gemm",
        ],
    )

    onnx_model_single = split2slice(onnx_model_single)
    remove_trailing_softmax(onnx_model_single)
    onnx.checker.check_model(onnx_model_single)

    remove_trailing_softmax(onnx_model_batched)
    onnx.checker.check_model(onnx_model_batched)

    return {
        "batched": onnx_model_batched,
        "single": onnx_model_single,
    }


def add_multi_stage_nodes(model):
    new_nodes = []
    for idx, n in enumerate(model.graph.node):
        if n.op_type in audio_ops:
            logger.warning("skipping audio operator %s", n.op_type)
            continue

        new_nodes.append(n)

        if n.op_type in MULTIPLE_STAGE_OPS.keys():
            orig_node_name = n.name or n.op_type
            orig_output_name = n.output[0]

            n.output[:] = [f"{orig_output_name}:stage1"]

            last_op_stage_idx = MULTIPLE_STAGE_OPS[n.op_type].num_stages
            for op_stage_idx in range(2, last_op_stage_idx + 1):
                new_node = onnx.NodeProto()
                new_node.name = f"{orig_node_name}:stage{op_stage_idx}"
                new_node.op_type = f"{n.op_type}Stage{op_stage_idx}"
                new_node.input[:] = [f"{orig_output_name}:stage{op_stage_idx - 1}"]
                if op_stage_idx == last_op_stage_idx:
                    new_node.output[:] = [orig_output_name]
                else:
                    new_node.output[:] = [f"{orig_output_name}:stage{op_stage_idx}"]

                if MULTIPLE_STAGE_OPS[n.op_type].include_all_inputs:
                    new_node.input.extend(n.input[1:])

                new_nodes.append(new_node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)


def onnxruntime_prepare_model(model):
    return backend.prepare(
        onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
    )


def onnxruntime_get_intermediate_tensor(
    model, image
) -> Iterator[tuple[str, str, np.ndarray]]:
    # Creating a new model with all nodes as outputs
    # https://github.com/microsoft/onnxruntime/issues/1455#issuecomment-979901463
    tmp_model = onnx.ModelProto()
    tmp_model.CopyFrom(model)

    orig_outputs = list(tmp_model.graph.output)
    orig_output_names = [node.name for node in orig_outputs]
    del tmp_model.graph.output[:]
    for node in tmp_model.graph.node:
        for output in node.output:
            if output not in orig_output_names:
                tmp_model.graph.output.append(onnx.ValueInfoProto(name=output))
    tmp_model.graph.output.extend(orig_outputs)

    rep = onnxruntime_prepare_model(tmp_model)
    outputs = rep.run(image)
    for idx, output in enumerate(outputs):
        output_name = tmp_model.graph.output[idx].name
        node = find_node_by_output(tmp_model.graph.node, output_name)
        yield output_name, node.op_type, output


def change_batch_size(onnx_model_batched: onnx.ModelProto):
    onnx_model_single = onnx.ModelProto()
    onnx_model_single.CopyFrom(onnx_model_batched)

    g = onnx_model_single.graph
    initializer_names = set([initializer.name for initializer in g.initializer])
    constant_names = set(
        [node.output[0] for node in g.node if node.op_type == "Constant"]
    )
    for value_info in itertools.chain(g.value_info, g.input, g.output):
        if value_info.name in initializer_names or value_info.name in constant_names:
            continue
        shape = value_info.type.tensor_type.shape
        if shape.dim and shape.dim[0].dim_param:
            shape.dim[0].dim_value = 1

    # make sure above steps did not break the model
    onnx.shape_inference.infer_shapes(onnx_model_single)

    return onnx_model_single


def print_float(val):
    print("%13.6f" % val, end="")


def print_tensor(tensor, print_histogram):
    shape = np.shape(tensor)
    print(f"Shape: {shape}")
    dimensions = np.shape(shape)[0]
    if dimensions == 4:
        N, C, H, W = shape
        for n in range(N):
            print(f"Matrix {n}")
            for c in range(C):
                print(f"Channel {c}")
                for h in range(H):
                    for w in range(W):
                        print_float(tensor[n, c, h, w])
                    print()
                print()
            print()
    elif dimensions == 3:
        N, C, W = shape
        for n in range(N):
            for c in range(C):
                print(f"Channel {c}")
                for w in range(W):
                    print_float(tensor[n, c, w])
                print()
                print()
            print()
    elif dimensions == 2:
        H, W = shape
        for h in range(H):
            for w in range(W):
                print_float(tensor[h, w])
            print()
    elif dimensions == 1:
        if shape[0] >= 1024:
            print(f"Skipping very long vector with length {shape[0]}")
            return
        for idx in range(shape[0]):
            print_float(tensor[idx])
            if idx % 16 == 15:
                print()
        print()
    else:
        print(f"Skip: unsupported {dimensions}-dimensional array")
    if dimensions >= 1 and np.prod(shape) != 0:
        if print_histogram:
            threshold = 1
            abs_tensor = np.absolute(tensor)
            total = np.prod(tensor.shape)
            while True:
                count = np.count_nonzero(
                    np.where(abs_tensor >= threshold, tensor, np.zeros(tensor.shape))
                )
                if not count:
                    break
                print(f">= {threshold}: {count} / {100.0 * count / total:.2f}%")
                threshold *= 2
        print(f"Max={np.max(tensor)}, min={np.min(tensor)}")


def run_model_single(
    model: onnx.ModelProto,
    model_data: ModelData,
    verbose: bool = True,
    save_file: Optional[os.PathLike[str]] = None,
) -> np.ndarray:
    # Testing
    images, labels = next(iter(model_data.data_loader(limit=1)))
    images = images.numpy()

    last_layer_out = None
    if verbose:
        print("Input")
        print_tensor(images, False)
    if save_file:
        model_output_pb2 = import_model_output_pb2()
        model_output = model_output_pb2.ModelOutput()
    for layer_name, op_type, layer_out in onnxruntime_get_intermediate_tensor(
        model, images
    ):
        if verbose:
            print(f"{op_type} layer: {layer_name}")
            print_tensor(layer_out, op_type in ("Conv", "Gemm", "MatMul"))
        if save_file:
            layer_out_obj = model_output_pb2.LayerOutput()
            layer_out_obj.name = layer_name
            layer_out_obj.op_type = op_type
            layer_out_obj.dims.extend(layer_out.shape)
            # needs int to handle zero-dimension tensor (scalar)
            linear_shape = [int(np.prod(layer_out.shape))]
            layer_out_obj.value.extend(np.reshape(layer_out, linear_shape))
            model_output.layer_out.append(layer_out_obj)
        last_layer_out = layer_out
    if save_file:
        with open(save_file, "wb") as f:
            f.write(model_output.SerializeToString())
    assert last_layer_out is not None
    return last_layer_out


def run_model_batched(
    model: onnx.ModelProto, model_data: ModelData, verbose: bool = True
) -> float:
    images, labels = next(iter(model_data.data_loader(limit=None)))
    images = images.numpy()

    correct = 0
    layer_outs = onnxruntime_prepare_model(model).run(images)[0]
    for idx, layer_out in enumerate(layer_outs):
        predicted = np.argmax(layer_out)
        if verbose:
            correct_marker = 1 if predicted == labels[idx] else 0
            print(
                f"idx={idx} label={labels[idx]} predicted={predicted} correct={correct_marker}"
            )
        if predicted == labels[idx]:
            correct += 1
    total = len(labels)
    accuracy = correct / total
    if verbose:
        print(f"correct={correct} total={total} rate={accuracy}")
    return accuracy


def remap_inputs(model: onnx.ModelProto, input_mapping: dict[str, str]):
    new_inputs = list(input_mapping.values())
    for new_input in new_inputs:
        model.graph.input.append(onnx.ValueInfoProto(name=new_input))
    for node in model.graph.node:
        node.input[:] = [input_mapping.get(inp, inp) for inp in node.input]
        node.output[:] = [
            output + "_unused" if output in new_inputs else output
            for output in node.output
        ]
    for idx, inp in enumerate(model.graph.input):
        if inp.name in input_mapping.keys():
            del model.graph.input[idx]

    return onnxoptimizer.optimize(model, ["eliminate_deadend"])


def to_bytes(arr, size=16):
    arr = np.array(arr).flatten()
    FORMAT_CHARS = {
        8: "B",  # unsigned char
        16: "h",
        32: "i",
        64: "q",
    }
    if size not in FORMAT_CHARS:
        raise ValueError(f"Unsupported size {size}")
    # https://stackoverflow.com/a/34794744
    return struct.pack("%u%c" % (len(arr), FORMAT_CHARS[size]), *arr)


def import_model_output_pb2():
    try:
        orig_sys_path = sys.path.copy()
        sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "build"))
        import model_output_pb2

        return model_output_pb2
    finally:
        sys.path = orig_sys_path


def split2slice(orig_model: onnx.ModelProto):
    model = onnx.ModelProto()
    model.CopyFrom(orig_model)

    new_nodes = []
    for node in model.graph.node:
        if node.op_type != "Split":
            new_nodes.append(node)
            continue

        # Slice: start, and, axis are inputs since opset 10
        # Split: split is an input since opset 13
        axis = get_attr(node, "axis")
        if axis is None:
            axis = 0

        # Lengths of the parts can be specified using input ‘split’. Otherwise, the tensor is split to equal sized parts.
        if len(node.input) == 2:
            split_data = find_initializer(model, node.input[1])
            assert split_data is not None
            split = list(onnx.numpy_helper.to_array(split_data))
        else:
            input_value_info = find_tensor_value_info(model, node.input[0])
            input_shape = input_value_info.type.tensor_type.shape
            output_split_dim_size, remaining = divmod(
                input_shape.dim[axis].dim_value, len(node.output)
            )
            assert remaining == 0
            split = [output_split_dim_size] * len(node.output)

        cur_start = 0
        cur_end = split[0]
        for idx, output in enumerate(node.output):
            output_node = find_node_by_input(model.graph.node, output)
            assert output_node is not None
            model.graph.initializer.extend(
                [
                    onnx.helper.make_tensor(
                        f"{node.name}_start{idx}",
                        onnx.TensorProto.INT64,
                        dims=[1],
                        vals=[cur_start],
                    ),
                    onnx.helper.make_tensor(
                        f"{node.name}_end{idx}",
                        onnx.TensorProto.INT64,
                        dims=[1],
                        vals=[cur_end],
                    ),
                    onnx.helper.make_tensor(
                        f"{node.name}_axis{idx}",
                        onnx.TensorProto.INT64,
                        dims=[1],
                        vals=[axis],
                    ),
                ]
            )
            cur_start = cur_end
            if idx < len(node.output) - 1:
                cur_end += split[idx + 1]
            new_node_inputs = [
                node.input[0],
                f"{node.name}_start{idx}",
                f"{node.name}_end{idx}",
                f"{node.name}_axis{idx}",
            ]
            new_node_output = f"{node.name}_output{idx}"
            new_nodes.append(
                onnx.helper.make_node(
                    "Slice",
                    inputs=new_node_inputs,
                    outputs=[new_node_output],
                    name=f"{node.name}_slice{idx}",
                )
            )
            replaced_inputs = [
                new_node_output if inp == output else inp for inp in output_node.input
            ]
            del output_node.input[:]
            output_node.input.extend(replaced_inputs)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # Run shape inference for newly-added Slice node outputs
    model = onnx.shape_inference.infer_shapes(model)

    return model


def get_parameter_dims(
    onnx_model: onnx.ModelProto, parameter_name
) -> list[Union[str, int]]:
    initializer = find_initializer(onnx_model, parameter_name)
    if initializer:
        return initializer.dims

    input_value_info = find_tensor_value_info(onnx_model, parameter_name)
    input_shape: onnx.TensorShapeProto = input_value_info.type.tensor_type.shape
    return [dim.dim_value or dim.dim_param for dim in input_shape.dim]


def remove_trailing_softmax(onnx_model: onnx.ModelProto):
    """Modify the model, such that if the last node is Softmax, remove that last node to avoid unnecessary computation."""
    output_name = onnx_model.graph.output[0].name
    output_node = find_node_by_output(onnx_model.graph.node, output_name)
    assert output_node

    if output_node.op_type == "Softmax":
        new_output_name = output_node.input[0]
        new_output_value_info = find_tensor_value_info(onnx_model, new_output_name)

        # change the desired output node
        del onnx_model.graph.output[:]
        onnx_model.graph.output.append(new_output_value_info)

        # remove the last node
        orig_nodes = onnx_model.graph.node[:]
        del onnx_model.graph.node[:]
        onnx_model.graph.node.extend(orig_nodes[:-1])


# Convert CamelCase to snake_case
# https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
def op_name(op: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", op).lower()


def sort_nodes(nodes: list[onnx.NodeProto]) -> list[onnx.NodeProto]:
    ts = graphlib.TopologicalSorter()

    for node_idx, node in enumerate(nodes):
        for node_input in node.input:
            input_node_idx, _ = find_node_and_idx_by_output(nodes, node_input)
            if input_node_idx == -1:
                # Not found in nodes, probably a graph input or a constant
                logger.debug(f"sort_nodes: skipping {node_input}")
                continue
            ts.add(node_idx, input_node_idx)

    new_nodes = []

    for sorted_node_idx in ts.static_order():
        new_nodes.append(nodes[sorted_node_idx])

    return new_nodes
