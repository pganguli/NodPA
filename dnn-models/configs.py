"""
Model configuration registry for the NodPA transform and inference pipeline.

Each config entry (cifar10-dnp, har-dnp, kws-dnp) specifies the ONNX model file,
Q15 quantization scales, NVM slot count, dataset loader, and pruning settings.
Imported by transform.py and exp/original_model_run.py.
"""

import functools
from typing import Protocol, TypedDict
import sys

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

from datasets import (
    load_data_cifar10,
    load_data_google_speech,
    load_har,
)
from utils import ModelData

ARM_PSTATE_LEN = 8704
# Acceleration output buffer size
# TODO: make these adjustable on runtime
OUTPUT_LEN = 256

vm_size = {
    # (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)
    "msp430": 1884,
    # determined by trial and error
    "msp432": 25000,  # includes space for pState
}


class DataLoader(Protocol):
    def __call__(self, train: bool, target_size: tuple[int, int]) -> ModelData: ...


class ConfigType(TypedDict):
    onnx_model: str
    onnx_model_single: NotRequired[str]
    scale: float
    input_scale: float
    num_slots: int
    data_loader: DataLoader
    n_all_samples: int
    op_filters: int
    # Filled by transform.py
    total_sample_size: NotRequired[int]
    gemm_tile_length: NotRequired[int]

    # For dynamic channel pruning
    pruning_threshold: NotRequired[float]


configs: dict[str, ConfigType] = {
    "cifar10-dnp": {
        "onnx_model": "cifar10_resnet10-batched",
        "onnx_model_single": "cifar10_resnet10-single",
        "scale": 1,
        "input_scale": 3,
        "num_slots": 4,
        "data_loader": functools.partial(load_data_cifar10, normalized=True),
        "n_all_samples": 10000,
        "op_filters": 2,
        "pruning_threshold": 0.5,
        "swap_add": False,
        "force_scales": {
            "/layers/layers.1/layers.1.0/conv2/Conv": "INPUT_1_SCALE",
        },
    },
    "kws-dnp": {
        "onnx_model": "kws_kws-batched",
        "onnx_model_single": "kws_kws-single",
        "scale": 1,
        "input_scale": 120,
        "num_slots": 3,
        "data_loader": functools.partial(load_data_google_speech, for_cnn=True),
        "n_all_samples": 4890,
        "op_filters": 2,  # decision head is not compatible with op_filters=4
        "pruning_threshold": 0.5,
        "max_output_channel": 16,
    },
    "har-dnp": {
        "onnx_model": "har_har_cnn-batched",
        "onnx_model_single": "har_har_cnn-single",
        "scale": 1,
        "input_scale": 48,
        "num_slots": 3,
        "data_loader": load_har,
        "n_all_samples": 2947,
        "op_filters": 4,
        "pruning_threshold": 0.5,
    },
}
