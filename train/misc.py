"""
Shared training utilities for dynamic-pruning model training.

Provides: Logger singleton, AverageMeter, accuracy computation, dataset loading
(CIFAR-10/HAR/KWS), model initialization, model transformation (inserting decision
heads), and a shared argparse factory used by train_baseline.py, main.py, and export.py.
"""

import argparse
from PIL import Image
import os
import shutil
import pickle as pkl
import time
from datetime import datetime
from torchvision import transforms, datasets
import torch
import pathlib
import numpy as np
from torch.utils.data import TensorDataset
import sys
import filelock
import platformdirs

import models
from decision import (
    apply_func,
    collect_params,
    normalize_head_weights,
    replace_func,
)

THIS_DIR = pathlib.Path(__file__).absolute().parent


def action_num(arch):
    if arch.startswith("resnet"):
        return 40
    elif arch in ("har_cnn", "kws"):
        return 5
    raise Exception("Unknown model architecture {}".format(arch))


def learning_rate(arch):
    if arch == "kws":
        return 0.01
    elif arch == "har_cnn" or arch.startswith("resnet"):
        return 0.1
    raise Exception("Unknown model architecture {}".format(arch))


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name="log"):
        if self._logger is None:
            import logging

            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel("INFO")
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        now = datetime.now()
        display_now = str(now).split(" ")[1][:-3]
        self.init(os.path.expanduser("~/tmp_log"), "tmp.log")
        self._logger.info("[" + display_now + "]" + " " + str_info)


logger = Logger()


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def load_pickle(path, verbose=True):
    begin_st = time.time()
    with open(path, "rb") as f:
        if verbose:
            print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    if verbose:
        print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v


def dump_pickle(obj, path):
    with open(path, "wb") as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


def prepare_logging(args):
    args.logdir = os.path.join("./logs", args.logdir)

    logger.init(args.logdir, "log")

    ensure_dir(args.logdir)
    logger.info("=================FLAGS==================")
    for k, v in args.__dict__.items():
        logger.info("{}: {}".format(k, v))
    logger.info("========================================")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_basic_argument_parser(default_wd: float):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kws", type=str)
    parser.add_argument("--arch", "-a", default="kws", type=str)
    parser.add_argument("--sparsity_level", default=0.1, type=float)
    parser.add_argument("--mm", default=0.9, type=float)
    parser.add_argument("--wd", default=default_wd, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--pruning_threshold", default=0.5, type=float)

    return parser


def preprocess_kws_dataset(original_dataset):
    GOOGLE_SPEECH_SAMPLE_RATE = 16000
    # From https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Pretrained_models/labels.txt
    new_labels = "_silence_ _unknown_ yes no up down left right on off stop go".split(
        " "
    )

    import tensorflow as tf

    with open(kws_dnn_model(), "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)

    with tf.compat.v1.Session() as sess:
        mfcc_tensor = sess.graph.get_tensor_by_name("Mfcc:0")

        labels = []
        mfccs = None  # The dimensions are unknown yet
        dataset_size = len(original_dataset)
        for n in range(dataset_size):
            # The first few _unknown_ samples are not recognized by Hello Edge's DNN model - use good ones instead
            data = original_dataset[dataset_size - 1 - n]
            waveform, sample_rate, label, _, _ = data
            assert sample_rate == GOOGLE_SPEECH_SAMPLE_RATE
            decoded_wav = np.squeeze(waveform)
            # Some files in speech_commands_v0.02 are less than 1 second. By comparing with corresponding files in
            # speech_commands_test_set_v0.02, zeros are padded at last apparently
            decoded_wav = np.pad(
                decoded_wav, ((0, GOOGLE_SPEECH_SAMPLE_RATE - len(decoded_wav)),)
            )
            decoded_wav = np.expand_dims(decoded_wav, axis=-1)

            # See the logic at https://github.com/tensorflow/datasets/blob/v4.6.0/tensorflow_datasets/audio/speech_commands.py#L128-L140
            if label in new_labels:
                label = new_labels.index(label)
            elif label in ("_silence_", "_background_noise_"):
                label = new_labels.index("_silence_")
            else:
                label = new_labels.index("_unknown_")
            labels.append(label)

            mfcc = sess.run(
                mfcc_tensor,
                {
                    "decoded_sample_data:0": decoded_wav,
                    "decoded_sample_data:1": GOOGLE_SPEECH_SAMPLE_RATE,
                },
            )
            if mfccs is None:
                mfccs_shape = list(mfcc.shape)
                mfccs_shape[0] = dataset_size
                mfccs = np.zeros(mfccs_shape)
            mfccs[n, :, :] = mfcc

    return mfccs, labels


def load_data_google_speech(train: bool):
    xdg_cache_home = platformdirs.user_cache_path()
    with filelock.FileLock(xdg_cache_home / "SpeechCommands.lock"):
        kws_cache_filename = xdg_cache_home / "SpeechCommands-cache-v1.pth"
        if kws_cache_filename.exists():
            mfccs, labels = torch.load(kws_cache_filename, weights_only=False)
        else:
            original_dataset = torchaudio.datasets.SPEECHCOMMANDS(
                root=xdg_cache_home,
                download=True,
                subset="training" if train else "testing",
            )
            mfccs, labels = preprocess_kws_dataset(original_dataset)
            torch.save((mfccs, labels), kws_cache_filename)
    dataset = TensorDataset(
        torch.from_numpy(np.expand_dims(mfccs.astype(np.float32), axis=1)),
        torch.tensor(labels),
    )
    return dataset


def prepare_data(dataset, train_batch_size):
    print("==> Preparing data..")

    if dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        trainset = datasets.CIFAR10(
            root="./data/cifar10", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=2
        )

        testset = datasets.CIFAR10(
            root="./data/cifar10", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2
        )

    elif dataset == "cifar100":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        trainset = datasets.CIFAR100(
            root="./data/cifar100", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=2
        )

        testset = datasets.CIFAR100(
            root="./data/cifar100", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2
        )

    elif dataset == "har":
        # Inspired by https://blog.csdn.net/bucan804228552/article/details/120143943
        try:
            orig_sys_path = sys.path.copy()
            sys.path.append(
                str(THIS_DIR.parent / "dnn-models" / "deep-learning-HAR" / "utils")
            )
            from utilities import read_data, standardize

            archive_dir = os.path.expanduser("~/.cache/UCI HAR Dataset")

            X_train, train_labels, _ = read_data(archive_dir, split="train")
            _, X_train = standardize(np.random.rand(*X_train.shape), X_train)
            trainset = TensorDataset(
                torch.from_numpy(X_train.astype(np.float32)),
                torch.from_numpy(train_labels - 1),
            )
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=train_batch_size, shuffle=True, num_workers=2
            )

            X_test, test_labels, _ = read_data(archive_dir, split="test")
            _, X_test = standardize(np.random.rand(*X_test.shape), X_test)
            testset = TensorDataset(
                torch.from_numpy(X_test.astype(np.float32)),
                torch.from_numpy(test_labels - 1),
            )
            testloader = torch.utils.data.DataLoader(
                trainset, batch_size=100, shuffle=True, num_workers=2
            )

        finally:
            sys.path = orig_sys_path

    elif dataset == "kws":
        trainloader = torch.utils.data.DataLoader(
            load_data_google_speech(train=True),
            batch_size=train_batch_size,
            shuffle=True,
        )
        testloader = torch.utils.data.DataLoader(
            load_data_google_speech(train=False), batch_size=100, shuffle=True
        )
    else:
        raise RuntimeError(f"Unknown dataset {dataset}")

    return trainloader, testloader


def initialize_model(dataset, arch, num_classes):
    print("==> Initializing model...")
    if dataset in ["cifar10", "cifar100"]:
        model = models.__dict__["cifar_" + arch](num_classes)

    elif dataset in ["har"]:
        model = models.har_cnn()

    elif dataset in ["kws"]:
        model = models.KWS_CNN_S()

    return model


def transform_model(model, arch, action_num):
    if arch.startswith("resnet"):
        from decision import init_decision_basicblock, decision_basicblock_forward

        init_func = init_decision_basicblock
        new_forward = decision_basicblock_forward
        module_type = "BasicBlock"

    elif arch in ("har_cnn", "kws"):
        from decision import init_decision_conv_block, decision_conv_block_forward

        init_func = init_decision_conv_block
        new_forward = decision_conv_block_forward
        module_type = "ConvBlock"

    else:
        from decision import init_decision_convbn, decision_convbn_forward

        init_func = init_decision_convbn
        new_forward = decision_convbn_forward
        module_type = "ConvBNReLU"

    print("==> Transforming model...")

    apply_func(model, module_type, init_func, action_num=action_num)
    apply_func(model, "DecisionHead", collect_params)
    replace_func(model, module_type, new_forward)
    apply_func(model, "DecisionHead", normalize_head_weights)
