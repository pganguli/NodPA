# dynamic-pruning

This is a customized tool for dynamic pruning of various models.
The tool is based on https://github.com/frankwang345/dynamic-pruning, which is in turn codes for [Dynamic Network Pruning with Interpretable Layerwise Channel Selection](https://aaai.org/ojs/index.php/AAAI/article/view/6098)

## Scripts

- `train_baseline.py`: pretrain baseline models (no dynamic pruning)
- `main.py`: train dynamic models with decision units
- `export.py`: export a trained dynamic model to ONNX
- `finetune.py`: fine-tune models after dynamic pruning training

All evaluation networks were trained from scratch for 10 epochs using the SGD optimizer, with a learning rate of 0.1 for ResNet and HAR, and 0.01 for KWS.

## Prerequisites

Training requires a CUDA-capable GPU. Install the training dependencies (separate from the main `requirements-base.txt`):

```bash
pip install -r train/requirements.txt
```

## Training Workflow

**Step 1 — Pre-train baseline model:**

```bash
python train/train_baseline.py --arch resnet10 --dataset cifar10
python train/train_baseline.py --arch har_cnn  --dataset har
python train/train_baseline.py --arch kws      --dataset kws
```

**Step 2 — Train dynamic model with pruning decision units:**

```bash
python train/main.py --arch resnet10 --dataset cifar10
python train/main.py --arch har_cnn  --dataset har
python train/main.py --arch kws      --dataset kws
```

**Step 3 — Export to ONNX (run from the `train/` directory):**

```bash
python export.py --arch resnet10 --dataset cifar10
python export.py --arch har_cnn  --dataset har
python export.py --arch kws      --dataset kws
```

Outputs: `dnn-models/<dataset>_<arch>-single.onnx` and `dnn-models/<dataset>_<arch>-batched.onnx`.

**Optional — Fine-tune after dynamic training:**

```bash
python train/finetune.py --arch resnet10 --dataset cifar10
```

## Key Arguments

- `--sparsity_level` (default 0.5): target fraction of channels to skip per inference
- `--epochs` (default 10): number of training epochs
- `--pruning_threshold` (default 0.5): Q15 gate threshold for channel selection
