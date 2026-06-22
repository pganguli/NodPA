"""
Train a model with dynamic channel-pruning decision units.

Loads a pre-trained checkpoint from logs/pretrained/ and jointly trains the
pruning decision heads and the backbone network with a sparsity-regularisation term.
Requires a CUDA-capable GPU.

Usage: python train/main.py --arch resnet10 --dataset cifar10 [--sparsity_level 0.5]
"""

import torch.nn.functional as F
import numpy as np
import torch
import os

from decision import (
    default_graph,
    apply_func,
    set_deterministic_value,
    normalize_head_weights,
    set_pruning_threshold,
)
import misc

np.set_printoptions(precision=2, linewidth=160)
print = misc.logger.info

parser = misc.get_basic_argument_parser(default_wd=1e-9)
parser.add_argument("--lambd", default=1.0, type=float)

args = parser.parse_args()

args.num_classes = 10 if args.dataset == "cifar10" else 100

args.device = "cuda"
torch.backends.cudnn.benchmark = True

args.logdir = "decision-%d/%s-%s/sparsity-%.2f" % (
    misc.action_num(args.arch),
    args.dataset,
    args.arch,
    args.sparsity_level,
)
misc.prepare_logging(args)

trainloader, testloader = misc.prepare_data(args.dataset, args.train_batch_size)

model = misc.initialize_model(args.dataset, args.arch, args.num_classes)

model_params = []
for p in model.parameters():
    model_params.append(p)

print("==> Loading pretrained model...")
model.load_state_dict(
    torch.load("logs/pretrained/%s/%s/checkpoint.pth" % (args.dataset, args.arch))
)

misc.transform_model(model, args.arch, misc.action_num(args.arch))

model = model.to(args.device)

head_params = default_graph.get_tensor_list("head_params")
gate_params = default_graph.get_tensor_list("gate_params")

optimizer_gate = torch.optim.Adam(
    head_params + gate_params, lr=misc.learning_rate(args.arch)
)
optimizer_model = torch.optim.SGD(
    model_params,
    lr=misc.learning_rate(args.arch),
    momentum=args.mm,
    weight_decay=args.wd,
)


def train(epoch):
    model.train()
    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=False)
    apply_func(
        model,
        "DecisionHead",
        set_pruning_threshold,
        pruning_threshold=args.pruning_threshold,
    )
    for i, (data, target) in enumerate(trainloader):
        default_graph.clear_all_tensors()

        data = data.to(args.device)
        target = target.to(args.device)

        optimizer_gate.zero_grad()
        output = model(data)
        loss_ce = F.cross_entropy(output, target)
        selected_channels = default_graph.get_tensor_list("selected_channels")
        loss_reg = (
            args.lambd
            * (torch.cat(selected_channels, dim=1).abs().mean() - args.sparsity_level)
            ** 2
        )
        loss = loss_ce + loss_reg

        loss.backward()
        optimizer_gate.step()

        for p in gate_params:
            p.data.clamp_(0, 5)

        apply_func(model, "DecisionHead", normalize_head_weights)

        optimizer_model.zero_grad()
        output = model(data)
        loss_model = F.cross_entropy(output, target)
        loss_model.backward()
        optimizer_model.step()

        if i % args.log_interval == 0:
            concat_channels = torch.cat(selected_channels, dim=1)
            sparsity = (concat_channels != 0).float().mean()
            mean_gate = concat_channels.mean()
            acc = (output.max(1)[1] == target).float().mean()

            print(
                "Train Epoch: %d [%d/%d]\tLoss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, "
                "Sparsity: %.4f, Mean gate: %.4f, Accuracy: %.4f"
                % (
                    epoch,
                    i,
                    len(trainloader),
                    loss.item(),
                    loss_ce.item(),
                    loss_reg.item(),
                    sparsity.item(),
                    mean_gate.item(),
                    acc.item(),
                )
            )


def test():
    model.eval()
    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)
    apply_func(
        model,
        "DecisionHead",
        set_pruning_threshold,
        pruning_threshold=args.pruning_threshold,
    )
    test_loss_ce = []
    test_loss_reg = []
    test_sparsity = []
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            default_graph.clear_all_tensors()

            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            selected_channels = default_graph.get_tensor_list("selected_channels")
            concat_channels = torch.cat(selected_channels, dim=1)

            test_loss_ce.append(F.cross_entropy(output, target).item())
            test_loss_reg.append(args.lambd * concat_channels.abs().sum().item())
            test_sparsity.append((concat_channels != 0).float().mean().item())

            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    actions = torch.stack(default_graph.get_tensor_list("sampled_actions")).permute(
        1, 0
    )
    acc = correct / len(testloader.dataset)
    print(
        "Test set: Loss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, "
        "Sparsity: %.4f, Accuracy: %.4f"
        % (
            np.mean(test_loss_ce) + np.mean(test_loss_reg),
            np.mean(test_loss_ce),
            np.mean(test_loss_reg),
            np.mean(test_sparsity),
            acc,
        )
    )
    print("   First 10 sampled actions: \n" + str(actions[:10].cpu().numpy()))
    print("   First 10 targets: " + str(target[:10].cpu().numpy()) + "\n")
    return acc, np.mean(test_sparsity)


def save_checkpoint(state, filepath):
    torch.save(state, os.path.join(filepath, "checkpoint.pth.tar"))


for epoch in range(args.epochs):
    train(epoch)
    acc, sparsity = test()

    print(
        "Save best checkpoint @ Epoch %d, Accuracy = %.4f, Sparsity = %.4f\n"
        % (epoch, acc, sparsity)
    )

    save_checkpoint(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
        },
        filepath=args.logdir,
    )
