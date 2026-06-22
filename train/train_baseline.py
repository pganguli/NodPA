"""
Pre-train a static baseline model (no dynamic pruning) on CIFAR-10, HAR, or KWS.

Saves a checkpoint to logs/pretrained/<dataset>/<arch>/checkpoint.pth.
This checkpoint is required as the starting point for main.py (dynamic training).
Requires a CUDA-capable GPU.

Usage: python train/train_baseline.py --arch resnet10 --dataset cifar10
"""

import torch.nn.functional as F
import torch
import os

import misc

print = misc.logger.info

parser = misc.get_basic_argument_parser(default_wd=1e-4)

args = parser.parse_args()

args.num_classes = 10 if args.dataset == "cifar10" else 100

args.device = "cuda"
torch.backends.cudnn.benchmark = True

args.logdir = "pretrained/%s/%s" % (args.dataset, args.arch)
misc.prepare_logging(args)

trainloader, testloader = misc.prepare_data(args.dataset, args.train_batch_size)

model = misc.initialize_model(args.dataset, args.arch, args.num_classes)

model = model.to(args.device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=misc.learning_rate(args.arch),
    momentum=args.mm,
    weight_decay=args.wd,
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[80, 120], gamma=0.1
)


def train(epoch):
    model.train()
    for i, (data, target) in enumerate(trainloader):
        data = data.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1)[1]
        acc = (pred == target).float().mean()

        if i % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{}]\tLoss: {:.6f}, Accuracy: {:.4f}".format(
                    epoch, i, len(trainloader), loss.item(), acc.item()
                )
            )


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    test_loss /= len(testloader.dataset)
    acc = correct / len(testloader.dataset)
    print("Test set: Average loss: {:.4f}, Accuracy: {:.4f}\n".format(test_loss, acc))
    return acc


for epoch in range(args.epochs):
    scheduler.step()
    train(epoch)
    acc = test()

    torch.save(model.state_dict(), os.path.join(args.logdir, "checkpoint.pth"))

print("Final saved model test accuracy = %.4f" % acc)
