"""Neural network models for federated learning."""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from utils.utils import create_folder, get_logger, save_config

__all__ = [
    "CNN",
    "MLP",
    "ResNet",
    "build_resnet_from_name",
    "get_loss_func",
    "init_model",
    "replace_bn_with_gn",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def init_model(cfg, requires_grad=True):
    """Initialize a CNN, MLP, or ResNet model."""
    model_name = str(cfg.model).lower()
    if model_name == "cnn":
        model = CNN(cfg)
    elif model_name == "mlp":
        model = MLP(cfg)
    elif model_name.startswith("resnet"):
        model, _ = build_resnet_from_name(cfg.model, cfg.num_classes, dataset=cfg.dataset)
    else:
        raise ValueError(f"Invalid model: {cfg.model}. Supported models: cnn, mlp, resnet*")

    model.loss = get_loss_func(cfg)
    cfg.dir_res = create_folder(cfg)
    log = get_logger(cfg.dir_res)
    # log.info(f"Working in {cfg.dir_res}.\n")
    save_config(cfg)

    model.requires_grad_(requires_grad)
    return model.to(cfg.device)


class CNN(nn.Module):
    """Two-convolution CNN for CIFAR-10, CIFAR-100, and FEMNIST."""

    def __init__(self, cfg):
        super().__init__()
        self.ignore_head = bool(getattr(cfg, "ignore_head", False))
        self.loss = get_loss_func(cfg)

        input_shape = getattr(cfg, "input_shape", None)
        if input_shape is None:
            input_shape = {
                "femnist": (1, 28, 28),
                "cifar10": (3, 32, 32),
                "cifar100": (3, 32, 32),
            }.get(str(getattr(cfg, "dataset", "")).lower())
        if input_shape is None:
            raise ValueError("CNN supports only cifar10, cifar100, and femnist.")

        in_channels, height, width = map(int, list(input_shape))
        self.conv1 = nn.Conv2d(in_channels, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # Infer the classifier input width so the same CNN supports FEMNIST and CIFAR.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            feat_dim = self._forward_conv(dummy).flatten(1).size(1)

        self.fc1 = nn.Linear(feat_dim, 384)
        self.fc2 = nn.Linear(384, 192)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(192, cfg.num_classes)

    def _forward_conv(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.flatten(1)
        x = self.relu(self.dropout1(self.fc1(x)))
        x = self.relu(self.dropout2(self.fc2(x)))
        if not self.ignore_head:
            x = self.fc3(x)
        return x


class MLP(nn.Module):
    """Two-hidden-layer MLP baseline for flattened image inputs."""

    def __init__(self, cfg):
        super().__init__()
        self.ignore_head = bool(getattr(cfg, "ignore_head", False))
        self.loss = get_loss_func(cfg)

        input_dim = getattr(cfg, "input_dim", None)
        if input_dim is None:
            input_shape = getattr(cfg, "input_shape", None)
            if input_shape is not None:
                input_dim = int(torch.tensor(list(input_shape)).prod().item())
        if input_dim is None:
            input_dim = {
                "femnist": 28 * 28,
                "cifar10": 3 * 32 * 32,
                "cifar100": 3 * 32 * 32,
            }.get(str(getattr(cfg, "dataset", "")).lower())

        if input_dim is None:
            raise ValueError("MLP needs cfg.input_dim, cfg.input_shape, or a known dataset.")

        hidden_dim = int(getattr(cfg, "hidden_dim", 200))
        self.fc1 = nn.Linear(int(input_dim), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, cfg.num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        if not self.ignore_head:
            x = self.fc3(x)
        return x


def _weights_init(module):
    """Initialize network weights using Kaiming (He) initialization."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(module.weight)


class LambdaLayer(nn.Module):
    """A wrapper module that applies an arbitrary lambda function to input tensors."""

    def __init__(self, lambd):
        """Initialize the LambdaLayer with a callable function."""
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        """Apply the stored lambda function to the input tensor."""
        return self.lambd(x)


class BasicBlock(nn.Module):
    """Basic residual block for ResNet architectures."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        """Initialize a BasicBlock."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                # CIFAR ResNet option A uses strided identity plus channel padding.
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (
                            0,
                            0,
                            0,
                            0,
                            planes // 4,
                            planes // 4,
                        ),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        """Forward pass through the BasicBlock."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """CIFAR-style ResNet architecture for 32x32 images."""

    def __init__(self, block, num_blocks, num_classes=10):
        """Initialize a CIFAR-style ResNet."""
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a stage (layer) consisting of multiple residual blocks."""
        # Only the first block in a stage downsamples; later blocks preserve resolution.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the ResNet."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10):
    """Construct a CIFAR-style ResNet-20 model."""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    """Construct a CIFAR-style ResNet-32 model."""
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes=10):
    """Construct a CIFAR-style ResNet-44 model."""
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(num_classes=10):
    """Construct a CIFAR-style ResNet-56 model."""
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(num_classes=10):
    """Construct a CIFAR-style ResNet-110 model."""
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202(num_classes=10):
    """Construct a CIFAR-style ResNet-1202 model."""
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes)


def replace_bn_with_gn(module):
    """Recursively replace all BatchNorm2d layers with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            # GroupNorm with one group behaves like channel-wise LayerNorm for NCHW tensors.
            setattr(module, name, nn.GroupNorm(num_groups=1, num_channels=num_features))
        else:
            replace_bn_with_gn(child)


def build_resnet_from_name(model_name, num_classes, dataset=None):
    """Build a ResNet instance with BatchNorm layers replaced by GroupNorm."""
    name = str(model_name).lower()
    match = re.match(r"^resnet(\d+)", name)
    if not match:
        raise ValueError(f"Invalid ResNet model name: {model_name}")

    resnet_depth = int(match.group(1))

    import torchvision.models as tv_models

    resnet_factories = {
        18: lambda: tv_models.resnet18(),
        20: lambda: resnet20(num_classes=num_classes),
        32: lambda: resnet32(num_classes=num_classes),
        44: lambda: resnet44(num_classes=num_classes),
        50: lambda: tv_models.resnet50(),
        56: lambda: resnet56(num_classes=num_classes),
        110: lambda: resnet110(num_classes=num_classes),
        1202: lambda: resnet1202(num_classes=num_classes),
    }
    if resnet_depth not in resnet_factories:
        raise ValueError(f"Unsupported ResNet depth: {resnet_depth}")

    model = resnet_factories[resnet_depth]()

    if resnet_depth in {18, 50}:
        ds = str(dataset).lower() if dataset is not None else ""
        if ds in {"cifar10", "cifar100"}:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        elif ds == "femnist":
            model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()

        model.fc = nn.Linear(model.fc.in_features, num_classes)

    replace_bn_with_gn(model)

    return model, resnet_depth


def get_loss_func(cfg):
    """Return the classification loss used for local training."""
    if getattr(cfg, "loss", "cn") != "cn":
        raise ValueError("Only cross-entropy loss ('cn') is supported.")
    return nn.CrossEntropyLoss()
