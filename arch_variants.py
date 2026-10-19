import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

NETWORKS = [
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "mobilenet_v2",
    "squeezenet1_1",
    "shufflenet_v2_x1_0",
    "googlenet",
    "ownnet",
]
VARIANTS = ["baseline", "remove", "add"]
NETWORK_VARIANTS = {
    **{network: VARIANTS for network in NETWORKS if network != "ownnet"},
    "ownnet": ["final"],
}


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.mean(dim=(2, 3), keepdim=False)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class SimpleConvBNReLU(nn.Sequential):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


class FeatureHeadModel(nn.Module):
    def __init__(self, backbone, feature_channels, description, use_dense_features=False):
        super().__init__()
        self.backbone = backbone
        self.description = description
        self.head = nn.Linear(feature_channels, 1)
        self.use_dense_features = use_dense_features

    def forward(self, x):
        if self.use_dense_features:
            y = self.backbone.features(x)
        else:
            y = self.backbone(x)
        if y.dim() == 4:
            y = F.adaptive_avg_pool2d(y, (1, 1)).flatten(1)
        elif y.dim() == 3:
            y = y.flatten(1)
        return self.head(y)


class OwnNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.features_edit = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            SEBlock(256, reduction=16),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.features_edit(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.head(x)


def _set_identity_head(model):
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    if hasattr(model, "head"):
        model.head = nn.Identity()
    if hasattr(model, "aux1"):
        model.aux1 = None
    if hasattr(model, "aux2"):
        model.aux2 = None


def _measure_feature_channels(model, device="cpu"):
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224, device=device)
        y = model(x)
        if y.dim() == 4:
            y = F.adaptive_avg_pool2d(y, (1, 1)).flatten(1)
        elif y.dim() == 3:
            y = y.flatten(1)
        return int(y.shape[1])


def _make_resnet50(variant):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if variant == "remove":
        model.layer4 = nn.Identity()
        description = "drop layer4; use layer1-3 output"
    elif variant == "add":
        model.layer4 = nn.Sequential(model.layer4, SEBlock(2048))
        description = "add SE block after layer4"
    else:
        description = "baseline body; new 1-output head"
    _set_identity_head(model)
    feature_channels = _measure_feature_channels(model)
    return FeatureHeadModel(model, feature_channels, description), feature_channels, description


def _make_densenet121(variant):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    if variant == "remove":
        model.features.denseblock4 = nn.Identity()
        model.features.norm5 = nn.Identity()
        description = "drop denseblock4 and final norm5"
    elif variant == "add":
        model.features.norm5 = nn.Sequential(
            model.features.norm5,
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        description = "add Conv-BN-ReLU after final feature map"
    else:
        description = "baseline body; new 1-output head"
    _set_identity_head(model)
    feature_channels = _measure_feature_channels(model)
    if variant == "add":
        return FeatureHeadModel(model, feature_channels, description, use_dense_features=True), feature_channels, description
    return FeatureHeadModel(model, feature_channels, description), feature_channels, description


def _make_efficientnet_b0(variant):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    if variant == "remove":
        model.features = nn.Sequential(*list(model.features.children())[:-1])
        description = "drop the last 1x1 conv in features"
    elif variant == "add":
        model.features = nn.Sequential(
            *list(model.features.children()),
            nn.Conv2d(1280, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )
        description = "add Conv-BN-ReLU to final features"
    else:
        description = "baseline body; new 1-output head"
    _set_identity_head(model)
    feature_channels = _measure_feature_channels(model)
    return FeatureHeadModel(model, feature_channels, description), feature_channels, description


def _make_mobilenet_v2(variant):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    if variant == "remove":
        model.features = nn.Sequential(*list(model.features.children())[:-1])
        description = "drop the last 1x1 conv in features"
    elif variant == "add":
        model.features = nn.Sequential(
            *list(model.features.children()),
            nn.Conv2d(1280, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )
        description = "add Conv-BN-ReLU to final features"
    else:
        description = "baseline body; new 1-output head"
    _set_identity_head(model)
    feature_channels = _measure_feature_channels(model)
    return FeatureHeadModel(model, feature_channels, description), feature_channels, description


def _make_squeezenet1_1(variant):
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    backbone = model.features
    if variant == "remove":
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        description = "drop the last Fire module"
    elif variant == "add":
        backbone = nn.Sequential(
            *list(backbone.children()),
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        description = "add Conv-BN-ReLU before pooling"
    else:
        description = "baseline body; new 1-output head"
    feature_channels = 512
    return FeatureHeadModel(backbone, feature_channels, description), feature_channels, description


def _make_shufflenet_v2_x1_0(variant):
    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    if variant == "remove":
        model.stage4 = nn.Identity()
        model.conv5 = nn.Identity()
        description = "drop the final stage (stage4) and the final projection"
    elif variant == "add":
        model.stage4 = nn.Sequential(model.stage4, SimpleConvBNReLU(464, kernel_size=1))
        description = "add Conv-BN-ReLU after final stage while preserving stage4 depth"
    else:
        description = "baseline body; new 1-output head"
    _set_identity_head(model)
    feature_channels = _measure_feature_channels(model)
    return FeatureHeadModel(model, feature_channels, description), feature_channels, description


def _make_googlenet(variant):
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=True)
    model.aux_logits = False
    model.aux1 = None
    model.aux2 = None
    if variant == "remove":
        model.inception5b = nn.Identity()
        description = "drop inception5b"
    elif variant == "add":
        model.inception5b = nn.Sequential(model.inception5b, SimpleConvBNReLU(1024, kernel_size=1))
        description = "add Conv-BN-ReLU after inception5b"
    else:
        description = "baseline body; new 1-output head"
    _set_identity_head(model)
    feature_channels = _measure_feature_channels(model)
    return FeatureHeadModel(model, feature_channels, description), feature_channels, description


def _make_ownnet(variant):
    if variant != "final":
        raise ValueError(f"Unsupported ownnet variant: {variant}")
    backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    description = "DenseNet121 features + Conv-BN-ReLU(1024->256) + SE(reduction=16) + dropout(0.3)"
    return OwnNet(backbone), 256, description


MODEL_BUILDERS = {
    "resnet50": _make_resnet50,
    "densenet121": _make_densenet121,
    "efficientnet_b0": _make_efficientnet_b0,
    "mobilenet_v2": _make_mobilenet_v2,
    "squeezenet1_1": _make_squeezenet1_1,
    "shufflenet_v2_x1_0": _make_shufflenet_v2_x1_0,
    "googlenet": _make_googlenet,
    "ownnet": _make_ownnet,
}


def build_model(network, variant):
    if network not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported network: {network}")
    if variant not in NETWORK_VARIANTS[network]:
        raise ValueError(f"Unsupported variant: {variant}")
    model, feature_channels, description = MODEL_BUILDERS[network](variant)
    if network == "densenet121" and variant == "add":
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
    return model, feature_channels, description


def validate_variant(network, variant):
    model, feature_channels, description = build_model(network, variant)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1), f"{network}/{variant} produced {tuple(y.shape)} instead of (2, 1)"
    return feature_channels, description


def write_summary_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["network", "variant", "feature_channels", "param_count_total", "param_count_trainable", "description"])
        for row in rows:
            writer.writerow(row)


def main():
    rows = []
    failures = []
    for network in NETWORKS:
        for variant in NETWORK_VARIANTS[network]:
            try:
                model, feature_channels, description = build_model(network, variant)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                x = torch.randn(2, 3, 224, 224)
                model.eval()
                with torch.no_grad():
                    y_eval = model(x)
                if y_eval.shape != (2, 1):
                    raise AssertionError(f"eval mode expected {(2,1)}, got {tuple(y_eval.shape)}")
                model.train()
                y_train = model(x)
                if y_train.shape != (2, 1):
                    raise AssertionError(f"train mode expected {(2,1)}, got {tuple(y_train.shape)}")
                rows.append([network, variant, feature_channels, total_params, trainable_params, description])
            except Exception as exc:  # pragma: no cover
                failures.append((network, variant, str(exc)))
                print(f"FAILED {network}/{variant}: {exc}")

    output_path = Path("runs/arch_comparison/architecture_summary.csv")
    write_summary_csv(rows, output_path)
    print(f"Saved architecture summary to {output_path}")
    print("network,variant,feature_channels,param_count_total,param_count_trainable,description")
    for row in rows:
        print(",".join(str(v) for v in row))
    if failures:
        print("\nFAILURES:")
        for network, variant, message in failures:
            print(f"- {network}/{variant}: {message}")
    return rows, failures


if __name__ == "__main__":
    main()
