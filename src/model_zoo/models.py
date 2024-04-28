import timm
import torch
import torch.nn as nn


def define_model(
    model_name: str,
    num_classes: int,
):
    model = timm.create_model(model_name, pretrained=True)

    # Get the parameters from the original first layer
    original_first_layer = model.conv_stem
    out_channels = original_first_layer.out_channels
    kernel_size = original_first_layer.kernel_size
    stride = original_first_layer.stride
    padding = original_first_layer.padding
    bias = original_first_layer.bias is not None

    # Create a new Conv2d layer with 1 input channel instead of 3
    model.conv_stem = nn.Conv2d(1, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)

    # Initialize the new conv_stem weights from the original weights
    # by averaging the weights across the input channel dimension
    with torch.no_grad():
        model.conv_stem.weight = nn.Parameter(original_first_layer.weight.mean(dim=1, keepdim=True))

    # Change the last layer to output num_classes
    in_features = model.get_classifier().in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)

    return model