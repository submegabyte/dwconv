## https://chatgpt.com/share/67a2e6e9-b820-800c-aa7a-89cf118ef18f

import torch
import torch.nn as nn

# Define a channel-wise (depthwise) convolution
depthwise_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3)

# Create a dummy input tensor (batch_size=1, channels=3, height=32, width=32)
x = torch.randn(1, 3, 32, 32)

# Apply depthwise convolution
y = depthwise_conv(x)

print(y.shape)  # Output: torch.Size([1, 3, 32, 32])

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 convolution

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Example usage
conv = DepthwiseSeparableConv(3, 16)
x = torch.randn(1, 3, 32, 32)
y = conv(x)
print(y.shape)  # Output: torch.Size([1, 16, 32, 32])
