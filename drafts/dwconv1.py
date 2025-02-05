## https://chatgpt.com/share/67a2e5a4-70d4-800c-ac5b-6ad479c80916

import torch
import torch.nn as nn

# Example input: Batch of 1, 3 input channels, 5x5 spatial size
x = torch.randn(1, 3, 5, 5)

# Channel-wise convolution: 3 input channels, 3 output channels, kernel size 3x3
conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3)

# Output
y = conv(x)

print(y.shape)  # Output: torch.Size([1, 3, 3, 3])

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)  # 1x1 conv for channel mixing

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Example usage
conv = DepthwiseSeparableConv(3, 6)
y = conv(x)
print(y.shape)  # torch.Size([1, 6, 5, 5])
