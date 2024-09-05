import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, base_filters=16, n_blocks=4):
        super(SimpleCNN, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, base_filters, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList()
        
        for i in range(n_blocks):
            block = nn.Sequential(
                nn.Conv2d(base_filters * (2**i), base_filters * (2**(i+1)), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_filters * (2**(i+1))),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.blocks.append(block)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = base_filters * (2**n_blocks)

    def forward(self, x):
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

    def get_embedding_dim(self):
        return self.out_channels
