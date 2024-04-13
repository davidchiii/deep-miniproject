from models.modifiedresnet import ModifiedBasicBlock

import torch.nn as nn
import torch.nn.functional as F

class DropoutResNet(nn.Module):
    def __init__(self, block, num_blocks, p, num_classes=10):
        super(DropoutResNet, self).__init__()
        self.in_planes = 64

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout layer
        self.dropout = nn.Dropout(p)

        # Fully connected layer
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Average pooling layer
        out = self.avgpool(out)

        # Flatten the output
        out = out.view(out.size(0), -1)

        # Dropout layer
        out = self.dropout(out)

        # Fully connected layer
        out = self.linear(out)
        return out
    
    
def DropoutResNet18(p):
    return  DropoutResNet(ModifiedBasicBlock, [4,4,3], p)