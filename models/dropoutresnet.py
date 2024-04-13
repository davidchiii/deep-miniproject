import torch.nn as nn
import torch.nn.functional as F
from models.modifiedresnet import ModifiedBasicBlock

class DropoutResNet(nn.Module):
    def __init__(self, block, num_blocks, p, num_classes=10):
        super(DropoutResNet, self).__init__()
        self.in_planes = 64

        # Initialize convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
       
        # Initialize batch normalization layer
        self.bn1 = nn.BatchNorm2d(64)
        
        # Initialize the residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        
        # Initialize Adaptive pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        
        # Initialize Dropout layer
        self.dropout = nn.Dropout(p)
        
        # Initialize Fully connected layer
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # Create a list of strides with the given stride value and 1s for the remaining blocks
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for stride in strides:
            # Append a block to the layers list
            layers.append(block(self.in_planes, planes, stride))
            
            # Update the value of in_planes for the next block
            self.in_planes = planes * block.expansion
        
        # Return the layers as a sequential container
        return nn.Sequential(*layers)

    def forward(self, x):
        # Apply convolutional layer, batch normalization, and ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Pass through the residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Apply adaptive average pooling        
        out = self.avgpool(out)

        # Reshape the tensor
        out = out.view(out.size(0), -1)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Apply fully connected layer
        out = self.linear(out)
        return out
    
    
def DropoutResNet18(p):
    return  DropoutResNet(ModifiedBasicBlock, [4,4,3], p)