import torch.nn as nn
import torch.nn.functional as F

class ModifiedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, f=3, k=1):
        super(ModifiedBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=f, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=f,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=k, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class ModifiedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, c=[64,128,256], f=3, k=1):
        super(ModifiedResNet, self).__init__()
        self.in_planes = c[0]

        self.conv1 = nn.Conv2d(3, c[0], kernel_size=f,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c[0])
        self.layer1 = self._make_layer(block, c[0], num_blocks[0], stride=1, f=f, k=k)
        self.layer2 = self._make_layer(block, c[1], num_blocks[1], stride=2, f=f, k=k)
        self.layer3 = self._make_layer(block, c[2], num_blocks[2], stride=2, f=f, k=k)
        self.linear = nn.Linear(c[2]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, f=3, k=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, f=f, k=k))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 5)        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    
def ModifiedResNet18(c, f, k):
    return  ModifiedResNet(ModifiedBasicBlock, [4,4,3], c=c, f=f, k=k)