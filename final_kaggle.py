import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from models.modifiedresnet import ModifiedResNet18 
from models.dropoutresnet import DropoutResNet18
import torch
import torchvision
import torch.nn.parallel
import torchvision.transforms as transforms
from collections import OrderedDict

# Function for loading Kaggle data
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#  Converting to tensor
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load and preprocess kaggle dataset
cifar10_batch = load_cifar_batch('./deep-learning-mini-project-spring-24-nyu/cifar_test_nolabels.pkl')
images = cifar10_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) 

# Assign device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Model
net = DropoutResNet18(0.4)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/best_epoch_3.pth', map_location=device)

# Run inference on Kaggle Set
net.load_state_dict(checkpoint['net'])
net.eval()
with torch.no_grad():
    for i in range(10000):
        image = images[i]
        input = torch.unsqueeze(test_transform(image),0)
        output = net(input)
        _, predicted = output.max(1)
        print(predicted.item())
