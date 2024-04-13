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

#  Color augmentation if needed
transform_test = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# Load and preprocess kaggle dataset
cifar10_batch = load_cifar_batch('./deep-learning-mini-project-spring-24-nyu/cifar_test_nolabels.pkl')
images = torch.from_numpy(cifar10_batch[b'data'])
images = images.reshape((10000, 3, 32, 32))
images = images.float()
# images = transform_test(images)

# Assign device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Model
net = DropoutResNet18(0.4)
# net = ModifiedResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/best_epoch_3.pth', map_location=device)
# print(checkpoint['acc'])
# checkpoint = torch.load('./checkpoint/ckpt_dropout_0.4.pth', map_location=device)
# checkpoint = torch.load('./checkpoint/pool.pth', map_location=device)


# Run inference on Kaggle Set
net.load_state_dict(checkpoint['net'])
with torch.no_grad():
    images = images.to(device)
    output = net(images)
    _, predicted = output.max(1)
    predicted = predicted.tolist()
    for i in predicted:
        print(i)
    