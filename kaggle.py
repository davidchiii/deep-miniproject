import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
from models.modifiedresnet import ModifiedResNet18 

# load model and convert model to cpu:
import torch
from torchvision import models
import torch.nn.parallel

from collections import OrderedDict


def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


cifar10_batch = load_cifar_batch('./deep-learning-mini-project-spring-24-nyu/cifar_test_nolabels.pkl')
images = torch.from_numpy(cifar10_batch[b'data'])
images = images.reshape((10000, 3, 32, 32))
images = images.float()

# cifar10_dir = './deep-learning-mini-project-spring-24-nyu/cifar-10-python/cifar-10-batches-py'
# cifar10_batch = load_cifar_batch(os.path.join(cifar10_dir, 'test_batch'))
# images = torch.from_numpy(cifar10_batch[b'data'])
# labels = cifar10_batch[b'labels']
# images = images.reshape((10000, 3, 32, 32))
# images = images.float()

images = torch.split(images, 1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = ModifiedResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/baseline_ckpt.pth')
net.load_state_dict(checkpoint['net'])
# print(checkpoint['acc'])
outputs = []
for batch in images:
    output = net(batch)
    _, predicted = output.max(1)
    outputs.append(predicted.item())
    # print(predicted.item())
    torch.cuda.empty_cache()
    
# correct = sum(1 for x, y in zip(outputs, labels) if x == y)
# print(correct / 10000)