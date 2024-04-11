import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
from models.modifiedresnet import ModifiedResNet18 

# load model and convert model to cpu:
import torch
import torchvision
from torchvision import models
import torch.nn.parallel
import torchvision.transforms as transforms


from collections import OrderedDict

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

transform_test = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_batch = load_cifar_batch('./deep-learning-mini-project-spring-24-nyu/cifar_test_nolabels.pkl')

# cifar10_dir = './deep-learning-mini-project-spring-24-nyu/cifar-10-python/cifar-10-batches-py'
# cifar10_batch = load_cifar_batch(os.path.join(cifar10_dir, 'test_batch'))

images = torch.from_numpy(cifar10_batch[b'data'])
images = images.reshape((10000, 3, 32, 32))
images = images.float()
images = transform_test(images)

# labels = torch.FloatTensor(cifar10_batch[b'labels'])
# labels = torch.split(labels, 100)


# images = torch.split(images, 10000)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = ModifiedResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
'''
transform_test = transforms.Compose([
    transforms.ToTensor(),
])   
testset = torchvision.datasets.CIFAR10(
    root='./deep-learning-mini-project-spring-24-nyu/cifar-10-python', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
'''
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
# checkpoint = torch.load('./checkpoint/ckpt_f_None.pth', map_location=device)
checkpoint = torch.load('./checkpoint/baseline_ckpt.pth', map_location=device)

net.load_state_dict(checkpoint['net'])

outputs = []
correct = 0


'''
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(100.*correct/total)
        # print(predicted.tolist())

'''
# net.eval()
with torch.no_grad():
    images = images.to(device)
    output = net(images)
    _, predicted = output.max(1)
    predicted = predicted.tolist()
    for i in predicted:
        print(i)
    '''
    for i in range(len(images)):
        input = images[i]
        input = input.to(device)
        output = net(input)
        _, predicted = output.max(1)

        # target = labels[i]
        # target = target.to(device)
        # correct += predicted.eq(target).sum().item()
        
        predicted = predicted.tolist()
        for i in predicted:
            outputs.append(i)

        # torch.cuda.empty_cache()
    for c in outputs:
        print(c)
    '''
# print(correct / 10000)
