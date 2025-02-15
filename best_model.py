'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import os
import argparse

from models.dropoutresnet import DropoutResNet18


def train(epoch):
    print('\nEpoch: %d' % epoch)
    
    # switch to train mode
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # start training epoch
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        # move to device
        inputs, targets = inputs.to(device), targets.to(device)
       
        # forward pass
        optimizer.zero_grad()
        outputs = net(inputs)
        
        # backpropogate
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # calculate loss and accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc

    # switch to eval mode
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
        
            # start test epoch
            inputs, targets = inputs.to(device), targets.to(device)

            # forward pass
            outputs = net(inputs)
            
            # calculate loss and accuracy
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    
    # Save best epoch checkpoint
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/best_epoch.pth')
        best_acc = acc
    
    # Save model
    if epoch == 199:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/final_best.pth')



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    # Load variabels for training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..')
    
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create dataloaders
    trainset = torchvision.datasets.CIFAR10(
        root='./deep-learning-mini-project-spring-24-nyu/cifar-10-python', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    # Load dataset with tranformation
    testset = torchvision.datasets.CIFAR10(
        root='./deep-learning-mini-project-spring-24-nyu/cifar-10-python', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # List of classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> Building model..')
    # Initialize the model
    net = DropoutResNet18(0.4)
    net = net.to(device)
    
    # If the model is on cuda, use DataParallel
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Load command line arguments
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/best_epoch.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Print out the model parameters
    summary(net, (3,32,32))

    # Running Training and Inference
    for epoch in range(start_epoch, 200):
        train(epoch)
        test(epoch)
        scheduler.step()