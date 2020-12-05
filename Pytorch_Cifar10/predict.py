from torch.autograd import Variable
from Pytorch_Cifar10.data_loader import test_loader
import Pytorch_Cifar10.net as Net
import torch

total = 0
correct = 0

for images,labels in test_loader:
    images = Variable(images.view(images.size(0),-1))
    outputs = Net.net(images)

    _,predicts = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicts == labels).sum()

print('Accuracy = %.2f' % (100 * correct // total))