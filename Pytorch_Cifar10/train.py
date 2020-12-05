from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch
import Pytorch_Cifar10.net as Net
from Pytorch_Cifar10.data_loader import train_loader

learning_rate = 1e-3
num_epoches = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(Net.net.parameters(),lr=learning_rate)
for epoch in range(num_epoches):
    print('current epoch = %d' % epoch)
    for i, (images,labels) in enumerate(train_loader): #利用enumerate取出一个可迭代对象的内容
        images = Variable(images.view(images.size(0),-1))
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = Net.net(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        if i%100 == 0:
            print('current loss = %.5f' % loss.item())
print('Finished training')