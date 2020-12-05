import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

batch_size = 100
input_size = 784 #mnist的像素为28*28
hidden_size = 500
num_classes = 10 #输出为10个类别分别对应于0-9
learning_rate = 1e-1 #学习率
num_epoches = 5

train_dataset = dsets.MNIST(root='./pymnist',
                            train=True,
                            transform = transforms.ToTensor(),
                            download=True)
test_dataset = dsets.MNIST(root='./pymnist',
                            train=False,
                            transform = transforms.ToTensor(),
                            download=True)

#加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size = batch_size,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size = batch_size,
                                            shuffle=True)


#创建神经网络模型
class Neural_net(nn.Module):
    #初始化函数，接受自定义输入特征的维数，隐含层特征维数以及输出层特征维数
    def __init__(self,input_num,hidden_size,out_put):
        super(Neural_net, self).__init__()
        self.layer1 = nn.Linear(input_num, hidden_size) #从输入到隐藏层的线性处理
        self.layer2 = nn.Linear(hidden_size, out_put)   #从隐藏层到输出层的线性处理

    def forward(self,x):
        out = self.layer1(x)    #从输入层到隐藏层的线性计算
        out = torch.relu(out)   #隐藏层激活
        out = self.layer2(out)  #输出层，注意，输出层直接接Loss
        return out

net = Neural_net(input_size,hidden_size,num_classes)
# print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate)  #使用随机梯度下降
for epoch in range(num_epoches):
    print('current epoch = %d' % epoch)
    for i,(images,labels) in enumerate(train_loader): #利用enumerate取出一个可迭代对象的内容
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels)

        outputs = net(images)   #将数据集传入网络做前向计算
        loss = criterion(outputs,labels)    #计算Loss
        optimizer.zero_grad()   #在做反向传播之前先清除下网络状态
        loss.backward() #反向传播
        optimizer.step()    #更新参数

        if i % 100 == 0:
            print('current loss = %.5f' % loss.item())

print('finished training')

total = 0
correct = 0

for images, labels in test_loader:
    images = Variable(images.view(-1,28*28))
    outputs = net(images)

    _,predicts = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicts == labels).sum()
print('Accuracy = %.2f' % (100 * correct // total))
