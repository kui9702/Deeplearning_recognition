import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import Pytorch_VGG16.VGG16 as Net
import os

# 使用torchvision可以很方便地下载Cifar10数据集，而torchvision下载的数据集为[0,1]的
# PILimage格式，我们需要将张量Tensor归一化到[-1,1]

transform = transforms.Compose(
    [transforms.ToTensor(),  # 将PILImage转化为张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将[0,1]归一化到[-1，1]
     ])

trainset = torchvision.datasets.CIFAR10(root=r'cifar-10-batches-py',
                                        train=True,
                                        download=True,
                                        transform=transform)  # 按照上面定义的transform格式转化的数据

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=0)  # 载入训练数据的子任务数

testset = torchvision.datasets.CIFAR10(root=r'cifar-10-batches-py',
                                       train=False,
                                       download=True,
                                       transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=0)  # 载入训练数据的子任务数

cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                   'horse', 'ship', 'truck')

dataiter = iter(trainloader)    #从训练数据中随机取出一些数据
images, labels = dataiter.next()
images.shape #(4L,3L,32L,32L)
torchvision.utils.save_image(images[1],'test.jpg')
cifar10_classes[labels[1]]

# x = torch.randn(2,3,32,32)
# y = net(x)
# print(y.size())
criterion = nn.CrossEntropyLoss()  # 定义损失函数：交叉熵
optimizer = optim.SGD(Net.net.parameters(), lr=0.001, momentum=0.9)  # 定义优化方法:随机梯度下降

# 卷积神经网络开始训练
for epoch in range(5):
    train_loss = 0.0
    for batch_idx, data in enumerate(trainloader, 0):
        # 初始化
        inputs, labels = data  # 获取数据
        optimizer.zero_grad()  # 先将梯度置为0

        # 优化过程
        outputs = Net.net(inputs)  # 将数据输入到网络，得到第一轮网络前向传播的预测结果outputs
        loss = criterion(outputs, labels)  # 预测结果outputs和labels通过之前定义的交叉熵计算损失
        loss.backward()  # 误差反向传播
        optimizer.step()  # 随机梯度下降方向(之前定义的)优化权重

        # 查看网络训练状态
        train_loss += loss.item()
        if batch_idx % 2000 == 1999:  # 没迭代2000个batch打印一次以查看当前网络的收敛状态
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, train_loss // 2000))
            train_loss = 0.0
    print('Saving epoch %d model ...' % (epoch + 1))
    state = {
        'net': Net.net.state_dict(),
        'epoch': epoch + 1
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/cifar10_epoch_%d.ckpt' % (epoch + 1))
print("Finish Training")
