import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms


batch_size = 100
#MNIST dataset
train_dataset = dsets.CIFAR10(root=r'pycifar',    #选择数据的根目录
                              train= True,                  #选择训练集
                              transform=transforms.ToTensor(),    #转换成Tensor变量
                              download=True
                              )
test_dataset = dsets.CIFAR10(root=r'pycifar',
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)    #将数据打乱

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

