from torch.autograd import Variable
import torch.nn as nn
import torch


input_size = 3072
hidden_size = 500
hidden_size2 = 200
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

#定义两层神经网络
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,hidden_size2,num_calsses):
        super(Net,self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Linear(hidden_size,hidden_size2)
        self.layer3 = nn.Linear(hidden_size2,num_calsses)

    def forward(self,x):
        out = torch.relu(self.layer1(x))
        out = torch.relu(self.layer2(out))
        out = self.layer3(out)
        return out

net = Net(input_size,hidden_size,hidden_size2,num_classes)