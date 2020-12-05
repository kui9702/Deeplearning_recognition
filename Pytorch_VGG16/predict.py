import torch
from Pytorch_VGG16.VGG16 import net
from Pytorch_VGG16.train import testloader

checkpoint = torch.load('./checkpoint/cifar_epoch_5.ckpt')
net.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                   'horse', 'ship', 'truck')

dataiter = iter(testloader)
test_images, test_label = dataiter.next()

outputs = net(test_images)  # 查看网络预测效果
_, predicted = torch.max(outputs, 1)  # 获取分数最高的类别

correct = 0
total = 0
with torch.nograd():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 当标记的label种类和预测的种类一致是认为正确，并计数

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct // total))

# 查看每个类的预测结果
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].items()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        cifar10_classes[i], 100 * class_correct[i] // class_total[i]
    ))
