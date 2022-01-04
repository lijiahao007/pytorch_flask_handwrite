import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # 用于显示图片
import torch.nn.functional as F

# 检测gpu是否可用
device = torch.device('cuda')
# if torch.cuda.is_available():
#     count = torch.cuda.device_count()
#     print("GPU数量：", count)
#     [print(torch.cuda.get_device_name(i)) for i in range(count)]
#     device = torch.device('cuda')

# 读取数据集MNIST
dataset_train = torchvision.datasets.MNIST(root='datam', train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
dataset_test = torchvision.datasets.MNIST(root='datam', train=False, transform=torchvision.transforms.ToTensor(),
                                          download=False)

# 将数据集按批量大小加载到数据集中
# DataLoader: 将数据集分批，以此读取多个数据能够提高GPU的利用率，shuffle(是否打乱数据)。
batch_size = 100
data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size,
                                                shuffle=True)  # 600*100*([[28*28],x])
data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)


num_epochs = 5
learning_rate = 0.001


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(512, 256, bias=True)
        self.dropout2 = nn.Dropout(0.1)
        self.layer3 = nn.Linear(256, 128, bias=True)
        self.dropout3 = nn.Dropout(0.1)
        self.layer4 = nn.Linear(128, 64, bias=True)
        self.dropout4 = nn.Dropout(0.1)
        self.layer5 = nn.Linear(64, 10)

    def forward(self, x):
        # 获取输入一个样本，得到一个输出
        out = F.relu(self.layer1(x))
        out = self.dropout1(out)
        out = F.relu(self.layer2(out))
        out = self.dropout2(out)
        out = F.relu(self.layer3(out))
        out = self.dropout3(out)
        out = F.relu(self.layer4(out))
        out = self.dropout4(out)
        out = self.layer5(out)
        return out


model = NeuralNet().to(device)  # 模型实例
print(model)

# 设置损失函数: 交叉熵损失函数
citerion = nn.CrossEntropyLoss()
# 设置优化方法：随机梯度下苏纳法
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练
total_step = len(data_loader_train)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader_train):
        # 固定每个image 28*28，但是不知道多少行
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # 计算误差
        loss = citerion(outputs, labels)
        # 梯度归零，后面误差反向传播的时候再计算
        optimizer.zero_grad()
        # 误差反馈
        loss.backward()
        # 更新权重
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(
                'Epoch[{}/{}], Step[{}/{}], Loss:{:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 测试模型
model.eval()  # 模型验证
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in data_loader_test:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        # 取每行最大的数作为预测结果.
        # (torch.max 返回最大值和索引值，我们取最大值对应的索引值作为预测结果)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)  # size(0)返回行数
        correct += (predicted == labels).sum().item()
    print("Accuracy of the network on the 10000 test images: {}%".format(100 * correct / total))

# # 保存模型
torch.save(model.state_dict(), 'model.ckpt')




