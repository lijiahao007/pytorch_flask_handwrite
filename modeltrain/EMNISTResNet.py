import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU

train_set = datasets.EMNIST(root="datam", split="balanced", train=True, download=True,
                            transform=transforms.ToTensor())  # (n, 28, 28)
test_set = datasets.EMNIST(root="datam", split="balanced", train=False, download=True,
                           transform=transforms.ToTensor())  # (n, 28, 28)

batch_size = 256
learn_rate = 0.001

train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                          shuffle=True, num_workers=4)  # (n, 28, 28) => n/batch_size * (batch_size, 1, 28, 28 )
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                         shuffle=False, num_workers=4)  # (n, 28, 28) => n/batch_size * (batch_size, 1, 28, 28 )

# 5.构建网络模型
class ResBlk(nn.Module):  # 定义Resnet Block模块
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):  # 进入网络前先得知道传入层数和传出层数的设定
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()  # 初始化

        # we add stride support for resbok, which is distinct from tutorials.
        # 根据resnet网络结构构建2个（block）块结构 第一层卷积 卷积核大小3*3,步长为1，边缘加1
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        # 将第一层卷积处理的信息通过BatchNorm2d
        self.bn1 = nn.BatchNorm2d(ch_out)
        # 第二块卷积接收第一块的输出，操作一样
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # 确保输入维度等于输出维度
        self.extra = nn.Sequential()  # 先建一个空的extra
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):  # 定义局部向前传播函数
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))  # 对第一块卷积后的数据再经过relu操作
        out = self.bn2(self.conv2(out))  # 第二块卷积后的数据输出
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out  # 将x传入extra经过2块（block）输出后与原始值进行相加
        out = F.relu(out)  # 调用relu，这里使用F.调用

        return out


class ResNet18(nn.Module):  # 构建resnet18层

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(  # 首先定义一个卷积层
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        # followed 4 blocks 调用4次resnet网络结构，输出都是输入的2倍
        # [b, 32, h, w] => [b, 64, h ,w]
        self.blk1 = ResBlk(32, 64, stride=1)
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk2 = ResBlk(64, 128, stride=1)
        # # [b, 128, h, w] => [b, 256, h, w]
        self.blk3 = ResBlk(128, 256, stride=1)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk4 = ResBlk(256, 256, stride=1)

        self.linearLayer1 = nn.Linear(256, 47)  # 最后是全连接层

    def forward(self, x):  # 定义整个向前传播
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))  # 先经过第一层卷积

        # [b, 32, h, w] => [b, 512, h, w]
        x = self.blk1(x)  # 然后通过4次resnet网络结构
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # F.adaptive_avg_pool2d功能尾巴变为1,1，[b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)  # 平铺一维值
        x = self.linearLayer1(x)  # 全连接层
        return x


model = ResNet18().to(DEVICE)
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(DEVICE)
# 定义模型优化器：输入模型参数，定义初始学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
# 定义学习率调度器：输入包装的模型，定义学习率衰减周期step_size，gamma为衰减的乘法因子
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# tensrBoard可视化
# summaryWriter = SummaryWriter(log_dir="logs")


def train(num_epochs, _model, _device, _train_loader, _optimizer, _lr_scheduler):
    _model.train()  # 设置模型为训练模式, 会对dropout和 batchNorm 有影响
    train_loader_len = _train_loader.__len__()
    for epoch in range(num_epochs):
        # 从迭代器抽取图片和标签
        for i, (images, labels) in enumerate(_train_loader):
            samples = images.to(_device)
            labels = labels.to(_device)
            # 此时样本是一批图片，在CNN的输入中，我们需要将其变为四维，
            # reshape第一个-1 代表自动计算批量图片的数目n
            # 最后reshape得到的结果就是n张图片，每一张图片都是单通道的28 * 28，得到四维张量
            output = _model(samples.reshape(-1, 1, 28, 28))  # (n, 1, 28, 28)
            # 计算损失函数值
            loss = criterion(output, labels)
            # 优化器内部参数梯度必须变为0
            optimizer.zero_grad()
            # 损失值后向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch:{}/{}, step:{}, loss:{:.4f}".format(epoch + 1, num_epochs, i + 1, loss.item()))
                # summaryWriter.add_scalar("train_loss3:", loss.item(), epoch * train_loader_len  + i + 1)

        _lr_scheduler.step()  # 设置学习率调度器,5轮调整一次

def test(_test_loader, _model, _device):
    _model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0

    with torch.no_grad():  # 如果不需要 backward更新梯度，那么就要禁用梯度计算，减少内存和计算资源浪费。
        for data, target in _test_loader:
            data, target = data.to(_device), target.to(_device)
            output = _model(data.reshape(-1, 1, 28, 28))
            loss += criterion(output, target).item()  # 添加损失值
            pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # .cpu()是将参数迁移到cpu上来。

    loss /= len(_test_loader.dataset)

    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))

if __name__ == '__main__':

    train(5, model, DEVICE, train_loader, optimizer, exp_lr_scheduler)
    test(test_loader, model, DEVICE)

    torch.save(model.state_dict(), 'model2.ckpt')