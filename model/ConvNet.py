import torch
import torchvision
import torch.nn as nn
from constant import get_image_test, get_device
import os

DEVICE = get_device()


# 设计模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 提取特征层
        self.features = nn.Sequential(
            # 卷积层
            # 输入图像通道为 1，因为我们使用的是黑白图，单通道的
            # 输出通道为32（代表使用32个卷积核），一个卷积核产生一个单通道的特征图
            # 卷积核kernel_size的尺寸为 3 * 3，stride 代表每次卷积核的移动像素个数为1
            # padding 填充，为1代表在图像长宽都多了两个像素
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            # ((28-3+2*1)/1)+1=28  28*28*1  》 28*28*32

            # 批量归一化，跟上一层的out_channels大小相等，以下的通道规律也是必须要对应好的
            nn.BatchNorm2d(num_features=32),  # 28*28*32  》 28*28*32

            # 激活函数，inplace=true代表直接进行运算
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # ((28-3+2*1)/1)+1=28     28*28*32 》 28*28*32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 最大池化层
            # kernel_size 为2 * 2的滑动窗口
            # stride为2，表示每次滑动距离为2个像素
            # 经过这一步，图像的大小变为1/4，即 28 * 28 -》 7 * 7
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 分类层
        self.classifier = nn.Sequential(
            # Dropout层
            # p = 0.5 代表该层的每个权重有0.5的可能性为0
            nn.Dropout(p=0.5),
            # 这里是通道数64 * 图像大小7 * 7，然后输入到512个神经元中
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

    # 前向传递函数
    def forward(self, x):
        # 经过特征提取层
        x = self.features(x)
        # 输出结果必须展平成一维向量
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def test(model):
    image_test, label_test = get_image_test(DEVICE, (1, 1, 28, 28))
    output = model(image_test)
    predicted = output.argmax(1)
    return predicted == label_test


def load_model(path):
    model = ConvNet().to(DEVICE)
    model.load_state_dict(torch.load(path))
    model.eval()
    if test(model):
        print("模型加载成功")
    else:
        print("模型加载失败")
    return model


def model_predict(img, model):
    if isinstance(model, ConvNet):
        imgFinal = torch.tensor(img, device=DEVICE, dtype=torch.float32).reshape(1, 1, 28, 28)
        output = model(imgFinal)
        pred = output.argmax(dim=1)
        return int(pred.data)
    return -1


def get_model_files():
    # 返回该模型对应的所有模型文件
    model_dir = os.path.split(__file__)[0]
    model_ConvNet = os.path.join(model_dir, "model_file", "model_ConvNet.ckpt")
    return [model_ConvNet]
