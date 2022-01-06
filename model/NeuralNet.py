import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import get_image_test, get_device
import os

DEVICE = get_device()


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


def test(model):
    image_test, label_test = get_image_test(DEVICE, (1, 28 * 28))
    output = model(image_test)
    _, predicted = torch.max(output, 1)
    return predicted == label_test


def load_model(path):
    # 加载模型
    model = NeuralNet().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()  # 固定dropout层和归一化层
    if test(model):
        print("模型加载成功")
    else:
        print("模型加载失败")
    return model


def model_predict(img, model):
    # 传入(1,28,28) 给出预测结果
    if isinstance(model, NeuralNet):
        imgFinal = torch.tensor(img, device=DEVICE, dtype=torch.float32).reshape(28 * 28)
        output = model(imgFinal)
        _, predicted = torch.max(output.data, 0)
        return int(predicted.data)
    return -1


def get_model_files():
    # 返回该模型对应的所有模型文件
    model_dir = os.path.split(__file__)[0]
    model_NeuralNet = os.path.join(model_dir, "model_file", "model_NeuralNet.ckpt")
    return [model_NeuralNet]
