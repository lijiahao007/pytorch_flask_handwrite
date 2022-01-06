from flask import Flask, render_template, request, jsonify
from datetime import timedelta
import re
import base64
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageToMnist import transGreyImgToMNIST
import time


# MNIST
# 全连接网络
# from model.NeuralNet import load_model, model_predict, get_model_files

# 卷积网络
# from model.ConvNet import load_model, model_predict, get_model_files

# 残差网络
# from model.ResNet import load_model, model_predict, get_model_files

# EMNIST手写英文、数字识别
# 残差网络
# from model.EMNIST_ConvNet import load_model, model_predict, get_model_files

# 卷积网络
from model.EMNIST_ResNet import load_model, model_predict, get_model_files


model_file = get_model_files()[0]
print("load_model from ", model_file)
model = load_model(model_file)  # 加载模型

app = Flask(__name__,
            template_folder='templates',
            static_folder='static',
            static_url_path='/static')

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/')
def canvas():  # put application's code here
    return render_template('canvas.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # 获取图像，使用模型预测， 返回预测结果
    start = time.time()
    datas = request.get_data()

    # 将图片base64编码部分摘出来
    imgBytes = re.search(b'[[](.*?)[]]', datas).group(1)
    imgBytesArr = imgBytes.split(b',')
    [imgBytesArr.pop(i) for i in range(len(imgBytesArr) - 2, -1, -2)]
    imgBytesArr = [imgBytesArr[i].removesuffix(b'\"') for i in range(len(imgBytesArr))]

    # 逐个图像进行处理
    res = []
    for i, imgByte in enumerate(imgBytesArr):
        # 将图像写入
        with open("imgs/output{}0.png".format(i), "wb") as output:
            output.write(base64.decodebytes(imgByte))

        # 使用opencv来处理图像
        img_bytes = base64.b64decode(imgByte)  # bytes -> bytes
        img_array = np.frombuffer(img_bytes, np.uint8)  # bytes->nparray
        img_grey = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # 灰度图模式读取
        plt.imsave("imgs/output{}1.png".format(i), img_grey)
        imgTran = cv2.resize(img_grey, (28, 28)) / 255  # 转换大小, (因为训练集中的灰度是使用小数的，所以这里也要除以255转化转换成小数)
        # 处理图像，让黑色部分更黑，白色部分更白
        imgTran[imgTran == imgTran.max()] += 0.1
        imgTran[imgTran > 1] = 1
        imgTran[imgTran == imgTran.min()] = 0
        plt.imsave("imgs/output{}0.png".format(i), imgTran)

        # 使用模型进行预测
        res.append(model_predict(imgTran, model))

    # 将预测结果返回
    code = ''
    for val in res:
        code += str(val)

    response = {'code': code}
    print(response, end=' ')
    end = time.time()
    print("predict times:{}s".format(end - start))
    return jsonify(response), 200, [("Access-Control-Allow-Origin", "*")]


@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    # 读取图片
    start = time.time()
    data = request.get_data()
    imgBytes = re.search(b'base64,(.*)', data).group(1)
    with open("imgs1/outputAllCanvas.png", "wb") as output:
        output.write(base64.decodebytes(imgBytes))

    img_bytes = base64.b64decode(imgBytes)  # bytes -> bytes
    img_array = np.frombuffer(img_bytes, np.uint8)  # bytes->nparray
    img_grey = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # 灰度图模式读取
    plt.imsave("imgs1/outputAllCanvasGray.png", img_grey)
    mnistImgs = transGreyImgToMNIST(img_grey)  # mnistImgs ：(n, 28, 28) n是识别到图片的数量
    code = ''
    for mnistImg in mnistImgs:
        # mnistImg: (28, 28)
        res = model_predict(mnistImg, model)
        code += str(res)

    response = {'code': code}
    print(response, end=' ')
    end = time.time()
    print("upload time:{}s".format(end - start))
    return jsonify(response), 200, [("Access-Control-Allow-Origin", "*")]


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
