import cv2
import matplotlib.pyplot as plt
import numpy as np

# 把整张图片转换成多张MNIST格式的图片
def transGreyImgToMNIST(img_grey, size=(28, 28)):
    # 二值化 (因为cv2.findContours需要传入一个二值化的图案)
    img = accessBinary(img_grey)

    # 检查图像边界，contours是寻找轮廓的图像
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    borders = []
    minArea = 50
    for contour in contours:
        # 将边缘拟合成一个边框
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > minArea:
            border = [(x, y), (x + w, y + h)]
            borders.append(border)

    # 将border排序，（从左到右）
    borders = sortBorder(borders)

    # (n, 28, 28) 的 imgData, 代表边框中的n个数字
    imgData = np.zeros((len(borders), size[0], size[0]), dtype='uint8')
    for i, border in enumerate(borders):
        # 把目标区域抠出来
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        # 根据最大边缘拓展像素
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2

        # 设置补成正方形
        paddingNum = 30
        if borderImg.shape[0] < borderImg.shape[1]:  # 宽大于高
            # 该方法能够位图像设置边框
            borderImg = cv2.copyMakeBorder(borderImg, paddingNum + extendPiexl,
                                           paddingNum + extendPiexl, paddingNum,
                                           paddingNum, cv2.BORDER_CONSTANT)
        else:
            borderImg = cv2.copyMakeBorder(borderImg, paddingNum, paddingNum,
                                           extendPiexl + paddingNum,
                                           extendPiexl + paddingNum,
                                           cv2.BORDER_CONSTANT)
        plt.imsave("imgs1/output{}0.png".format(i),borderImg)
        targetImg = cv2.resize(borderImg, size)  # img => (28, 28)
        targetImg = targetImg / 255  # 将灰度映射到(0~1)
        imgData[i] = targetImg
        plt.imsave("imgs1/output{}1.png".format(i), targetImg)

    return imgData  # (n, 28, 28)


def init_find_union(n):
    fa = []
    rank = []
    for i in range(n):
        fa.append(i)
        rank.append(i)
    return fa, rank


def find(x, fa):
    if fa[x] == x:
        return x
    else:
        # 每个节点直连根节点
        fa[x] = find(fa[x], fa)
        return fa[x]


def union(i, j, fa, rank):
    x = find(i, fa)
    y = find(j, fa)
    if rank[x] <= rank[y]:
        fa[x] = y
    else:
        fa[y] = x
    if rank[x] == rank[y] and x != y:
        rank[y] += 1


def isInSameLine(border1, border2):
    if border1[0][1] > border2[1][1] or border1[1][1] < border2[0][1]:
        return False
    return True


def sortBorder(borders):
    n = len(borders)
    fa, rank = init_find_union(n)
    for i in range(n):
        x = find(i, fa)
        for j in range(i + 1, n):
            y = find(j, fa)
            if x != y and isInSameLine(borders[x], borders[y]):
                union(i, j, fa, rank)

    rows = []
    indexes = [0] * n
    for i in range(n - 1, -1, -1):
        root = find(i, fa)
        if root == i:
            rows.append([borders[i]])
            indexes[i] = len(rows) - 1
        else:
            rows[indexes[root]].append(borders[i])

    rows.sort(key=lambda rowTmp: rowTmp[0][0][1], reverse=False)
    for i in range(len(rows)):
        rows[i].sort(key=lambda borderTmp: borderTmp[0][0], reverse=False)

    res = []
    for row in rows:
        for border in row:
            res.append(border)

    return res


# 显示结果及边框
def showResults(img, borders, results=None):
    # 绘制
    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (255, 255, 255))  # 在图像上画图
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
    cv2.imshow('test', img)
    cv2.waitKey(0)  # 使用imshow显示图像


# 反相灰度图，将黑白阈值颠倒
def accessPiexl(img):
    # 这个遍历要3s，太慢了，所以就不用这个了，
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
        for j in range(width):
            if img[i][j] == 255:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img


# 将图像膨胀一下
def accessBinary(img, threshold=128):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img
