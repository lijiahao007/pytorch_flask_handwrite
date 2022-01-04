let canvas = document.getElementById("drawing-board");
let ctx = canvas.getContext('2d');

// SignaturePad里面封装了很多canvas的方法，能够方便我们使用
const mnistPad = new SignaturePad(canvas, {
    backgroundColor: '#FFFFFFFF', // 背景是不透明白色的
    penColor: 'rgba(0, 0, 0, 255)', // 笔的颜色是黑色的
    minWidth: 6, // 线的最小宽度
    maxWidth: 8, // 线的最大宽度x
});


// 根据页面大小自动调整canvas的大小
function resizeCanvas() {
    let pageWidth = document.documentElement.clientWidth;
    let pageHeight = document.documentElement.clientHeight;

    // 记录图像，在调整canvas size之后重新写入
    let data = mnistPad.toData()
    canvas.width = pageWidth;
    canvas.height = pageHeight;

    // This library does not listen for canvas changes, so after the canvas is automatically
    // cleared by the browser, SignaturePad#isEmpty might still return false, even though the
    // canvas looks empty, because the internal data of this library wasn't cleared. To make sure
    // that the state of this library is consistent with visual state of the canvas, you
    // have to clear it manually.
    mnistPad.clear()  // 这个是必须的，否则在一开始的时候
    mnistPad.fromData(data)
}

window.onresize = resizeCanvas;
resizeCanvas();


// 功能按钮
let range = document.getElementById("range");
let brush = document.getElementById("brush");
let eraser = document.getElementById("eraser");
let reSetCanvas = document.getElementById("clear");
let undo = document.getElementById("undo");
let save = document.getElementById("save");
let upload = document.getElementById("upload")
let recognition = document.getElementById("recognition")
let resultDiv = document.getElementsByClassName("recognition_result")[0]
let resultText = document.getElementById("recognition_result")
let result = ''

// 颜色按钮，有7个
let aColorBtn = document.getElementsByClassName("color-item");
let activeColor = '#000000FF'; // 初始激活颜色为黑色
getColor() // 设置颜色列表监听器


// 调整笔宽的range input
range.onchange = function () {
    let rangeVal = parseInt(this.value)
    mnistPad.minWidth = rangeVal
    mnistPad.maxWidth = rangeVal + 2
}


// 画笔
brush.onclick = function () {
    this.classList.add("active") // 给brush按钮添加激活类，修改样式
    eraser.classList.remove("active") // erase 按钮删除激活类，修改样式
    // ctx.globalCompositeOperation = 'source-over' // 显示画的内容
    mnistPad.penColor = activeColor
    getColor()
}


// 橡皮檫
eraser.onclick = function () {
    this.classList.add("active")
    brush.classList.remove("active")
    // ctx.globalCompositeOperation = 'destination-out' // 画的内容(源图像)透明，且canvas只显示源图像之外的部分
    mnistPad.penColor = '#FFFFFFFF'
    removeColorBtnListener();
}


// 清空按钮
reSetCanvas.onclick = function () {
    // if (eraser.classList.contains("active")) {
    //     ctx.globalCompositeOperation = 'source-over'
    //     mnistPad.clear()
    //     ctx.globalCompositeOperation = 'destination-out'
    // } else {
    //     mnistPad.clear()
    // }
    mnistPad.clear()
}


// 撤销
undo.onclick = function () {
    let data = mnistPad.toData();
    if (data) {
        data.pop(); // remove the last dot or line
        mnistPad.fromData(data);
    }
}


// 保存(保存图片)
save.onclick = function () {
    if (mnistPad.isEmpty()) {
        WebToast({
            message: "请写入数字",
            time: 2000
        })
        return
    }
    // 保存png图片
    const dataURL = mnistPad.toDataURL('image/png');
    download(dataURL, "signature.png");
}


function download(dataURL, filename) {
    // 下载 dataURL 对应的文件
    var blob = dataURLToBlob(dataURL);
    var url = window.URL.createObjectURL(blob);

    // 创建一个<a>来下载文件
    var a = document.createElement("a");
    a.style = "display: none";
    a.href = url;
    a.download = filename;

    document.body.appendChild(a);
    a.click();

    // 浏览器执行url
    window.URL.revokeObjectURL(url);
}


function dataURLToBlob(dataURL) {
    // 将dataUrl 转化成 Blob二进制数据
    var parts = dataURL.split(';base64,'); // 把文件类型和图片数据分离
    var contentType = parts[0].split(":")[1]; // 获取文件类型
    var raw = window.atob(parts[1]); // 解码 base64编码的图片
    var rawLength = raw.length;
    var uInt8Array = new Uint8Array(rawLength);

    for (var i = 0; i < rawLength; ++i) {
        uInt8Array[i] = raw.charCodeAt(i);
    }

    return new Blob([uInt8Array], {type: contentType});
}


// 颜色选择
function getColor() {
    for (let i = 0; i < aColorBtn.length; i++) {
        // 设置颜色列表的点击监听器
        aColorBtn[i].onclick = function () {
            for (let i = 0; i < aColorBtn.length; i++) {
                aColorBtn[i].classList.remove("active"); // 其他按钮删除激活类
                this.classList.add("active"); // 被选择的元素添加激活类
                activeColor = this.style.backgroundColor;
                mnistPad.penColor = activeColor
            }
        }
    }
}


// 清除颜色选择监听器
function removeColorBtnListener() {
    for (let i = 0; i < aColorBtn.length; i++) {
        aColorBtn[i].onclick = function () {
            WebToast({
                message: "请先选择画笔",
                time: 2000
            })
        }
    }
}


// 上传
upload.onclick = function () {
    if (mnistPad.isEmpty()) {
        WebToast({
            message: "请输入一个数字",
            time: 2000
        })
    } else {
        resultText.innerHTML = "Predicting..."

        let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        const tmpCanvas = document.createElement("canvas")
        tmpCanvas.height = canvas.height
        tmpCanvas.width = canvas.width
        imgData = mnistPad.makeImageDataBlackBackgroundWhiteDigit(imgData)
        tmpCanvas.getContext("2d").putImageData(imgData, 0, 0)
        let dataUrl = tmpCanvas.toDataURL()

        const http = new XMLHttpRequest()
        const formData = new FormData();
        formData.append("data", dataUrl)
        http.open("POST", "http://127.0.0.1:5000/upload/")
        http.send(formData)

        http.onreadystatechange = function () {
            if (http.readyState === 4) {
                if ((http.status >= 200 && http.status < 300) || http.status === 304) {
                    var content = JSON.parse(http.responseText) // 需要json解析字符串
                    console.log(content)
                    resultDiv.classList.add("success")
                    WebToast({
                        time: 2000,
                        message: "识别结果：" + content["code"]
                    })
                    resultText.innerHTML = content["code"]
                    result = content["code"]
                }
            }
        };
    }
}


// 识别
recognition.onclick = function () {
    if (mnistPad.isEmpty()) {
        WebToast({
            message: "请输入一个数字",
            time: 2000
        })
    } else {
        resultText.innerHTML = "Predicting..."

        let rects = mnistPad.getStrokesArea() // 获取每个笔画的范围
        rects = mnistPad.mergeStrokes(rects) // 将有交集的笔画合并
        let dataUrls = []
        for (let i = 0; i < rects.length; i++) { // 将所有笔画区域变成方框
            let rect = mnistPad.makeGravityPointCenter(rects[i])
            let dataUrl = mnistPad.getGridAreaImgUrl(rect)
            dataUrls.push(dataUrl)
        }


        const http = new XMLHttpRequest()
        const formData = new FormData();
        const blob = new Blob([JSON.stringify(dataUrls)], {type: 'application/json'})
        formData.append("data", blob)
        http.open("POST", "http://127.0.0.1:5000/predict/")
        http.send(formData)

        http.onreadystatechange = function () {
            if (http.readyState === 4) {
                if ((http.status >= 200 && http.status < 300) || http.status === 304) {
                    var content = JSON.parse(http.responseText) // 需要json解析字符串
                    console.log(content)
                    resultDiv.classList.add("success")
                    WebToast({
                        time: 2000,
                        message: "识别结果：" + content["code"]
                    })
                    resultText.innerHTML = content["code"]
                    result = content["code"]
                }
            }
        };
    }
}


resultDiv.onclick = function () {
    WebToast({
        time: 1500,
        message: "识别结果：" + result
    })
    resultDiv.classList.remove("success")
}


// 给SignaturePad 添加一个将边框变成正方形的函数
SignaturePad.prototype.change2grid = function (area) {
    const w = area.w,
        h = area.h,
        x = area.x,
        y = area.y;
    let xc = x,
        yc = y,
        wc = w,
        hc = h;
    if (h >= w) {
        xc = x - (h - w) * 0.5;
        wc = h;
    } else {
        yc = y - (w - h) * 0.5;
        hc = w;
    }
    return {
        x: xc,
        y: yc,
        w: wc,
        h: hc
    }
}


SignaturePad.prototype.makeGravityPointCenter = function (area) {
    // 使图像的重心在grid中心
    const imgData = ctx.getImageData(area.x, area.y, area.w, area.h)
    let mM = 0, mX = 0, mY = 0
    for (let i = 0; i < imgData.data.length; i += 4) {
        let t = 1
        if (imgData.data[i] == 255 && imgData.data[i + 1] == 255 && imgData.data[i + 2] == 255) {
            t = 0
        }
        let posY = Math.ceil((i + 1) / 4 / area.w)
        let posX = Math.ceil((i + 1) / 4 - (posY - 1) * area.w)
        mM += t
        mX += posX * t
        mY += posY * t
    }
    let xCenter = mX / mM
    let yCenter = mY / mM

    let width = Math.max(Math.abs(xCenter), Math.abs(xCenter - area.w))
    let height = Math.max(Math.abs(yCenter), Math.abs(yCenter - area.h))

    return {
        x: xCenter - width + area.x,
        y: yCenter - height + area.y,
        w: width * 2,
        h: height * 2
    }
}


SignaturePad.prototype.drawPoint = function (x, y, color) {
    ctx.strokeStyle = color
    ctx.beginPath()
    ctx.arc(x, y, 4, 0, 2 * Math.PI)
    ctx.stroke()
}


SignaturePad.prototype.drawRect = function (area, color = 'black') {
    this._ctx.strokeStyle = color
    this._ctx.strokeRect(area.x, area.y, area.w, area.h)
}


SignaturePad.prototype.makeImageDataBlackBackgroundWhiteDigit = function (imgData) {
    // 让imageData的背景变成黑色，字体变成白色

    // 先把所有背景变为不透明
    for (let i = 0; i < imgData.data.length; i += 4) {
        imgData.data[i + 3] = 255
        if (imgData.data[i] === 255 && imgData.data[i + 1] === 255 && imgData.data[i + 2] === 255) {
            // 白色背景转为黑色
            imgData.data[i] = 0
            imgData.data[i + 1] = 0
            imgData.data[i + 2] = 0

        } else {
            // 其他颜色字体转为白色
            imgData.data[i] = 255
            imgData.data[i + 1] = 255
            imgData.data[i + 2] = 255
        }
    }
    return imgData
}


SignaturePad.prototype.getGridAreaImgUrl = function (area) {
    // 根据area获取图像，放到方形的新canvas中，返回dataUrl

    // 获取图像
    let imgData = ctx.getImageData(area.x, area.y, area.w, area.h)
    // 转换成黑底白字
    imgData = this.makeImageDataBlackBackgroundWhiteDigit(imgData)

    // 创建一个新的canvas对象，将目标图像放到那里，然后在调用toDataURL
    let paddingNum = 20
    let grid = this.change2grid(area)
    const tmpCanvas = document.createElement("canvas")
    tmpCanvas.height = grid.h + 2 * paddingNum
    tmpCanvas.width = grid.w + 2 * paddingNum
    let x = (area.w > area.h ? 0 : (area.h - area.w) / 2) + paddingNum
    let y = (area.w > area.h ? (area.w - area.h) / 2 : 0) + paddingNum
    let ctxTmp = tmpCanvas.getContext("2d")
    ctxTmp.fillStyle = 'rgba(0,0,0,255)' // 设置成黑色不透明
    ctxTmp.fillRect(0, 0, tmpCanvas.width, tmpCanvas.height)
    ctxTmp.putImageData(imgData, x, y)
    const dataUrl = tmpCanvas.toDataURL()
    console.log(dataUrl)
    return dataUrl
}


SignaturePad.prototype.getStrokesArea = function () {
    // 返回每一个笔画的区域。
    var strokes = mnistPad.toData()
    var strokesArea = []
    for (var i = 0; i < strokes.length; i++) {
        // orignChild是每一个笔画
        const orignChild = strokes[i];

        if (orignChild.penColor === '#FFFFFFFF') {
            // erase笔画 不当作是一个笔画
            continue
        }

        let xs = []
        let ys = []
        for (let j = 0; j < orignChild.points.length; j++) {
            xs.push(orignChild.points[j].x);
            ys.push(orignChild.points[j].y);
        }
        let min_x = Math.min.apply(null, xs)
        let min_y = Math.min.apply(null, ys)
        let max_x = Math.max.apply(null, xs)
        let max_y = Math.max.apply(null, ys)
        strokesArea.push({
            x: min_x,
            y: min_y,
            w: max_x - min_x,
            h: max_y - min_y
        })
    }

    return strokesArea
}


SignaturePad.prototype.mergeStrokes = function (strokesArea) {
    // 判断哪些笔画是在一起的
    // 并查集
    let fa = [] // fa[i] 表示第i个元素的根节点是那个（属于哪个集合）
    let rank = [] // 记录每个节点所在的深度
    // init 初始化，使每个笔画各自为一个集合
    for (let i = 0; i < strokesArea.length; i++) {
        fa.push(i)
        rank.push(i)
    }

    // 定义find函数，find(i)返回这个元素所属集合的根节点
    let find = function (x) {
        if (fa[x] === x) {
            return x
        } else {
            // 每个节点直连根节点
            fa[x] = find(fa[x])
            return fa[x]
        }
    }

    // 定义union函数，将两个集合合并，
    let union = function (i, j) {
        let x = find(i)
        let y = find(j)
        if (rank[x] <= rank[y]) {
            fa[x] = y
        } else {
            fa[y] = x
        }
        if (rank[x] === rank[y] && x !== y) {
            rank[y]++
        }
    }

    // 将范围重合的笔画合并
    for (let i = 0; i < strokesArea.length; i++) {
        let x = find(i)
        for (let j = i + 1; j < strokesArea.length; j++) {
            let y = find(j)
            if (x !== y && this.isIntersection(strokesArea[i], strokesArea[j])) {
                // 如果有联系，就合并
                union(i, j)
            }
        }
    }

    // 将所有同一个集合的笔画area合并
    let rootSet = new Set()
    for (let i = 0; i < fa.length; i++) {
        let root = fa[i]
        if (root !== i) {
            // 该点不是根节点，将该点的范围扩展到根节点中
            let area1 = strokesArea[root]
            let area2 = strokesArea[i]
            let min_x = Math.min(area1.x, area2.x)
            let min_y = Math.min(area1.y, area2.y)
            let max_x = Math.max(area1.x + area1.w, area2.x + area2.w)
            let max_y = Math.max(area1.y + area1.h, area2.y + area2.h)
            strokesArea[root] = {
                x: min_x,
                y: min_y,
                w: max_x - min_x,
                h: max_y - min_y
            }
        } else {
            // 如果该店是根节点，就加入rootSet中
            rootSet.add(root)
        }
    }
    let strokesAreaAfterMerge = []
    rootSet.forEach((root) => {
        strokesAreaAfterMerge.push(strokesArea[root])
    })

    // 添加padding
    let paddingNum = 10
    strokesAreaAfterMerge.forEach((area) => {
        area.x = area.x - paddingNum
        area.y = area.y - paddingNum
        area.w = area.w + 2 * paddingNum
        area.h = area.h + 2 * paddingNum
    })

    // ============== 分行 =================
    // 1. fa rank 重新初始化
    fa.splice(0, fa.length)
    rank.splice(0, rank.length)
    for (let i = 0; i < strokesAreaAfterMerge.length; i++) {
        fa.push(i)
        rank.push(i)
    }

    // 2. 划分哪些数字属于哪一行
    let isInOneLine = function (area1, area2) {
        // 判断两个方框是否在同一行
        if (area1.y > area2.y + area2.h || area2.y > area1.y + area1.h) {
            return false
        }
        return true
    }
    for (let i = 0; i < strokesAreaAfterMerge.length; i++) {
        let x = find(i)
        for (let j = i + 1; j < strokesAreaAfterMerge.length; j++) {
            let y = find(j)
            if (x !== y && isInOneLine(strokesAreaAfterMerge[x], strokesAreaAfterMerge[y])) {
                union(i, j)
            }
        }
    }

    // 3. 将属于同一行的划分出来
    let rows = []
    let indexes = new Array(fa.length)
    for (let i = fa.length - 1; i >= 0; i--) {
        let root = find(i)
        if (root === i) {
            rows.push([strokesAreaAfterMerge[i]])
            indexes[i] = rows.length - 1
        } else {
            rows[indexes[root]].push(strokesAreaAfterMerge[i])
        }
    }

    // 4. 将行 按从上到下 排列
    rows.sort((row1, row2) => {
        return row1[0].y - row2[0].y
    })

    // 5. 每行从左到右排列
    for (let i = 0; i < rows.length; i++) {
        rows[i].sort((area1, area2) => {
            return area1.x - area2.x
        })
    }

    // 6. flatten
    let res = []
    for (let i = 0; i < rows.length; i++) {
        for (let j = 0; j < rows[i].length; j++) {
            if (rows[i][j].w * rows[i][j].h > 50) {
                // 需要超过一定范围的笔画才算
                res.push(rows[i][j])
            }
        }
    }

    return res
}


SignaturePad.prototype.isIntersection = function (area1, area2) {
    // 判断两个范围是否有交集
    // 先考虑没有交集的情况， 即area2在area1的上下左右
    let leftTop1 = {x: area1.x, y: area1.y}
    let rightTop1 = {x: area1.x + area1.w, y: area1.y}
    let leftBottom1 = {x: area1.x, y: area1.y + area1.h}

    let leftTop2 = {x: area2.x, y: area2.y}
    let rightTop2 = {x: area2.x + area2.w, y: area2.y}
    let leftBottom2 = {x: area2.x, y: area2.y + area2.h}

    // area2 在 area1的上侧
    let flag = leftBottom2.y < leftTop1.y
    // area2 在 area1 的下册
    flag ||= leftTop2.y > leftBottom1.y
    // area2 在 area1 的左册
    flag ||= rightTop2.x < leftTop1.x
    // area2 在 area1 的右册
    flag ||= leftTop2.x > rightTop1.x
    // 取反返回
    return !flag
}