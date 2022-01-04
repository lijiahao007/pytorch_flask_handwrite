var WebToast = function (params) {
    /*设置信息框停留的默认时间*/
    var time = params.time;
    if (time == undefined || time == '') {
        time = 1500;
    }
    var el = document.createElement("div");
    el.setAttribute("class", "web-toast");
    el.innerHTML = params.message;
    document.body.appendChild(el);
    el.classList.add("fadeIn");
    setTimeout(function () {
        el.classList.remove("fadeIn");
        el.classList.add("fadeOut");
        /*监听动画结束，移除提示信息元素*/
        el.addEventListener("animationend", function () {
            document.body.removeChild(el);
        });
        el.addEventListener("webkitAnimationEnd", function () {
            document.body.removeChild(el);
        });

    }, time);
}