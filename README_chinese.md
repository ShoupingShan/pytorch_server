# flask_pytorch
使用Flask搭建Pytorch服务器                                                               [English Version](./README.md)

## 运行
```sh
python app.py
```
正常启动画面

![flask](./static/img/flask.jpg)

## 功能

1. 使用pickle模拟数据库操作
2. 实时爬取西安资讯
3.  用户反馈
4.  用户上传图片预测结果反馈


## 配置文件
>base_url = 'https://.com/'#部署公网在线服务，否则微信小程序无法显示图片
>
>base_url = 'http://127.0.0.1:5000/' #本地测试
>
>adminGroup = ['User1', 'User2', 'User3'] #管理员群组的微信昵称
# 相关
本项目参考了 [flask_pytorch](https://github.com/WenmuZhou/flask_pytorch).