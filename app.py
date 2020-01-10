# -*- coding: utf-8 -*-
# @Time    : 2018/4/24 10:04
# @Author  : zhoujun
import flask
from flask_uploads import UploadSet, IMAGES, configure_uploads, ALL
from flask import request, Flask, redirect, url_for, render_template
import time, os, sys,random
from model import Net
import base64, cv2
from PIL import Image
import numpy as np
import config

from werkzeug.utils import secure_filename


app = Flask(__name__, static_url_path='')
app.config.from_object(config)
photos = UploadSet('PHOTO')
configure_uploads(app, photos)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demo', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        img = request.files['img'].filename
        topk = request.form['topk']
        img = secure_filename(img)
        new_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '_' + img
        filename = photos.save(request.files['img'], name=new_name)
        img = photos.path(filename)
        img = np.array(Image.open(img).convert('RGB'))
        data = predict_img(img, top_k=int(topk), search_length=20)
        img_path = photos.url(filename)
        return flask.jsonify({"result":data,"img_path":img_path})
    else:
        img_path = None
        result = []
    return render_template('upload.html', img_path=img_path, result=result)
@app.route('/search', methods=['GET', 'POST'])
def search():
    data = {'state': False}
    if request.method == 'POST':
        pass
    return data

@app.route('/query_all_result', methods=['GET', 'POST'])
def query_all_result():
    data = {'state': False}
    if request.method == 'POST':
        pass
    return data

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    data = {'state': False}
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        t = time.time()
        f_name=str(int(t))+str(random.randint(1000 , 9999))+'.'+f.filename.rsplit('.',1)[1]
        upload_path = os.path.join(basepath, 'data', 'up_images')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        image_path = os.path.join(upload_path, f_name)
        f.save(image_path)
        img = np.array(Image.open(image_path).convert('RGB'))
        try:
            topk = request.form['topk']
        except:
            topk = 5
        data = predict_img(img, top_k=int(topk), search_length=20, file_name=f_name)
        img_path = photos.url('./12.png')
        print(img_path)
    return flask.jsonify(data)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = {'state': False}
    if request.method == 'POST':
        print('image size: ','*'*10)
        img = request.files['image'].read()
        shape = (int(request.form['h']), int(request.form['w']), int(request.form['c']))
        try:
            topk = request.form['topk']
        except:
            topk = 1
        img = np.fromstring(img, np.uint8)
        img = img.reshape(shape)
        print('image size: ','*'*10,img.shape)
        # img = np.array(Image.open(img).convert('RGB'))
        data = predict_img(img, top_k=int(topk), search_length=20)
    return flask.jsonify(data)


def predict_img(img, top_k=1, search_length=20, CAM=True, file_name=None):
    data = dict()
    start = time.time()
    result = model.predict(img, top_k=top_k, search_length=search_length, CAM=CAM)
    cost_time = time.time() - start
    data['predictions'] = list()
    data['matches'] = list()
    # data['predictions']['prob'] = list()
    # data['predictions']['times'] = list()
    prob_result, cos_result = result[0], result[1]
    if CAM:
        cam= result[2]
        basepath = os.path.dirname(__file__)
        cv2.imwrite(os.path.join(basepath, 'uploads', file_name), cam)
        #data['cam'] = 'http://192.168.1.101:80' + '/' + file_name
        data['cam'] = 'http://192.168.31.96:80' + '/' + file_name
    for label, prob in prob_result:
        prob_predict = {'label': label, 'probability': ("%.4f" % prob)}
        data['predictions'].append(prob_predict)
    for label, times in cos_result:
        cos_predict = {'label': label, 'times': ("%d" % times)}
        data['matches'].append(cos_predict)

    md_name = data['matches'][0]['label'].split('/')[-1] + '.md'
    baike = os.path.join(basepath, 'baike', md_name)
    with open(baike, 'r', encoding='UTF-8') as f:
        bk = f.read()
    data['state'] = True
    data['content'] = bk
    data['time'] = cost_time
    data['code'] = 1000
    print('*'*10,'\n')
    print(data)
    print('*'*10,'\n')
    return data


        

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    gpu_id = None
    model_path = './model_best.pth.tar'
    feature_path = './features_train_baidu.pkl'
    # image = os.path.join('./12.png')
    # img = np.array(Image.open(image).convert('RGB'))
    # refer = pr_dic
    model = Net(model_path, feature_path)
    # result = model.predict(img, top_k=3, search_length=20)
    # print('debug')
    app.run()
