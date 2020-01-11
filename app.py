# -*- coding: utf-8 -*-
# @Time    : 2018/4/24 10:04
# @Author  : zhoujun
import flask
from flask_uploads import UploadSet, IMAGES, configure_uploads, ALL
from flask import request, Flask, redirect, url_for, render_template
import time, os, sys,random
from model import Net
import base64, cv2, json
from model import label_id_name_dict
from PIL import Image
import numpy as np
import config
from utils import Database

from werkzeug.utils import secure_filename
# baseurl = 'http://192.168.31.96:80/'
baseurl = 'http://192.168.1.102:80/'
reverse_dict = dict([(v,k) for (k,v) in label_id_name_dict.items()])
app = Flask(__name__, static_url_path='')
app.config.from_object(config)
photos = UploadSet('PHOTO')
configure_uploads(app, photos)
DB = Database()

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

@app.route('/queryDetail', methods=['GET', 'POST'])
def queryDetail():
    data = dict()
    data['state'] = True
    data['data'] = dict()
    if request.method == 'GET':
        user_name = request.args.get('userName') #SHP
        query_id = request.args.get('id')
        history = DB.query_by_id(user_name, query_id)
        data['data'] =history
        data['code'] = 1000
        data['predictions'] = [{'label':'121', 'probability':0.9}, {'label':'434', 'probability':0.29}]
        data['matches'] = [{'label':'121', 'times':19, 'similar':'http://192.168.1.101:80/source/15787318349849.jpg'}, {'label':'434', 'times':1, 'similar':'http://192.168.1.101:80/source/15787318349849.jpg'}, \
            {'label':'121', 'times':19, 'similar':'http://192.168.1.101:80/source/15786952789964.jpg'}, {'label':'434', 'times':1, 'similar':'http://192.168.1.101:80/source/15786952789964.jpg'}]
    return flask.jsonify(data)


@app.route('/record_user', methods=['GET', 'POST'])
def record_user():
    data = dict()
    data['state'] = True
    data['data'] = dict()
    if request.method == 'GET':
        user_name = request.args.get('userId') #SHP
        pageNum = int(request.args.get('pageNum')) - 1
        pageSize = int(request.args.get('pageSize'))
        total, history = DB.query(user_name, pageNum, pageSize)
        if total == None or history == None:
            data['code'] = -1000
        else:
            data['code'] = 1000
            data['data']['total'] = total
            data['data']['list'] = history

    return flask.jsonify(data)

@app.route('/search', methods=['GET', 'POST'])
# 查询指定名称
def search():
    data = dict()
    data['state'] = True
    if request.method == 'GET':
        search_name = request.args.get('name')
        user_name = request.args.get('userId')
        search_name = search_name.strip()
        if search_name in ['', ' ', '/', '_']:
            data['code'] = -1000
        else:
            data['code'] = 1000
        search_result = []
        for k, v in label_id_name_dict.items():
            if search_name in v or v in search_name:
                search_result.append(k)
        data['data'] = []
        for i in search_result:
            cate_name = label_id_name_dict[i]
            coverImage = baseurl + 'category/' + cate_name.replace('/', '_') + '.jpg'
            dic = {'id':i, 'coverImageUrl':coverImage , 'name':cate_name.split('/')[-1]}
            data['data'].append(dic)
    return flask.jsonify(data)

@app.route('/query_by_category', methods=['GET', 'POST'])
def query_by_category():
    data = dict()
    data['state'] = True
    if request.method == 'GET':
        query_id = request.args.get('templateId')
        cate_name = label_id_name_dict[query_id]
        cate = label_id_name_dict[query_id].split('/')[0]
        name = label_id_name_dict[query_id].split('/')[-1]
        coverImage = baseurl + 'category/' + cate_name.replace('/', '_') + '.jpg'
        data['code'] = 1000
        data['data'] = dict()
        data['data']['coverImageUrl'] = coverImage
        data['data']['name'] = name
        data['data']['cate'] = cate
        md_name = name + '.md'
        basepath = os.path.dirname(__file__)
        baike = os.path.join(basepath, 'baike', md_name)
        with open(baike, 'r', encoding='UTF-8') as f:
            bk = f.read()
        data['data']['content'] = bk
    return flask.jsonify(data)

@app.route('/query_all_result', methods=['GET', 'POST'])
def query_all_result():
    data = dict()
    data['state'] = False
    data['code'] = 1000
    data['data'] = dict()
    data['data']['total'] = len(label_id_name_dict.keys())
    if request.method == 'GET':
        pageNum = int(request.args.get('pageNum')) - 1
        pageSize = int(request.args.get('pageSize'))
        start_index = pageNum * pageSize
        end_index = min(len(label_id_name_dict.keys()), (pageNum + 1) * pageSize)
        data['data']['list'] = []
        for i in range(start_index, end_index):
            coverImage = baseurl + 'category/' + label_id_name_dict[str(i)].replace('/', '_') + '.jpg'
            dic = {'id':str(i), 'coverImageUrl':coverImage, 'name':label_id_name_dict[str(i)], \
                'createTime':time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'content':''}
            data['data']['list'].append(dic)

    return flask.jsonify(data)

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    data = {'state': False}
    if request.method == 'POST':
        user_name = request.values.get('userName')
        location = request.values.get('location')
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        t = time.time()
        f_name=str(int(t))+str(random.randint(1000 , 9999))+'.'+f.filename.rsplit('.',1)[1]
        upload_path = os.path.join(basepath, 'uploads', 'source')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        image_path = os.path.join(upload_path, f_name)
        source_path = baseurl + 'source/' + f_name
        f.save(image_path)
        img = np.array(Image.open(image_path).convert('RGB'))
        try:
            topk = request.form['topk']
        except:
            topk = 5
        data = predict_img(img, top_k=int(topk), search_length=20, file_name=f_name)
        # pred_id_template = data['matches'][0]['label']
        cam = data['cam']
        match_images = data['match_images']
        softmax_prob = data['predictions']
        match_times = data['matches']
        '''
        user_name   #当前查询用户昵称
        source_path  #用户查询图片本地保存地址
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) #用户查询时间戳
        location  #用户查询时所在的地点
        match_times #使用矩阵匹配的结果{'label':'', 'times':''}
        softmax_prob #使用softmax计算的结果{'label':'', 'probability':0.9}
        match_images #图像相似度检索top20图片索引
        cam #类激活图保存地址
        '''
        DB.update(user_name, 
        source_path, 
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
        location, 
        match_times,
        softmax_prob,
        match_images,
        cam)
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
    prob_result, cos_result, match_image_name = result[0], result[1], result[2]
    match_image_name = [baseurl + 'train/' + i for i in match_image_name]
    if CAM:
        cam= result[3]
        basepath = os.path.dirname(__file__)
        cv2.imwrite(os.path.join(basepath, 'uploads', 'cam' ,file_name), cam)
        #data['cam'] = 'http://192.168.1.101:80' + '/' + file_name
        data['cam'] = baseurl + 'cam/' + file_name
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
    data['match_images'] = match_image_name
    data['code'] = 1000
    # print('*'*10,'\n')
    # print(data)
    # print('*'*10,'\n')
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
