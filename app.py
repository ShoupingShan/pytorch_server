# -*- coding: utf-8 -*-
# @Time    : 2019/1/12 22：01
# @Author  : shp
import flask
from flask_uploads import UploadSet, IMAGES, configure_uploads, ALL
from flask import request, Flask, redirect, url_for, render_template
import time, os, sys,random
from model import Net
import base64, cv2, json, shutil
from model import label_id_name_dict
from PIL import Image
import numpy as np
import config
from utils import Database
from utils import Xian
from utils import Guest
from utils import Feedback
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
adminGroup = config.adminGroup
baseurl = config.base_url

reverse_dict = dict([(v,k) for (k,v) in label_id_name_dict.items()])
app = Flask(__name__, static_folder='./uploads')
app.config.from_object(config)
photos = UploadSet('PHOTO')
configure_uploads(app, photos)
DB = Database()
NEWS = Xian()
FD = Feedback()
GST = Guest()

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

@app.route('/deleteItem', methods=['GET', 'POST'])
def deleteItem():
    data = dict()
    data['state'] = True
    data['data'] = dict()
    if request.method == 'GET':
        user_name = request.args.get('userId') #SHP
        id = int(request.args.get('deleteId'))
        try:
            DB.deleteItem(user_name, id)
            data['code'] = 1000
        except:
            data['code'] = - 1000
        GST.update(user_name, time.time(), 'Delete')
        GST.auto_save()
    return flask.jsonify(data)

@app.route('/get_chart', methods=['GET', 'POST'])
def get_chart():
    data = dict()
    data['state'] = True
    data['data'] = dict()
    if request.method == 'GET':
        user_name = request.args.get('userId')
        mode = request.args.get('mode')
        try:
            with open('./train_log.txt', 'r', encoding='UTF-8') as f:
                history = f.readlines()
            tags = history[0].strip().split('\t')
            data['lr'] = []
            data['train_loss'] = []
            data['eval_loss'] = []
            data['train_acc'] = []
            data['eval_acc'] = []
            data['epoch'] = []
            for index, line in enumerate(history[1:]):
                item = line.strip().split('\t')
                data['lr'].append(float(item[0]))
                data['train_loss'].append(float(item[1]))
                data['eval_loss'].append(float(item[2]))
                data['train_acc'].append(float(item[3]))
                data['eval_acc'].append(float(item[4]))
                data['epoch'].append(str(index + 1))
            data['code'] = 1000
        except:
            data['code'] = -1000
        # print('Debug', data)
    return flask.jsonify(data)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    data = dict()
    data['state'] = True
    data['data'] = dict()
    if request.method == 'GET':
        user_name = request.args.get('userId')
        user_feedback = request.args.get('user_feedback')
        feedback_image_name = os.path.split(request.args.get('name'))[-1]
        save_feedback_folder = os.path.join('./uploads/feedback/' + user_name)
        if not os.path.exists(save_feedback_folder):
            os.makedirs(save_feedback_folder)
        image_path_save = os.path.join(save_feedback_folder, feedback_image_name)
        image_path_source = os.path.join('./uploads/source/' + user_name + '/' + feedback_image_name)
        image_path = 'uploads/feedback/' + user_name + '/' + feedback_image_name
        shutil.copy(image_path_source, image_path_save)
        predictions = request.args.get('prediction')
        current_time = time.time()
        if predictions == 'undefined':
            data['code'] = -1000
        else:            
            try:
                FD.updata(user_name, feedback_image_name, image_path, current_time, user_feedback, predictions)
                data['code'] = 1000
            except:
                data['code'] = -1000
        GST.update(user_name, time.time(), 'Feedback')
        GST.auto_save()
    return flask.jsonify(data)

@app.route('/queryDetail', methods=['GET', 'POST'])
def queryDetail():
    data = dict()
    data['state'] = True
    data['data'] = dict()
    if request.method == 'GET':
        user_name = request.args.get('userName') #SHP
        query_id = request.args.get('id')
        history = DB.query_by_id(user_name, query_id)
        history['match_images'] = [baseurl + i for i in history['match_images']]
        history['shotImage'] = baseurl + history['shotImage']
        history['sourceImage'] = baseurl + history['sourceImage']
        history['cam'] = baseurl + history['cam']
        data['data'] =history
        data['code'] = 1000
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
        deleteNum = int(request.args.get('deleteNum'))
        total, history = DB.query(user_name, pageNum, pageSize, deleteNum=deleteNum)
        for index, item in enumerate(history):
            history[index]['shotImage'] = baseurl + item['shotImage']
            history[index]['sourceImage'] = baseurl + item['sourceImage']
            # history[index]['cam'] = baseurl + history[index]['cam']
        if total == None or history == None:
            data['code'] = -1000
        else:
            data['code'] = 1000
            data['data']['total'] = total
            data['data']['list'] = history
        if user_name in adminGroup:
            data['isAdmin'] = True
        else:
            data['isAdmin'] = False

    return flask.jsonify(data)

# 查询指定名称
@app.route('/search', methods=['GET', 'POST'])
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
            coverImage = baseurl + 'uploads/category/' + cate_name.replace('/', '_') + '.jpg'
            dic = {'id':i, 'coverImageUrl':coverImage , 'name':cate_name.split('/')[-1]}
            data['data'].append(dic)
        GST.update(user_name, time.time(), 'Search_cate')
        GST.auto_save()
    return flask.jsonify(data)

@app.route('/admin_feedback_by_username', methods=['GET', 'POST'])
def admin_feedback_by_username():
    data = dict()
    data['state'] = True
    if request.method == 'GET':
        search_name = request.args.get('user_name')
        user_name = request.args.get('userId')
        search_name = search_name.strip()
        if search_name in ['', ' ', '/', '_']:
            data['code'] = -1000 #查找不合法
        else:
            search_result =  FD.query_by_user(search_name)
            if len(search_result) == 0:
                data['code'] = -2000 #未查询到相关信息
            else:
                data['data'] = []
                for i, item in enumerate(search_result):
                    local_time =  time.localtime(float(item['feedbackTime']))
                    feedback_time = time.strftime('%Y-%m-%d %H:%M', local_time)
                    feedbackImage = baseurl + item['imagePath']
                    dic = {'id':i,'time':feedback_time, 'coverImageUrl':feedbackImage ,
                    'feedback': item['user_feedback'], 'prediction':item['prediction'], 'user':search_name}
                    data['data'].append(dic)
                data['code'] = 1000
    return flask.jsonify(data)

@app.route('/admin_feedback_by_time', methods=['GET', 'POST'])
def admin_feedback_by_time():
    data = dict()
    data['state'] = True
    if request.method == 'GET':
        try:
            during_time = float(request.args.get('during_time')) #天
        except:
            data['code'] = -1000#查找不合法
            return flask.jsonify(data)
        user_name = request.args.get('userId')
        search_result =  FD.query_by_time(during_time)
        if len(search_result) == 0:
            data['code'] = -2000 #未查询到相关信息
        else:
            data['data'] = []
            for i, item in enumerate(search_result):
                local_time =  time.localtime(float(item['feedbackTime']))
                feedback_time = time.strftime('%Y-%m-%d %H:%M', local_time)
                feedbackImage = baseurl + item['imagePath']
                dic = {'id':i,'time':feedback_time, 'coverImageUrl':feedbackImage ,
                'feedback': item['user_feedback'], 'prediction':item['prediction'], 'user':item['userId']}
                data['data'].append(dic)
            data['code'] = 1000
    return flask.jsonify(data)

@app.route('/admin_feedback_statistic', methods=['GET', 'POST'])
def admin_feedback_statistic():
    data = dict()
    data['state'] = True
    if request.method == 'GET':
        user_name = request.args.get('userId')
        try:
            refer_date, refer_user, today, total = FD.query_statistic()
            data['data'] = {}
            data['data']['date_category'] = [i[0] for i in refer_date]
            data['data']['date_times'] = [i[1] for i in refer_date]
            data['data']['user_category'] = [i[0] for i in refer_user]
            data['data']['user_times'] = [i[1] for i in refer_user]
            data['today'] = today
            data['total'] = total
            data['code'] = 1000
        except:
            data['code'] = -1000
    return flask.jsonify(data)

@app.route('/admin_guest_statistic', methods=['GET', 'POST'])
def admin_guest_statistic():
    data = dict()
    data['state'] = True
    if request.method == 'GET':
        user_name = request.args.get('userId')
        try:
            refer_date, refer_user, today, total = FD.query_statistic()
            data['data'] = {}
            data['data']['date_category'] = [i[0] for i in refer_date]
            data['data']['date_times'] = [i[1] for i in refer_date]
            data['data']['user_category'] = [i[0] for i in refer_user]
            data['data']['user_times'] = [i[1] for i in refer_user]
            data['today'] = today
            data['total'] = total
            data['code'] = 1000
        except:
            data['code'] = -1000
    return flask.jsonify(data)

@app.route('/search_news_by_key_word', methods=['GET', 'POST'])
def search_news_by_key_word():
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
        search_result, search_index =  NEWS.query_by_key_word(search_name)
        
        data['data'] = []
        for i, item in enumerate(search_result):
            cate_name = item['title']
            coverImage = baseurl + 'uploads/images/' + os.path.split(item['cover'])[-1]
            dic = {'id':search_index[i],'time':item['timestamp'], 'coverImageUrl':coverImage , 'name':cate_name.split('/')[-1]}
            data['data'].append(dic)
        GST.update(user_name, time.time(), 'Search_news')
        GST.auto_save()
    return flask.jsonify(data)

#根据id查询新闻资讯
@app.route('/query_news_by_id', methods=['GET', 'POST'])
def query_news_by_id():
    data = dict()
    data['state'] = True
    if request.method == 'GET':
        query_id = int(request.args.get('templateId'))
        userId = request.args.get('userName')
        query_result = NEWS.query_detail(query_id, baseurl)
        coverImage = baseurl + 'uploads/images/' + os.path.split(query_result['cover'])[-1]
        data['code'] = 1000
        data['data'] = dict()
        data['data']['coverImageUrl'] = coverImage
        data['data']['name'] = query_result['total_title']
        # data['data']['name'] = query_result['title']
        data['data']['time'] = query_result['timestamp']
        data['data']['content'] = query_result['link']
        data['data']['webpage'] = query_result['webpage']

        GST.update(userId, time.time(), 'News')
        GST.auto_save()
        # print(data['data']['content'])
    return flask.jsonify(data)

@app.route('/query_by_category', methods=['GET', 'POST'])
def query_by_category():
    data = dict()
    data['state'] = True
    if request.method == 'GET':
        query_id = request.args.get('templateId')
        userId = request.args.get('userName')
        cate_name = label_id_name_dict[query_id]
        cate = label_id_name_dict[query_id].split('/')[0]
        name = label_id_name_dict[query_id].split('/')[-1]
        coverImage = baseurl + 'uploads/category/' + cate_name.replace('/', '_') + '.jpg'
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
        GST.update(userId, time.time(), 'Baike')
        GST.auto_save()
    return flask.jsonify(data)

@app.route('/query_all_information', methods=['GET', 'POST'])
def query_all_information(): #查询所有的新闻信息
    data = dict()
    data['state'] = False
    data['code'] = 1000
    data['data'] = dict()
    if request.method == 'GET':
        pageNum = int(request.args.get('pageNum')) - 1
        pageSize = int(request.args.get('pageSize'))
        if pageNum == 0:
            NEWS.check_update()
        result, length = NEWS.query_by_page(pageNum, pageSize)
        data['data']['list'] = []
        for i, item in enumerate(result):
            index = i + pageNum * pageSize
            coverImage = baseurl + 'uploads/images/' + os.path.split(item['cover'])[-1]
            dic = {'id':str(index), 'coverImageUrl':coverImage, 'name':item['title'], \
                'createTime':item['timestamp'], 'content':'', 'time':item['timestamp'], 'webpage':item['webpage']}
            data['data']['list'].append(dic)
    data['data']['total'] = length
    # print(data)
    return flask.jsonify(data)

@app.route('/query_all_result', methods=['GET', 'POST'])
def query_all_result(): #查询所有的类别信息
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
            coverImage = baseurl + 'uploads/category/' + label_id_name_dict[str(i)].replace('/', '_') + '.jpg'
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
        print('userName: ', user_name, 'location: ', location)
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        t = time.time()
        f_name=str(int(t))+str(random.randint(1000 , 9999))+'.'+f.filename.rsplit('.',1)[1]
        upload_path = os.path.join(basepath, 'uploads', 'source', user_name)
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        image_path = os.path.join(upload_path, f_name)
        save_small_dir = os.path.join(basepath, 'uploads', 'small', user_name)
        if not os.path.exists(save_small_dir):
            os.makedirs(save_small_dir)
        small_image_path = os.path.join(save_small_dir, f_name)
        f.save(image_path)
        img1 = Image.open(image_path).convert('RGB')
        img = np.array(img1)
        img1 = img1.resize((100, 100), Image.ANTIALIAS)
        img1.save(small_image_path)
        # print('DEBUG ', small_image_path)
        small_image_path = 'uploads/small/' + user_name + '/' + f_name
        # print('DEBUG: ', image_path, small_image_path)
        try:
            topk = request.form['topk']
        except:
            topk = 5
        data = predict_img(img, top_k=int(topk), search_length=20, file_name=f_name, user_name=user_name)
        # pred_id_template = data['matches'][0]['label']
        cam = data['cam']
        # print('DEBUG CAM', cam)
        data['cam'] = baseurl + data['cam']
        match_images = data['match_images_save']
        softmax_prob = data['predictions']
        match_times = data['matches']
        '''
        user_name   #当前查询用户昵称
        small_image_path  #用户查询图片本地保存缩略图地址
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) #用户查询时间戳
        location  #用户查询时所在的地点
        match_times #使用矩阵匹配的结果{'label':'', 'times':''}
        softmax_prob #使用softmax计算的结果{'label':'', 'probability':0.9}
        match_images #图像相似度检索top20图片索引
        cam #类激活图保存地址
        '''
        DB.update(user_name, 
        small_image_path, 
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
        location, 
        match_times,
        softmax_prob,
        match_images,
        cam)
        GST.update(user_name, time.time(), 'Classification')
        GST.auto_save()
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
        # print('image size: ','*'*10,img.shape)
        # img = np.array(Image.open(img).convert('RGB'))
        data = predict_img(img, top_k=int(topk), search_length=20)
    return flask.jsonify(data)


def predict_img(img, top_k=1, search_length=20, CAM=True, file_name=None, user_name=''):
    data = dict()
    start = time.time()
    result = model.predict(img, top_k=top_k, search_length=search_length, CAM=CAM)
    cost_time = time.time() - start
    data['predictions'] = list()
    data['matches'] = list()
    prob_result, cos_result, match_image_name = result[0], result[1], result[2]
    match_image_name_upload = [baseurl + 'uploads/train/' + i for i in match_image_name]
    match_image_name_save = ['uploads/train/' + i for i in match_image_name]
    basepath = os.path.dirname(__file__)
    if CAM:
        cam= result[3]
        save_cam_dir = os.path.join(basepath, 'uploads', 'cam', user_name)
        if not os.path.exists(save_cam_dir):
            os.makedirs(save_cam_dir)
        plt.imshow(cam)
        plt.axis('off')

        plt.gcf().set_size_inches(512 / 100, 512 / 100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.savefig(os.path.join(save_cam_dir ,file_name), dpi=300)
        # cv2.imwrite(os.path.join(save_cam_dir ,file_name), cam)
        data['cam'] = 'uploads/cam/' + user_name + '/' + file_name
    for label, prob in prob_result:
        prob_predict = {'label': label, 'probability': ("%.4f" % prob)}
        data['predictions'].append(prob_predict)
    for label, times in cos_result:
        cos_predict = {'label': label, 'times': ("%d" % times)}
        data['matches'].append(cos_predict)

    md_name = data['predictions'][0]['label'].split('/')[-1] + '.md'
    print(md_name.split('.')[0])
    baike = os.path.join(basepath, 'baike', md_name)
    with open(baike, 'r', encoding='UTF-8') as f:
        bk = f.read()
    data['state'] = True
    data['content'] = bk
    data['time'] = cost_time
    data['match_images_save'] = match_image_name_save
    data['match_images'] = match_image_name_upload
    data['code'] = 1000
    return data

@app.route('/admin_guest_by_username', methods=['GET', 'POST'])
def admin_guest_by_username():
    data = dict()
    data['state'] = True
    if request.method == 'GET':
        search_name = request.args.get('user_name')
        user_name = request.args.get('userId')
        search_name = search_name.strip()
        if search_name in ['', ' ', '/', '_']:
            data['code'] = -1000 #查找不合法
            return flask.jsonify(data)
        else:
            refer_days, refer_detail, today_num, total_num =  GST.statistic_by_user(search_name)
            # print(refer_days, refer_detail, today_num, total_num)
            data['time_cate'] = [i[0] for i in refer_days]
            data['time_value'] = [i[1] for i in refer_days]
            data['detail_cate'] = [i[0] for i in refer_detail]
            data['detail_value'] = [i[1] for i in refer_detail]
            data['today'] = today_num
            data['total'] = total_num
            data['code'] = 1000
    return flask.jsonify(data)

@app.route('/admin_guest_by_time', methods=['GET', 'POST'])
def admin_guest_by_time():
    data = dict()
    data['state'] = True
    if request.method == 'GET':
        try:
            during_time = float(request.args.get('during_time')) #天
        except:
            data['code'] = -1000#查找不合法
            return flask.jsonify(data)
        user_name = request.args.get('userId')
        refer_user, refer_detail, today_num, total_num =  GST.statistic_by_time(during_time)
        data['user_cate'] = [i[0] for i in refer_user]
        data['user_value'] = [i[1] for i in refer_user]
        data['detail_cate'] = [i[0] for i in refer_detail]
        data['detail_value'] = [i[1] for i in refer_detail]
        data['today'] = today_num
        data['total'] = total_num
        data['code'] = 1000
    return flask.jsonify(data)

        

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    gpu_id = None
    model_path = './model_best.pth.tar'
    feature_path = './features_train_baidu.pkl'
    '''
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    transforms_ = transforms.Compose([
		transforms.Resize(512),
		transforms.CenterCrop(320),
		#ScaleResize((320, 320)),
		transforms.ToTensor(),
		normalize
	])
    img = np.array(Image.open('15809656297338.jpg').convert('RGB'))
    model = Net(model_path, feature_path)
    result = model.predict(img, top_k=5, search_length=20, CAM=True)
    '''
    model = Net(model_path, feature_path)
    app.run()
