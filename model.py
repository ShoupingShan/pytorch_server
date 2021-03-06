from PIL import Image
from collections import OrderedDict
import torch.nn as nn
import torch
import time
import traceback
import numpy as np
from collections import Counter
import torchvision.transforms as transforms
from cal_similarity import getCosDist, get_similarity
import torch.nn.functional as F
import pickle, os, sys
from utils import get_CAM
from models.res2net_se import se_res2net50_26w_8s
from models.eff_model import efficientnet_b5
label_id_name_dict = \
            {
                "0": "工艺品/仿唐三彩",
                "1": "工艺品/仿宋木叶盏",
                "2": "工艺品/布贴绣",
                "3": "工艺品/景泰蓝",
                "4": "工艺品/木马勺脸谱",
                "5": "工艺品/柳编",
                "6": "工艺品/葡萄花鸟纹银香囊",
                "7": "工艺品/西安剪纸",
                "8": "工艺品/陕历博唐妞系列",
                "9": "景点/关中书院",
                "10": "景点/兵马俑",
                "11": "景点/南五台",
                "12": "景点/大兴善寺",
                "13": "景点/大观楼",
                "14": "景点/大雁塔",
                "15": "景点/小雁塔",
                "16": "景点/未央宫城墙遗址",
                "17": "景点/水陆庵壁塑",
                "18": "景点/汉长安城遗址",
                "19": "景点/西安城墙",
                "20": "景点/钟楼",
                "21": "景点/长安华严寺",
                "22": "景点/阿房宫遗址",
                "23": "民俗/唢呐",
                "24": "民俗/皮影",
                "25": "特产/临潼火晶柿子",
                "26": "特产/山茱萸",
                "27": "特产/玉器",
                "28": "特产/阎良甜瓜",
                "29": "特产/陕北红小豆",
                "30": "特产/高陵冬枣",
                "31": "美食/八宝玫瑰镜糕",
                "32": "美食/凉皮",
                "33": "美食/凉鱼",
                "34": "美食/德懋恭水晶饼",
                "35": "美食/搅团",
                "36": "美食/枸杞炖银耳",
                "37": "美食/柿子饼",
                "38": "美食/浆水面",
                "39": "美食/灌汤包",
                "40": "美食/烧肘子",
                "41": "美食/石子饼",
                "42": "美食/神仙粉",
                "43": "美食/粉汤羊血",
                "44": "美食/羊肉泡馍",
                "45": "美食/肉夹馍",
                "46": "美食/荞面饸饹",
                "47": "美食/菠菜面",
                "48": "美食/蜂蜜凉粽子",
                "49": "美食/蜜饯张口酥饺",
                "50": "美食/西安油茶",
                "51": "美食/贵妃鸡翅",
                "52": "美食/醪糟",
                "53": "美食/金线油塔",
                "54": "景点/鼓楼",
                "55": "美食/小炒泡馍",
                "56": "美食/葫芦头泡馍"
            }
pr_dic = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, 
            '19': 11, '2': 12, '20': 13,'21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21, 
            '29': 22, '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, '37': 31, '38': 32, 
            '39': 33, '4': 34, '40': 35, '41': 36, '42': 37, '43': 38, '44': 39, '45': 40, '46': 41, '47': 42, '48': 43, 
            '49': 44, '5': 45, '50': 46, '51': 47, '52': 48, '53': 49, '54': 50, '55': 51, '56': 52, '6': 53, '7': 54, 
            '8': 55, '9': 56}

class Net:
    def __init__(self, model_path, feature_path, gpu_id=0, idx2label=label_id_name_dict, refer=pr_dic):
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.feature_path = feature_path
        self.idx2label = idx2label
        self.refer_map_huawei = dict([(v,k) for (k,v) in label_id_name_dict.items()])
        self.refer = dict([(v,k) for (k,v) in refer.items()])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        self.transforms = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(320),
            #ScaleResize((320, 320)),
            transforms.ToTensor(),
            self.normalize
        ])
        self.model = efficientnet_b5()
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, 57)
        if torch.cuda.is_available() and gpu_id is not None:
            self.use_cuda = True
            print('Using GPU for inference')
            self.device = torch.device("cuda:%s" % (self.gpu_id))
            checkpoint = torch.load(self.model_path)
            self.model = torch.nn.DataParallel(self.model).cuda(gpu_id)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.use_cuda = False
            print('Using CPU for inference')
            self.device = torch.device("cpu")
            checkpoint = torch.load(self.model_path, map_location='cpu')
            state_dict = OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                state_dict[tmp] = value
            self.model.load_state_dict(state_dict)

    def predict(self, img, top_k=1, search_length=20, CAM=False):
        self.model.eval()
        pic = Image.fromarray(img)
        pic = self.transforms(pic)
        pic = pic.unsqueeze(0)

        pic = torch.Tensor(pic)
        if self.use_cuda:
            pic = pic.cuda()
        # try:
        train_dict = pickle.load(open(self.feature_path, 'rb'))
        train_label = [k for (k, v) in train_dict.items()]
        train_features = np.array([v for (k, v) in train_dict.items()])
        train_image_name = np.array([str(os.path.split(i)[-1]) for i in train_label])
        train_label = np.array([int(self.refer_map_huawei[str(os.path.split(i)[0])]) for i in train_label])
        if torch.cuda.is_available():
            train_features = torch.Tensor(train_features).cuda()
            train_label = torch.Tensor(train_label).cuda()
            # train_image_name = torch.Tensor(train_image_name).cuda()
        else:
            train_features = torch.Tensor(train_features)
            train_label = torch.Tensor(train_label)
            # train_image_name = torch.Tensor(train_image_name)
        # except Exception as err:
        #     errout = traceback.format_exc()
        if CAM:
            cam, pred_score, eval_features = get_CAM(self.model, '_bn1', '_dropout', img, pic)
            
        with torch.no_grad():
            # pred_score = self.model(pic)
            pred_score = F.softmax(pred_score.data, dim=1)
            prob, cate = torch.topk(pred_score.data, k=top_k)
            # if self.use_cuda:
            #     eval_features = self.model.module.features
            # else:
            #     eval_features = self.model.features
            #     print('Forward fea:', eval_features)
            # try:
            cos_matrix = getCosDist(eval_features, train_features)
            # print(eval_features)
            # print(train_features)
            similarity = get_similarity((cos_matrix))
            distance, index = torch.topk(similarity, search_length, dim=1)
            labels = list(train_label[index].cpu().numpy())
            index = index.cpu().numpy()
            # print(labels)
            image_names = list(train_image_name[index[0]])
            # print(image_names)
            num_all_pred_cate = len(set(list(labels[0])))
            top_k_dis = min(top_k, num_all_pred_cate)
            '''
            Counter('abcdeabcdabcaba').most_common(3)
            [('a', 5), ('b', 4), ('c', 3)]
            '''
            labels_cos = [Counter(i).most_common(top_k_dis) for i in labels]
            print(labels_cos)
            # except Exception as err:
            #     errout = traceback.format_exc()

            if self.use_cuda:
                prob = prob.cpu().numpy().tolist()
                cate = cate.cpu().numpy().tolist()
            else:
                prob = prob.numpy().tolist()
                cate = cate.numpy().tolist()
            # collect results
            label_cos_all = []
            times_cos_all = [] #match times in dictionary
            for pred in labels_cos: # for each input image
                label = []
                times = []
                for idx in pred:
                    label.append(self.idx2label[str(int(idx[0]))] if self.idx2label is not None else str(int(idx[0])))
                    times.append(int(idx[1]))
                times_cos_all.append(times)
                label_cos_all.append(label)
            label_prob_all = []
            score_prob_all = []
            for _prob, _cate in zip(prob, cate):# for each input image
                label = []
                times = []
                for p, c in zip(_prob, _cate):
                    label.append(self.idx2label[self.refer[int(c)]] if self.idx2label is not None else self.refer[int(c)])
                    times.append(p)
                score_prob_all.append(times)
                label_prob_all.append(label)
            result_cos = zip(label_cos_all[0], times_cos_all[0])
            result_prob = zip(label_prob_all[0], score_prob_all[0])
        if CAM:
            return [result_prob, result_cos, image_names, cam]
        else:
            return [result_prob, result_cos, image_names]

if __name__ == '__main__':
    model_path = './model_best.pth.tar'
    feature_path = './features_train_baidu.pkl'
    idx2label = label_id_name_dict
    image = os.path.join('./1.jpg')
    img = np.array(Image.open(image).convert('RGB'))
    refer = pr_dic
    model = Net(model_path, feature_path, idx2label=idx2label, refer=refer)
    result = model.predict(img, top_k=3, search_length=20, CAM=True)
    print(result[2])
