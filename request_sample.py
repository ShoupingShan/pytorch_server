# -*- coding: utf-8 -*-
# @Time    : 2018/4/24 10:04
# @Author  : zhoujun
import argparse
import requests
from PIL import Image
import numpy as np
URL = 'http://127.0.0.1:5000/predict'

def predict_result(image_path):

    # img = open(image_path,'rb').read()
    img = np.array(Image.open(image_path).convert('RGB'))
    params = {'topk':3, 'h':img.shape[0], 'w':img.shape[1], 'c':img.shape[2]}
    img = img.tostring()
    # img = image_path
    msg = {'image':img}
    try:
        r = requests.post(URL,files=msg,data=params)
        print(r.url)
        r = r.json()
        if r['state']:
            print('Success! Time:%.3fs'%(r['time']))
            print('Softmax:\n')
            print('Rank Label Probability')
            for index, item in enumerate(r['predictions']):
                print(str(index) + ' ' + str(item['label']) + ' ' + str(item['probability']) +'\n')
            print('Matrix Match:\n')
            print('Rank Label Times')
            for index, item in enumerate(r['matches']):
                print(str(index) + ' ' + str(item['label']) + ' ' + str(item['times']) +'\n' )

        else:
            print('failed')
    except:
        print('failed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('-f','--file', type=str,default='12.png', help='test image file')

    args = parser.parse_args()
    predict_result(args.file)