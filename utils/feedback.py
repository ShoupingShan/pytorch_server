__author__='SHP'
import urllib.request
import bs4, pickle
import re, os
import requests
import time, random
base_url = 'http://news.xiancn.com/'
class Feedback:
    def __init__(self, pkl_path='./data/database'):
        if not os.path.exists(pkl_path):
            os.makedirs(pkl_path)
        if not os.path.exists(os.path.join(pkl_path, 'Feedback.pkl')):
            with open(os.path.join(pkl_path, 'Feedback.pkl'), 'wb') as f:
                content = []
                pickle.dump(content, f)
        self.DB_path = os.path.join(pkl_path, 'Feedback.pkl')
        with open(self.DB_path, 'rb') as f:
            self.data = pickle.load(f)
        self.locked = False

    def _check_status(self):
        while(self.locked):
            print('Lock Phenomenon Tiggered')
            time.sleep(0.001)
        self.locked = True

    def _release_lock(self):
        self.locked = False

    def updata(self, userId, imageName, imagePath, feedbackTime, user_feedback, prediction):
        self._check_status()
        item = {
            'userId': userId,
            'imageName': imageName,
            'imagePath' : imagePath,
            'feedbackTime' : feedbackTime,
            'user_feedback':user_feedback,
            'prediction':prediction,
        }
        self.data.append(item)
        with open(self.DB_path, 'wb') as f:
            pickle.dump(self.data, f)
        self._release_lock()

    def query_by_time(self, times):
        self._check_status()
        with open(self.DB_path, 'rb') as f:
            self.data = pickle.load(f)
        current_time = time.time()
        seconds = times * 3600 * 24
        start_time = current_time - seconds
        query_result = [i for i in self.data if i['feedbackTime'] >= start_time]
        query_result.reverse()
        self._release_lock()
        return query_result



    def query_by_user(self, userId):
        self._check_status()
        with open(self.DB_path, 'rb') as f:
            self.data = pickle.load(f)
        query_result = [i for i in self.data if i['userId'] == userId]
        query_result.reverse()
        self._release_lock()
        return query_result
    
    def query_all(self, pageNum, pageSize):
        self._check_status()
        with open(self.DB_path, 'rb') as f:
            self.data = pickle.load(f)
        start_index = min(len(self.data), pageNum * pageSize)
        end_index = min(len(self.data), (pageNum + 1) * pageSize)
        query_result = self.data[start_index:end_index]
        self._release_lock()
        return query_result

    def query_statistic(self):
        self._check_status()
        with open(self.DB_path, 'rb') as f:
            self.data = pickle.load(f)
        time_list = [i['feedbackTime'] for i in self.data]
        local_list = [time.localtime(i) for i in time_list]
        day_list = [time.strftime('%Y-%m-%d', i) for i in local_list]
        today = time.strftime('%Y-%m-%d', time.localtime())
        date = {}
        user = {}
        for index, item in enumerate(day_list):
            if item not in date:
                date[item] = 1
            else:
                date[item] += 1
            if self.data[index]['userId'] not in user:
                user[self.data[index]['userId']] = 1
            else:
                user[self.data[index]['userId']] += 1
        if today in date.keys():
            today_times = date[today]
        else:
            today_times = 0
        refer_date = sorted(date.items(),key = lambda x:x[0],reverse = False)
        refer_user = sorted(user.items(),key = lambda x:x[1],reverse = True)
        self._release_lock()
        return refer_date, refer_user, today_times, len(self.data)
    

if __name__ == '__main__':
    spider=Xian('./dataset')
    spider.check_update()
    print('Update successfully')
    a = spider.query_by_page(0, 10)
    print(a)
