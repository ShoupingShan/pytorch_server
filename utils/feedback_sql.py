__author__='SHP'
import urllib.request
import bs4, pickle
import re, os
import requests
import time, random, datetime
from .mysql import DB
class Feedback:
    def __init__(self):
        self.db = DB()

    def update(self, userId, imageName, imagePath, feedbackTime, user_feedback, prediction):
        t = time.localtime(feedbackTime)
        t = time.strftime("%Y-%m-%d %H:%M:%S", t)
        sql = '''
        INSERT INTO feedback(time, imageName, imagePath, prediction, userId, userFeedback) VALUES 
        ('%s', '%s', '%s', '%s', '%s', '%s');
        '''%(t, imageName, imagePath, prediction, userId, user_feedback)
        status = self.db.fix_db(sql)
        if not status:
            print('UPDATE Feedback Error!')


    def query_by_time(self, times):
        current_time = time.time()
        seconds = times * 3600 * 24
        start_time = current_time - seconds
        t = time.localtime(start_time)
        t = time.strftime("%Y-%m-%d %H:%M:%S", t)
        sql = 'SELECT * from feedback where time >= "%s" order by time desc;'%(t)
        status, query_result = self.db.search_db(sql)
        if not status:
            print('Query Feedback by Time Error!')
        res = []
        for item in query_result:
            temp = {
            'userId': item[5],
            'imageName': item[2],
            'imagePath' : item[3],
            'feedbackTime' : item[1],
            'user_feedback':item[6],
            'prediction':item[4],
            }
            res.append(temp)

        return res



    def query_by_user(self, userId):
        sql = 'SELECT * from feedback where userId = "%s" order by time desc;'%(userId)
        status, query_result = self.db.search_db(sql)
        if not status:
            print('Query Feedback by User Error!')
        res = []
        for item in query_result:
            temp = {
            'userId': item[5],
            'imageName': item[2],
            'imagePath' : item[3],
            'feedbackTime' : item[1],
            'user_feedback':item[6],
            'prediction':item[4],
            }
            res.append(temp)
        return res
    
    def query_all(self, pageNum, pageSize):
        sql = 'SELECT count(*) from feedback;'
        status, query_result = self.db.search_db(sql)
        length = query_result[0][0]
        start_index = min(length, pageNum * pageSize)
        sql = 'SELECT * from feedback limit %d, %d'%(start_index, pageSize)
        status, query_result = self.db.search_db(sql)
        if not status:
            print('Query Feedback All Error!')
        res = []
        for item in query_result:
            temp = {
            'userId': item[5],
            'imageName': item[2],
            'imagePath' : item[3],
            'feedbackTime' : item[1],
            'user_feedback':item[6],
            'prediction':item[4],
            }
            res.append(temp)
        return res

    def query_statistic(self):
        sql = 'SELECT count(*) from feedback;'
        status, query_result = self.db.search_db(sql)
        length = query_result[0][0]
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        today = today.strftime("%Y-%m-%d")
        tomorrow = tomorrow.strftime("%Y-%m-%d")
        sql = 'SELECT count(*) from feedback where time between "%s" and "%s";'%(today, tomorrow)
        status, today_times = self.db.search_db(sql)
        today_times = today_times[0][0]
        sql = 'select userId ,count(*) as a from feedback group by userId order by a desc;'
        status, refer_user = self.db.search_db(sql)
        sql = 'select DATE_FORMAT(time,"%Y-%m-%d") as t,count(*) from feedback group by t order by t desc;'
        status, refer_date = self.db.search_db(sql)
        return refer_date, refer_user, today_times, length
    

if __name__ == '__main__':
    db = Feedback()
    res = db.query_by_time(100)
    print(res)
