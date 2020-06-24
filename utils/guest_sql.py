import pickle, os
import time, random, datetime
from .mysql import DB
class Guest:
    def __init__(self, query_path=os.path.join('data', 'database', 'guest_records.pkl')):
        self.db = DB()

    def update(self, user, query_time, query_detail):
        t = time.localtime(query_time)
        t = time.strftime("%Y-%m-%d %H:%M:%S", t)
        sql = 'INSERT INTO guest_record(detail, time, userId) VALUES ("%s", "%s", "%s");'%(query_detail, t, str(user))
        status = self.db.fix_db(sql)
        if not status:
            print('UPDATE Guest record Error!')
        return status


    def statistic_by_time(self, during_time):
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        today = today.strftime("%Y-%m-%d")
        tomorrow = tomorrow.strftime("%Y-%m-%d")
        sql = 'SELECT count(*) from guest_record where time between "%s" and "%s";'%(today, tomorrow)
        status, today_num = self.db.search_db(sql)
        today_num = today_num[0][0]
        sql = 'select userId ,count(*) as a from guest_record where time between "%s" and "%s" group by userId order by a desc;'%(today, tomorrow)
        status, refer_user = self.db.search_db(sql)
        if len(refer_user) == 0:
            refer_user = [('Not Found', 0)]
        sql = 'select detail ,count(*) as a from guest_record where time between "%s" and "%s" group by detail order by a desc;'%(today, tomorrow)
        status, refer_detail = self.db.search_db(sql)
        if len(refer_user) == 0:
            refer_detail = [('Not Found', 0)]
        sql = 'select count(distinct userId) from guest_record;'
        status, today_user = self.db.search_db(sql)
        today_user = today_user[0][0]
        return refer_user, refer_detail, today_num, today_user

    def statistic_by_user(self, user_name):
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        today = today.strftime("%Y-%m-%d")
        tomorrow = tomorrow.strftime("%Y-%m-%d")
        sql = 'SELECT count(*) from guest_record where time between "%s" and "%s";'%(today, tomorrow)
        status, today_num = self.db.search_db(sql)
        today_num = today_num[0][0]
        sql = 'select count(distinct userId) from guest_record where time between "%s" and "%s";'%(today, tomorrow)
        status, today_user = self.db.search_db(sql)
        today_user = today_user[0][0]

        sql = 'select DATE_FORMAT(time,"%%Y-%%m-%%d") as t, count(*) from guest_record where userId="%s" group by t order by t desc;'%(user_name)
        status, refer_days = self.db.search_db(sql)
        if len(refer_days) == 0:
            refer_days = [('Not Found', 0)]

        sql = 'select detail, count(*) as t from guest_record where userId="%s" group by detail order by t desc;'%(user_name)
        status, refer_detail = self.db.search_db(sql)
        if len(refer_detail) == 0:
            refer_detail = [('Not Found', 0)]
        
        return refer_days, refer_detail, today_num, today_user
    

if __name__ == '__main__':
    db = Guest()
    user, query_time, query_detail = 'TEST', '2020-06-23 01:17:01', 'News'
    res = db.update(user, query_time, query_detail)
    # res = db.statistic_by_user('TEST')
    print(res)