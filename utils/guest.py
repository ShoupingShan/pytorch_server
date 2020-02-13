import pickle, os
import time
class Guest:
    def __init__(self, query_path=os.path.join('data', 'database', 'guest_records.pkl')):
        self.query_path = query_path
        self.locked = False
        self.time_before = time.time()
        if not os.path.exists(self.query_path):
            record = []
            with open(self.query_path, 'wb') as f:
                pickle.dump(record, f)
                print('Init quest database!')
        with open(self.query_path, 'rb') as f:
            self.DB = pickle.load(f)
    def _check_status(self):
        while(self.locked):
            print('Guest dataset Lock Phenomenon Tiggered')
            time.sleep(0.001)
        self.locked = True
    def _release_lock(self):
        self.locked = False

    def auto_save(self):
        current_time = time.time()
        self._check_status()
        if current_time - self.time_before > 120:
            self.time_before = current_time
            with open(self.query_path, 'wb') as f:
                pickle.dump(self.DB, f)
        self._release_lock()
    
    def update(self, user, query_time, query_detail):
        self._check_status()
        log = {
            'userId' : str(user),
            'time':query_time,
            'detail':query_detail,
        }
        self.DB.append(log)
        self._release_lock()

    
    def statistic_by_time(self, during_time):
        self._check_status()
        user = {}
        detail = {}
        current_time = time.time()
        today = time.strftime('%Y-%m-%d', time.localtime(current_time))
        today_num = 0
        today_user = []
        for log in self.DB[::-1]:
            
            update_time = time.strftime('%Y-%m-%d', time.localtime(log['time']))
            if update_time == today:
                today_num += 1
                today_user.append(log['userId'])

            if current_time - log['time'] > during_time * 3600 * 24:
                pass
            else: 
                if log['userId'] not in user.keys():
                    user[log['userId']] = 1  #该用户历史请求总数
                else:
                    user[log['userId']] += 1
                if log['detail'] not in detail.keys():
                    detail[log['detail']] = 1  #该用户历史请求总数
                else:
                    detail[log['detail']] += 1
        if(len(list(user.keys())) > 0):
            refer_user = sorted(user.items(),key = lambda x:x[1],reverse = True)
        else:
            refer_user = [('Not Found', 0)]
        if(len(list(detail.keys())) > 0):
            refer_detail = sorted(detail.items(),key = lambda x:x[1],reverse = True)
        else:
            refer_detail = [('Not Found', 0)]
        self._release_lock()
        return refer_user, refer_detail, today_num, len(list(set(today_user)))

    def statistic_by_user(self, user_name):
        self._check_status()
        current_time = time.time()
        today = time.strftime('%Y-%m-%d', time.localtime(current_time))
        today_num = 0
        today_user = []
        days = {}
        detail = {}
        for log in self.DB:
            day = time.strftime('%Y-%m-%d', time.localtime(log['time']))
            if day == today:
                today_num += 1
                today_user.append(log['userId'])
            if log['userId'] != user_name:
                pass
            else:
                if day not in days.keys():
                    days[day] = 1  #该用户历史请求总数
                else:
                    days[day] += 1
                if log['detail'] not in detail.keys():
                    detail[log['detail']] = 1  #该用户历史请求总数
                else:
                    detail[log['detail']] += 1
        if(len(list(days.keys())) > 0):
            refer_days = sorted(days.items(),key = lambda x:x[0],reverse = False)
        else:
            refer_days = [('Not Found', 0)]
        if(len(list(detail.keys())) > 0):
            refer_detail = sorted(detail.items(),key = lambda x:x[1],reverse = True)
        else:
            refer_detail = [('Not Found', 0)]
        
        self._release_lock()
        return refer_days, refer_detail, today_num, len(list(set(today_user)))
    

if __name__ == '__main__':
    pass