import pickle, os
import time
class Database:
    def __init__(self, query_path=os.path.join('data', 'database', 'query_records.pkl')):
        # query_records = os.path.join('data', 'database', 'query_records.pkl')
        self.query_path = query_path
        self.locked = False
        if not os.path.exists(self.query_path):
            record = {}
            with open(self.query_path, 'wb') as f:
                pickle.dump(record, f)
                print('Init database!')
        with open(self.query_path, 'rb') as f:
            self.DB = pickle.load(f)
            print('Load database!')
    def _check_status(self):
        while(self.locked):
            print('Lock Phenomenon Tiggered')
            time.sleep(0.001)
        self.locked = True
    def _release_lock(self):
        self.locked = False

    def update(self, nickname, image_path, update_time, upload_location, match_times, softmax_prob, match_images, cam=''):
        self._check_status()
        with open(self.query_path, 'rb') as f:
            self.DB = pickle.load(f)
            # print('Load database!')
        if nickname not in self.DB.keys():
            self.DB[nickname] = {}
            self.DB[nickname]['query_list'] = []
            sourceImage = image_path.replace('small', 'source')
            dic = {'createTime':update_time, \
                'name':os.path.split(image_path)[-1], \
                'shotImage':image_path, \
                'location':upload_location, \
                'templateId': 0, \
                'sourceImage':sourceImage, \
                'pred_result':match_times[0]['label'], \
                'match_times':match_times, \
                'softmax_prob':softmax_prob, \
                'match_images' : match_images, \
                'content':'', \
                'cam':cam}
            self.DB[nickname]['query_list'].append(dic)
            self.DB[nickname]['total'] = 1
        else:
            current_id = self.DB[nickname]['total']
            sourceImage = image_path.replace('small', 'source')
            dic = {'createTime':update_time, \
                'name':os.path.split(image_path)[-1], \
                'shotImage':image_path, 'location':upload_location, \
                'templateId': current_id, 'sourceImage':sourceImage,\
                'pred_result':match_times[0]['label'], \
                'match_times':match_times, \
                'softmax_prob':softmax_prob, \
                'match_images' : match_images, \
                'content':'', \
                'cam':cam}
            self.DB[nickname]['query_list'].append(dic) #该查找在数据库的索引
            self.DB[nickname]['total'] += 1
            assert self.DB[nickname]['total'] == len(self.DB[nickname]['query_list']), 'database query length is illegal'
        with open(self.query_path, 'wb') as f:
            pickle.dump(self.DB, f)
        self._release_lock()
    def clear(self, nickname):
        self._check_status()
        with open(self.query_path, 'rb') as f: #获取最新的数据库结果
            self.DB = pickle.load(f)
        if nickname not in self.DB.keys():
            print('%s is not saved in database before')
            # self.DB[nickname] = {}
        else:
            self.DB[nickname] = {}
            self.DB[nickname]['total'] = 0
        with open(self.query_path, 'wb') as f:
            pickle.dump(self.DB, f)
        self._release_lock()
    
    def deleteItem(self, nickname, id):
        self._check_status()
        with open(self.query_path, 'rb') as f: #获取最新的数据库结果
            self.DB = pickle.load(f)
        if nickname not in self.DB.keys():
            print('%s is not saved in database before')
            # self.DB[nickname] = {}
        else:
            id = self.DB[nickname]['total'] - id - 1
            print('id=',id)
            self.DB[nickname]['query_list'].pop(id)
            self.DB[nickname]['total'] -= 1
            for i in range(id, self.DB[nickname]['total']):
                self.DB[nickname]['templateId'] = i
        with open(self.query_path, 'wb') as f:
            pickle.dump(self.DB, f)
        self._release_lock()

    def query(self, nickname, pageNum=0, pageSize=10, deleteNum=0):
        self._check_status()
        with open(self.query_path, 'rb') as f: #获取最新的数据库结果
            self.DB = pickle.load(f)
        if nickname not in self.DB.keys():
            print('%s is not saved in database before')
            # self.DB[nickname] = {}
            self._release_lock()
            return None, []
        else:
            reverse_list = self.DB[nickname]['query_list'].copy()
            reverse_list.reverse()
            total = self.DB[nickname]['total']
            start_index = min(total, pageNum * pageSize - deleteNum)
            end_index = min(total, (pageNum + 1)*pageSize - deleteNum)
            history = reverse_list[start_index:end_index]
            self._release_lock()
            return total, history
        # with open(self.query_path, 'wb') as f:
        #     pickle.dump(self.DB, f)
    
    def query_by_id(self, nickname, id):
        self._check_status()
        with open(self.query_path, 'rb') as f: #获取最新的数据库结果
            self.DB = pickle.load(f)
        if nickname not in self.DB.keys():
            print('%s is not saved in database before')
            self._release_lock()
            return None, []
        else:
            history = self.DB[nickname]['query_list'][int(id)]
            self._release_lock()
            return history
        # with open(self.query_path, 'wb') as f:
        #     pickle.dump(self.DB, f)


        



