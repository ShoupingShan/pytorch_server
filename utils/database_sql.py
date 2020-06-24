import pickle, os
import time, random
import sys
from .mysql import DB
class Database:
    def __init__(self, query_path=os.path.join('data', 'database', 'query_records.pkl')):
        self.db = DB()
        self.query_path = query_path
    def insert_match_image(self, data):
        part_sql = ''
        for d in data:
            part_sql += ' \'' + str(d) + '\','
        part_sql = part_sql[:-1]
        part_sql1 = ''
        for i in range(1,21):
            part_sql1 += ' top'+str(i)+','
        part_sql1 = part_sql1[:-1]
        sql = '''INSERT INTO match_image(%s) VALUES (%s);''' % (part_sql1, part_sql)
        status = self.db.fix_db(sql)
        if not status:
            print('INSERT MATCH IMAGE ERROR')
    def insert_softmax(self, data):
        # sql = '''INSERT INTO softmax(top1, top2, top3, top4, top5) VALUES (%d, %d, %d, %d, %d);''' % (index, index + 1, index + 2, index + 3, index + 4)
        sql = '''INSERT INTO softmax(label_top1, probability_top1, label_top2, probability_top2,
        label_top3, probability_top3, label_top4, probability_top4, label_top5, probability_top5) VALUES 
        ('%s', '%s', '%s', '%s','%s', '%s','%s', '%s','%s', '%s');''' % (data[0]['label'], data[0]['probability'], data[1]['label'], data[1]['probability'], data[2]['label'], data[2]['probability'], data[3]['label'], data[3]['probability'], data[4]['label'], data[4]['probability'])
        status = self.db.fix_db(sql)
        if not status:
            print('INSERT SOFTMAX ERROR')
    def insert_match_times(self, data):
        sql = '''INSERT INTO match_time(label, times) VALUES ('%s', '%s');''' % (data['label'], data['times'])
        status = self.db.fix_db(sql)
        if not status:
            print('INSERT MATCH TIMES ERROR')
    
    def update(self, nickname, image_path, update_time, upload_location, match_times, softmax_prob, match_images, cam=''):
        self.insert_match_times(match_times[0])
        self.insert_match_image(match_images)
        self.insert_softmax(softmax_prob)
        sourceImage = image_path.replace('small', 'source')
        sql = '''
        INSERT INTO query_record(nickName, cam, createTime, location, 
        imageName, pred_result, shotImage, sourceImage) VALUES (
            "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s"
        );
        ''' % (nickname, cam, update_time, upload_location, 
                os.path.split(image_path)[-1], match_times[0]['label'], image_path, sourceImage)
        status = self.db.fix_db(sql)
        if not status:
            print('INSERT QUERY RECORD ERROR')

    def deleteItem(self, nickname, id):
        sql = 'SELECT * from query_record where id=(select id from (select (@i:=@i+1)pm,s.* from query_record s,(select @i:=0)t where nickName="%s" order by createTime desc) as f where f.pm=%d);'%(nickname, int(id) + 1)
        status, result= self.db.search_db(sql)
        result = result[0]
        delete_id = result[0]
        if not status:
            print('SEARCH ITEM ERROR')
        sql = 'DELETE from query_record where id=%d'%(delete_id)
        status = self.db.fix_db(sql)
        if not status:
            print('DELETE ITEM ERROR')
        
        sql = 'DELETE from match_image where id=%d;'%(delete_id)
        status = self.db.fix_db(sql)
        if not status:
            print('DELETE match_image ERROR')

        sql = 'DELETE from match_time where id=%d;'%(delete_id)
        status = self.db.fix_db(sql)
        if not status:
            print('DELETE match_time ERROR')
        
        sql = 'DELETE from softmax where id=%d;'%(delete_id)
        status = self.db.fix_db(sql)
        if not status:
            print('DELETE softmax ERROR')

        small_image = os.path.join(result[7])
        source_image = os.path.join(result[8])
        cam_image = os.path.join(result[2])
        for image in [small_image, source_image, cam_image]:
            if os.path.exists(image):
                os.remove(image)
        

    def query(self, nickname, pageNum=0, pageSize=10, deleteNum=0):

        sql = 'SELECT count(*) from query_record where nickName="%s";'%(nickname)
        status, query_result = self.db.search_db(sql)
        length = query_result[0][0]
        start_index = pageNum * pageSize - deleteNum
        sql = 'SELECT * from query_record where nickName="%s" order by createTime desc limit %d, %d;'%(nickname, start_index, pageSize)
        status, result = self.db.search_db(sql)
        if not status:
            print('SEARCH QUERY ERROR')
        sql = 'select * from softmax where id in (SELECT id from (select id from query_record where nickName="%s"  order by createTime desc limit %d, %d) as t);'%(nickname ,start_index, pageSize)
        status, res_softmax = self.db.search_db(sql)
        if not status:
            print('SEARCH SOFTMAX ERROR')
        
        sql = 'select * from match_time where id in (SELECT id from (select id from query_record where nickName="%s"  order by createTime desc limit %d, %d) as t);'%(nickname ,start_index, pageSize)
        status, result_mt = self.db.search_db(sql)
        if not status:
            print('SEARCH match_time ERROR')

        sql = 'select * from match_image where id in (SELECT id from (select id from query_record where nickName="%s"  order by createTime desc limit %d, %d) as t);'%(nickname, start_index, pageSize)
        status, result_mi = self.db.search_db(sql)
        if not status:
            print('SEARCH match_image ERROR')
        history = []
        for qu, sm, mt, mi in zip(result, res_softmax, result_mt, result_mi):
            list_mi = []
            for i in range(1, 21):
                list_mi.append(mi[i])
            dict_mt = {}
            dict_mt['label'] = mt[1]
            dict_mt['times'] = mt[2]
            list_mt = [dict_mt]

            list_sm = []
            
            for i in range(1, 11, 2):
                temp = {}
                temp['label'] = sm[i]
                temp['probability'] = sm[i + 1]
                list_sm.append(temp)

            temp = {}
            temp['cam'] = qu[2]
            temp['content'] = ''
            temp['createTime'] = qu[3].strftime("%Y-%m-%d %H:%M:%S")
            temp['location'] = qu[4]
            temp['name'] = qu[5]
            temp['pred_result'] = qu[6]
            temp['shotImage'] = qu[7]
            temp['sourceImage'] = qu[8]
            temp['match_images'] = list_mi
            temp['softmax_prob'] = list_sm
            temp['match_times'] = list_mt
            history.append(temp)
        return length, history

    def query_by_id(self, nickname, ids):
        sql = 'SELECT * from query_record where id=(select id from (select (@i:=@i+1)pm,s.* from query_record s,(select @i:=0)t where nickName="%s" order by createTime) as f where f.pm= %d);'%(nickname, int(ids) + 1)
        status, result= self.db.search_db(sql)
        result = result[0]
        query_id = result[0]
        if not status:
            print('SEARCH BY ID ERROR')
        sql = 'SELECT * from softmax where id=%d'%(int(query_id))
        status, res_softmax = self.db.search_db(sql)
        res_softmax = res_softmax[0]
        if not status:
            print('SEARCH SOFTMAX ERROR')
        
        sql = 'SELECT * from match_time where id=%d'%(int(query_id))
        status, result_mt = self.db.search_db(sql)
        result_mt = result_mt[0]
        if not status:
            print('SEARCH match_time ERROR')

        sql = 'SELECT * from match_image where id=%d'%(int(query_id))
        status, result_mi = self.db.search_db(sql)
        result_mi = result_mi[0]
        if not status:
            print('SEARCH match_image ERROR')

        qu, sm, mt, mi = result, res_softmax, result_mt, result_mi
        list_mi = []
        for i in range(1, 21):
            list_mi.append(mi[i])
        dict_mt = {}
        dict_mt['label'] = mt[1]
        dict_mt['times'] = mt[2]
        list_mt = [dict_mt]

        list_sm = []
        
        for i in range(1, 11, 2):
            temp = {}
            temp['label'] = sm[i]
            temp['probability'] = sm[i + 1]
            list_sm.append(temp)

        temp = {}
        temp['cam'] = qu[2]
        temp['createTime'] = qu[3].strftime("%Y-%m-%d %H:%M:%S")
        temp['location'] = qu[4]
        temp['name'] = qu[5]
        temp['pred_result'] = qu[6]
        temp['shotImage'] = qu[7]
        temp['sourceImage'] = qu[8]
        temp['content'] = ''
        temp['match_images'] = list_mi
        temp['softmax_prob'] = list_sm
        temp['match_times'] = list_mt
        history = temp
        return history

if __name__ == "__main__":
    db = Database()
    # res = db.query_by_id('SHP',1)
    res = db.query('SHP')
    print(res)
    