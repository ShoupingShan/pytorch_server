from mysql import *
import pickle
import os
import time

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class Pickle2SQL:
    def __init__(self):
        self.db = DB()

    def build_feedback(self, data):
        status = self.db.create_table('''CREATE TABLE feedback(
            id int PRIMARY KEY auto_increment, 
            time timestamp, 
            imageName VARCHAR(40),
            imagePath VARCHAR(80), 
            prediction VARCHAR(40),
            userId VARCHAR(20),
            userFeedback TEXT);''')
        if status:
            for index, item in enumerate(data):
                t = time.localtime(item['feedbackTime'])
                t = time.strftime("%Y-%m-%d %H:%M:%S", t)
                sql = '''INSERT INTO feedback(time, imageName, imagePath, prediction, userId, userFeedback) VALUES ('%s', '%s', '%s', '%s', '%s', '%s');''' % (t, item['imageName'], item['imagePath'], item['prediction'], item['userId'], item['user_feedback'])
                status = self.db.fix_db(sql)
                if not status:
                    print('INSERT ERROR')

    def build_guest_record(self, data):
        status = self.db.create_table('''CREATE TABLE guest_record(
            id int PRIMARY KEY auto_increment, 
            detail VARCHAR(20), 
            time timestamp,
            userId VARCHAR(20));
            ''')
        if status:
            for index, item in enumerate(data):
                t = time.localtime(item['time'])
                t = time.strftime("%Y-%m-%d %H:%M:%S", t)
                sql = '''INSERT INTO guest_record(detail, time, userId) VALUES ('%s', '%s', '%s');''' % (item['detail'], t, item['userId'])
                status = self.db.fix_db(sql)
                if not status:
                    print('INSERT GUEST ERROR')

    def build_news(self, data):
        status = self.db.create_table('''CREATE TABLE news(
            id int PRIMARY KEY auto_increment, 
            cover VARCHAR(80),
            link VARCHAR(80), 
            time timestamp,
            title VARCHAR(40),
            total_title VARCHAR(80),
            webpage VARCHAR(20)
            );
            ''')
        if status:
            for key, item in data['content'].items():
                t = time.localtime(item['timestamp'])
                t = time.strftime("%Y-%m-%d %H:%M:%S", t)
                cover = item['cover'].replace('\\','/')
                link = item['link'].replace('\\','/')
                sql = '''INSERT INTO news(cover, link, time, title, total_title, webpage) VALUES ('%s', '%s', '%s', '%s', '%s', '%s');''' % (cover, link, t, item['title'], item['total_title'], item['webpage'])
                status = self.db.fix_db(sql)
                if not status:
                    print('INSERT NEWS ERROR')

    def build_users(self):
        status = self.db.create_table('''CREATE TABLE users(
            id int PRIMARY KEY auto_increment, 
            nickName VARCHAR(20)
            );
            ''')

    def insert_users(self, item):
        sql = '''INSERT INTO users(nickName) VALUES ('%s');''' % (item)
        status = self.db.fix_db(sql)
        if not status:
            print('INSERT USERS ERROR')
    
    def build_match_image(self):
        status =  self.db.create_table('''CREATE TABLE match_image(
            id int PRIMARY KEY auto_increment,
            top1 VARCHAR(80),
            top2 VARCHAR(80),
            top3 VARCHAR(80),
            top4 VARCHAR(80),
            top5 VARCHAR(80),
            top6 VARCHAR(80),
            top7 VARCHAR(80),
            top8 VARCHAR(80),
            top9 VARCHAR(80),
            top10 VARCHAR(80),
            top11 VARCHAR(80),
            top12 VARCHAR(80),
            top13 VARCHAR(80),
            top14 VARCHAR(80),
            top15 VARCHAR(80),
            top16 VARCHAR(80),
            top17 VARCHAR(80),
            top18 VARCHAR(80),
            top19 VARCHAR(80),
            top20 VARCHAR(80)
            );
            ''')

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

    def build_softmax(self):
        status =  self.db.create_table('''CREATE TABLE softmax(
            id int PRIMARY KEY auto_increment,
            label_top1 VARCHAR(40),
            probability_top1 VARCHAR(10),
            label_top2 VARCHAR(40),
            probability_top2 VARCHAR(10),
            label_top3 VARCHAR(40),
            probability_top3 VARCHAR(10),
            label_top4 VARCHAR(40),
            probability_top4 VARCHAR(10),
            label_top5 VARCHAR(40),
            probability_top5 VARCHAR(10)
            );
            ''')

    def insert_softmax(self, data):
        # sql = '''INSERT INTO softmax(top1, top2, top3, top4, top5) VALUES (%d, %d, %d, %d, %d);''' % (index, index + 1, index + 2, index + 3, index + 4)
        sql = '''INSERT INTO softmax(label_top1, probability_top1, label_top2, probability_top2,
        label_top3, probability_top3, label_top4, probability_top4, label_top5, probability_top5) VALUES 
        ('%s', '%s', '%s', '%s','%s', '%s','%s', '%s','%s', '%s');''' % (data[0]['label'], data[0]['probability'], data[1]['label'], data[1]['probability'], data[2]['label'], data[2]['probability'], data[3]['label'], data[3]['probability'], data[4]['label'], data[4]['probability'])
        status = self.db.fix_db(sql)
        if not status:
            print('INSERT SOFTMAX ERROR')

    def build_match_time(self):
        status =  self.db.create_table('''CREATE TABLE match_time(
            id int PRIMARY KEY auto_increment,
            label VARCHAR(40),
            times VARCHAR(2)
            );
            ''')
    
    def insert_match_times(self, data):
        sql = '''INSERT INTO match_time(label, times) VALUES ('%s', '%s');''' % (data['label'], data['times'])
        status = self.db.fix_db(sql)
        if not status:
            print('INSERT MATCH TIMES ERROR')
    
    def build_query_record(self):
        self.build_match_image()
        self.build_softmax()
        self.build_match_time()
        self.build_users()
        status = self.db.create_table('''CREATE TABLE query_record(
            id int PRIMARY KEY auto_increment, 
            nickName VARCHAR(20),
            cam VARCHAR(80),
            createTime TIMESTAMP DEFAULT NULL,
            location VARCHAR(150),
            imageName VARCHAR(25),
            pred_result VARCHAR(40),
            shotImage VARCHAR(80),
            sourceImage VARCHAR(80)
            );
            ''')
        return status

    def insert_query_resord(self, data):
        for name, user in data.items():
            self.insert_users(name)
            for query in user['query_list']:
                self.insert_match_times(query['match_times'][0])
                self.insert_match_image(query['match_images'])
                self.insert_softmax(query['softmax_prob'])
                sql = '''
                INSERT INTO query_record(nickName, cam, createTime, location, 
                imageName, pred_result, shotImage, sourceImage) VALUES (
                    "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s"
                );
                ''' % (name, query['cam'], query['createTime'], query['location'], 
                query['name'], query['pred_result'], query['shotImage'], query['sourceImage'])
                status = self.db.fix_db(sql)
                if not status:
                    print('INSERT QUERY RECORD ERROR')

    def select(self, sql_name):
        sql = '''SELECT * from %s'''%(sql_name)
        status, result = self.db.search_db(sql)
        # print(result)
            



if __name__ == "__main__":
    feedback = os.path.join(os.getcwd(), 'data/database/Feedback.pkl')
    guest_record = os.path.join(os.getcwd(), 'data/database/guest_records.pkl')
    news = os.path.join(os.getcwd(), 'data/database/News.pkl')
    query_records = os.path.join(os.getcwd(), 'data/database/query_records.pkl')
    feedback = load_pickle(feedback)
    guest_record = load_pickle(guest_record)
    news = load_pickle(news)
    query_records = load_pickle(query_records)
    SQL = Pickle2SQL()
    SQL.build_feedback(feedback)
    SQL.select('feedback')
    SQL.build_guest_record(guest_record)
    SQL.select('guest_record')
    SQL.build_news(news)
    SQL.select('news')
    SQL.build_query_record()
    SQL.insert_query_resord(query_records)
    SQL.select('query_record')