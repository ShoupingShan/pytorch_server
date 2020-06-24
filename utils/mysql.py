import pymysql
class DB:
    #连接数据库，并返回连接到的数据库对象
    def con_db(self,hostname='127.0.0.1',username='root',password='Shp39521',database='xian',port=3306, charset='utf8'):#默认参数
        db = pymysql.connect(hostname,username,password,database,port)
        return db

    #查询数据库
    def search_db(self,sql):
        db = self.con_db()
        cu = db.cursor()#得到一个游标
        result = []
        success = True
        try:
            cu.execute(sql)#通过游标执行sql语句
            result = cu.fetchall()#获取sql执行结果
            db.commit()#提交数据库
        except:
            db.rollback()
            success = False
        db.close()#关闭数据库连接对象
        return success, list(result)


    #增删改数据库
    def fix_db(self,sql):
        db = self.con_db()
        cu = db.cursor()  # 得到一个游标
        success = True
        try:
            cu.execute(sql)  # 通过游标执行sql语句
            db.commit()  # 提交数据库
        except:
            db.rollback()
            success = False
        db.close()  # 关闭数据库连接对象
        return success
    
    #创建数据表
    def create_table(self, sql):
        db = self.con_db()
        cu = db.cursor()
        success = True
        try:
            cu.execute(sql)
            db.commit()  # 提交数据库
        except:
            db.rollback()
            success = False
        db.close()  # 关闭数据库连接对象
        return success

if __name__ == '__main__':
    d =DB()
    status, res = d.search_db('SELECT * from query_record where id=(select id from (select (@i:=@i+1)pm,s.* from query_record s,(select @i:=0)t where nickName="SHP" order by createTime desc) as f where f.pm=1);')
    print(res[0][0] == None)