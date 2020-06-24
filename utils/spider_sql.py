__author__='SHP'
import urllib.request
import bs4, pickle
import re, os
import requests
import time, random
import sys
from .mysql import DB
sys.setrecursionlimit(1000000)
base_url = 'http://news.xiancn.com/'
# base_url = 'http://news.xiancn.com/node_2057.htm'
class Xian:
    def __init__(self, pkl_path='./data/database'):
        self.db = DB()
        if not os.path.exists(pkl_path):
            os.makedirs(pkl_path)
        self.pageIndex=1
        self.locked = False
        self.user_agent='Mozilla/4.0(compatible;MSIE 5.5;Windows NT)'
        self.headers={'User-Agent':self.user_agent}
        self.stories=[]
        self.enable=False
        self.check_update()

    def _check_status(self):
        while(self.locked):
            print('Lock Phenomenon Tiggered')
            time.sleep(0.001)
        self.locked = True

    def _release_lock(self):
        self.locked = False

    #传入某一页的索引获得页面代码
    def getPage(self,pageIndex):
        try:
            temp = ''
            if pageIndex != 1:
                temp = '_' + str(pageIndex)
            url = 'http://news.xiancn.com/node_2057' + temp + '.htm'
            request=urllib.request.Request(url,headers=self.headers)
            response=urllib.request.urlopen(request)
            #print response.read()
            pageCode=response.read().decode('utf-8')
            return pageCode
        except urllib.error.URLError as e:
            if hasattr(e,"reason"):
                print(u"连接服务器失败,错误原因:",e.reason)
                return None

    def getPageItems(self, pageIndex):
        time.sleep(0.1 + random.random())
        pageCode=self.getPage(pageIndex)
        if not pageCode:
            print ("页面加载失败")
            return None
        soup = bs4.BeautifulSoup(pageCode,"html.parser")
        items = soup.select('a')
        news_items = []
        page_items = []
        for item in items:
            if 'class' not in item.attrs.keys() and 'target' in item.attrs.keys():
                news_items.append(item)
            if 'target' not in item.attrs.keys() and 'class' not in item.attrs.keys():
                page_items.append(item)
        title = [str(i) for index, i in enumerate(news_items) if index %2 != 0]
        cover = [i for index, i in enumerate(news_items) if index %2 == 0]
        link = [base_url + i.attrs['href'] for i in cover]
        # print('DEBUG')
        coverImage = []
        for i in cover:
            item = i.select('img')
            if len(item) > 0:
                coverImage.append(base_url + item[0].attrs['src'])
            else:
                coverImage.append('https://ss0.bdstatic.com/94oJfD_bAAcT8t7mm9GUKT-xh_/timg?image&quality=100&size=b4000_4000&sec=1592854311&di=0645d8d4c9001ea5a10b18441c0fe93f&src=http://5b0988e595225.cdn.sohucs.com/images/20180828/43bbc000b5ff4222b7e83aff69410805.jpeg')
        # coverImage = [base_url + i.select('img')[0].attrs['src'] for i in cover]
        title_word = []
        pattern=re.compile('"_blank">(.*?)</a>',re.S)
        for t in title:
            title_word.append(re.findall(pattern,t)[0])
        #存储每页的新闻
        pageStories=[]
        for _title, _cover, _link in zip(title_word, coverImage, link):
            pageStories.append([_title, _cover, _link])
        return pageStories

    def download_img(self, img_url, save_folder='./'):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        header = self.headers
        r = requests.get(img_url, headers=header, stream=True)
        print(r.status_code) # 返回状态码
        if r.status_code == 200:
            image_name = os.path.split(img_url)[-1]
            if not os.path.exists(os.path.join(save_folder, image_name)):
                open(os.path.join(save_folder, image_name), 'wb').write(r.content) # 将内容写入图片
                print("Download %s"%(image_name))
            del r
            return image_name
        else:
            del r
            return None

    def download_h5(self, link_url, news_folder='./news'):
        if not os.path.exists(os.path.join(news_folder, 'image')):
            os.makedirs(os.path.join(news_folder, 'image'))
        h5_name = link_url.split('/')[-1].split('.')[0] + '.md'
        h5_path = os.path.join(news_folder, h5_name)
        try:
            url = link_url
            news_base_url = 'http://news.xiancn.com'
            request=urllib.request.Request(url,headers=self.headers)
            response=urllib.request.urlopen(request)
            pageCode=response.read().decode('utf-8')
            soup = bs4.BeautifulSoup(pageCode,"html.parser")
            update_items = soup.find_all('div', {'id':'maintt'})
            total_title = update_items[0].find_all('div',{'class':'biaoti'})[0].contents[0]
            sub_title = update_items[0].find_all('div',{'class':'mess'})[0]
            update_time = sub_title.contents[0][3:19]#截取发布时间
            upload_web = sub_title.find('a')
            try:
                update_webpage = upload_web.contents[0]
            except:
                update_webpage = sub_title.contents[0][26:]
            timeArray = time.strptime(update_time, '%Y-%m-%d %H:%M')
            timestamp = time.mktime(timeArray)
            items = soup.find_all('div', {'id':'content'})
            all_image = []
            images = items[0].find_all('img')
            for ima in images:
                image_realurl = news_base_url + ima.attrs['src'].split('..')[-1]
                image_path = os.path.join(news_folder, 'image')
                image_name = self.download_img(image_realurl, save_folder=image_path)
                ima.attrs['src'] = os.path.join('image', image_name)
                all_image.append(os.path.join(image_path, image_name))
            all_news = []
            all_news.append(items[0].prettify())
            number_page = 1
            while len(items[0].find_all('div', {'id':'displaypagenum'})) != 0:
                number_page += 1
                url = os.path.split(link_url)[0] + '/' + os.path.split(link_url)[-1].split('.')[0] + '_' + str(number_page) + '.htm'
                request=urllib.request.Request(url,headers=self.headers)
                response=urllib.request.urlopen(request)
                pageCode=response.read().decode('utf-8')
                soup = bs4.BeautifulSoup(pageCode,"html.parser")
                items = soup.find_all('div', {'id':'content'})
                images = items[0].find_all('img')
                for ima in images:
                    image_realurl = news_base_url + ima.attrs['src'].split('..')[-1]
                    image_path = os.path.join(news_folder, 'image')
                    image_name = self.download_img(image_realurl, save_folder=image_path)
                    ima.attrs['src'] = os.path.join('image', image_name)
                    all_image.append(os.path.join(image_path, image_name))
                all_news.append(items[0].prettify())
                pages = items[0].find_all('div', {'id':'displaypagenum'})[0]
                pagenum = pages.find_all('a')
                if pagenum[-1].string != '尾页':
                    break
            content = '\n'.join(all_news)
            with open(h5_path, 'w', encoding='UTF-8') as f:
                f.write(content)

            # print('jump successful')
        except urllib.error.URLError as e:
            if hasattr(e,"reason"):
                print(u"连接服务器失败,错误原因:",e.reason)
        

        return h5_name, timestamp, update_webpage, total_title
        

    #加载并提取页面的内容,加入到列表
    def loadPage(self):
        #如果当前未看的页数少于2页,则加载下一页
        if self.enable==True:
            if len(self.stories)<2:
                pageStories=self.getPageItems(self.pageIndex)
                if len(pageStories) > 0:
                    self.stories.append(pageStories)
                    self.pageIndex+=1

    def getOneStory(self,pageStories,page):
        for index, story in enumerate(pageStories):
            # if index == 0:
            #     continue
            time.sleep(0.1 + random.random())
            self.loadPage()
            sql = 'select title from news;'
            status, res = self.db.search_db(sql)
            res = [i[0] for i in res]#[('A',), ('B',), ('C',)]-->['A','B','C']
            
            if story[0] not in res:
                print('Find new information, update it!')
                save_folder=os.path.join('./uploads/images/')
                news_folder= os.path.join('./uploads/news/')
                image_name = self.download_img(story[1], save_folder=save_folder)
                h5_name, timestamp, webpage, total_title = self.download_h5(story[2], news_folder)
                t = time.localtime(timestamp)
                t = time.strftime("%Y-%m-%d %H:%M", t)
                sql = '''INSERT INTO news(cover, link, time, title, total_title, webpage) VALUES
                ("%s", "%s", "%s", "%s", "%s", "%s")'''%(
                    os.path.join(save_folder, image_name), os.path.join(news_folder, h5_name),
                    t, story[0], total_title, webpage)
                status = self.db.fix_db(sql)
                if not status:
                    print('INSERT NEWS ERROR')
            else:
                print('News has been checked!')
                self.enable=False
                return
            if page > 5:
                # print('DEBUG')
                self.enable=False
                return


    def check_update(self):
        print (u"检查更新")
        self.enable=True
        self.pageIndex=1
        self.stories = []
        self.loadPage()
        newPage=0
        try: #防止网站被屏蔽，无法获取更新,同时防止锁一直被占用，无法更新
            while self.enable:
                if len(self.stories)>0:
                    pageStories=self.stories[0]
                    newPage+=1
                    #将全局list中第一个元素删除,因为已经取出
                    del self.stories[0]
                    self.getOneStory(pageStories,newPage)
        except:
            print('网站限制')
            pass
        self.pageIndex=1
        self.stories = []
        return
        

    def query_by_page(self, pageNum, pageSize):
        sql = 'SELECT count(*) from news;'
        status, query_result = self.db.search_db(sql)
        length = query_result[0][0]
        start_index = min(length, pageNum * pageSize)
        sql = 'select * from news order by time desc limit %d, %d;'%(start_index, pageSize)
        status, result = self.db.search_db(sql)
        if not status:
            print('QUERY BY PAGE ERROR!')
        query_result = []
        for item in result:
            temp = {}
            temp['title'] = item[4]
            temp['total_title'] = item[5]
            temp['webpage'] = item[6]
            temp['cover'] = item[1]
            temp['timestamp'] = item[3].strftime("%Y-%m-%d %H:%M")
            temp['link'] = item[2]
            query_result.append(temp)
        return query_result, length

    def query_by_key_word(self, key_word):

        # sql = 'select * from news where title like "%%%s%%" order by time desc;'%(key_word)
        
        sql ='select * from (select (@i:=@i+1)pm, s.* from news as s ,(select @i:=0)t order by time desc) as f where f.title like "%%%s%%";'%(key_word)
        status, result = self.db.search_db(sql)
        if not status:
            print('QUERY BY WORD ERROR!')
        query_result = []
        query_index = []
        for item in result:
            temp = {}
            temp['title'] = item[5]
            temp['total_title'] = item[6]
            temp['webpage'] = item[7]
            temp['cover'] = item[2]
            temp['timestamp'] = item[4].strftime("%Y-%m-%d %H:%M")
            temp['link'] = item[3]
            query_index.append(item[0] - 1)
            query_result.append(temp)
        return query_result, query_index
    
    def query_detail(self, index, baseurl):
        # sql = 'SELECT * from news where id=%d order by time desc'%(index)
        sql ='SELECT * from news where id=(select id from (select (@i:=@i+1)pm,s.* from news s,(select @i:=0)t order by time desc) as f where f.pm=%d);'%(index)
        status, result = self.db.search_db(sql)
        query_result = {}

        if not status:
            print('QUERY BY DEATIL ERROR!')
        if len(result) != 0:
            item = result[0]
            temp = {}
            temp['title'] = item[4]
            temp['total_title'] = item[5]
            temp['webpage'] = item[6]
            temp['cover'] = item[1]
            temp['timestamp'] = item[3].strftime("%Y-%m-%d %H:%M")
            temp['link'] = item[2]
            query_result = temp
        with open(query_result['link'].split('../')[-1], 'r', encoding='UTF-8') as f:
            html = f.read()
        soup = bs4.BeautifulSoup(html, "html.parser")
        items = soup.find_all('div', {'id':'content'})
        # soup.decompose()
        for index, item in enumerate(items):
            align = item.find_all('p')
            for i in align:
                if 'align' in i.attrs.keys():
                    i.attrs['align'] = 'left'
            images = item.find_all('img')
            for ima in images:
                image_name = ima.attrs['src'].split('\\')[-1]
                # print(image_name)
                image_path = baseurl + 'uploads/news/image/' + image_name
                ima.attrs['src'] = image_path
                if 'align' in ima.attrs.keys():
                    ima.attrs['align'] = 'center'
                if 'height' in ima.attrs.keys():
                    ima.attrs['height'] = 'auto'
                if 'style' in ima.attrs.keys():
                    ima.attrs['style'] = "max-width:100%;height:auto;display:block"
            if index != len(items) - 1: #删除多余编辑
                editor = item.find_all('div', {'class':'text'})[0]
                editor.contents = ''
            try:
                pagenum = item.find_all('div', {'id':'displaypagenum'})[0]
                pagenum.contents = ''
            except:
                pass
        if 'total_title' not in query_result.keys():
            query_result['total_title'] = query_result['title']
        query_result['timestamp'] = query_result['timestamp']
        query_result['link'] = soup.prettify()
        return query_result

if __name__ == '__main__':
    spider=Xian('./dataset')
    # spider.check_update()
    a = spider.query_by_page(0, 10)
    print(a)
    print('Update successfully')

