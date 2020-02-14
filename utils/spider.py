__author__='SHP'
import urllib.request
import bs4, pickle
import re, os
import requests
import time, random
base_url = 'http://news.xiancn.com/'
class Xian:
    def __init__(self, pkl_path='./data/database'):
        if not os.path.exists(pkl_path):
            os.makedirs(pkl_path)
        if not os.path.exists(os.path.join(pkl_path, 'News.pkl')):
            with open(os.path.join(pkl_path, 'News.pkl'), 'wb') as f:
                content = {}
                content['map'] = {}
                content['content'] = {}
                pickle.dump(content, f)
        self.DB_path = os.path.join(pkl_path, 'News.pkl')
        with open(self.DB_path, 'rb') as f:
            self.data = pickle.load(f)
        self.content = self.data['content']
        self.map = self.data['map']
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
        coverImage = [base_url + i.select('img')[0].attrs['src'] for i in cover]
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
            sub_title = update_items[0].find_all('div',{'class':'mess'})[0]
            update_time = sub_title.contents[0][3:19]#截取发布时间
            upload_web = sub_title.find('a')
            update_webpage = upload_web.contents[0]
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
        

        return h5_name, timestamp, update_webpage
        

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
            time.sleep(0.01 + random.random())
            self.loadPage()
            if story[0] not in self.content.keys():
                print('Find new information, update it!')
                self.content[story[0]] = {}
                save_folder=os.path.join('./uploads/images')
                news_folder= os.path.join('./uploads/news')
                image_name = self.download_img(story[1], save_folder=save_folder)
                h5_name, timestamp, webpage = self.download_h5(story[2], news_folder)
                self.content[story[0]]['title'] = story[0]
                self.content[story[0]]['cover'] = os.path.join(save_folder, image_name)
                self.content[story[0]]['link'] = os.path.join(news_folder, h5_name)
                self.content[story[0]]['timestamp'] = timestamp #新闻更新时间
                self.content[story[0]]['webpage'] = webpage #新闻更新时间
                self.map[story[0]] = timestamp
            else:
                print('News has been checked!')
                self.enable=False
                return
            if page > 10:
                self.enable=False
                return


    def check_update(self):
        self._check_status()
        print (u"检查更新")
        self.enable=True
        with open(self.DB_path, 'rb') as f:
            self.data = pickle.load(f)
        self.content = self.data['content']
        self.map = self.data['map']
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
            pass
        self.pageIndex=1
        self.stories = []
        with open(self.DB_path, 'wb') as f:
            data = {}
            data['content'] = self.content
            data['map'] = self.map
            pickle.dump(data, f)
            print('Database saved')
        self._release_lock()
        

    def query_by_page(self, pageNum, pageSize):
        self._check_status()
        with open(self.DB_path, 'rb') as f:
            self.data = pickle.load(f)
        self.content = self.data['content']
        self.map = self.data['map']
        refer = sorted(self.map.items(),key = lambda x:x[1],reverse = True)
        start_index = pageNum * pageSize
        end_index = min(len(refer), (pageNum + 1) * pageSize)
        news_titles = [i[0] for i in refer[start_index: end_index]]
        query_result = [self.content[i] for i in news_titles]
        for item in query_result:
            if 'webpage' not in item.keys():
                item['webpage'] = '未记录'
           
            if type(item['timestamp']) is str:
                pass
            else:
                local_time =  time.localtime(float(item['timestamp']))
                date = time.strftime('%Y-%m-%d %H:%M', local_time)
                item['timestamp'] = date
        self._release_lock()
        return query_result, len(refer)

    def query_by_key_word(self, key_word):
        self._check_status()
        with open(self.DB_path, 'rb') as f:
            self.data = pickle.load(f)
        self.content = self.data['content']
        self.map = self.data['map']
        refer = sorted(self.map.items(),key = lambda x:x[1],reverse = True)
        query_tuple = [(self.content[i[0]], index) for index, i in enumerate(refer) if key_word in i[0]]
        query_index = [i[1] for i in query_tuple]
        query_result = [i[0] for i in query_tuple]
        for item in query_result:
            if 'webpage' not in item.keys():
                item['webpage'] = '未记录'
            if type(item['timestamp']) is str:
                pass
            else:
                local_time =  time.localtime(float(item['timestamp']))
                date = time.strftime('%Y-%m-%d %H:%M', local_time)
                item['timestamp'] = date
        self._release_lock()
        return query_result, query_index
    
    def query_detail(self, index, baseurl):
        self._check_status()
        with open(self.DB_path, 'rb') as f:
            self.data = pickle.load(f)
        self.content = self.data['content']
        self.map = self.data['map']
        refer = sorted(self.map.items(),key = lambda x:x[1],reverse = True)
        query_result = self.content[refer[index][0]]
        with open(query_result['link'].split('../')[-1], 'r', encoding='UTF-8') as f:
            html = f.read()
        # print(html)
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
                image_path = baseurl + 'uploads/news/image/' + image_name
                ima.attrs['src'] = image_path
            if index != len(items) - 1: #删除多余编辑
                editor = item.find_all('div', {'class':'text'})[0]
                editor.contents = ''
            try:
                pagenum = item.find_all('div', {'id':'displaypagenum'})[0]
                pagenum.contents = ''
            except:
                pass
        local_time =  time.localtime(float(query_result['timestamp']))
        date = time.strftime('%Y-%m-%d %H:%M', local_time)
        query_result['timestamp'] = date
        query_result['link'] = soup.prettify()
        if 'webpage' not in query_result.keys():
                query_result['webpage'] = '未记录'
        self._release_lock()
        return query_result

if __name__ == '__main__':
    spider=Xian('./dataset')
    spider.check_update()
    print('Update successfully')
    a = spider.query_by_page(0, 10)
    print(a)
