import urllib.request
import bs4, pickle
import re, os
import requests
import time, random

class Baidu:
    def __init__(self, base_url='https://baike.baidu.com/item/'):
        self.user_agent='Mozilla/4.0(compatible;MSIE 5.5;Windows NT)'
        self.headers={'User-Agent':self.user_agent}
        self.base_url = base_url
        self.search_name = ''
        self.url = ''
    def getPage(self, search_name):
        # self.search_name = str(search_name.encode('utf-8').decode(encoding='ascii'))
        self.search_name = search_name
        try:
            self.url = 'https://baike.baidu.com/item/' + self.search_name
            request=requests.get(self.url,headers=self.headers)
            response= request.content
            pageCode=response.decode('utf-8')
            return pageCode
        except urllib.error.URLError as e:
            if hasattr(e,"reason"):
                print(u"连接服务器失败,错误原因:",e.reason)
            return None

    def parse_html(self, html):

        try:
            pageCode= html
            with open('test.html', 'w', encoding='UTF-8') as f:
                f.write(pageCode)
            soup = bs4.BeautifulSoup(pageCode,"html.parser")
            content = soup.find('div', {'class':'main-content'})
            album = content.find('div', {'class': 'top-tool'})
            if album is not None:
                album.contents = ''
            album = content.find('a', {'class': 'edit-lemma cmn-btn-hover-blue cmn-btn-28 j-edit-link'})
            if album is not None:
                album.contents = ''
            album = content.find('a', {'class': 'lemma-discussion cmn-btn-hover-blue cmn-btn-28 j-discussion-link'})
            if album is not None:
                album.contents = ''
            album = content.find('a', {'class': 'lock-lemma'})
            if album is not None:
                album.contents = ''
           
            
            album = content.find('div', {'class': 'album-list'})
            if album is not None:
                album.contents = ''
            album = content.find('div', {'class': 'clearfix'})
            if album is not None:
                album.contents = ''
            album = content.find_all('div', {'class': 'rs-container-foot'})
            for item in album:
                item.contents = ''
            album = content.find_all('script')
            for item in album:
                item.contents = ''
            album = content.find_all('div', {'class': 'rs-container-top'})
            for item in album:
                item.contents = ''
            
            album = content.find_all('div', {'class': 'main_tab main_tab-diyTab-1'})
            for item in album:
                item.contents = ''
            tashuo = content.find('div', {'class': 'tashuo-bottom'})
            if tashuo is not None:
                tashuo.contents = ''
            paper = content.find('div', {'class': 'paper-title'})
            if paper is not None:
                paper.contents = ''
            edit = content.find_all('a', {'class': 'edit-icon j-edit-link'})
            for item in edit:
                item.contents = ''
            save_path = os.path.join('HTML', self.search_name + '.md')
            self.save_md(save_path, content.prettify())
        except urllib.error.URLError as e:
            if hasattr(e,"reason"):
                print(u"连接服务器失败,错误原因:",e.reason)
        return 0

    def save_md(self, save_path, html):
        folder = os.path.split(save_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(save_path, 'w', encoding='UTF-8') as f:
            f.write(html)

if __name__ == '__main__':
    Spider = Baidu()
    from tqdm import tqdm
    # with open('./test.html', 'r', encoding='utf-8') as f:
    #         page = f.read()
    # # page = Spider.getPage('凉皮')
    # html = Spider.parse_html(page)


    html_folder = os.path.join('./baike_origin')
    for item in tqdm(os.listdir(html_folder)):
        page_path = os.path.join(html_folder, item)
        Spider.search_name = item[:-5]
        with open(page_path, 'r', encoding='utf-8') as f:
            page = f.read()
        html = Spider.parse_html(page)