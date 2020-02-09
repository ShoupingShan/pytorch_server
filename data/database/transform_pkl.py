import pickle
if __name__ == '__main__':
    a = pickle.load(open('./query_records.pkl', 'rb'))
    b = a['SHP']['query_list']
    for item in b:
        shotImage = item['shotImage']
        item['shotImage'] = shotImage.replace('http://192.168.1.102:80/source', 'https://hunshuimoyu.picp.vip/uploads/small')
        match_images = item['match_images']
        for index, ima in enumerate(match_images):
            match_images[index] = ima.replace('http://192.168.1.102:80', 'https://hunshuimoyu.picp.vip/uploads')
        cam = item['cam']
        item['cam'] = cam.replace('http://192.168.1.102:80', 'https://hunshuimoyu.picp.vip/uploads')
    a['SHP']['query_list'] = b
    with open('query_records2.pkl', 'wb') as f:
        pickle.dump(a, f)