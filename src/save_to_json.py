import json
from bs4 import BeautifulSoup
from tqdm import tqdm

activity = open('../data/MyActivity.html',mode='r',encoding='utf-8').read()
print('read file')
soup = BeautifulSoup(activity,'lxml')
data = {}


for idx,entry in tqdm(enumerate(soup.find('div',class_='mdl-grid').find_all('div',class_='mdl-grid'))):
    # print(entry)
    data[idx] = {
        'title':entry.p.get_text(),
        'value':entry.a.get_text(),
        }

json.dump(data,open('../data/data.json','w'))