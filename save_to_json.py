import json
from bs4 import BeautifulSoup

activity = open('data/MyActivity.html',mode='r',encoding='utf-8').read()
soup = BeautifulSoup(activity,'html.parser')

print(soup)

# soup.

# print(activity)
