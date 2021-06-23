# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 20:29:57 2021

@author: kaush
"""

import requests
import json
import pandas as pd

text_data=pd.read_excel('data.xlsx')['Text']

dictToSend = {}
for i in range(20):
    dictToSend['article_'+str(i)]=text_data[20+i]

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
url='http://localhost:5000/predict'
res = requests.post(url, json=dictToSend,headers=headers)
print('response from server')

print(res)
print(res.text)
