# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:37:53 2020

@author: 28369
"""

import json

names = ['joker','joe','nacy','timi']

filename='names.json'
with open(filename,'w') as file_obj:
    json.dump(names,file_obj) # 将names写入filename中


with open(filename) as file_obj:
    names = json.load(file_obj)
    
print(names)















