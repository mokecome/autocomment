# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:36:57 2021

@author: Bill
"""

import sys
import json
import pickle
from keras.models import Model
from keras.layers import Input, Dense,Layer,TimeDistributed,Conv1D
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
import keras
import keras_multi_head
from keras_multi_head import MultiHeadAttention
from functools import partial, update_wrapper
import tensorflow as tf
from keras.models import load_model
import numpy as np
import tensorflow as tf
import keras 
from keras.callbacks import ModelCheckpoint
import random
from keras.layers.merge import dot

import math
import annoy
from annoy import AnnoyIndex
import pandas as pd
#提前計算user-item_list(要有) 但是新用戶沒有數據  
#線上實時計算 建樹據類-相似區域

class chat:
    def __init__(self):
        self.index_answer={}
        self.hanzi_index={}
        self.model={}
        self.t=0
        self.input_length=10	
        
        ans=pd.read_csv('all_answers', encoding='utf-8',delim_whitespace=True,names=['n','a'])
        ans=ans.dropna().reset_index(drop=True)#去除不完整數據
        self.index_answer=dict(zip(ans['n'],ans['a'])) 
        
        with open("hanzi_index",'r',encoding='utf-8') as f:
            self.hanzi_index=json.load(f)
        self.model=load_model("q_model.h5",custom_objects={'MultiHeadAttention':keras_multi_head.MultiHeadAttention},compile=False)
    
        f = 16
        self.t = AnnoyIndex(f, 'angular') 
        self.t.load("annoy_model")
        
    def normal(vector):
        ss=math.sqrt(sum([s*s for s in vector]))
        return [s/ss for s in vector]    
        
        
        
    def chat_reply(self,msg):
        sent=msg
        sent_index=np.array([[self.hanzi_index[s] for s in sent if s in self.hanzi_index ]])
        sent_index=keras.preprocessing.sequence.pad_sequences(sent_index, maxlen=self.input_length, value=0.,padding='post')#只能接受长度相等的序列输入
        vector=self.model.predict(sent_index).tolist()[0] #新句是舊的組成
        vector=chat.normal(vector)
        result=self.t.get_nns_by_vector(vector,10, search_k=-1, include_distances=False)
        result=[i for i in result]
        result=random.sample(result,1)
       
        return self.index_answer[str(*result)]
    
    
if __name__ == '__main__':
    CHAT=chat() 
    print(CHAT.chat_reply('请问鱼的耳朵'))             


'''
召回:多  快

向量計算
在線:對新用戶友好 ----annoy檢索logn
離線:速度快--------- 事先建表

'''       