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


class chat:
    def __init__(self):
        self.index_answer={}
        self.hanzi_index={}
        self.model={}
        self.t=0
        self.input_length=10	
        
        with open("all_answers",'r',encoding='utf-8') as f:
            lines=f.readlines() 
        self.index_answer=dict([line.strip().split("\t") for line in lines]) #字典的dic新增 用列表表達式    
        with open("hanzi_index",'r',encoding='utf-8') as f:
            self.hanzi_index=json.load(f)
        self.model=load_model("q_model.h5",custom_objects={'MultiHeadAttention':keras_multi_head.MultiHeadAttention},compile=False)
    
        f = 16
        self.t = AnnoyIndex(f, 'angular') 
        self.t.load("annoy_model")

    def normal(self,vector):
        ss=math.sqrt(sum([s*s for s in vector]))
        return [s/ss for s in vector]    
        
        
        
    def chat_reply(self,msg):
        sent=msg
        sent_index=np.array([[self.hanzi_index[s] for s in sent if s in self.hanzi_index ]])
        sent_index=keras.preprocessing.sequence.pad_sequences(sent_index, maxlen=self.input_length, value=0.,padding='post')#只能接受长度相等的序列输入
        vector=self.model.predict(sent_index).tolist()[0]
        vector=chat.normal(self,vector) #`类` 是模板，`对象` 是根据 `类` 这个模板创建出来的，应该先有类，再有对象    #實例方法 1.实例对象调用2.以类名称调用:類.方法(需傳遞實例參數self) 即實例化後的對象     
        result=self.t.get_nns_by_vector(vector,10, search_k=-1, include_distances=False)
        result=[i for i in result]
        result=random.sample(result,1)
       
        return self.index_answer[str(*result)]
    
    
if __name__ == '__main__':
    CHAT=chat() 
    print(CHAT.chat_reply('你是誰'))                    