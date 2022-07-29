#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import tensorflow as tf
import scipy.misc
import os
import csv
import itertools
from collections import deque
import pandas as pd
from operator import itemgetter


# In[2]:


# どの表を使用するかを決定
global use_color, use_size, use_shape, use_taste, use_hardness, use_move
use_color = True
use_shape = True
use_taste = True
use_move = True
use_verb = True

# # 強化学習DRQN
# ## priority experience memoryの作成
# ### 優先度をつけることにより効率的に学習してほしい
# ### シンプルな手法を使用する
# ### Prioritized Experience Replay    https://arxiv.org/pdf/1511.05952.pdf
# ### 問題と名詞・形容詞の出現率のランダム性はseed値で固定 

class Database():
    def __init__(self):
        global use_color, use_size, use_shape, use_taste, use_hardness, use_move
        self.data = pd.read_csv("./table/main_new.csv", index_col=0)
        self.verb = pd.read_csv("./table/verb.csv", index_col=0)
        self.request = []
        self.object = self.data.index
        self.feature = self.data.columns
        self.feature_length = 0
        self.object_length = len(self.object)
        self.action = []
        self.name = list(dict.fromkeys(self.data['name']))
        self.name_noun = list(dict.fromkeys(self.data['name']))
        use_data = self.data.query('deleted == "f"')
        self.use_object = use_data.index

        if(use_color):
            self.color = pd.read_csv("./table/new_color_histgram.csv", index_col=0)
            self.color_feature = self.color.columns
            self.color_size = len(self.color.columns)
            self.feature_length += self.color_size
            self.action.append('color')
        else:
            self.color_size = 0
        if(use_shape):
            self.shape = pd.read_csv("./table/new_shape.csv", index_col=0)
            self.shape_feature = self.shape.columns
            self.shape_size = len(self.shape.columns)
            self.feature_length += self.shape_size
            self.action.append('shape')
        else:
            self.shape_size = 0
        if(use_taste):
            self.taste = pd.read_csv("./table/new_taste.csv", index_col=0)
            self.taste_feature = self.taste.columns
            self.taste_size = len(self.taste.columns)
            self.feature_length += self.taste_size
            self.action.append('taste')
        else:
            self.taste_size = 0
        if(use_move):
            self.move = pd.read_csv("./table/new_move_k_medoids_100.csv", index_col=0)
            self.move_feature = self.move.columns
            self.move_size = len(self.move.columns)
            self.feature_length += self.move_size
            self.action.append('move')
        else:
            self.move_size = 0
        if(use_verb):
            verb_list = pd.read_csv("./table/verb.csv", index_col=0)
            self.name_verb = list(dict.fromkeys(verb_list['name']))
            for data in self.name_verb:
                self.name.append(data)
            
        self.name.append('not_sure')
        self.name_noun.append('not_sure')
        self.name_verb.append('not_sure')
        #print(self.name)
        #print(self.name_verb)
    
    # ここで特徴を取得している
    def Overfetch(self,object_number,request,parent_order):
        """
        if request not in self.request:
            self.request.append(request)
        """
        base =  np.zeros(self.feature_length)
        if parent_order == 0:
            get_number = random.randint(object_number, object_number+2)
            if get_number >= len(self.data):
                return base
            objFeatures = self.data.loc[get_number]
        else:
            name = self.name[object_number]
            bool_list = self.verb["name"] == name
            get_number = random.choice(self.verb[bool_list].index.values)
            if get_number >= len(self.data):
                return base
            objFeatures = self.data.loc[get_number]
        #objFeatures = self.data.loc[object_number]
        #base =  np.zeros(self.feature_length)
        k1 = self.color_size
        k2 = k1 + self.shape_size
        k3 = k2 + self.taste_size
        k4 = k3 + self.move_size
        if request == 'color':
                base[0:k1] = objFeatures[2:2+k1]
        elif request == 'shape':
                base[k1:k2]  = objFeatures[2+k1:2+k2]
        elif request == 'taste':
                base[k2:k3] = objFeatures[2+k2:2+k3]
        elif request == 'move':
                base[k3:k4] = objFeatures[2+k3:2+k4]
        """
        for req in self.request:
            if req == 'color':
                base[0:k1] = objFeatures[2:2+k1]
            elif req == 'shape':
                base[k1:k2]  = objFeatures[2+k1:2+k2]
            elif req == 'taste':
                base[k2:k3] = objFeatures[2+k2:2+k3]
            elif req == 'move':
                base[k3:k4] = objFeatures[2+k3:2+k4]
        """

        return base
    
    """
    def trueName(self,object_number,part):
        return self.data.loc[object_number][0]
    
    def clear(self):
        self.request.clear()
    """


# In[3]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'Database.ipynb'])


# In[4]:


#test = Database()


# In[5]:


#test.Overfetch(7, 'color', 1)


# In[ ]:




