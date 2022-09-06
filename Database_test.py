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
from sklearn.ensemble import RandomForestClassifier as RFC


# In[2]:


# どの表を使用するかを決定
global use_color, use_size, use_shape, use_taste, use_hardness, use_move, use_verb
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
    def __init__(self, div_size):
        global use_color, use_size, use_shape, use_taste, use_hardness, use_move, use_verb
        self.data = pd.read_csv("./table/main_new.csv", index_col=0)
        self.verb = pd.read_csv("./table/verb.csv", index_col=0)
        self.request = []
        self.object = self.data.index
        self.feature_length = 0
        self.object_length = len(self.object)
        self.action = []
        self.name = list(dict.fromkeys(self.data['name']))
        self.name_noun = list(dict.fromkeys(self.data['name']))
        use_data = self.data.query('deleted == "f"')
        self.use_object = use_data.index
        obj_list = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]

        if(use_color):
            color_all = pd.read_csv("./table/new_color_histgram.csv", index_col=0)
            feature_list = color_all.values.tolist()
            rfc = RFC(random_state=0)
            rfc.fit(feature_list, obj_list)

            important_index = [i for importance, i in sorted(zip(rfc.feature_importances_, range(rfc.n_features_)),
                                                             key=lambda x:x[0], reverse=True)]
            #self.color = color_all.iloc[:,important_index[:20]]
            if len(color_all.columns) <= 50:
                self.color = color_all.iloc[:,important_index]
            elif len(color_all.columns) // div_size <= 50:
                self.color = color_all.iloc[:,important_index[:50]]
            else:
                self.color = color_all.iloc[:,important_index[:len(color_all.columns)//div_size]]
            self.color_feature = self.color.columns
            self.color_size = len(self.color.columns)
            self.feature_length += self.color_size
            self.action.append('color')
        else:
            self.color_size = 0
        if(use_shape):
            shape_all = pd.read_csv("./table/new_shape.csv", index_col=0)
            feature_list = shape_all.values.tolist()
            rfc = RFC(random_state=0)
            rfc.fit(feature_list, obj_list)

            important_index = [i for importance, i in sorted(zip(rfc.feature_importances_, range(rfc.n_features_)),
                                                             key=lambda x:x[0], reverse=True)]
            #self.shape = shape_all.iloc[:,important_index[:20]]
            if len(shape_all.columns) <= 50:
                self.shape = shape_all.iloc[:,important_index]
            elif len(shape_all.columns) // div_size <= 50:
                self.shape = shape_all.iloc[:,important_index[:50]]
            else:
                self.shape = shape_all.iloc[:,important_index[:len(shape_all.columns)//div_size]]
            self.shape_feature = self.shape.columns
            self.shape_size = len(self.shape.columns)
            self.feature_length += self.shape_size
            self.action.append('shape')
        else:
            self.shape_size = 0
        if(use_taste):
            taste_all = pd.read_csv("./table/new_taste.csv", index_col=0)
            feature_list = taste_all.values.tolist()
            rfc = RFC(random_state=0)
            rfc.fit(feature_list, obj_list)

            important_index = [i for importance, i in sorted(zip(rfc.feature_importances_, range(rfc.n_features_)),
                                                             key=lambda x:x[0], reverse=True)]
            #self.taste = taste_all.iloc[:,important_index[:20]]
            if len(taste_all.columns) <= 50:
                self.taste = taste_all.iloc[:,important_index]
            elif len(taste_all.columns) // div_size <= 50:
                self.taste = taste_all.iloc[:,important_index[:50]]
            else:
                self.taste = taste_all.iloc[:,important_index[:len(taste_all.columns)//div_size]]
            self.taste_feature = self.taste.columns
            self.taste_size = len(self.taste.columns)
            self.feature_length += self.taste_size
            self.action.append('taste')
        else:
            self.taste_size = 0
        if(use_move):
            move_all = pd.read_csv("./table/new_move_k_medoids_100.csv", index_col=0)
            feature_list = move_all.values.tolist()
            rfc = RFC(random_state=0)
            rfc.fit(feature_list, obj_list)

            important_index = [i for importance, i in sorted(zip(rfc.feature_importances_, range(rfc.n_features_)),
                                                             key=lambda x:x[0], reverse=True)]
            #self.move = move_all.iloc[:,important_index[:20]]
            if len(move_all.columns) <= 50:
                self.move = move_all.iloc[:,important_index]
            elif len(move_all.columns) // div_size <= 50:
                self.move = move_all.iloc[:,important_index[:50]]
            else:
                self.move = move_all.iloc[:,important_index[:len(move_all.columns)//div_size]]
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
    
    # ここで特徴を取得している
    def Overfetch(self,object_number,requests,rand,parent_order):
        """
        if request not in self.request:
            self.request.append(request)
        """
        random.seed(rand)
        get_number = random.randint(object_number, object_number+2) # 同種の中からランダムに特徴を与える(オレンジならオレンジ3種類の中から)
        base =  np.zeros(self.feature_length)
        if parent_order == 0:
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
        for request in requests:
            if request == 'color':
                    base[0:k1] = objFeatures[2:2+k1]
            elif request == 'shape':
                    base[k1:k2]  = objFeatures[2+k1:2+k2]
            elif request == 'taste':
                    base[k2:k3] = objFeatures[2+k2:2+k3]
            elif request == 'move':
                    base[k3:k4] = objFeatures[2+k3:2+k4]

        return base

    def Overfetch_V2(self,object_number,requests,rand,parent_order=0):
        """
        if request not in self.request:
            self.request.append(request)
        """
        random.seed(rand)
        get_number = random.randint(object_number, object_number+2) # 同種の中からランダムに特徴を与える(オレンジならオレンジ3種類の中から)
        base =  np.zeros(self.feature_length)
        if parent_order == 0:
            if get_number >= len(self.data):
                return base
            objFeatures = self.data.loc[get_number]
            # print(self.data.loc[get_number]['name'])
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
        for request in requests:
            if request == 'color':
                    base[0:k1] = objFeatures[2:2+k1]
            elif request == 'shape':
                    base[k1:k2]  = objFeatures[2+k1:2+k2]
            elif request == 'taste':
                    base[k2:k3] = objFeatures[2+k2:2+k3]
            elif request == 'move':
                    base[k3:k4] = objFeatures[2+k3:2+k4]

        return base
# In[3]:


# import subprocess
# subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'Database_test.ipynb'])


# In[ ]:





# In[ ]:




