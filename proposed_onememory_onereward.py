#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding:utf-8
# [0]必要なライブラリのインポート
# import gym  # 倒立振子(cartpole)の実行環境
import numpy as np
import random as rd
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import img_to_array, load_img
from collections import deque
# from gym import wrappers  # gymの画像保存
from keras import backend as K
# from IPython.display import clear_output
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import copy
import glob
from datetime import datetime

from Database_test import *
from Pre_Model import *

import pandas as pd
import codecs


# In[2]:

# 時間計測スタート #
starttime = time.time()

DIR_PATH = './result_9/'
os.makedirs(DIR_PATH, exist_ok=True)
os.makedirs(DIR_PATH+'check_points/', exist_ok=True)
IMAGE_PATH = './jaffedbase/jaffedbase/'


# In[3]:


def plot_history(epochs, acc, i, figure_name='onememory_onereward'):
    # print(history.history.keys())
    
    # clear_output(wait = True)
    # 精度の履歴をプロット
    plt.clf()
    plt.close() # 前回のループのpltをcloseすることでメモリ解放？https://qiita.com/Masahiro_T/items/bdd0482a8efd84cdd270

    plt.plot(epochs, acc)
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([-0.02,1.02])
    plt.savefig(DIR_PATH+'figure_'+figure_name+'_' + datetime.now().strftime('%Y%m%d') + f'_{i}' + '.png')
    # plt.savefig(DIR_PATH + 'model_accuracy.png')
    # plt.savefig('figure_acc/figure_' + datetime.now().strftime('%Y%m%d') + '.png')
    # plt.show()


# In[4]:


def get_fe(path):
    fe_list = ['HA', 'SU', 'AN', 'DI', 'SA', 'NE'] #happy, surprise, angry, disgust, sad, neutral
    fe_data = []
    for data in fe_list:
        temp_img = load_img(glob.glob(path+'KA.'+data+'1.*')[0], color_mode = "grayscale", target_size=(32,32))
        temp_img_array = img_to_array(temp_img)
        fe_data.append(temp_img_array)
    
    fe_data = np.asarray(fe_data)
    fe_data = fe_data.astype('float32')
    fe_data = fe_data / 255.0
    
    return fe_data


# In[5]:


DQN_MODE = False    # TrueがDQN、FalseがDDQNです
PER_MODE = True
MODEL_LOAD = False

NUM_EPISODES = 20000  # 総試行回数(episode数)
# NUM_EPISODES = 1  # 総試行回数(episode数)
MAX_NUMBER_OF_STEPS = 5  # 1試行のstep数
MAX_NUMBER_OF_LOOPS = 2 # 1stepの品詞の出力数(名詞→動詞なので2)
GAMMA = 0.99    # 割引係数
ISLEARNED = False  # 学習が終わったフラグ
ISRENDER = False  # 描画フラグ
# acc = []
# epochs = []
# ---
#HIDDEN_SIZE = 1536               # LSTMの隠れ層のニューロンの数
#HIDDEN_SIZE = 64
"""HIDDEN_SIZE = 48"""
HIDDEN_SIZE = 1024
#HIDDEN_SIZE = 256
#EMBEDDING_SIZE = 2048 # データの圧縮次元数
#EMBEDDING_SIZE = 512
"""EMBEDDING_SIZE = 256"""
EMBEDDING_SIZE = 2048
"""HIDDEN_SIZE_2 = 32"""
HIDDEN_SIZE_2 = 512
LEARNING_RATE = 0.00001         # Q-networkの学習係数
MEMORY_SIZE = 32            # バッファーメモリの大きさ 96 → 48
BATCH_SIZE = 8                # Q-networkを更新するバッチの大記載
TEST_EPOCHS_INTERVAL = 50 # 何エポック毎にテストするか
TEST_EPOCHS = 50 # テスト回数
PRIORITIZED_MODE_BORDER = 0.3 # prioritized experience replayに入る正答率
SET_TARGETQN_INTERVAL = 10 # 何エピソードでmainQNとtargetQNを同期するか
reward_sum_n = 1 # 名詞の正解数
reward_sum_v = 1 # 動詞の正解数
DIV_SIZE = 1 # 特徴量の分割サイズ
PER_ERROR = 1.0

P_HIDDEN_SIZE = 32

#P_HIDDEN_SIZE = 32 # 親の特徴量に対する隠れ層のユニット数

Data = Database(DIV_SIZE)

obj_val_idx_list = [] # 特徴選択の推移を見るため、特徴選択を保存するリストを用意する

#features = Data.feature
features_length = Data.feature_length # 物体の特徴の特徴量の次元数
actions = Data.action
actions_length = len(actions) # 選択可能な行動数(物体の特徴の数):4

symbols = Data.name # 名詞の選択肢
symbols_noun = Data.name_noun
symbols_verb = Data.name_verb # 動詞の選択肢
objects_length = len(symbols) # 物体の数:5(名詞)+4(動詞)+1(分からない)

# guess_length = 1 # '分からない'用の次元数
cur_loop_length = 1 # ループ回数を保存する次元

parent_length = 32

# state_length = actions_length + features_length + objects_length + guess_length
state_length = actions_length + features_length + objects_length + cur_loop_length
output_length = actions_length + objects_length

parent_FE = get_fe(IMAGE_PATH) #表情一覧を取得

# 親の意図を確率変数で変化させる
pos = ['noun', 'verb']
noun_p = 1
verb_p = 0

# 名詞と動詞の正答を保存するDataFrame
cols = ['nounorverb', 'ans']
df = pd.DataFrame(index=[], columns=cols)

GPU_ID = 0
physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[GPU_ID], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[GPU_ID], True)


mainQN = QNetwork(hidden_size=HIDDEN_SIZE, state_size=state_length, step_size=MAX_NUMBER_OF_STEPS, action_size=actions_length, 
                  object_size=objects_length, output_size=output_length, parent_size=parent_length, feature_size=features_length, embedding_size=EMBEDDING_SIZE, 
                  learning_rate=LEARNING_RATE, hidden_size_2=HIDDEN_SIZE_2,  p_hidden_size=P_HIDDEN_SIZE)     # メインのQネットワーク
targetQN = QNetwork(hidden_size=HIDDEN_SIZE, state_size=state_length, step_size=MAX_NUMBER_OF_STEPS, action_size=actions_length, 
                    object_size=objects_length, output_size=output_length, parent_size=parent_length, feature_size=features_length, embedding_size=EMBEDDING_SIZE,
                    learning_rate=LEARNING_RATE, hidden_size_2=HIDDEN_SIZE_2, p_hidden_size=P_HIDDEN_SIZE)   # 価値を計算するQネットワーク
# plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
memory_episode = Memory(max_size=MEMORY_SIZE)
memory_step = Memory(max_size=1)
memory = Memory(max_size=MEMORY_SIZE)
memory_TDerror = Memory_TDerror(max_size=MEMORY_SIZE, state_size=state_length, output_size=output_length, step_size=MAX_NUMBER_OF_STEPS, parent_size=parent_length)
actor = Actor(features_length=features_length, objects_length=objects_length, actions_length=actions_length, output_size=output_length, MODEL_LOAD=MODEL_LOAD)

memory_TDerror_noun = Memory_TDerror(max_size=MEMORY_SIZE, state_size=state_length, output_size=output_length, step_size=MAX_NUMBER_OF_STEPS, parent_size=parent_length)
memory_TDerror_verb = Memory_TDerror(max_size=MEMORY_SIZE, state_size=state_length, output_size=output_length, step_size=MAX_NUMBER_OF_STEPS, parent_size=parent_length)
memory_episode_noun = Memory(max_size=MEMORY_SIZE)
memory_episode_verb = Memory(max_size=MEMORY_SIZE)

# 名詞だけを学習するループ
# ckpt_path = DIR_PATH + f'check_points/my_checkpoint_onlynoun_{NUM_EPISODES}'
ckpt_path = DIR_PATH + f'check_points/my_checkpoint_onlynoun_10000'
for i in range(1, 2):
    acc = []
    epochs = []
    name_select_list = np.empty((NUM_EPISODES, MAX_NUMBER_OF_LOOPS, MAX_NUMBER_OF_STEPS), dtype=object) # 名称選択の保存リスト
    feature_select_list = np.empty_like(name_select_list) # 特徴選択の保存リスト

    # memory_episode_list = [memory_episode_noun, memory_episode_verb]
    # memory_TDerror_list = [memory_TDerror_noun, memory_TDerror_verb]

    if os.path.isfile(ckpt_path + '.index'):
        if i == 0:
            print('not load weights for model, start learning!!')
        if i == 1:
            mainQN.model.load_weights(ckpt_path)
            print('model load weights!!')   
    
    ############ 名詞を学習した ############

    starttime = time.time()

    # NUM_EPISODES = 10000  # 総試行回数(episode数)

    # 名詞→動詞の順に学習するループ
    for episode in range(NUM_EPISODES-10000, NUM_EPISODES):
        # 正解の場合フラグを立てる
        correct = 0

        print('episode : ', episode)
        np.random.seed(episode)
        rand = np.random.randint(0, 65535)

        pred_noun_verb = []
        ans_noun_verb = []

        # step終了を判断する変数
        # terminal = [0, 0]
        
        # act_val = np.zeros(output_length)
        # act_name = np.zeros(output_length)
        obj = [0] * objects_length
        fea_vec = [0] * actions_length # 特徴の種類
        fea_val = [0] * features_length # 特徴量の値
        requests = []
        mask1 = [1] * actions_length + [0] * objects_length # 特徴選択用のmask
        mask2 = [0] * actions_length + [1] * objects_length # 名前予測用のmask
        # guess = [0] # not_sureを選んだ回数
        cur_loop = [0] # ループ回数を保存してstateに組み込む
        not_sure_count = 0 # not_sureをそのエピソードで選んだかどうかを保存するフラグ
        #parent_order = np.reshape(parent_order, [1, parent_length])
                
        memory_none = [np.concatenate([fea_vec, fea_val, obj, cur_loop]), np.concatenate([fea_vec, fea_val, obj, cur_loop]), 
            [0] * output_length, [0] * output_length, np.zeros(output_length), np.zeros(output_length), 0, 0, 0, np.zeros(parent_length*parent_length)]
        
        # 名詞と動詞を入れるメモリを用意するため、リストを倍にする
        memory_in = [memory_none] * 2 * MAX_NUMBER_OF_STEPS
                
        action_step_state = np.zeros((1, 2 * MAX_NUMBER_OF_STEPS, state_length))
        out = np.zeros((1, 2 * MAX_NUMBER_OF_STEPS, output_length))

        # 答えを決める
        objectIndex = [0, 0]
        objectIndex[0] = random.randint(0, len(symbols_noun)-2)
        objectIndex[1] = objectIndex[0]
        # if objectIndex[0] <= 1:
        #     objectIndex[1] = 5
        # else:
        #     objectIndex[1] = objectIndex[0] + 4
        print(f'ans : {symbols[objectIndex[0]], symbols[objectIndex[1]]}')
        
        for step in range(MAX_NUMBER_OF_STEPS):
            # reward = [0, 0]
            # terminal = [0, 0]
            reward = 0
            terminal = 0

            # oneloop_time = time.time()
            for loop in range(MAX_NUMBER_OF_LOOPS):
                # print(f'episode : {episode}, step : {step}, loop : {loop}')
                cur_loop[0] = loop
                parent_intent = [0, 0]
                parent_intent[loop] = 1
                parent_select = 0
                # parent_select = loop

                infant_intent = [0, 0]
                idx = random.randint(0, 1)
                infant_intent[idx] = 1
            
                #parent_order = [0, 0, 0, 0, 0, 0] #happy, surprise, angry, disgust, sad, neutral
                if parent_intent == infant_intent:
                    idx = random.randint(0, 1)
                    parent_order = parent_FE[idx]
                else:
                    idx = random.randint(0, 5)
                    parent_order = parent_FE[idx]

                parent_order = np.reshape(parent_order, [1, parent_length, parent_length, 1])
                
                pre_fea_vec = copy.deepcopy(fea_vec)
                # print(f'fea_vec : {fea_vec}')
                pre_obj_vec = copy.deepcopy(obj)
                state1 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
                state1 = np.reshape(state1, [1, state_length])
                mask1 = np.reshape(mask1, [1, output_length])
                action_step_state[0][step] = state1 # 動詞学習の際に名詞学習の結果を上書きしている？要検討
                out_1 = np.concatenate([pre_fea_vec, pre_obj_vec])
                out[0][step] = np.reshape(out_1, [1, output_length])

                obj_val_idx, retTargetQs = actor.get_value(action_step_state, out, mask1, parent_order, episode, mainQN) # 時刻tで取得する特徴量を決定
                feature_select_list[episode, loop, step] = actions[obj_val_idx]
                retTargetQs = retTargetQs[retTargetQs != 0]

                fea_vec[obj_val_idx] = 1
                request = actions[obj_val_idx]
                requests.append(request)
                fea_val = Data.Overfetch(objectIndex[loop], list(set(requests)), rand, parent_select)       
                state2 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
                state2 = np.reshape(state2, [1,state_length])
                mask2 = np.reshape(mask2, [1, output_length])
                action_step_state[0][step+1] = state2
                out_2 = np.concatenate([fea_vec, pre_obj_vec])
                out[0][step+1] = np.reshape(out_2, [1, output_length])
                # print(f'action_step_state : {len(action_step_state[loop][0][step])}, out : {len(out[loop][0][step])}, mask2 : {len(mask2)}, parent_order : {len(parent_order)}')
                obj_name_idx = actor.get_name(action_step_state, out, mask2, parent_order, episode, mainQN) # 時刻tで取得する物体の名称を決定

                name = symbols[obj_name_idx - actions_length]
                name_select_list[episode, loop, step] = name
                obj[obj_name_idx - actions_length] = 1
                if name == 'not_sure':
                    not_sure_count = 1
                
                pred_noun_verb.append(name)
                ans_noun_verb.append(symbols[objectIndex[loop]])

                reward, terminal = one_reward_func(objectIndex[loop], (obj_name_idx - actions_length), not_sure_count, step, MAX_NUMBER_OF_STEPS,
                                                                        False, requests, request, obj_val_idx, 3, symbols, symbols_verb, loop, name, reward, terminal)

                not_sure_count = 0 # 「分からない」フラグを元に戻す

                memory_in[step] = [state1.reshape(-1), state2.reshape(-1), out_1, out_2, mask1.reshape(-1), mask2.reshape(-1), obj_val_idx, reward, terminal, parent_order.reshape(-1)]
                # for j, (state_b, next_state_b, out_b, next_out_b, mask_b, next_mask_b, action_b, reward_b, terminal_b, parent_order_b) in enumerate(eps):
                # ValueError: too many values to unpack (expected 10)
            
                state1 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
                next_out = fea_vec+obj

                memory_in[step+1] = [state2.reshape(-1), state1.reshape(-1), out_2, next_out, mask2.reshape(-1), mask1.reshape(-1), obj_name_idx, reward, terminal, parent_order.reshape(-1)]

                # TDerror_calc = time.time()
                memory_step.add(memory_in)
                # memory_episode_list[loop].add(memory_in[loop])
                # TDerror = memory_TDerror_list[loop].get_TDerror(memory_episode_list[loop], GAMMA, mainQN, targetQN)
                # memory_TDerror_list[loop].add(TDerror)
                # print(memory_TDerror_list[loop].len())
                # print(f'one time TDerror calc : {time.time() - TDerror_calc}')

                # print(f'memory_episode.len() : {memory_episode_list[loop].len()}')

                if (memory_episode.len() > BATCH_SIZE) and terminal == 2:
                    if PER_MODE == True:
                        # print('experience_replay')
                        # exp_replay_time = time.time()
                        history = mainQN.prioritized_experience_replay(memory_episode, BATCH_SIZE, GAMMA, targetQN, memory_TDerror)
                        # print(f'exp replay time : {time.time() - exp_replay_time}')
                    else:
                        history = mainQN.replay(memory_episode, BATCH_SIZE, GAMMA, targetQN)
                else:
                    # replay_time = time.time()
                    history = mainQN.replay(memory_step, 1, GAMMA, targetQN)
                    # print(f'normal replay time : {time.time() - replay_time}')
                    
                if terminal == 2:
                    # memory_episode_list[loop].add(memory_in[loop])
                    # TDerror = memory_TDerror.get_TDerror(memory_episode_list[loop], GAMMA, mainQN, targetQN)
                    # memory_TDerror.add(TDerror)

                    if episode % SET_TARGETQN_INTERVAL == 0:
                        # print('SET_TARGETQN')
                        targetQN.model.set_weights(mainQN.model.get_weights()) # 行動決定と価値計算のQネットワークを同じにする
                        memory_TDerror.update_TDerror(memory_episode, GAMMA, mainQN, targetQN)
            
            print(f'terminal : {terminal}, reward : {reward}')
            print(f'predict : {pred_noun_verb}')
            if terminal == 2: # 名詞と動詞どちらも正解かstepの上限まで達したら
                # print(f'predict : {pred_noun_verb}, ans : {ans_noun_verb}')
                # 正解の場合フラグを立てる
                if pred_noun_verb == ans_noun_verb:
                    correct = 1
                break
            # print(f'one time loop : {time.time() - oneloop_time}')
        if episode % TEST_EPOCHS_INTERVAL == 0:
            print('test!')
            reward_sum, reward_sum_n, reward_sum_v = mainQN.test_onememory_onereward(Data, actor, symbols, symbols_noun, symbols_verb, actions,  
                                                            mask1, mask2, TEST_EPOCHS, MAX_NUMBER_OF_STEPS, MAX_NUMBER_OF_LOOPS, episode, DIR_PATH,
                                                                    PER_ERROR, parent_FE)
            print(f'reward_sum : {reward_sum}')
            if len(epochs) == 0:
                epoch = 0
            else:
                epoch = max(epochs) + TEST_EPOCHS
            acc.append(reward_sum/(TEST_EPOCHS))
            print(f'test accuracy : {reward_sum/(TEST_EPOCHS)}')
            epochs.append(epoch)
            
            plot_history(epochs, acc, i)
                
            if acc[-1] >= PRIORITIZED_MODE_BORDER:
                PER_MODE = True
                
        if episode == 1000 or episode == 3000 or episode == 7000:
        # if episode % 100 == 0:
            mainQN.model.save_weights(DIR_PATH+f'check_points/my_checkpoint_onememory_onereward_{episode}_{i}')

        # print(name_select_list)
        # print(feature_select_list)
    np.save(DIR_PATH + 'numpy/' + f'proposed_onememory_name_{NUM_EPISODES}_{i}.npy', name_select_list)
    np.save(DIR_PATH + 'numpy/' + f'proposed_onememory_feature_{NUM_EPISODES}_{i}.npy', feature_select_list)

    mainQN.model.save_weights(DIR_PATH+f'check_points/my_checkpoint_onememory_onereward_{NUM_EPISODES}_{i}')
        

    # 時間計測の結果を出力
    print('----------------------------', file=codecs.open(DIR_PATH+'elapsed_time'+datetime.now().strftime('%Y%m%d')+'.txt', 'a', 'utf-8'))
    print('time : ', time.time() - starttime, file=codecs.open(DIR_PATH+'elapsed_time'+datetime.now().strftime('%Y%m%d')+'.txt', 'a', 'utf-8'))