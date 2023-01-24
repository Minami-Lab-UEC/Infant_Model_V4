#!/usr/bin/env python
# coding: utf-8
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


# 時間計測スタート #
starttime = time.time()

DIR_PATH = './result_9/'
os.makedirs(DIR_PATH, exist_ok=True)
os.makedirs(DIR_PATH+'check_points/', exist_ok=True)
IMAGE_PATH = './jaffedbase/jaffedbase/'



def plot_history(epochs, acc, i=0):
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
    plt.savefig(DIR_PATH+'figure_onlynoun_except_' + datetime.now().strftime('%Y%m%d') + f'_{i}' + '.png')
    # plt.savefig(DIR_PATH + 'model_accuracy.png')
    # plt.savefig('figure_acc/figure_' + datetime.now().strftime('%Y%m%d') + '.png')
    # plt.show()



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

DQN_MODE = False    # TrueがDQN、FalseがDDQNです
PER_MODE = True
MODEL_LOAD = False

NUM_EPISODES = 10000  # 総試行回数(episode数)
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

GPU_ID = 2
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

# 名詞だけを学習するループ
# ckpt_path = DIR_PATH + f'check_points/my_checkpoint_onlynoun_{NUM_EPISODES}'
ckpt_path = DIR_PATH + f'check_points/my_checkpoint_onlynoun_10000'

mainQN.model.load_weights(ckpt_path)
print('model load weights!!')
# print('model not load !!')

acc = []
epochs = []

for episode in range(NUM_EPISODES):
    # 正解の場合フラグを立てる
    correct = 0

    print('episode : ', episode)
    np.random.seed(episode)
    rand = np.random.randint(0, 65535)
    
    parent_intent = [1, 0]
    parent_select = 0

    objectIndex = random.randint(0, len(symbols_noun)-2)
    if objectIndex == 1:
        objectIndex = 0 # orrange delete
        
    infant_intent = [0, 0]
    idx = random.randint(0, 1)
    infant_intent[idx] = 1
    
    #parent_order = [0, 0, 0, 0, 0, 0] #happy, surprise, angry, disgust, sad, neutral
    if parent_intent == infant_intent:
        print('same intent')
        idx = random.randint(0, 1)
        parent_order = parent_FE[idx]
    else:
        print('diff intent')
        idx = random.randint(0, 5)
        parent_order = parent_FE[idx]

    act_val = np.zeros(output_length)
    act_name = np.zeros(output_length)
    obj = [0] * objects_length
    fea_vec = [0] * actions_length # 特徴の種類
    fea_val = [0] * features_length # 特徴量の値
    requests = []
    mask1 = [1] * actions_length + [0] * objects_length # 特徴選択用のmask
    mask2 = [0] * actions_length + [1] * objects_length # 名前予測用のmask
    # guess = [0] # not_sureを選んだ回数
    cur_loop = [0] # loop回数を保存する、名詞のみ学習のときには使わない
    not_sure_count = 0 # not_sureをそのエピソードで選んだかどうかを保存するフラグ
    #parent_order = np.reshape(parent_order, [1, parent_length])
    parent_order = np.reshape(parent_order, [1, parent_length, parent_length, 1])
    
    memory_none = [np.concatenate([fea_vec, fea_val, obj, cur_loop]), np.concatenate([fea_vec, fea_val, obj, cur_loop]), 
                [0] * output_length, [0] * output_length, np.zeros(output_length), np.zeros(output_length), 0, 0, 0, np.zeros(parent_length*parent_length)]
    memory_in = [memory_none] * 2 * MAX_NUMBER_OF_STEPS
    
    action_step_state = np.zeros((1, 2 * MAX_NUMBER_OF_STEPS, state_length))
    out = np.zeros((1, 2 * MAX_NUMBER_OF_STEPS, output_length))
    
    obj_val_idx_l = np.array([0] * actions_length) # エピソード内のステップごとに選択した特徴選択を保存する

    for step in range(MAX_NUMBER_OF_STEPS):
        pre_fea_vec = copy.deepcopy(fea_vec)
        pre_obj_vec = copy.deepcopy(obj)
        state1 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
        state1 = np.reshape(state1, [1, state_length])
        mask1 = np.reshape(mask1, [1, output_length])
        action_step_state[0][step] = state1
        out_1 = np.concatenate([pre_fea_vec, pre_obj_vec])
        out[0][step] = np.reshape(out_1, [1, output_length])
        
        obj_val_idx, retTargetQs = actor.get_value(action_step_state, out, mask1, parent_order, episode, mainQN) # 時刻tで取得する特徴量を決定
        retTargetQs = retTargetQs[retTargetQs != 0]
        obj_val_idx_l = np.vstack([obj_val_idx_l, retTargetQs])

        fea_vec[obj_val_idx] = 1
        request = actions[obj_val_idx]
        requests.append(request)
        fea_val = Data.Overfetch(objectIndex, list(set(requests)), rand, parent_select)       
        state2 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
        state2 = np.reshape(state2, [1,state_length])
        mask2 = np.reshape(mask2, [1, output_length])
        action_step_state[0][step+1] = state2
        out_2 = np.concatenate([fea_vec, pre_obj_vec])
        out[0][step+1] = np.reshape(out_2, [1, output_length])
        obj_name_idx = actor.get_name(action_step_state, out, mask2, parent_order, episode, mainQN) # 時刻tで取得する物体の名称を決定
        
        # name : モデル予測
        name = symbols[obj_name_idx - actions_length]
        obj[obj_name_idx - actions_length] = 1
        if name == 'not_sure':
            not_sure_count = 1
        
        # 報酬を設定し、与える
        reward, terminal = reward_func(objectIndex, (obj_name_idx - actions_length), not_sure_count, step, MAX_NUMBER_OF_STEPS,
                                                                    False, requests, request, obj_val_idx, 3, symbols, symbols_verb, parent_select, name)

        not_sure_count = 0 # 「分からない」フラグを元に戻す
        
        memory_in[step] = [state1.reshape(-1), state2.reshape(-1), out_1, out_2, mask1.reshape(-1), mask2.reshape(-1), obj_val_idx, reward, terminal, parent_order.reshape(-1)]
        
        state1 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
        next_out = fea_vec+obj
        
        memory_in[step+1] = [state2.reshape(-1), state1.reshape(-1), out_2, next_out, mask2.reshape(-1), mask1.reshape(-1), obj_name_idx, reward, terminal, parent_order.reshape(-1)]
        
        memory_step.add(memory_in)
        memory_episode.add(memory_in)
        TDerror = memory_TDerror.get_TDerror(memory_episode, GAMMA, mainQN, targetQN)
        memory_TDerror.add(TDerror)
        
        if (memory_episode.len() > BATCH_SIZE) and terminal == 1:
            if PER_MODE == True:
                print('experience_replay')
                history = mainQN.prioritized_experience_replay(memory_episode, BATCH_SIZE, GAMMA, targetQN, memory_TDerror)
            else:
                history = mainQN.replay(memory_episode, BATCH_SIZE, GAMMA, targetQN)
        else:
            history = mainQN.replay(memory_step, 1, GAMMA, targetQN)
                
        if terminal == 1:
            # 正解の場合フラグを立てる
            print(f'[{pos[parent_select]}] predict : {name}, ans : {symbols[objectIndex]}')
            if objectIndex == (obj_name_idx - actions_length):
                correct = 1
            if episode % SET_TARGETQN_INTERVAL == 0:
                targetQN.model.set_weights(mainQN.model.get_weights()) # 行動決定と価値計算のQネットワークを同じにする
                memory_TDerror.update_TDerror(memory_episode, GAMMA, mainQN, targetQN)
            break
    obj_val_idx_l = np.delete(obj_val_idx_l, 0, 0) # ダミーで入れた最初の行を削除する
    obj_val_idx_list.append(obj_val_idx_l) # 毎エピソードで選択した特徴選択を保存する

    record = pd.Series([parent_select, correct], index=df.columns)
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    ass = action_step_state
    o = out
    
    if episode % TEST_EPOCHS_INTERVAL == 0:
        reward_sum, reward_sum_n, reward_sum_v = mainQN.test(Data, actor, symbols, symbols_noun, symbols_verb, actions,  
                                                    mask1, mask2, TEST_EPOCHS, MAX_NUMBER_OF_STEPS, episode, DIR_PATH,
                                                            PER_ERROR, parent_FE)
        if len(epochs) == 0:
            epoch = 0
        else:
            epoch = max(epochs) + TEST_EPOCHS
        acc.append(reward_sum/(TEST_EPOCHS))
        print(f'test accuracy : {reward_sum/(TEST_EPOCHS)}')
        epochs.append(epoch)
        
        if acc[-1] >= PRIORITIZED_MODE_BORDER:
            PER_MODE = True
        
        plot_history(epochs, acc)
        
        if episode == 1000 or episode == 5000 or episode == 8000:
            mainQN.model.save_weights(DIR_PATH+f'check_points/my_checkpoint_onlynoun_except_{episode}')
    
with open(DIR_PATH+'+LSTM_acc.pickle', mode='wb') as f:
    pickle.dump(acc, f)

# 名詞と動詞の正答率の推移をプロット
# noun_transition : list, episodeごとに正答率を計算しappend
# verb_transition : list, episodeごとに正答率を計算しappend
df.to_pickle(DIR_PATH+'acc_transition.pkl')

mainQN.model.save_weights(DIR_PATH+f'check_points/my_checkpoint_onlynoun_except_{NUM_EPISODES}')
    
# plt.savefig(DIR_PATH+'figure_onlynoun_' + datetime.now().strftime('%Y%m%d' + '.png'))

# if MODEL_LOAD == False:
#     os.makedirs(DIR_PATH+'check_points/', exist_ok=True)
#     mainQN.model.save_weights(DIR_PATH+'check_points/my_checkpoint')

# 時間計測の結果を出力
print('----------------------------', file=codecs.open(DIR_PATH+'elapsed_time'+datetime.now().strftime('%Y%m%d')+'.txt', 'a', 'utf-8'))
print('time : ', time.time() - starttime, file=codecs.open(DIR_PATH+'elapsed_time'+datetime.now().strftime('%Y%m%d')+'.txt', 'a', 'utf-8'))

print(f'noun learned!! time : {time.time() - starttime}')