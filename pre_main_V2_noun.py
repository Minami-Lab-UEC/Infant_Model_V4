#!/usr/bin/env python
# coding: utf-8

# In[1]:

# 2022/07/01 名詞だけの学習がうまくいくか実験するためのプログラム

# coding:utf-8
# [0]必要なライブラリのインポート
# import gym  # 倒立振子(cartpole)の実行環境
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
# from gym import wrappers  # gymの画像保存
from keras import backend as K
from IPython.display import clear_output
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import copy
import datetime
import codecs

from Database_test import *
from Pre_Model_V2 import *


# In[2]:


dir_path = './result_HCS_after_60/'
os.makedirs(dir_path, exist_ok=True)
ckpt_path = './result_HCS_after_49/check_points/my_checkpoint'


# In[3]:


def plot_history(epochs, acc):
    # print(history.history.keys())
    
    clear_output(wait = True)
    # 精度の履歴をプロット
    plt.plot(epochs, acc, color='green')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([-0.02,1.02])
    # plt.show()
    # plt.savefig(dir_path + 'model_accuracy.png')
    # plt.savefig('figure_' + datetime.date.today().strftime('%Y%m%d') + '.png') # savefigすると毎回図を保存してしまう


# In[4]:


DQN_MODE = False    # TrueがDQN、FalseがDDQNです
LENDER_MODE = True # Falseは学習後も描画なし、Falseは学習終了後に描画する
PER_MODE = True
MODEL_LOAD = False

# env = gym.make('CartPole-v0')
# num_episodes = 20000  # 総試行回数(episode数)
num_episodes = 10000 # 総試行回数(episode数)
max_number_of_steps = 5  # 1試行のstep数
goal_average_reward = 195  # この報酬を超えると学習終了
num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
gamma = 0.99    # 割引係数
islearned = False  # 学習が終わったフラグ
isrender = False  # 描画フラグ
acc = []
epochs = []
# ---
#hidden_size = 1536               # LSTMの隠れ層のニューロンの数
hidden_size = 1024
#hidden_size = 256
embedding_size = 2048 # データの圧縮次元数
# embedding_size = 256
#embedding_size = 1024
hidden_size_2 = 512
learning_rate = 0.00001         # Q-networkの学習係数
#memory_size = 20            # バッファーメモリの大きさ
memory_size = 96
#batch_size = 16                # Q-networkを更新するバッチの大記載
batch_size = 64
test_epochs_interval = 50 # 何エポック毎にテストするか 50
test_epochs = 50 # テスト回数
prioritized_mode_border = 0.3 # prioritized experience replayに入る正答率
set_targetQN_interval = 10 # 何エピソードでmainQNとtargetQNを同期するか
div_size = 1 # 特徴量の分割サイズ
start_change = 10000 # 固定を始めるエピソード数

Data = Database(div_size)

#features = Data.feature
features_length = Data.feature_length # 物体の特徴の特徴量の次元数
actions = Data.action
actions_length = len(actions) # 選択可能な行動数(物体の特徴の数):4

# symbols = Data.name
symbols = Data.name_noun # 名詞の選択肢
objects_length = len(symbols) # 物体の数:5+1(分からない)

guess_length = 1 # '分からない'用の次元数

state_length = actions_length + features_length + objects_length + guess_length
output_length = actions_length + objects_length

# obj_val_idx_list = np.zeros((1, max_number_of_steps, actions_length)) # 特徴選択の推移を見るため、特徴選択を保存するリストを用意する, ダミーをまず入れる
obj_val_idx_list = [[[]]] # 特徴選択の推移を見るため、特徴選択を保存するリストを用意する, ダミーをまず入れる

correct_count = [0] * (objects_length - 1) # 各ラベルの正答数
count = [0] * (objects_length - 1) # 各ラベルの総数
# In[5]:


mainQN = QNetwork(hidden_size=hidden_size, state_size=state_length, step_size=max_number_of_steps, action_size=actions_length,
                  object_size=objects_length, output_size=output_length, feature_size=features_length, embedding_size=embedding_size, 
                  learning_rate=learning_rate, hidden_size_2=hidden_size_2)     # メインのQネットワーク
targetQN = QNetwork(hidden_size=hidden_size, state_size=state_length, step_size=max_number_of_steps, action_size=actions_length, 
                    object_size=objects_length, output_size=output_length, feature_size=features_length, embedding_size=embedding_size,
                    learning_rate=learning_rate, hidden_size_2=hidden_size_2)   # 価値を計算するQネットワーク

if MODEL_LOAD == True:
    mainQN.model.load_weights(ckpt_path)
    targetQN.model.set_weights(mainQN.model.get_weights())

# plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
memory_episode = Memory(max_size=memory_size)
memory_step = Memory(max_size=1)
memory = Memory(max_size=memory_size)
memory_TDerror = Memory_TDerror(max_size=memory_size, state_size=state_length, output_size=output_length, step_size=max_number_of_steps)
actor = Actor(features_length=features_length, objects_length=objects_length, actions_length=actions_length, output_size=output_length, MODEL_LOAD=MODEL_LOAD)


# In[ ]:


for episode in range(num_episodes):
    #Data.clear()
    np.random.seed(episode)
    rand = np.random.randint(0, 65535)
    objectIndex = random.randint(0, objects_length-2) # 当ててほしい物体のindex
    
    # count[objectIndex] += 1 # 各ラベルのカウント
    
    act_val = np.zeros(output_length)
    act_name = np.zeros(output_length)
    obj = [0] * objects_length
    fea_vec = [0] * actions_length # 特徴の種類
    fea_val = [0] * features_length # 特徴量の値
    requests = []
    mask1 = [1] * actions_length + [0] * objects_length # 特徴選択用のmask
    #mask1 = [1] * actions_length + [-float('inf')] * objects_length
    mask2 = [0] * actions_length + [1] * objects_length # 名前予測用のmask
    #mask2 = [-float('inf')] * actions_length + [1] * objects_length
    guess = [0] # not_sureを選んだ回数
    not_sure_count = 0 # not_sureをそのエピソードで選んだかどうかを保存するフラグ
    
    memory_none = [np.concatenate([fea_vec, fea_val, obj, guess]), np.concatenate([fea_vec, fea_val, obj, guess]), [0] * output_length, 
                   [0] * output_length, np.zeros(output_length), np.zeros(output_length), 0, 0, 0]
    memory_in = [memory_none] * 2 * max_number_of_steps
    
    action_step_state = np.zeros((1, 2 * max_number_of_steps, state_length))
    out = np.zeros((1, 2 * max_number_of_steps, output_length))
    #print(str(episode)+' episode start.')
    
    # obj_val_idx_l = np.array([0] * actions_length) # エピソード内のステップごとに選択した特徴選択を保存する
    obj_val_idx_l = [[]] # エピソード内のステップごとに選択した特徴を保存する
    for step in range(max_number_of_steps):
        pre_fea_vec = copy.deepcopy(fea_vec)
        pre_obj_vec = copy.deepcopy(obj)
        #fea_val = [0] * features_length # 特徴量の値
        state1 = np.concatenate([fea_vec, fea_val, obj, guess])
        state1 = np.reshape(state1, [1, state_length])
        mask1 = np.reshape(mask1, [1, output_length])
        action_step_state[0][step] = state1
        out_1 = np.concatenate([pre_fea_vec, pre_obj_vec])
        out[0][step] = np.reshape(out_1, [1, output_length])
        obj_val_idx, retTargetQs = actor.get_value(action_step_state, out, mask1, episode, mainQN) # 時刻tで取得する特徴量を決定
        print("{}:{} retTargetQs : {}".format(episode, step, retTargetQs))
        retTargetQs = retTargetQs[retTargetQs != 0] # 確率が0のものがたまに含まれるので削除
        # obj_val_idx_l = np.vstack([obj_val_idx_l, retTargetQs])
        
        fea_vec[obj_val_idx] = 1
        request = actions[obj_val_idx]
        # print("request : ", request)
        requests.append(request)
        print("requests : ", requests)
        fea_val = Data.Overfetch_V2(objectIndex, list(set(requests)), rand)       
        state2 = np.concatenate([fea_vec, fea_val, obj, guess])
        state2 = np.reshape(state2, [1,state_length])
        mask2 = np.reshape(mask2, [1, output_length])
        action_step_state[0][step+1] = state2
        out_2 = np.concatenate([fea_vec, pre_obj_vec])
        out[0][step+1] = np.reshape(out_2, [1, output_length])
        obj_name_idx = actor.get_name(action_step_state, out, mask2, episode, mainQN) # 時刻tで取得する物体の名称を決定
        
        #print(obj_name_idx - actions_length)
        
        name = symbols[obj_name_idx - actions_length]
        obj[obj_name_idx - actions_length] = 1
        if name == 'not_sure':
            guess[0] += 1
            not_sure_count = 1
        
        # 報酬を設定し、与える
        # reward, terminal = reward_func(objectIndex, (obj_name_idx - actions_length), guess[0], step, max_number_of_steps, False, requests, request)
        reward, terminal = reward_func(objectIndex, (obj_name_idx - actions_length), not_sure_count, step, max_number_of_steps, False, requests, request) # guess[0]ではなくnot_sure_countを与える 220721
        not_sure_count = 0 # 「分からない」フラグを元に戻す 220721
        # if terminal == 1:
        #     if objectIndex == (obj_name_idx - actions_length):
        #         correct_count[objectIndex] += 1 # 各ラベルの正答数をカウント
        
        #act_val[obj_val_idx], act_name[obj_name_idx] = 1, 1
        
        memory_in[step] = [state1.reshape(-1), state2.reshape(-1), out_1, out_2,  mask1.reshape(-1), mask2.reshape(-1), obj_val_idx, reward, terminal]
        #memory.add((state1, state2, mask1, mask2, obj_val_idx, reward, terminal)) # メモリの更新(特徴量)
        
        state1 = np.concatenate([fea_vec, fea_val, obj, guess])
        next_out = fea_vec+obj
        
        memory_in[step+1] = [state2.reshape(-1), state1.reshape(-1), out_2, next_out, mask2.reshape(-1), mask1.reshape(-1), obj_name_idx, reward, terminal]
        
        memory_step.add(memory_in)
        
        #memory.add((state2, state1, mask2, mask1, obj_name_idx, reward, terminal)) # メモリの更新(物体の名称)
        
        if (memory_episode.len() > batch_size) and terminal == 1:
            if PER_MODE == True:
                memory_episode.add(memory_in)
                TDerror = memory_TDerror.get_TDerror(memory_episode, gamma, mainQN, targetQN)
                memory_TDerror.add(TDerror)
                print('experience_replay progress : ')
                history = mainQN.prioritized_experience_replay(memory_episode, batch_size, gamma, targetQN, memory_TDerror)
            else:
                history = mainQN.replay(memory_episode, batch_size, gamma, targetQN)
        else:
            print('replay progress : ')
            history = mainQN.replay(memory_step, 1, gamma, targetQN)
            
        # test 20220413
        # if step == 2:
        #     terminal = 1

        if terminal == 1:
            # for _ in range(max_number_of_steps - step - 1):
            #     obj_val_idx_l = np.vstack([obj_val_idx_l, obj_val_idx_l[-1]]) # 途中でエピソードが打ち切られる場合には、最後のステップの結果をコピーする
            obj_val_idx_l.append(requests)

            """
            memory_episode.add(memory_in)
            TDerror = memory_TDerror.get_TDerror(memory_episode, gamma, mainQN, targetQN)
            memory_TDerror.add(TDerror)
            """
            
            if episode % set_targetQN_interval == 0:
                targetQN.model.set_weights(mainQN.model.get_weights()) # 行動決定と価値計算のQネットワークを同じにする
                memory_TDerror.update_TDerror(memory_episode, gamma, mainQN, targetQN)
                if episode >= start_change:
                    mainQN.change_learn()
            break
    # obj_val_idx_l = np.delete(obj_val_idx_l, 0, 0) # ダミーで入れた最初の行を削除する
    obj_val_idx_l.pop(0) # ダミーデータの削除
    # obj_val_idx_list = np.vstack([obj_val_idx_list, [obj_val_idx_l]]) # 毎エピソードで選択した特徴選択を保存する
    obj_val_idx_list.append(obj_val_idx_l) # 毎エピソードで選択した特徴を保存する
            
    
    if episode % test_epochs_interval == 0:
        reward_sum, count, correct_count = mainQN.test(Data, actor, symbols, actions, mask1, mask2, test_epochs, max_number_of_steps, episode, dir_path, True, count, correct_count)
        if len(epochs) == 0:
            epoch = 0
        else:
            epoch = max(epochs) + test_epochs
        acc.append(reward_sum/(test_epochs))
        epochs.append(epoch)
        
        if acc[-1] >= prioritized_mode_border:
            PER_MODE = True
        
        plot_history(epochs, acc)
        
    """
    if len(epochs) == 0:
        epoch = 0
    else:
        epoch = max(epochs) + 1
    acc.append(reward_sum/(epoch+1))
    epochs.append(epoch)
    
    plot_history(epochs, acc)
    """
    
# obj_val_idx_list = np.delete(obj_val_idx_list, 0, 0) # ダミーで入れた最初の行を削除する
obj_val_idx_list.pop(0) # ダミーで入れた最初の行を削除する

np.save(dir_path + "obj_val_idx_list", obj_val_idx_list) # 特徴選択の推移を後で計算するため特徴選択のリストを保存する
with open(dir_path+'+LSTM_acc.pickle', mode='wb') as f:
    pickle.dump(acc, f)
    
plt.savefig('figure_' + datetime.date.today().strftime('%Y%m%d') + '.png')

if MODEL_LOAD == False:
    os.makedirs(dir_path+'check_points/', exist_ok=True)
    mainQN.model.save_weights(dir_path+'check_points/my_checkpoint')

print('----------------------------', file=codecs.open('perlabelaccuracy'+datetime.date.today().strftime('%Y%m%d')+'.txt', 'a', 'utf-8'))
print('count : ', count, file=codecs.open('perlabelaccuracy'+datetime.date.today().strftime('%Y%m%d')+'.txt', 'a', 'utf-8'))
print('correct_count : ', correct_count, file=codecs.open('perlabelaccuracy'+datetime.date.today().strftime('%Y%m%d')+'.txt', 'a', 'utf-8'))
print('per label accuracy : ', np.array(correct_count) / np.array(count), file=codecs.open('perlabelaccuracy'+datetime.date.today().strftime('%Y%m%d')+'.txt', 'a', 'utf-8'))
# In[ ]:





# In[ ]:




