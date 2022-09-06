#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding:utf-8
# [0]必要なライブラリのインポート
# import gym  # 倒立振子(cartpole)の実行環境
import numpy as np
import time
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from collections import deque
import random
import copy

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, LSTM, Bidirectional
from tensorflow.keras.activations import softmax, tanh
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.utils import plot_model

from tensorflow_addons.activations import mish
from tensorflow_addons.optimizers import RectifiedAdam


# In[2]:


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[3]:
def reward_func_backup220729(objectIndex, predict_idx, guess, step, max_step, test_mode, requests, request):
    reward = 0
    terminal = 0
    
    if step+1 == max_step:
        if test_mode == True:
            if objectIndex == predict_idx:
                reward = 1
                # correct_count[objectIndex] += 1 # 各ラベルの正解数をカウント
            else:
                reward = -1
                if guess > 0: # わからないを選択したとき 6/24
                    reward = 0.2
                else:
                    reward -= 0.2
        else:
            #reward = -5
            if objectIndex == predict_idx:
                reward = 1
                # correct_count[objectIndex] += 1 # 各ラベルの正解数をカウント
                # if request in requests: # 22/06/24 動作名が当たっている かつ 特徴選択がうまくいっているなら報酬を与える。特徴選択がだめなら報酬は与えない。
                #     reward = 1
                # else:
                #     reward = 0
            else:
                reward = -1
                if guess > 0: # わからないを選択したとき 6/24
                    reward = 0.2
                else:
                    reward -= 0.2
        terminal = 1
    elif objectIndex == predict_idx:
        if test_mode == True:
            reward = 1
        else:
            #reward = 1.0 - 0.2 * step
            reward = 1
            # correct_count[objectIndex] += 1 # 各ラベルの正解数をカウント
            # if request in requests: # 22/06/24 動作名が当たっている かつ 特徴選択がうまくいっているなら報酬を与える。特徴選択がだめなら報酬は与えない。
            #     reward = 1
            # else:
            #     reward = 0
        terminal = 1
    else:
        if test_mode == False:
            if guess > 0: # わからないを選択したとき 6/24
                reward = 0.2
            else:
                reward -= 0.2
            # if request in requests: # 特徴選択については報酬を設定しない 6/24
            #     # reward = -3
            #     reward = -0.5
            # else:
            #     # reward = 0.5
            #     reward = 0 # 22/06/24 動作名が当たっていない かつ 特徴選択がうまくいっていないなら報酬を与えない。
    
    return reward, terminal


def reward_func(objectIndex, predict_idx, guess, step, max_step, test_mode, requests, request):
    reward = 0
    terminal = 0
    
    if step+1 == max_step:
        if objectIndex == predict_idx:
            reward = 1
        else:
            reward = -1
            if guess == 1:
                reward = -0.2
            # elif guess == 0:
            #     reward = -1.2
        terminal = 1
    elif objectIndex == predict_idx:
        reward = 1
        terminal = 1
    else:
        reward = -1
        if guess == 1:
            reward = -0.2
        # elif guess == 0:
        #     reward = -1.2

    return reward, terminal


# In[4]:


def loss_func(y_true, y_pred):
    error = tf.abs(y_pred - y_true)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    return loss


# In[5]:


# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=83300, step_size=5, feature_size=0, 
                 action_size=4, object_size=6, output_size=10, embedding_size=200, hidden_size=100, hidden_size_2=16):
        self.state_size = state_size
        self.step_size = step_size
        self.action_size = action_size
        self.object_size = object_size
        self.output_size = output_size
        self.feature_length = feature_size
        
        inputs = Input(shape=(2*step_size, state_size,))
        #embedding = Dense(embedding_size, input_shape=(2*step_size, state_size,), activation='relu')(inputs)
        embedding = Dense(embedding_size, input_shape=(2*step_size, state_size,))(inputs)
        #embedding = tanh(embedding)
        embedding = mish(embedding)
        lstm = LSTM(hidden_size, batch_input_shape=(None, step_size, embedding_size), return_sequences=False)(embedding)
        #lstm, h_state, c_state = LSTM(hidden_size, batch_input_shape=(None, step_size, embedding_size), 
        #                              return_sequences=True, return_state=True)(embedding)
        #lstm = Bidirectional(LSTM(hidden_size, batch_input_shape=(None, step_size, embedding_size), 
        #                                              return_sequences=False))(embedding)
        out = Input(shape=(2*step_size, output_size,))
        lstm_2 = LSTM(hidden_size_2, input_shape=(None, step_size, output_size), return_sequences=False)(out)
        #lstm_2, h_state_2, c_state_2 = LSTM(hidden_size_2, input_shape=(None, step_size, output_size), 
        #                                    return_sequences=True, return_state=True)(out)
        #lstm_2 = Bidirectional(LSTM(hidden_size_2, batch_input_shape=(None, step_size, output_size),
        #                                                        return_sequences=False))(out)
        lstm = Concatenate()([lstm, lstm_2])
        #Q_F = Dense(action_size, activation='relu')(lstm)
        self.Q_F = Dense(action_size)(lstm)
        self.Q_F.trainable = True
        #Q_F = mish(Q_F)
        Q_F = softmax(self.Q_F)
        #Q_F.trainable = False
        #Q_N = Dense(object_size, activation='relu')(lstm)
        self.Q_N = Dense(object_size)(lstm)
        self.Q_N.trainable = True
        #Q_N = mish(Q_N)
        Q_N = softmax(self.Q_N)
        #Q_all = Concatenate()([Q_F, Q_N])
        predictions = Concatenate()([Q_F, Q_N])
        
        #predictions = Dense(output_size, activation='softmax')(Q_all)
        mask = Input(shape=(output_size,))
        predictions = Multiply()([predictions, mask])
        self.model = Model(inputs=[inputs, out, mask], outputs=predictions)
        #opt = Adam(lr=learning_rate)
        #opt = Adamax(lr=learning_rate)
        opt = RectifiedAdam(lr=learning_rate)
        self.model.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])
        
        self.model.summary()
        
    def change_learn(self):
        if self.Q_F.trainable == True:
            self.Q_F.trainable = False
            self.Q_N.trainable = True
        else:
            self.Q_F.trainable = True
            self.Q_N.trainable = False

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 2*self.step_size, self.state_size))
        state_t = np.zeros((1, 2*self.step_size, self.state_size))
        next_state_t = np.zeros((1, 2*self.step_size, self.state_size))
        
        out_vec = np.zeros((batch_size, 2*self.step_size, self.output_size))
        out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
        next_out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
        
        #targets = np.zeros((batch_size, 2*self.step_size, self.output_size)) 
        targets = np.zeros((batch_size, self.output_size)) # 怪しい
        targets_restore = np.zeros((batch_size, 2*self.step_size, self.output_size))
        masks = np.ones((batch_size, self.output_size))
        
        mini_batch = memory.sample(batch_size)
        
        for i, eps in enumerate(mini_batch):
            for j, (state_b, next_state_b, out_b, next_out_b, mask_b, next_mask_b, action_b, reward_b, terminal_b) in enumerate(eps):
                
                inputs[i][j:j+1] = state_b
                out_vec[i][j:j+1] = out_b
                next_state_t[j:j+1] = np.reshape(state_b, [1, self.state_size])
                next_out_vec_t[j:j+1] = np.reshape(out_b, [1, self.output_size])
                target = reward_b

                if not terminal_b:
                    # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値関数のQネットワークは分離）
                    next_state_b = np.reshape(next_state_b, [1, self.state_size])
                    next_mask_b = np.reshape(next_mask_b, [1, self.output_size])
                    next_state_t[j+1:j+2] = next_state_b
                    #next_state_t[j+1:j+2] = np.reshape(next_state_b, [1, self.state_size])
                    next_out_vec_t[j+1:j+2] = np.reshape(next_out_b, [1, self.output_size]) 
                    retmainQs = self.model.predict([next_state_t, next_out_vec_t, next_mask_b], verbose=0)
                    next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                    target = reward_b + gamma * targetQN.model.predict([next_state_t, next_out_vec_t, next_mask_b], verbose=0)[0][next_action]

                state_b = np.reshape(state_b, [1, self.state_size])
                mask_b = np.reshape(mask_b, [1, self.output_size])
                state_t[j:j+1] = state_b
                out_vec_t[j:j+1] = np.reshape(out_b, [1, self.output_size])
                #targets[i][j] = self.model.predict([state, mask_b])    # Qネットワークの出力(Q値)
                #targets[i][j][action_b] = target               # 教師信号
                #targets[i] = self.model.predict([state, mask_b])    # 怪しい
                targets_restore[i][j] = self.model.predict([state_t, out_vec_t, mask_b], verbose=0)
                #targets[i][action_b] = target               # 怪しい
                targets_restore[i][j][action_b] = target
                
            targets[i] = np.mean(targets_restore[i], axis=0)
        history = self.model.fit([inputs, out_vec, masks], targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        
        return history
    
    # 優先度付き経験再生
    def prioritized_experience_replay(self, memory, batch_size, gamma, targetQN, memory_TDerror):
        
        sum_absolute_TDerror = memory_TDerror.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, batch_size)
        generatedrand_list = np.sort(generatedrand_list)
        
        batch_memory = []
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(memory_TDerror.buffer[idx]) + 0.0001
                idx += 1
                
            batch_memory.append(memory.buffer[idx])
            
        inputs = np.zeros((batch_size, 2*self.step_size, self.state_size))
        targets = np.zeros((batch_size, self.output_size))
        inputs = np.zeros((batch_size, 2*self.step_size, self.state_size))
        state_t = np.zeros((1, 2*self.step_size, self.state_size))
        next_state_t = np.zeros((1, 2*self.step_size, self.state_size))
        
        out_vec = np.zeros((batch_size, 2*self.step_size, self.output_size))
        out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
        next_out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
        
        targets = np.zeros((batch_size, self.output_size)) # 怪しい
        targets_restore = np.zeros((batch_size, 2*self.step_size, self.output_size))
        masks = np.ones((batch_size, self.output_size))
        
        for i, eps in enumerate(batch_memory):
            for j, (state_b, next_state_b, out_b, next_out_b, mask_b, next_mask_b, action_b, reward_b, terminal_b) in enumerate(eps):
                
                inputs[i][j:j+1] = state_b
                out_vec[i][j:j+1] = out_b
                next_state_t[j:j+1] = np.reshape(state_b, [1, self.state_size])
                next_out_vec_t[j:j+1] = np.reshape(out_b, [1, self.output_size])
                target = reward_b

                if not terminal_b:
                    # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値関数のQネットワークは分離）
                    next_state_b = np.reshape(next_state_b, [1, self.state_size])
                    next_mask_b = np.reshape(next_mask_b, [1, self.output_size])
                    next_state_t[j+1:j+2] = next_state_b
                    #next_state_t[j+1:j+2] = np.reshape(next_state_b, [1, self.state_size])
                    next_out_vec_t[j+1:j+2] = np.reshape(next_out_b, [1, self.output_size]) 
                    retmainQs = self.model.predict([next_state_t, next_out_vec_t, next_mask_b], verbose=0)
                    next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                    target = reward_b + gamma * targetQN.model.predict([next_state_t, next_out_vec_t, next_mask_b], verbose=0)[0][next_action]

                state_b = np.reshape(state_b, [1, self.state_size])
                mask_b = np.reshape(mask_b, [1, self.output_size])
                state_t[j:j+1] = state_b
                out_vec_t[j:j+1] = np.reshape(out_b, [1, self.output_size])
                targets_restore[i][j] = self.model.predict([state_t, out_t, mask_b], verbose=0)
                targets_restore[i][j][action_b] = target
                
            targets[i] = np.mean(targets_restore[i], axis=0)
        history = self.model.fit([inputs, out_vec, masks], targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        
        return history
        
    
    def test(self, Data, actor, symbols, actions, mask1, mask2, test_epochs, max_number_of_steps, num_episode, dir_path, optune, count, correct_count):
        reward_sum = 0
        result_df = pd.DataFrame(columns=['ans','fea_1','name_1','fea_2','name_2','fea_3','name_3','fea_4','name_4','fea_5','name_5',
                                              'reward_1', 'reward_2', 'reward_3', 'reward_4', 'reward_5'], dtype=object)
        
        for episode in range(test_epochs):
            rand = np.random.randint(0, 65535)
            obj = [0] * self.object_size
            fea_vec = [0] * self.action_size # 特徴の種類
            fea_val = [0] * self.feature_length # 特徴量の値
            requests = []
            guess = [0] # not_sureを選んだ回数
            not_sure_count = 0 # not_sureをそのエピソードで選んだかどうかを保存するフラグ
            tmp_info = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                 np.nan, np.nan, np.nan, np.nan, np.nan], dtype=object)
            objectIndex = random.randint(0, self.object_size-2) # 当ててほしい物体のindex

            count[objectIndex] += 1 # 各ラベルのカウント 

            tmp_info[0] = symbols[objectIndex]
            action_step_state = np.zeros((1, 2*self.step_size, self.state_size))
            out = np.zeros((1, 2*self.step_size, self.output_size))
            for step in range(max_number_of_steps):
                pre_fea_vec = copy.deepcopy(fea_vec)
                pre_obj_vec = copy.deepcopy(obj)
                #fea_val = [0] * self.feature_length # 特徴量の値
                state1 = np.concatenate([fea_vec, fea_val, obj, guess])
                state1 = np.reshape(state1, [1, self.state_size])
                mask1 = np.reshape(mask1, [1, self.output_size])
                action_step_state[0][step] = state1
                out_1 = np.concatenate([pre_fea_vec, pre_obj_vec])
                out[0][step] = np.reshape(out_1, [1, self.output_size])
                obj_val_idx = np.argmax(self.model.predict([action_step_state, out,  mask1], verbose=0))# 時刻tで取得する特徴量を決定
                
                tmp_info[2*step+1] = actions[obj_val_idx]

                fea_vec[obj_val_idx] = 1
                request = actions[obj_val_idx]
                requests.append(request)
                fea_val = Data.Overfetch_V2(objectIndex, list(set(requests)), rand)       
                state2 = np.concatenate([fea_vec, fea_val, obj, guess])
                state2 = np.reshape(state2, [1, self.state_size])
                mask2 = np.reshape(mask2, [1, self.output_size])
                action_step_state[0][step+1] = state2
                out_2 = np.concatenate([fea_vec, pre_obj_vec])
                out[0][step+1] = np.reshape(out_2, [1, self.output_size])
                obj_name_idx = np.argmax(self.model.predict([action_step_state, out, mask2], verbose=0)) # 時刻tで取得する物体の名称を決定

                name = symbols[obj_name_idx - self.action_size]
                obj[obj_name_idx - self.action_size] = 0
                if name == 'not_sure':
                    guess[0] += 1
                    not_sure_count = 1
                    
                tmp_info[2*step+2] = name

                # 報酬を設定し、与える
                # reward, terminal = reward_func(objectIndex, (obj_name_idx - self.action_size), guess[0], step, max_number_of_steps, True, requests, request)
                reward, terminal = reward_func(objectIndex, (obj_name_idx - self.action_size), not_sure_count, step, max_number_of_steps, True, requests, request)
                not_sure_count = 0
                
                tmp_info[step+1+10] = reward
                
                state1 = np.concatenate([fea_vec, fea_val, obj, guess])
                
                if terminal:
                    if objectIndex == (obj_name_idx - self.action_size):
                        correct_count[objectIndex] += 1 # 各ラベルの正答数をカウント
                    tmp_se = pd.Series(tmp_info, index=result_df.columns, name=episode+1)
                    # result_df = result_df.append(tmp_se, ignore_index=True) # .appendが非推奨のため
                    result_df = pd.concat([result_df, pd.DataFrame([tmp_se])], ignore_index=True)
                    # print('tmp_se : ', tmp_se)
                    # print('result_df : ', result_df)
                    if reward == 1:
                        reward_sum += 1
                        break
        
        if optune == True:
            result_df.to_csv(dir_path+str(num_episode+1)+'_result.csv')
        
        return reward_sum, count, correct_count


# In[6]:


# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch_buffer = []
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        for i in idx:
            batch_buffer.append(self.buffer[i])
            
        return batch_buffer

    def len(self):
        return len(self.buffer)


# In[7]:


class Memory_TDerror(Memory):
    def __init__(self, max_size=1000, state_size=83300, output_size=10, step_size=5):
        super().__init__(max_size)
        self.state_size = state_size
        self.output_size = output_size
        self.step_size = step_size
        
    def get_TDerror(self, memory, gamma, mainQN, targetQN):
        TDerror = []
        state_t = np.zeros((1, 2*self.step_size, self.state_size))
        next_state_t = np.zeros((1, 2*self.step_size, self.state_size))
        out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
        next_out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
        
        for j, (state, next_state, out, next_out, mask, next_mask, action, reward, terminal) in enumerate(memory.buffer[memory.len() - 1]):
            next_state_t[j:j+1] = np.reshape(state, [1, self.state_size])
            next_out_vec_t[j:j+1] = np.reshape(state, [1, self.output_size])
            
            next_state = np.reshape(next_state, [1, self.state_size])
            next_mask = np.reshape(next_mask, [1, self.output_size])
            next_state_t[j+1:j+2] = next_state
            #next_state_t[j+1:j+2] = np.reshape(next_state, [1, self.state_size])
            next_out_vec_t[j+1:j+2] = np.reshape(next_out, [1, self.output_size])
            retmainQs = mainQN.model.predict([next_state_t, next_out_vec_t, next_mask], verbose=0)
            next_action = np.argmax(retmainQs)
            target = reward + gamma * targetQN.model.predict([next_state_t, next_out_vec_t, next_mask], verbose=0)[0][next_action]

            state = np.reshape(state, [1, self.state_size])
            mask = np.reshape(mask, [1, self.output_size])
            state_t[j:j+1] = state
            out_t[j:j+1] = np.reshape(out, [1, self.output_size])
            
            TDerror.append(target - targetQN.model.predict([state_t, out_t, mask], verbose=0)[0][action])
        
        return sum(TDerror) / len(TDerror)
        
    def update_TDerror(self, memory, gamma, mainQN, targetQN):
        for i in range(0, (self.len() - 1)):
            TDerror = []
            state_t = np.zeros((1, 2*self.step_size, self.state_size))
            next_state_t = np.zeros((1, 2*self.step_size, self.state_size))
            out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
            next_out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
            
            for j, (state, next_state, out, next_out, mask, next_mask, action, reward, terminal) in enumerate(memory.buffer[i]):
                next_state_t[j:j+1] = np.reshape(state, [1, self.state_size])
                next_out_vec_t[j:j+1] = np.reshape(state, [1, self.output_size])
                
                next_state = np.reshape(next_state, [1, self.state_size])
                next_mask = np.reshape(next_mask, [1, self.output_size])
                next_state_t[j+1:j+2] = next_state
                #next_state_t[j+1:j+2] = np.reshape(next_state, [1, self.state_size])
                next_out_vec_t[j+1:j+2] = np.reshape(next_out, [1, self.output_size])
                retmainQs = mainQN.model.predict([next_state_t, next_out_vec_t, next_mask], verbose=0)
                next_action = np.argmax(retmainQs)
                target = reward + gamma * targetQN.model.predict([next_state_t, next_out_vec_t, next_mask], verbose=0)[0][next_action]

                state = np.reshape(state, [1, self.state_size])
                mask = np.reshape(mask, [1, self.output_size])
                state_t[j:j+1] = state
                out_t[j:j+1] = np.reshape(out, [1, self.output_size])
                TDerror.append(target - targetQN.model.predict([state_t, out_t, mask], verbose=0)[0][action])
                
            self.buffer[i] = sum(TDerror) / len(TDerror)
            
    def get_sum_absolute_TDerror(self):
        sum_absolute_TDerror = 0
        for i in range(0, (self.len() - 1)):
            sum_absolute_TDerror += abs(self.buffer[i]) + 0.0001
            
        return sum_absolute_TDerror


# In[8]:


# [4]行動を決定するクラス
class Actor:
    def __init__(self, features_length, objects_length, actions_length, output_size, MODEL_LOAD):
        self.features_length = features_length
        self.objects_length = objects_length
        self.actions_length = actions_length
        self.output_size = output_size
        self.MODEL_LOAD = MODEL_LOAD
        
    def get_value(self, state, out, mask, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.1 + 0.9 / (1.0+(episode/500))
        
        if epsilon <= np.random.uniform(0, 1) and episode != 0 or self.MODEL_LOAD == True:
            # print("optimum")
            retTargetQs = mainQN.model.predict([state, out, mask], verbose=0)
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
        else:
            # print("random")
            # action = random.randint(0, self.actions_length-1)  # ランダムに行動する
            retTargetQs = np.random.rand(self.actions_length)  # ランダムに行動する
            retTargetQs /= retTargetQs.sum() # 合計で1になるようにする
            action = np.argmax(retTargetQs)

        # return action
        return action, retTargetQs # 特徴選択の確率をみたいため、retTargetQsを返り値に追加
    
    def get_name(self, state, out, mask, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.01 + 0.99 / (1.0+episode)

        if epsilon <= np.random.uniform(0, 1) and episode != 0 or self.MODEL_LOAD == True:
            retTargetQs = mainQN.model.predict([state, out, mask], verbose=0)
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
        else:
            action = random.randint(self.actions_length, self.actions_length+self.objects_length-1)  # ランダムに行動する

        return action


# In[9]:


# import subprocess
# subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'Pre_Model.ipynb'])


# In[ ]:




