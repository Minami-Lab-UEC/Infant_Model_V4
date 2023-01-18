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
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, LSTM, Bidirectional, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.activations import softmax, tanh, relu
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.utils import plot_model

from tensorflow_addons.activations import mish
from tensorflow_addons.optimizers import RectifiedAdam


def reward_func(objectIndex, predict_idx, guess, step, max_step, test_mode, requests, request,predict_val_idx, verb_idx, symbols, symbols_verb, parent_select, name):
    reward = 0 # 正解か不正解か
    terminal = 0 # 終了しているか否か

    # test_modeはpre_main.pyでFalseにされている

    if objectIndex == predict_idx:
        reward = 1
        terminal = 1
    else:
        reward = -1
        if guess == 1:
            reward = -0.2
    
    if step+1 == max_step:
        terminal = 1
    
    return reward, terminal

def reward_func_verbnoun(objectIndex, predict_idx, guess, step, max_step, test_mode, requests, request,predict_val_idx, verb_idx, symbols, symbols_verb, parent_select, name, reward, terminal):
    # test_modeはpre_main.pyでFalseにされている
    if objectIndex == predict_idx:
        reward[parent_select] = 1
        terminal[parent_select] = 1 
    else:
        reward[parent_select] = -1
        if guess == 1:
            reward[parent_select] = -0.2
    if step+1 == max_step:
        terminal = [1] * 2
    
    return reward, terminal

def reward_func_loop_weight(objectIndex, predict_idx, guess, step, max_step, test_mode, requests, request,predict_val_idx, verb_idx, symbols, symbols_verb, parent_select, name, reward, terminal):
    # test_modeはpre_main.pyでFalseにされている
    if objectIndex == predict_idx:
        reward[parent_select] = 1
        terminal[parent_select] = 1 
    else:
        reward[parent_select] = -1
        if guess == 1:
            reward[parent_select] = -0.2
        if parent_select == 0 and symbols[predict_idx] not in symbols_verb: # loop1回目で名詞を選べていたら
            reward[parent_select] = -0.2
        if parent_select == 1 and symbols[predict_idx] in symbols_verb: # loop2回目で動詞を選べていたら
            reward[parent_select] = -0.2
    if step+1 == max_step:
        terminal = [1] * 2
    
    return reward, terminal


def loss_func(y_true, y_pred):
    error = tf.abs(y_pred - y_true)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    return loss


# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=83300, step_size=5, loop_size=2, feature_size=0, 
                 action_size=4, object_size=6, output_size=10, parent_size=4, embedding_size=200, hidden_size=100, hidden_size_2=16,
                  p_hidden_size=32):
        self.state_size = state_size
        self.step_size = step_size
        self.loop_size = loop_size
        self.action_size = action_size
        self.object_size = object_size
        self.output_size = output_size
        self.parent_size = parent_size
        self.feature_length = feature_size
        
        inputs = Input(shape=(2*step_size, state_size,))
        embedding = Dense(embedding_size, input_shape=(2*step_size, state_size,))(inputs)
        embedding = tanh(embedding)
        
        lstm = LSTM(hidden_size, batch_input_shape=(None, step_size, hidden_size), return_sequences=False)(embedding)
        
        out = Input(shape=(2*step_size, output_size,))
        lstm_2 = LSTM(hidden_size_2, input_shape=(None, step_size, output_size), return_sequences=False)(out)
        
        parent_order = Input(shape=(parent_size, parent_size, 1,))
        
        p_conv_1 = Conv2D(p_hidden_size*2,6,input_shape=(None, parent_size, parent_size, 1))(parent_order)
        p_conv_1 = relu(p_conv_1)
        p_max_pool_1 = MaxPool2D(pool_size=(6,6))(p_conv_1)
        p_conv_2 = Conv2D(p_hidden_size,6,input_shape=(None, parent_size, parent_size, 1))(parent_order)
        p_conv_2 = relu(p_conv_2)
        p_max_pool_2 = MaxPool2D(pool_size=(6,6))(p_conv_2)
        flat = Flatten()(p_max_pool_2)
        #p_dense_1 = Dense(p_hidden_size)(parent_order)
        p_dense_1 = Dense(500)(flat)
        p_dense_1 = relu(p_dense_1)
        p_dense_2 = Dense(2)(p_dense_1)
        
        parent_output = softmax(p_dense_2)
        
        lstm = Concatenate()([lstm, lstm_2, parent_output])
        Q_F = Dense(action_size)(lstm)
        Q_F = softmax(Q_F)
        
        Q_N = Dense(object_size)(lstm)
        Q_N = softmax(Q_N)
        
        #Q_all = Concatenate()([Q_F, Q_N])
        predictions = Concatenate()([Q_F, Q_N])
        
        # predictions = Dense(output_size, activation='softmax')(Q_all)
        mask = Input(shape=(output_size,))
        predictions = Multiply()([predictions, mask])
        self.model = Model(inputs=[inputs, out, mask, parent_order], outputs=predictions)
        opt = Adam(learning_rate=learning_rate)
        #self.model.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])
        self.model.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])
        
        self.model.summary()
        
    def check_bias(self, data):
        names = [l.name for l in self.model.layers]
        print(names)
        layer_name = 'tf.nn.softmax'
        intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        return intermediate_layer_model(data)

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
        
        #parent_orders = np.zeros((batch_size, self.parent_size))
        parent_orders = np.zeros((batch_size, self.parent_size, self.parent_size, 1))
        
        for i, eps in enumerate(mini_batch):
            for j, (state_b, next_state_b, out_b, next_out_b, mask_b, next_mask_b, action_b, reward_b, terminal_b, parent_order_b) in enumerate(eps):
                
                inputs[i][j:j+1] = state_b
                out_vec[i][j:j+1] = out_b
                next_state_t[j:j+1] = np.reshape(state_b, [1, self.state_size])
                next_out_vec_t[j:j+1] = np.reshape(out_b, [1, self.output_size])
                target = reward_b
                parent_order_b = np.reshape(parent_order_b, [1, self.parent_size, self.parent_size, 1])
                parent_orders[i] = parent_order_b
                #parent_order_b = np.reshape(parent_order_b, [1, self.parent_size])
                #parent_order_b = np.reshape(parent_order_b, [1, self.parent_size, self.parent_size, 1])

                if not terminal_b:
                    # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値関数のQネットワークは分離）
                    next_state_b = np.reshape(next_state_b, [1, self.state_size])
                    next_mask_b = np.reshape(next_mask_b, [1, self.output_size])
                    
                    next_state_t[j+1:j+2] = next_state_b
                    next_out_vec_t[j+1:j+2] = np.reshape(next_out_b, [1, self.output_size]) 
                    retmainQs = self.model([next_state_t, next_out_vec_t, next_mask_b,  parent_order_b])
                    next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                    target = reward_b + gamma * targetQN.model([next_state_t, next_out_vec_t, next_mask_b, parent_order_b])[0][next_action]

                state_b = np.reshape(state_b, [1, self.state_size])
                mask_b = np.reshape(mask_b, [1, self.output_size])
                state_t[j:j+1] = state_b
                out_vec_t[j:j+1] = np.reshape(out_b, [1, self.output_size])
                #targets[i][j] = self.model([state, mask_b])    # Qネットワークの出力(Q値)
                #targets[i][j][action_b] = target               # 教師信号
                #targets[i] = self.model([state, mask_b])    # 怪しい
                targets_restore[i][j] = self.model([state_t, out_vec_t, mask_b, parent_order_b])
                #targets[i][action_b] = target               # 怪しい
                targets_restore[i][j][action_b] = target
                
            targets[i] = np.mean(targets_restore[i], axis=0)
        history = self.model.fit([inputs, out_vec, masks, parent_orders], targets, epochs=1)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        
        return history
    
    # 優先度付き経験再生
    def prioritized_experience_replay(self, memory, batch_size, gamma, targetQN, memory_TDerror):
        
        # TDerror_calc = time.time()
        sum_absolute_TDerror = memory_TDerror.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, batch_size)
        generatedrand_list = np.sort(generatedrand_list)
        # print(f'one time TDerror calc : {time.time() - TDerror_calc}')
        
        # batch_sel = time.time()
        batch_memory = Memory(max_size=batch_size)
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(memory_TDerror.buffer[idx]) + 0.0001
                idx += 1
                
            batch_memory.add(memory.buffer[idx])
        # print(f'one time batch select : {time.time() - batch_sel}')

        
        inputs = np.zeros((batch_size, 2*self.step_size, self.state_size))
        targets = np.zeros((batch_size, self.output_size))
        state_t = np.zeros((1, 2*self.step_size, self.state_size))
        next_state_t = np.zeros((1, 2*self.step_size, self.state_size))
        
        out_vec = np.zeros((batch_size, 2*self.step_size, self.output_size))
        out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
        next_out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
        
        targets_restore = np.zeros((batch_size, 2*self.step_size, self.output_size))
        masks = np.ones((batch_size, self.output_size))
        #parent_orders = np.zeros((batch_size, self.parent_size))
        parent_orders = np.zeros((batch_size, self.parent_size, self.parent_size, 1))
        
        for i, eps in enumerate(batch_memory.buffer):
            for j, (state_b, next_state_b, out_b, next_out_b, mask_b, next_mask_b, action_b, reward_b, terminal_b, parent_order_b) in enumerate(eps):
                # batch_calc = time.time()
                inputs[i][j:j+1] = state_b
                out_vec[i][j:j+1] = out_b
                next_state_t[j:j+1] = np.reshape(state_b, [1, self.state_size])
                next_out_vec_t[j:j+1] = np.reshape(out_b, [1, self.output_size])
                target = reward_b
                parent_order_b = np.reshape(parent_order_b, [1, self.parent_size, self.parent_size, 1])
                parent_orders[i] = parent_order_b
                #parent_order_b = np.reshape(parent_order_b, [1, self.parent_size])
                

                if not terminal_b:
                    # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値関数のQネットワークは分離）
                    next_state_b = np.reshape(next_state_b, [1, self.state_size])
                    next_mask_b = np.reshape(next_mask_b, [1, self.output_size])
                    next_state_t[j+1:j+2] = next_state_b
                    next_out_vec_t[j+1:j+2] = np.reshape(next_out_b, [1, self.output_size]) 
                    retmainQs = self.model([next_state_t, next_out_vec_t, next_mask_b, parent_order_b])
                    next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                    target = reward_b + gamma * targetQN.model([next_state_t, next_out_vec_t, next_mask_b, parent_order_b])[0][next_action]

                state_b = np.reshape(state_b, [1, self.state_size])
                mask_b = np.reshape(mask_b, [1, self.output_size])
                state_t[j:j+1] = state_b
                out_vec_t[j:j+1] = np.reshape(out_b, [1, self.output_size])
                targets_restore[i][j] = self.model([state_t, out_vec_t, mask_b, parent_order_b])
                targets_restore[i][j][action_b] = target
                # print(f'one time batch calc : {time.time() - batch_calc}')
            targets[i] = np.mean(targets_restore[i], axis=0)
                
        # fit_time = time.time()
        history = self.model.fit([inputs, out_vec, masks, parent_orders], targets, epochs=1)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        # print(f'one time fit : {time.time() - fit_time}')
        return history
        
    
    def test(self, Data, actor, symbols, symbols_noun, symbols_verb, actions, mask1, mask2, test_epochs, max_number_of_steps, num_episode, dir_path, per_error, parent_FE):
        reward_sum = 0
        reward_sum_n = 0
        reward_sum_v = 0
        
        for episode in range(test_epochs):
            rand = np.random.randint(0, 65535)
            obj = [0] * self.object_size
            fea_vec = [0] * self.action_size # 特徴の種類
            fea_val = [0] * self.feature_length # 特徴量の値
            requests = []
            # guess = [0] # not_sureを選んだ回数
            cur_loop = [0] # ループ回数を保存、名詞のみを学習するときは使わない
            not_sure_count = 0 # not_sureをそのエピソードで選んだかどうかを保存するフラグ
            
            parent_intent = [1, 0]
            parent_select = 0
            objectIndex = random.randint(0, len(symbols_noun)-2)
            # if episode < (test_epochs//2):
            #     objectIndex = random.randint(0, len(symbols_noun)-2)
            # else:
            #     parent_select = 1
            #     objectIndex = random.randint(len(symbols_noun)-1, len(symbols)-2)
            #     parent_intent = [0, 1]
                
            infant_intent = [0, 0]
            idx = random.randint(0, 1)
            infant_intent[idx] = 1
    
            #parent_order = [0, 0, 0, 0]
            if parent_intent == infant_intent:
                #parent_order = [1, 0, 0, 0] #joy, angry, sad, neutral
                idx = random.randint(0, 1)
                parent_order = parent_FE[idx]
            else:
                idx = random.randint(0, 5)
                parent_order = parent_FE[idx]

            action_step_state = np.zeros((1, 2*self.step_size, self.state_size))
            out = np.zeros((1, 2*self.step_size, self.output_size))
            parent_order = np.reshape(parent_order, [1, self.parent_size, self.parent_size, 1])
            for step in range(max_number_of_steps):
                pre_fea_vec = copy.deepcopy(fea_vec)
                pre_obj_vec = copy.deepcopy(obj)
                state1 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
                state1 = np.reshape(state1, [1, self.state_size])
                mask1 = np.reshape(mask1, [1, self.output_size])
                action_step_state[0][step] = state1
                out_1 = np.concatenate([pre_fea_vec, pre_obj_vec])
                out[0][step] = np.reshape(out_1, [1, self.output_size])
                
                obj_val_idx = np.argmax(self.model([action_step_state, out, mask1, parent_order]))# 時刻tで取得する特徴量を決定

                fea_vec[obj_val_idx] = 1
                request = actions[obj_val_idx]
                requests.append(request)
                fea_val = Data.Overfetch(objectIndex, list(set(requests)), rand, parent_select)       
                state2 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
                state2 = np.reshape(state2, [1, self.state_size])
                mask2 = np.reshape(mask2, [1, self.output_size])
                action_step_state[0][step+1] = state2
                out_2 = np.concatenate([fea_vec, pre_obj_vec])
                out[0][step+1] = np.reshape(out_2, [1, self.output_size])
                obj_name_idx = np.argmax(self.model([action_step_state, out, mask2, parent_order])) # 時刻tで取得する物体の名称を決定

                name = symbols[obj_name_idx - self.action_size]
                obj[obj_name_idx - self.action_size] = 0
                if name == 'not_sure':
                    not_sure_count = 1

                # 報酬を設定し、与える
                reward, terminal = reward_func(objectIndex, (obj_name_idx - self.action_size), not_sure_count, step, max_number_of_steps,
                                                                   True, requests, request, obj_val_idx, 3, symbols, symbols_verb, parent_select, name)
                not_sure_count = 0
                
                state1 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
                
                if terminal:
                    if reward == 1:
                        reward_sum += 1
                        if episode < (test_epochs//2):
                            reward_sum_n += 1
                        else:
                            reward_sum_v += 1
                        break
        
        return reward_sum, reward_sum_n, reward_sum_v

    def test_verbnoun(self, Data, actor, symbols, symbols_noun, symbols_verb, actions, mask1, mask2, test_epochs, max_number_of_steps, max_number_of_loops, num_episode, dir_path, per_error, parent_FE):
        reward_sum = 0
        reward_sum_n = 0
        reward_sum_v = 0
        
        for episode in range(test_epochs):
            rand = np.random.randint(0, 65535)
            obj = [[0] * self.object_size] * max_number_of_loops
            fea_vec = [[0] * self.action_size] * max_number_of_loops # 特徴の種類
            fea_val = [[0] * self.feature_length] * max_number_of_loops # 特徴量の値
            requests = [[], []]
            # guess = [0] # not_sureを選んだ回数
            cur_loop = [0] # ループ回数を保存、stateに組み込む
            not_sure_count = 0 # not_sureをそのエピソードで選んだかどうかを保存するフラグ

            action_step_state = np.zeros((max_number_of_loops, 1, 2*self.step_size, self.state_size))
            out = np.zeros((max_number_of_loops, 1, 2*self.step_size, self.output_size))

            # 答えを決める
            objectIndex = [0, 0]
            objectIndex[0] = random.randint(0, len(symbols_noun)-2)
            if objectIndex[0] <= 1:
                objectIndex[1] = 5
            else:
                objectIndex[1] = objectIndex[0] + 4
            
            for step in range(max_number_of_steps):
                reward = [0, 0]
                terminal = [0, 0]
                for loop in range(max_number_of_loops):
                    cur_loop[0] = loop
                    parent_intent = [0, 0]
                    parent_intent[loop] = 1
                    parent_select = loop
                     
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

                    parent_order = np.reshape(parent_order, [1, self.parent_size, self.parent_size, 1])

                    pre_fea_vec = copy.deepcopy(fea_vec[loop])
                    pre_obj_vec = copy.deepcopy(obj[loop])
                    state1 = np.concatenate([fea_vec[loop], fea_val[loop], obj[loop], cur_loop])
                    state1 = np.reshape(state1, [1, self.state_size])
                    mask1 = np.reshape(mask1, [1, self.output_size])
                    action_step_state[loop][0][step] = state1
                    out_1 = np.concatenate([pre_fea_vec, pre_obj_vec])
                    out[loop][0][step] = np.reshape(out_1, [1, self.output_size])
                    
                    obj_val_idx = np.argmax(self.model([action_step_state[loop], out[loop], mask1, parent_order]))# 時刻tで取得する特徴量を決定

                    fea_vec[loop][obj_val_idx] = 1
                    request = actions[obj_val_idx]
                    requests[loop].append(request)
                    fea_val[loop] = Data.Overfetch(objectIndex[loop], list(set(requests[loop])), rand, parent_select)       
                    state2 = np.concatenate([fea_vec[loop], fea_val[loop], obj[loop], cur_loop])
                    state2 = np.reshape(state2, [1, self.state_size])
                    mask2 = np.reshape(mask2, [1, self.output_size])
                    action_step_state[loop][0][step+1] = state2
                    out_2 = np.concatenate([fea_vec[loop], pre_obj_vec])
                    out[loop][0][step+1] = np.reshape(out_2, [1, self.output_size])
                    obj_name_idx = np.argmax(self.model([action_step_state[loop], out[loop], mask2, parent_order])) # 時刻tで取得する物体の名称を決定

                    name = symbols[obj_name_idx - self.action_size]
                    obj[loop][obj_name_idx - self.action_size] = 0
                    if name == 'not_sure':
                        not_sure_count = 1

                    # 報酬を設定し、与える
                    reward, terminal = reward_func_verbnoun(objectIndex[loop], (obj_name_idx - self.action_size), not_sure_count, step, max_number_of_steps,
                                                                    True, requests, request, obj_val_idx, 3, symbols, symbols_verb, parent_select, name, reward, terminal)
                    not_sure_count = 0
                    
                    state1 = np.concatenate([fea_vec[loop], fea_val[loop], obj[loop], cur_loop])

                if terminal == [1, 1]:
                    if reward == [1, 1]:
                        reward_sum += 1
                        # reward_sum_n, _vを計算する機構を用意する

        
        return reward_sum, reward_sum_n, reward_sum_v
    
    def test_onememory(self, Data, actor, symbols, symbols_noun, symbols_verb, actions, mask1, mask2, test_epochs, max_number_of_steps, max_number_of_loops, num_episode, dir_path, per_error, parent_FE):
        reward_sum = 0
        reward_sum_n = 0
        reward_sum_v = 0
        
        for episode in range(test_epochs):
            rand = np.random.randint(0, 65535)
            obj = [0] * self.object_size
            fea_vec = [0] * self.action_size # 特徴の種類
            fea_val = [0] * self.feature_length # 特徴量の値
            requests = []
            # guess = [0] # not_sureを選んだ回数
            cur_loop = [0] # ループ回数を保存、stateに組み込む
            not_sure_count = 0 # not_sureをそのエピソードで選んだかどうかを保存するフラグ

            action_step_state = np.zeros((max_number_of_loops, 1, 2*self.step_size, self.state_size))
            out = np.zeros((max_number_of_loops, 1, 2*self.step_size, self.output_size))

            # 答えを決める
            objectIndex = [0, 0]
            objectIndex[0] = random.randint(0, len(symbols_noun)-2)
            if objectIndex[0] <= 1:
                objectIndex[1] = 5
            else:
                objectIndex[1] = objectIndex[0] + 4
            
            for step in range(max_number_of_steps):
                reward = [0, 0]
                terminal = [0, 0]
                for loop in range(max_number_of_loops):
                    cur_loop[0] = loop
                    parent_intent = [0, 0]
                    parent_intent[loop] = 1
                    parent_select = loop
                     
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

                    parent_order = np.reshape(parent_order, [1, self.parent_size, self.parent_size, 1])

                    pre_fea_vec = copy.deepcopy(fea_vec)
                    pre_obj_vec = copy.deepcopy(obj)
                    state1 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
                    state1 = np.reshape(state1, [1, self.state_size])
                    mask1 = np.reshape(mask1, [1, self.output_size])
                    action_step_state[loop][0][step] = state1
                    out_1 = np.concatenate([pre_fea_vec, pre_obj_vec])
                    out[loop][0][step] = np.reshape(out_1, [1, self.output_size])
                    
                    obj_val_idx = np.argmax(self.model([action_step_state[loop], out[loop], mask1, parent_order]))# 時刻tで取得する特徴量を決定

                    fea_vec[obj_val_idx] = 1
                    request = actions[obj_val_idx]
                    requests.append(request)
                    fea_val = Data.Overfetch(objectIndex[loop], list(set(requests)), rand, parent_select)       
                    state2 = np.concatenate([fea_vec, fea_val, obj, cur_loop])
                    state2 = np.reshape(state2, [1, self.state_size])
                    mask2 = np.reshape(mask2, [1, self.output_size])
                    action_step_state[loop][0][step+1] = state2
                    out_2 = np.concatenate([fea_vec, pre_obj_vec])
                    out[loop][0][step+1] = np.reshape(out_2, [1, self.output_size])
                    obj_name_idx = np.argmax(self.model([action_step_state[loop], out[loop], mask2, parent_order])) # 時刻tで取得する物体の名称を決定

                    name = symbols[obj_name_idx - self.action_size]
                    obj[obj_name_idx - self.action_size] = 0
                    if name == 'not_sure':
                        not_sure_count = 1

                    # 報酬を設定し、与える
                    reward, terminal = reward_func_verbnoun(objectIndex[loop], (obj_name_idx - self.action_size), not_sure_count, step, max_number_of_steps,
                                                                    True, requests, request, obj_val_idx, 3, symbols, symbols_verb, parent_select, name, reward, terminal)
                    not_sure_count = 0
                    
                    state1 = np.concatenate([fea_vec, fea_val, obj, cur_loop])

                if terminal == [1, 1]:
                    if reward == [1, 1]:
                        reward_sum += 1
                        # reward_sum_n, _vを計算する機構を用意する

        
        return reward_sum, reward_sum_n, reward_sum_v

    def test_loopweight(self, Data, actor, symbols, symbols_noun, symbols_verb, actions, mask1, mask2, test_epochs, max_number_of_steps, max_number_of_loops, num_episode, dir_path, per_error, parent_FE):
        reward_sum = 0
        reward_sum_n = 0
        reward_sum_v = 0
        
        for episode in range(test_epochs):
            rand = np.random.randint(0, 65535)
            obj = [[0] * self.object_size] * max_number_of_loops
            fea_vec = [[0] * self.action_size] * max_number_of_loops # 特徴の種類
            fea_val = [[0] * self.feature_length] * max_number_of_loops # 特徴量の値
            requests = [[], []]
            # guess = [0] # not_sureを選んだ回数
            cur_loop = [0] # ループ回数を保存、stateに組み込む
            not_sure_count = 0 # not_sureをそのエピソードで選んだかどうかを保存するフラグ

            action_step_state = np.zeros((max_number_of_loops, 1, 2*self.step_size, self.state_size))
            out = np.zeros((max_number_of_loops, 1, 2*self.step_size, self.output_size))

            # 答えを決める
            objectIndex = [0, 0]
            objectIndex[0] = random.randint(0, len(symbols_noun)-2)
            if objectIndex[0] <= 1:
                objectIndex[1] = 5
            else:
                objectIndex[1] = objectIndex[0] + 4
            
            for step in range(max_number_of_steps):
                reward = [0, 0]
                terminal = [0, 0]
                for loop in range(max_number_of_loops):
                    cur_loop[0] = loop
                    parent_intent = [0, 0]
                    parent_intent[loop] = 1
                    parent_select = loop
                     
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

                    parent_order = np.reshape(parent_order, [1, self.parent_size, self.parent_size, 1])

                    pre_fea_vec = copy.deepcopy(fea_vec[loop])
                    pre_obj_vec = copy.deepcopy(obj[loop])
                    state1 = np.concatenate([fea_vec[loop], fea_val[loop], obj[loop], cur_loop])
                    state1 = np.reshape(state1, [1, self.state_size])
                    mask1 = np.reshape(mask1, [1, self.output_size])
                    action_step_state[loop][0][step] = state1
                    out_1 = np.concatenate([pre_fea_vec, pre_obj_vec])
                    out[loop][0][step] = np.reshape(out_1, [1, self.output_size])
                    
                    obj_val_idx = np.argmax(self.model([action_step_state[loop], out[loop], mask1, parent_order]))# 時刻tで取得する特徴量を決定

                    fea_vec[loop][obj_val_idx] = 1
                    request = actions[obj_val_idx]
                    requests[loop].append(request)
                    fea_val[loop] = Data.Overfetch(objectIndex[loop], list(set(requests[loop])), rand, parent_select)       
                    state2 = np.concatenate([fea_vec[loop], fea_val[loop], obj[loop], cur_loop])
                    state2 = np.reshape(state2, [1, self.state_size])
                    mask2 = np.reshape(mask2, [1, self.output_size])
                    action_step_state[loop][0][step+1] = state2
                    out_2 = np.concatenate([fea_vec[loop], pre_obj_vec])
                    out[loop][0][step+1] = np.reshape(out_2, [1, self.output_size])
                    obj_name_idx = np.argmax(self.model([action_step_state[loop], out[loop], mask2, parent_order])) # 時刻tで取得する物体の名称を決定

                    name = symbols[obj_name_idx - self.action_size]
                    obj[loop][obj_name_idx - self.action_size] = 0
                    if name == 'not_sure':
                        not_sure_count = 1

                    # 報酬を設定し、与える
                    reward, terminal = reward_func_loop_weight(objectIndex[loop], (obj_name_idx - self.action_size), not_sure_count, step, max_number_of_steps,
                                                                    True, requests, request, obj_val_idx, 3, symbols, symbols_verb, parent_select, name, reward, terminal)
                    not_sure_count = 0
                    
                    state1 = np.concatenate([fea_vec[loop], fea_val[loop], obj[loop], cur_loop])

                if terminal == [1, 1]:
                    if reward == [1, 1]:
                        reward_sum += 1
                        # reward_sum_n, _vを計算する機構を用意する

        
        return reward_sum, reward_sum_n, reward_sum_v    


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


class Memory_TDerror(Memory):
    def __init__(self, max_size=1000, state_size=83300, output_size=10, step_size=5, parent_size=2):
        super().__init__(max_size)
        self.state_size = state_size
        self.output_size = output_size
        self.step_size = step_size
        self.parent_size = parent_size
        
    def get_TDerror(self, memory, gamma, mainQN, targetQN):
        TDerror = []
        state_t = np.zeros((1, 2*self.step_size, self.state_size))
        next_state_t = np.zeros((1, 2*self.step_size, self.state_size))
        out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
        next_out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
        
        for j, (state, next_state, out, next_out, mask, next_mask, action, reward, terminal, parent_order) in enumerate(memory.buffer[memory.len() - 1]):
            next_state_t[j:j+1] = np.reshape(state, [1, self.state_size])
            
            next_state = np.reshape(next_state, [1, self.state_size])
            next_mask = np.reshape(next_mask, [1, self.output_size])
            parent_order = np.reshape(parent_order, [1, self.parent_size, self.parent_size, 1])
            next_state_t[j+1:j+2] = next_state
            next_out_vec_t[j+1:j+2] = np.reshape(next_out, [1, self.output_size])
            retmainQs = mainQN.model([next_state_t,  next_out_vec_t, next_mask, parent_order])
            next_action = np.argmax(retmainQs)
            target = reward + gamma * targetQN.model([next_state_t, next_out_vec_t, next_mask, parent_order])[0][next_action]

            state = np.reshape(state, [1, self.state_size])
            mask = np.reshape(mask, [1, self.output_size])
            state_t[j:j+1] = state
            out_vec_t[j:j+1] = np.reshape(out, [1, self.output_size])
            
            TDerror.append(target - targetQN.model([state_t, out_vec_t, mask, parent_order])[0][action])
        
        return sum(TDerror) / len(TDerror)
        
    def update_TDerror(self, memory, gamma, mainQN, targetQN):
        for i in range(0, (self.len() - 1)):
            TDerror = []
            state_t = np.zeros((1, 2*self.step_size, self.state_size))
            next_state_t = np.zeros((1, 2*self.step_size, self.state_size))
            out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
            next_out_vec_t = np.zeros((1, 2*self.step_size, self.output_size))
            
            for j, (state, next_state, out, next_out, mask, next_mask, action, reward, terminal, parent_order) in enumerate(memory.buffer[i]):
                next_state_t[j:j+1] = np.reshape(state, [1, self.state_size])
                next_state = np.reshape(next_state, [1, self.state_size])
                next_mask = np.reshape(next_mask, [1, self.output_size])
                parent_order = np.reshape(parent_order, [1, self.parent_size, self.parent_size, 1])
                next_state_t[j+1:j+2] = next_state
                next_out_vec_t[j+1:j+2] = np.reshape(next_out, [1, self.output_size])
                retmainQs = mainQN.model([next_state_t, next_out_vec_t, next_mask, parent_order])
                next_action = np.argmax(retmainQs)
                target = reward + gamma * targetQN.model([next_state_t, next_out_vec_t, next_mask, parent_order])[0][next_action]

                state = np.reshape(state, [1, self.state_size])
                mask = np.reshape(mask, [1, self.output_size])
                state_t[j:j+1] = state
                out_vec_t[j:j+1] = np.reshape(out, [1, self.output_size])
                TDerror.append(target - targetQN.model([state_t, out_vec_t, mask, parent_order])[0][action])
                
            self.buffer[i] = sum(TDerror) / len(TDerror)
            
    def get_sum_absolute_TDerror(self):
        sum_absolute_TDerror = 0
        for i in range(0, (self.len() - 1)):
            sum_absolute_TDerror += abs(self.buffer[i]) + 0.0001
            
        return sum_absolute_TDerror


# In[7]:


# [4]行動を決定するクラス
class Actor:
    def __init__(self, features_length, objects_length, actions_length, output_size, MODEL_LOAD):
        self.features_length = features_length
        self.objects_length = objects_length
        self.actions_length = actions_length
        self.output_size = output_size
        self.MODEL_LOAD = MODEL_LOAD
        
    def get_value(self, state, out, mask, parent_order, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.1 + 0.9 / (1.0+(episode/500))
        
        if epsilon <= np.random.uniform(0, 1) and episode != 0 or self.MODEL_LOAD == True:
            retTargetQs = mainQN.model([state, out, mask, parent_order])
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する

        else:
            # action = random.randint(0, self.actions_length-1)  # ランダムに行動する
            retTargetQs = np.random.rand(self.actions_length)  # ランダムに行動する
            action = np.argmax(retTargetQs)
        
        
        return action, retTargetQs # 特徴選択の確率をみたいため、retTargetQsを返り値に追加
    
    def get_name(self, state, out, mask, parent_order, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.01 + 0.99 / (1.0+episode)
        
        if epsilon <= np.random.uniform(0, 1) and episode != 0 or self.MODEL_LOAD == True:
            retTargetQs = mainQN.model([state, out, mask, parent_order])
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する

        else:
            action = random.randint(self.actions_length, self.actions_length+self.objects_length-1)  # ランダムに行動する
        
        return action


# In[8]:


# import subprocess
# subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'Pre_Model.ipynb'])


# In[ ]:




