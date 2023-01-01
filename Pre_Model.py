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


# In[2]:


def reward_func(pred_noun, pred_verb, ans_noun, ans_verb, guess, step, max_step, test_mode, requests, request,predict_val_idx, verb_idx, symbols, symbols_verb, parent_select):
    reward = 0 # 報酬
    terminal = 0 # 終了しているか否か
    reward_feature = 0 # 特徴選択時にきちんと親が指定した物体or動作に注目しているか
    reward_name = 0 # 名称選択時にきちんと親が指定した名詞or動詞に注目しているか
    correct = 0 # 正解か不正解か

    # test_modeはpre_main.pyでFalseにされている
    
    if parent_select == 0:
        if pred_noun == ans_noun:
            reward = 1
            correct = 1
            terminal = 1
        else:
            if step+1 == max_step:
                reward = -1
                terminal = 1
                if guess == 1:
                    reward = -0.2
        if pred_noun in symbols_verb:
            reward_name = -1
        if predict_val_idx == verb_idx:
            reward_future = -1
    else:
        if pred_verb == ans_verb:
            if pred_noun == ans_noun:
                reward = 10 # 名詞も動詞も両方正解でrewardをmaxあげる, そのエピソードの学習も終了
                correct = 1
                terminal = 1
            else:
                reward = 0.5 # 動詞が正解で、名詞が不正解なら半分rewardをあげる
                if step+1 == max_step:
                    terminal = 1
        else:
            if pred_noun == ans_noun:
                reward = 0.25 # 名詞だけ正解ならreward1/4あげる、動詞を当てるタスクなので少なめ
            if step+1 == max_step:
                reward = -1
                terminal = 1
                if guess == 1:
                    reward = -0.2
        if pred_verb not in symbols_verb:
            reward_name = -1
        if predict_val_idx != verb_idx:
            reward_future = -1
        
    return reward, reward_feature, reward_name, terminal, correct


# In[3]:


def loss_func(y_true, y_pred):
    error = tf.abs(y_pred - y_true)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    return loss


# In[4]:


# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=83300, step_size=5, feature_size=0, 
                 action_size=4, object_size=6, output_size=10, parent_size=4, embedding_size=200, hidden_size=100, hidden_size_2=16,
                  p_hidden_size=32):
        self.state_size = state_size
        self.step_size = step_size
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
        
        Q_N_1 = Dense(object_size)(lstm)
        Q_N_1 = softmax(Q_N_1) # 名詞用

        Q_N_2 = Dense(object_size)(lstm)
        Q_N_2 = softmax(Q_N_2) # 動詞用
        
        #Q_all = Concatenate()([Q_F, Q_N])
        predictions_1 = Concatenate()([Q_F, Q_N_1]) # 名詞用
        predictions_2 = Concatenate()([Q_F, Q_N_2]) # 動詞用
        
        # predictions = Dense(output_size, activation='softmax')(Q_all)
        mask = Input(shape=(output_size,))
        predictions_noun = Multiply()([predictions_1, mask])
        predictions_verb = Multiply()([predictions_2, mask])
        self.model = Model(inputs=[inputs, out, mask, parent_order], outputs=[predictions_noun, predictions_verb])
        opt = Adam(lr=learning_rate)
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
                    retmainQs, _ = self.model([next_state_t, next_out_vec_t, next_mask_b,  parent_order_b])
                    next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                    target = reward_b + gamma * targetQN.model([next_state_t, next_out_vec_t, next_mask_b, parent_order_b])[0][0][next_action]

                state_b = np.reshape(state_b, [1, self.state_size])
                mask_b = np.reshape(mask_b, [1, self.output_size])
                state_t[j:j+1] = state_b
                out_vec_t[j:j+1] = np.reshape(out_b, [1, self.output_size])
                #targets[i][j] = self.model([state, mask_b])    # Qネットワークの出力(Q値)
                #targets[i][j][action_b] = target               # 教師信号
                #targets[i] = self.model([state, mask_b])    # 怪しい
                targets_restore[i][j], _ = self.model([state_t, out_vec_t, mask_b, parent_order_b])
                #targets[i][action_b] = target               # 怪しい
                targets_restore[i][j][action_b] = target
                
            targets[i] = np.mean(targets_restore[i], axis=0)
        history = self.model.fit([inputs, out_vec, masks, parent_orders], targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        
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
        #parent_orders = np.zeros((batch_size, self.parent_size))
        parent_orders = np.zeros((batch_size, self.parent_size, self.parent_size, 1))
        
        for i, eps in enumerate(batch_memory):
            for j, (state_b, next_state_b, out_b, next_out_b, mask_b, next_mask_b, action_b, reward_b, terminal_b, parent_order_b) in enumerate(eps):
                
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
                    retmainQs, _ = self.model([next_state_t, next_out_vec_t, next_mask_b, parent_order_b])
                    next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                    target = reward_b + gamma * targetQN.model([next_state_t, next_out_vec_t, next_mask_b, parent_order_b])[0][0][next_action]

                state_b = np.reshape(state_b, [1, self.state_size])
                mask_b = np.reshape(mask_b, [1, self.output_size])
                state_t[j:j+1] = state_b
                out_vec_t[j:j+1] = np.reshape(out_b, [1, self.output_size])
                targets_restore[i][j], _ = self.model([state_t, out_vec_t, mask_b, parent_order_b])
                targets_restore[i][j][action_b] = target
                
            targets[i] = np.mean(targets_restore[i], axis=0)
        history = self.model.fit([inputs, out_vec, masks, parent_orders], targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        
        return history
        
    
    def test(self, Data, actor, symbols, symbols_noun, symbols_verb, actions, mask1, mask2, test_epochs, max_number_of_steps, num_episode, dir_path, per_error, parent_FE):
        reward_sum = 0
        reward_sum_n = 0
        reward_sum_v = 0
        
        bias_n = []
        bias_v = []
        
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
            
            parent_intent = [1, 0]
            parent_select = 0
            objectIndex_noun = random.randint(0, len(symbols_noun)-2)
            if episode < (test_epochs//2):
                objectIndex = objectIndex_noun
                objectIndex_verb = objectIndex_noun
            else:
                parent_select = 1
                objectIndex_verb = random.randint(len(symbols_noun)-1, len(symbols)-2)
                objectIndex = objectIndex_verb
                parent_intent = [0, 1]
                
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
                #parent_order[idx] = 1
            
            """
            parent_order = [1, 0] # 親が名詞か動詞のどちらを選んだか
            objectIndex = random.randint(0, self.object_size-1) # 当ててほしい物体のindex
            
            parent_select = 0
            if episode < (test_epochs//2):
                objectIndex = random.randint(0, len(symbols_noun)-2)
                if per_error <= np.random.uniform(0, 1):
                    parent_order = [0, 1]
            else:
                parent_select = 1
                objectIndex = random.randint(len(symbols_noun)-1, len(symbols)-2)
                if per_error <= np.random.uniform(0, 1):
                    parent_order = [1, 0]
                else:
                    parent_order = [0, 1]
            """
            #objectIndex = random.randint(0, self.object_size-1) # 当ててほしい物体のindex
            tmp_info[0] = symbols[objectIndex]
            action_step_state = action_step_state = np.zeros((1, 2*self.step_size, self.state_size))
            out = np.zeros((1, 2*self.step_size, self.output_size))
            parent_order = np.reshape(parent_order, [1, self.parent_size, self.parent_size, 1])
            for step in range(max_number_of_steps):
                pre_fea_vec = copy.deepcopy(fea_vec)
                pre_obj_vec = copy.deepcopy(obj)
                state1 = np.concatenate([fea_vec, fea_val, obj, guess])
                state1 = np.reshape(state1, [1, self.state_size])
                mask1 = np.reshape(mask1, [1, self.output_size])
                action_step_state[0][step] = state1
                out_1 = np.concatenate([pre_fea_vec, pre_obj_vec])
                out[0][step] = np.reshape(out_1, [1, self.output_size])
                
                retTargetQs, _ = self.model([action_step_state, out, mask1, parent_order])
                obj_val_idx = np.argmax(retTargetQs) # 時刻tで取得する特徴量を決定
                """
                if step != 4:
                    obj_val_idx = step
                else:
                    obj_val_idx = 3
                """
                tmp_info[2*step+1] = actions[obj_val_idx]

                fea_vec[obj_val_idx] = 1
                request = actions[obj_val_idx]
                requests.append(request)
                fea_val = Data.Overfetch(objectIndex, list(set(requests)), rand, parent_select)       
                state2 = np.concatenate([fea_vec, fea_val, obj, guess])
                state2 = np.reshape(state2, [1, self.state_size])
                mask2 = np.reshape(mask2, [1, self.output_size])
                action_step_state[0][step+1] = state2
                out_2 = np.concatenate([fea_vec, pre_obj_vec])
                out[0][step+1] = np.reshape(out_2, [1, self.output_size])
                retTargetQs_noun, retTargetQs_verb = self.model([action_step_state, out, mask2, parent_order])
                obj_name_idx_noun = np.argmax(retTargetQs_noun)
                obj_name_idx_verb = np.argmax(retTargetQs_verb)
                # 時刻tで取得する物体の名称を決定

                pred_noun = symbols[obj_name_idx_noun - self.action_size]
                pred_verb = symbols[obj_name_idx_verb - self.action_size]
                ans_noun = symbols[objectIndex_noun]
                ans_verb = symbols[objectIndex_verb]
                obj[obj_name_idx_noun - self.action_size] = 1
                obj[obj_name_idx_verb - self.action_size] = 1
                if pred_noun == 'not_sure' and pred_verb == 'not_sure':
                    # guess[0] += 1 # pre_main.pyでもコメントアウトしたため、こちらもコメントアウトすべき？
                    not_sure_count = 1

                if parent_select == 0:    
                    tmp_info[2*step+2] = pred_noun
                else:
                    tmp_info[2*step+2] = pred_verb

                # 報酬を設定し、与える
                reward, _, _, terminal, _ = reward_func(pred_noun, pred_verb, ans_noun, ans_verb, not_sure_count, step, max_number_of_steps,
                                                                   True, requests, request, obj_val_idx, 3, symbols, symbols_verb, parent_select)
                not_sure_count = 0

                tmp_info[step+1+10] = reward
                
                state1 = np.concatenate([fea_vec, fea_val, obj, guess])
                
                if terminal:
                    if episode < (test_epochs//2):
                        pass # biasに関する動きが意味わからんのでパス
                        # if bias_n == []:
                        #     bias_n = self.check_bias(data=[action_step_state, out, mask1, parent_order])
                    else:
                        pass # biasに関する動きが意味わからんのでパス
                        # if bias_n == []:
                        # if bias_v == []:
                        #     bias_v = self.check_bias(data=[action_step_state, out, mask1, parent_order])
                    #print(self.check_bias(data=[action_step_state, out, mask1, parent_order]))
                    tmp_se = pd.Series(tmp_info, index=result_df.columns, name=episode+1)
                    # result_df = result_df.append(tmp_se, ignore_index=True) # .appendが非推奨のため
                    result_df = pd.concat([result_df, pd.DataFrame([tmp_se])], ignore_index=True)
                    if reward == 1:
                        reward_sum += 1
                        if episode < (test_epochs//2):
                            reward_sum_n += 1
                        else:
                            reward_sum_v += 1
                        break
        
        result_df.to_csv(dir_path+str(num_episode+1)+'_result.csv')
        
        # with open(dir_path+str(num_episode+1)+'_result.csv', 'a') as f:
        #     f.write('\n')
        #     f.write('noun_bias_result: [' + str(bias_n[0][0]) + ',' + str(bias_n[0][1]) + ']' + '\n')
        #     f.write('verb_bias_result: [' + str(bias_v[0][0]) + ',' + str(bias_v[0][1]) + ']' + '\n')
        
        return reward_sum, reward_sum_n, reward_sum_v


# In[5]:


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


# In[6]:


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
            parent_order = np.reshape(parent_order, [1, self.parent_size])
            next_state_t[j+1:j+2] = next_state
            next_out_vec_t[j+1:j+2] = np.reshape(next_out, [1, self.output_size])
            retmainQs, _ = mainQN.model([next_state_t,  next_out_vec_t, next_mask, parent_order])
            next_action = np.argmax(retmainQs)
            target = reward + gamma * targetQN.model([next_state_t, next_out_vec_t, next_mask, parent_order])[0][0][next_action]

            state = np.reshape(state, [1, self.state_size])
            mask = np.reshape(mask, [1, self.output_size])
            state_t[j:j+1] = state
            out_t[j:j+1] = np.reshape(out, [1, self.output_size])
            
            TDerror.append(target - targetQN.model([state_t, out_t, mask, parent_order])[0][0][action])
        
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
                parent_order = np.reshape(parent_order, [1, self.parent_size])
                next_state_t[j+1:j+2] = next_state
                next_out_vec_t[j+1:j+2] = np.reshape(next_out, [1, self.output_size])
                retmainQs, _ = mainQN.model([next_state_t, next_out_vec_t, next_mask, parent_order])
                next_action = np.argmax(retmainQs)
                target = reward + gamma * targetQN.model([next_state_t, next_out_vec_t, next_mask, parent_order])[0][0][next_action]

                state = np.reshape(state, [1, self.state_size])
                mask = np.reshape(mask, [1, self.output_size])
                state_t[j:j+1] = state
                out_t[j:j+1] = np.reshape(out, [1, self.output_size])
                TDerror.append(target - targetQN.model([state_t, out_t, mask, parent_order])[0][0][action])
                
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
            retTargetQs, _ = mainQN.model([state, out, mask, parent_order])
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
            retTargetQs_noun, retTargetQs_verb = mainQN.model([state, out, mask, parent_order])
            action_noun = np.argmax(retTargetQs_noun)  # 最大の報酬を返す行動を選択する
            action_verb = np.argmax(retTargetQs_verb)  # 最大の報酬を返す行動を選択する

        else:
            action_noun = random.randint(self.actions_length, self.actions_length+self.objects_length-1)
            action_verb = random.randint(self.actions_length, self.actions_length+self.objects_length-1)   # ランダムに行動する
        
        return action_noun, action_verb


# In[8]:


# import subprocess
# subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'Pre_Model.ipynb'])


# In[ ]:




