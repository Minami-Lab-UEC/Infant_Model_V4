def reward_func_verbnoun(objectIndex, predict_idx, guess, step, max_step, test_mode, requests, request,predict_val_idx, verb_idx, symbols, symbols_verb, parent_select, name, reward, terminal):
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