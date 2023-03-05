import numpy as np
import collections as c

def pick_tree(pick):
	first_pick = [p_i[0] for p_i in pick]
	fp_dir = c.Counter(first_pick)
	most_fp = max(fp_dir, key=fp_dir.get) # 10000エピソード時は'shape'だった.
	print(fp_dir)

	second_pick = [p_i[1] for p_i in pick if p_i[0] == most_fp]
	sp_dir = c.Counter(second_pick)
	most_sp = max(sp_dir, key=sp_dir.get)
	print(sp_dir)

	therd_pick = [p_i[2] for p_i in pick if p_i[0] == most_fp and p_i[1] == most_sp]
	tp_dir = c.Counter(therd_pick)
	most_tp = max(tp_dir, key=tp_dir.get)
	print(tp_dir)

	forth_pick = [p_i[3] for p_i in pick if p_i[0] == most_fp and p_i[1] == most_sp and p_i[2] == most_tp]
	fop_dir = c.Counter(forth_pick)
	most_fop = max(fop_dir, key=fop_dir.get)
	print(fop_dir)

	fifth_pick = [p_i[4] for p_i in pick if p_i[0] == most_fp and p_i[1] == most_sp and p_i[2] == most_tp and p_i[3] == most_fop]
	fip_dir = c.Counter(fifth_pick)
	most_fip = max(fip_dir, key=fip_dir.get)
	print(fip_dir)

max_number_of_steps = 5 # 特徴選択のマックス回数は5

picklist = np.load('result_HCS_after_60/obj_val_idx_list.npy', allow_pickle=True)
picklist = list(map(lambda x: x[0] + ['None'] * (max_number_of_steps - len(x[0])), picklist))
for i in range(len(picklist) // 10000):
	print('-----------------------------')
	print('{}:{}episode :'.format(i*10000, (i+1)*10000))
	pick_tree(picklist[i*10000:(i+1)*10000])
	print('-----------------------------')


