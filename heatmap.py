# -*- coding: utf-8 -*-
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# list_2d = [[0, 1, 2], [3, 4, 5]]
obj_val_retTargetQs = np.load("result_HCS_after_60/obj_val_idx_list.npy")

# pd.DataFrame(data=obj_val_retTargetQs[0], columns=["color", "shape", "taste", "move"])
obj_val_retTargetQs_1 = pd.DataFrame(data=obj_val_retTargetQs[0], columns=["color", "shape", "taste", "move"]) # 最初の特徴選択
# obj_val_retTargetQs_5000 = pd.DataFrame(data=np.mean(obj_val_retTargetQs[0:5000], axis=0), columns=["color", "shape", "taste", "move"]) # 5000エピソード後の特徴選択
obj_val_retTargetQs_10000 = pd.DataFrame(data=np.mean(obj_val_retTargetQs[0:10000], axis=0), columns=["color", "shape", "taste", "move"]) # 10000エピソード後の特徴選択
obj_val_retTargetQs_30000 = pd.DataFrame(data=np.mean(obj_val_retTargetQs[:], axis=0), columns=["color", "shape", "taste", "move"]) # 10000エピソード後の特徴選択


# sns.heatmap(obj_val_retTargetQs_1)
# fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
fig, axes = plt.subplots(1, 3, figsize = (15, 5), sharex=True, sharey=True)
# axes.set_ylim(5, 0)
# plt.figure()
# sns.heatmap(list_2d)
sns.heatmap(obj_val_retTargetQs_1, ax = axes[0], annot=True)
# sns.heatmap(obj_val_retTargetQs_5000, ax = axes[1], annot=True)
sns.heatmap(obj_val_retTargetQs_10000, ax = axes[1], annot=True)
sns.heatmap(obj_val_retTargetQs_30000, ax = axes[2], annot=True)
# axes.set_ylim(5, 0)
plt.savefig("result_HCS_after_60/seaborn_heatmap_list.png")
# plt.close('all')