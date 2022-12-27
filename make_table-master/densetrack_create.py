from io import BytesIO
import cv2
import csv
import numpy as np
import io
import random
import os
from k_medoids import KMedoids
import kmedoids
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

header = []
body = []
TRAJECTORY_SIZE = 30
TRAJECTORY_DIM = 11
HOG_SIZE = 96
HOG_DIM = 12
HOF_SIZE = 108
HOF_DIM = 13
MBHX_SIZE = 96
MBHX_DIM = 14
MBHY_SIZE = 96
MBHY_DIM = 15
SAMPLE_SIZE = 100

MOVIE_DIR = '../track_sample_100over_select/'

def densetrack(s):

    # read dense trajectories to list
    d = np.load(s)

    # convert to float
    d = [list(map(lambda x:x.astype(float), d[v])) for v in range(0,len(d))]
    
    trajectory = np.zeros((len(d), TRAJECTORY_SIZE))
    for i, data in enumerate(d):
        trajectory[i] = data[TRAJECTORY_DIM].flatten()
  
    D = pairwise_distances(trajectory, metric='euclidean')
    
    # split into 2 clusters
    M, C = kmedoids.kMedoids(D, SAMPLE_SIZE, seed=0)
    
    M.sort()
  
    trj = np.zeros((SAMPLE_SIZE, TRAJECTORY_SIZE))
    hog = np.zeros((SAMPLE_SIZE, HOG_SIZE))
    hof = np.zeros((SAMPLE_SIZE, HOF_SIZE))
    mbhx = np.zeros((SAMPLE_SIZE, MBHX_SIZE))
    mbhy = np.zeros((SAMPLE_SIZE, MBHY_SIZE))

    for i, sample in enumerate(M):
        trj[i] = d[sample][TRAJECTORY_DIM].flatten()
        hog[i] = d[sample][HOG_DIM]
        hof[i] = d[sample][HOF_DIM]
        mbhx[i] = d[sample][MBHX_DIM]
        mbhy[i] = d[sample][MBHY_DIM]

    return trj, hog, hof, mbhx, mbhy

header.append("id")
for i in range(SAMPLE_SIZE):
    for j in range(TRAJECTORY_SIZE):
        header.append("trajectory_" + str(i) + "_" + str(j))
for i in range(SAMPLE_SIZE):
    for j in range(HOG_SIZE):
        header.append("hog_" + str(i) + "_" + str(j))
for i in range(SAMPLE_SIZE):
    for j in range(HOF_SIZE):
        header.append("hof_" + str(i) + "_" + str(j))
for i in range(SAMPLE_SIZE):
    for j in range(MBHX_SIZE):
        header.append("mbhx_" + str(i) + "_" + str(j))
for i in range(SAMPLE_SIZE):
    for j in range(MBHY_SIZE):
        header.append("mbhy_" + str(i) + "_" + str(j))

movie_list = os.listdir(MOVIE_DIR)
id = 30 # 動画特徴量を追加するため途中のidから
for m in movie_list:
    s = MOVIE_DIR + m # 動画特徴量へのフルパス

    trajectory, hog, hof, mbhx, mbhy = densetrack(s)
    
    trajectory = trajectory.flatten().astype(str)
    x = np.insert(trajectory, 0, str(id)) # id + trj
    hog = hog.flatten().astype(str)
    x = np.insert(x, TRAJECTORY_SIZE-1, hog)
    hof = hof.flatten().astype(str)
    x = np.insert(x, TRAJECTORY_SIZE+HOG_SIZE-1, hof)
    mbhx = mbhx.flatten().astype(str)
    x = np.insert(x, TRAJECTORY_SIZE+HOG_SIZE+HOF_SIZE-1, mbhx)
    mbhy = mbhy.flatten().astype(str)
    x = np.insert(x, TRAJECTORY_SIZE+HOG_SIZE+HOF_SIZE+MBHX_SIZE-1, mbhy)
    body.append(x)

    id += 1


with open('densetrack_create_takeshita/new_move_k_medoids_100_30_1222.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(len(movie_list)):
        writer.writerow(body[i])
