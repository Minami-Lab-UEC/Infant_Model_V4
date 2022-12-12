from io import BytesIO
import cv2
import csv
import numpy as np
import io
import random
from k_medoids import KMedoids
import kmedoids
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

header = []
body = []
trajectory_size = 180
hog_size = 96
hof_size = 108
mbhx_size = 96
mbhy_size = 96
sample_size = 100
data_size = 30

def densetrack(s):

    # read dense trajectories to list
    with open(s,'r') as f:
       d = [v.rstrip().split('\t') for v in f.readlines()]

    # convert to float
    d = [list(map(lambda x:float(x), d[v])) for v in range(0,len(d))]
    
    #random.seed(0)

    #print(len(d))
    #print(len(d[0]))
    #print(d[len(d)-1][0])
    trajectory = np.array([[0 for i in range(180)] for j in range(len(d))], dtype=float)
    for i, data in enumerate(d):
        trajectory[i][0:180] = np.array(data[10:190], dtype=float)

    """
    # km = KMeans(n_clusters=1000, init='random')
    
    # km = KMedoids(n_cluster=sample_size)
    # D = squareform(pdist(trajectory, metric='euclidean'))
    # predicted_labels = km.fit_predict(trajectory)
    centroids = km.cluster_centers_

    print(predicted_labels)
    """
    
    D = pairwise_distances(trajectory, metric='euclidean')
    
    # split into 2 clusters
    M, C = kmedoids.kMedoids(D, sample_size, seed=0)
    
    M.sort()
    #print(M)
     
    r_d = np.array([[0 for i in range(587)] for j in range(sample_size)], dtype=float)

    for i, index in enumerate(M):
        r_d[i][10:190] = d[index][10:190]
        r_d[i][190:286] = d[index][190:286]
        r_d[i][286:394] = d[index][286:394]
        r_d[i][394:490] = d[index][394:490]
        r_d[i][490:586] = d[index][490:586]

    # r_d = random.sample(d, sample_size)

    """
    #Trajectory 180dim: 2(bin) x 90(trajectory)
    print("Trajectory:")
    print(d[0][10:40])

    #HOG 96dim: 8(bin) x 2 x 2(spatio) x 3(temporal)
    print("HOG:")
    print(d[0][40:40+96])

    #HOF 108dim: 9(bin) x 2 x 2(spatio) x 3(temporal)
    print("HOF:")
    print(d[0][136:136+108])

    #MBHX 96dim: 8(bin) x 2 x 2(spatio) x 3(temporal)
    print("MBHX:")
    print(d[0][244:244+96])

    #MBHY 96dim: 8(bin) x 2 x 2(spatio) x 3(temporal)
    print("MBHY:")
    print(d[0][340:340+96])
    """

    trj = np.array([[0 for i in range(180)] for j in range(sample_size)], dtype=float)
    hog = np.array([[0 for i in range(96)] for j in range(sample_size)], dtype=float)
    hof = np.array([[0 for i in range(108)] for j in range(sample_size)], dtype=float)
    mbhx = np.array([[0 for i in range(96)] for j in range(sample_size)], dtype=float)
    mbhy = np.array([[0 for i in range(96)] for j in range(sample_size)], dtype=float)

    for i, data in enumerate(r_d):
        trj[i][0:180] = np.array(data[10:190], dtype=float)
        hog[i][0:96] = np.array(data[190:286], dtype=float)
        hof[i][0:108] = np.array(data[286:394], dtype=float)
        mbhx[i][0:96] = np.array(data[394:490], dtype=float)
        mbhy[i][0:96] = np.array(data[490:586], dtype=float)

    return trj, hog, hof, mbhx, mbhy

header.append("id")
for i in range(sample_size):
    for j in range(trajectory_size):
        header.append("trajectory_" + str(i) + "_" + str(j))
for i in range(sample_size):
    for j in range(hog_size):
        header.append("hog_" + str(i) + "_" + str(j))
for i in range(sample_size):
    for j in range(hof_size):
        header.append("hof_" + str(i) + "_" + str(j))
for i in range(sample_size):
    for j in range(mbhx_size):
        header.append("mbhx_" + str(i) + "_" + str(j))
for i in range(sample_size):
    for j in range(mbhy_size):
        header.append("mbhy_" + str(i) + "_" + str(j))

for k in range (data_size):
    s = './all_d_movie/' + str(k)

    trajectory, hog, hof, mbhx, mbhy = densetrack(s)
    
    trajectory = trajectory.flatten()
    trajectory = np.array(trajectory, dtype=str)
    x = np.insert(trajectory, 0, str(k))
    hog = hog.flatten()
    hog = np.array(hog, dtype=str)
    x = np.insert(x, trajectory_size-1, hog)
    hof = hof.flatten()
    hof = np.array(hof, dtype=str)
    x = np.insert(x, trajectory_size+hog_size-1, hof)
    mbhx = mbhx.flatten()
    mbhx = np.array(mbhx, dtype=str)
    x = np.insert(x, trajectory_size+hog_size+hof_size-1, mbhx)
    mbhy = mbhy.flatten()
    mbhy = np.array(mbhy, dtype=str)
    x = np.insert(x, trajectory_size+hog_size+hof_size+mbhx_size-1, mbhy)
    body.append(x)


with open('../Investigation/bin/table_last/new_move_k_medoids_100_30.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(data_size):
        writer.writerow(body[i])
