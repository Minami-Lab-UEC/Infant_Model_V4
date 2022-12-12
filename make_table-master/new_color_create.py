#-*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure, io
from io import BytesIO
import csv
import numpy as np

size = 150
win_size = (size, size)
hdims = 64

header = []
body = []
header_size = hdims
id = []
data_size = 30

header.append("id")
for i in range(header_size):
    header.append("r_" + str(i))
for i in range(header_size):
    header.append("g_" + str(i))
for i in range(header_size):
    header.append("b_" + str(i))

for k in range (data_size):
    #print(k)
    # 入力画像を読み込み
    pre_img = cv2.imread("../illust/" + str(k) +".jpg")

    img = cv2.resize(pre_img, win_size)

    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:110, 40:110] = 255
    masked_img = cv2.bitwise_and(img,img,mask = mask)

    
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]

    hist_m_r = cv2.calcHist([r],[0],mask,[hdims],[0,256])
    hist_m_g = cv2.calcHist([g],[0],mask,[hdims],[0,256])
    hist_m_b = cv2.calcHist([b],[0],mask,[hdims],[0,256])
    
    r = hist_m_r
    g = hist_m_g
    b = hist_m_b
    r = r.flatten()
    r = np.array(r, dtype=str)
    x = np.insert(r, 0, str(k))
    g = g.flatten()
    x = np.insert(x, hdims-1, g)
    b = b.flatten()
    x = np.insert(x, (2*hdims)-1, b)
    body.append(x)
    


with open('../Investigation/bin/table_last/new_color_histgram_30.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(data_size):
        writer.writerow(body[i])
