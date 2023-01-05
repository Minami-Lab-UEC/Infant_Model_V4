import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure, io
from io import BytesIO
import cv2
import csv
import numpy as np


orientations = 9
size = 150
win_size = (size, size)
pixels_per_cell = (16, 16)
cells_per_block = (8, 8)
header = []
body = []
header_size = size * size
id = []
data_size = 15

header.append("id")
for i in range(header_size):
    header.append("hog_" + str(i))

for k in range (data_size):
    s = '../illust/' + str(k) + '.jpg'

    # 画像ファイルパスから読み込み
    img = cv2.imread(s)
    #dst = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.3)
    img_resize = cv2.resize(img, win_size)
    image = color.rgb2gray(img_resize)

    #image = color.rgb2gray(data.astronaut())

    fd, hog_image = hog(image, orientations, pixels_per_cell,
                        cells_per_block, visualise=True, block_norm='L2-Hys')
    
    x = hog_image.flatten()
    x = np.array(x, dtype=str)
    x = np.insert(x, 0, str(k))
    body.append(x)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    print(hog_image_rescaled.shape)

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    """

with open('../Investigation/bin/table_last/new_shape.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(data_size):
        writer.writerow(body[i])
