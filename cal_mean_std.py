'''
    计算数据集均值与方差
'''
import numpy as np
import cv2 as cv
import os

# img_h, img_w = 32, 32
img_h, img_w = 32, 48   #根据自己数据集适当调整，影响不大
means, stdevs = [], []
img_list = []

imgs_path = './dataset/train/'
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv.imread(os.path.join(imgs_path,item))
    img = cv.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i,'/',len_)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))