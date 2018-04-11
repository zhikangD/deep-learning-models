from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import pickle
import tensorflow as tf
import numpy as np
import os
import random
import sys
import cv2
import random
from sklearn.utils import shuffle
W = 256
H = 64

def main():
    i = 0
    font = ImageFont.truetype("/home/ubuntu/zk/deep-learning-models/data/impact.ttf", 30)
    labels = []
    print('start generatiing data')
    while i < 10000:
        img = Image.new("RGB", (W, H), "black")
        draw = ImageDraw.Draw(img)
        rand = random.randint(10000, 99999)
        msg = str(rand)
        labels.append(msg)
        w, h = draw.textsize(msg)
        draw.text(((W - w) / 2, (H - h) / 2), msg, fill=(255, 255, 255), font=font)
        img.save('/home/ubuntu/zk/deep-learning-models/data/digits_sample/sample' + str(i) + '.jpg')
        i = i + 1

    # while i < 20000:
    #     img = Image.new("RGB", (W, H), "black")
    #     draw = ImageDraw.Draw(img)
    #     rand = random.randint(1000, 9999)
    #     msg = str(rand)
    #     labels.append(msg)
    #     w, h = draw.textsize(msg)
    #     draw.text(((W - w) / 2, (H - h) / 2), msg, fill=(255, 255, 255), font=font)
    #     img.save('/home/ubuntu/zk/deep-learning-models/data/digits_sample/sample' + str(i) + '.jpg')
    #     i = i + 1

    print("done")



    img_data_list = []
    for i in range(0, 10000):
        filename = '/home/ubuntu/zk/deep-learning-models/data/digits_sample/sample' + str(i) + '.jpg'
        img = cv2.imread(filename)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (192, 96)).astype('float32')
        img = img / 255
        x = np.expand_dims(img, axis=0)
        #     x = preprocess_input(x)
        x = img.reshape(1, 96, 192, 1)
        img_data_list.append(x)
    img_data = np.array(img_data_list)
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]
    print(np.shape(labels))
    print(img_data.shape)
    print('write to local')
    img_data_s, labels_s = shuffle(img_data, labels, random_state=2)
    pickle.dump(labels_s, open("/home/ubuntu/zk/deep-learning-models/data/20000labels_v1.p", "wb"))
    pickle.dump(img_data_s, open("/home/ubuntu/zk/deep-learning-models/data/20000img_gray.p", "wb"),
                protocol=4)


if __name__ == '__main__':
    main()