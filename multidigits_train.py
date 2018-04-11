import numpy as np
import os
import time
from keras.preprocessing import image
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
import sys
from keras import backend as K
import tensorflow as tf
from multidigits import DigitsModel
import pickle
import argparse

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def parse_args(args):
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data_dir', help='Path to dataset directory.')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=20)
    parser.add_argument('--batch_size', default=32)
    return parser.parse_args(args)

def get_cat(s):
    if s is None:
        res = np_utils.to_categorical(10, 12)
    elif s.isdigit():
        res = np_utils.to_categorical(int(s), 12)
    elif s.lower() in ['x', '-', '*']:
        res = np_utils.to_categorical(11, 12)
    else:
        res=None
        print('error: no category')
    return res

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    K.tensorflow_backend.set_session(get_session())


    img_data = pickle.load(open(args.data_dir+"bib_img_g.p", "rb"))
    bibs = pickle.load(open(args.data_dir+"labels_g.p", "rb"))



    digit_len = []
    digits = []
    for i in range(5):
        digits.append([])

    for bib in bibs:
        if len(bib) > 2 and bib.lower()[0:2] in ['no']:
            digit_len.append(np_utils.to_categorical(0, 6))
            for i in range(5):
                digits[i].append(get_cat(None))
        else:
            digit_len.append(np_utils.to_categorical(len(bib), 6))
            for i in range(len(bib)):
                digits[i].append(get_cat(bib[i]))
            for j in range(len(bib), 5):
                digits[j].append(get_cat(None))
    for i in range(5):
        digits[i] = np.array(digits[i])
    digits=np.array(digits)
    digit_len = np.array(digit_len)
    data_size = img_data.shape[0]
    split = int(data_size*0.8)
    train_digits = [digits[0][:split], digits[1][:split], digits[2][:split], digits[3][:split], digits[4][:split]]
    test_digits = [digits[0][split:], digits[1][split:], digits[2][split:], digits[3][split:], digits[4][split:]]



    model = DigitsModel()
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(img_data[:8000], [digit_len[:8000],digits[0][:8000],digits[1][:8000],digits[2][:8000],digits[3][:8000],digits[4][:8000]],
    #           batch_size=32, epochs=20, verbose=1,
    #           validation_data=(img_data[8000:], [digit_len[8000:],digits[0][8000:],digits[1][8000:],digits[2][8000:],digits[3][8000:],digits[4][8000:]]))
    model.fit(img_data[:split], train_digits, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
              validation_data=(img_data[split:], test_digits))
    model.save(args.data_dir+'digits_model.h5')

if __name__ == '__main__':
    main()