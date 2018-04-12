from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,BatchNormalization,Convolution2D,Dropout,Activation
import sys
from keras import backend as K
import tensorflow as tf
from multidigits import DigitsModel
import pickle
import argparse
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle
import os
import cv2

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
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

def parse_args(args):
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data_dir', help='Path to dataset directory.')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=20)
    parser.add_argument('--batch_size', default=32, type=int)
    return parser.parse_args(args)

def DigitsModel2(shape=(96,192,1), weight_file = None):
    data = Input(name='data', shape=shape)
    x = Conv2D(64, (3,3),activation='relu', padding='same', name='conv1')(data)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    x = Conv2D(128, (3, 3),activation='relu', padding='same', name='conv2')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    x = Dropout(0.25)(x)
    flatten = Flatten()(x)

    dense = Dense(512, activation='relu',name='fc1')(flatten)
    dense = Dropout(0.25)(dense)
    dense = Dense(512, activation='relu',name='fc2')(dense)
    dense = Dropout(0.25)(dense)

    digit_1 = Dense(12, activation='softmax', name='digit_1')(dense)
    digit_2 = Dense(12, activation='softmax', name='digit_2')(dense)
    digit_3 = Dense(12, activation='softmax', name='digit_3')(dense)
    digit_4 = Dense(12, activation='softmax', name='digit_4')(dense)
    digit_5 = Dense(12, activation='softmax', name='digit_5')(dense)

    model = Model(input=data,output=[digit_1,digit_2,digit_3,digit_4,digit_5])

    return model

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    K.tensorflow_backend.set_session(get_session())

    # img_data = pickle.load(open(args.data_dir+"20000img_gray.p", "rb"))
    # labels = pickle.load(open(args.data_dir+"20000labels_v1.p", "rb"))

    # img_data_1 = pickle.load(open(args.data_dir + "twisted_img_3c_1.p", "rb"))
    # img_data_2 = pickle.load(open(args.data_dir + "twisted_img_3c_2.p", "rb"))
    # img_data=np.concatenate((img_data_1,img_data_2),axis=0)
    # del img_data_1
    # del img_data_2
    # labels = pickle.load(open(args.data_dir+"twisted_labels.p", "rb"))



    # img_data, labels = shuffle(img_data, labels, random_state=2)
    #
    # pickle.dump(labels, open("/home/ubuntu/zk/deep-learning-models/data/20000labels_v1.p", "wb"))
    # pickle.dump(img_data, open("/home/ubuntu/zk/deep-learning-models/data/20000img_gray.p", "wb"),
    #             protocol=4)
    # del img_data
    # del labels
    # img_data = pickle.load(open(args.data_dir+"20000img_gray.p", "rb"))
    # labels = pickle.load(open(args.data_dir+"20000labels_v1.p", "rb"))

    labels = []
    img_data_list = []
    filedir = args.data_dir+'renders_v3/'
    files = os.listdir(filedir)
    for filename in files:
        labels.append(filename[7:11])
        img = cv2.imread(filedir + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (192, 96)).astype('float32')
        img = img / 255
        x = np.expand_dims(img, axis=0)
        x = img.reshape(1, 96, 192, 1)
        img_data_list.append(x)

    # data2 = pickle.load(open(args.data_dir+"data2_df.p", "rb"))
    img_data = np.array(img_data_list)
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]
    img_data, labels = shuffle(img_data, labels, random_state=2)


    rows=img_data.shape[1]
    cols= img_data.shape[2]
    channels=img_data.shape[3]

    digit_len = []
    digits = []
    digits.append([])
    digits.append([])
    digits.append([])
    digits.append([])
    digits.append([])

    for bib in labels:
        bib = str(bib)
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
    digit_len = np.array(digit_len)
    digits = np.array(digits)
    data_size = img_data.shape[0]
    split = int(data_size*0.8)
    train_digits = [digits[0][:split], digits[1][:split], digits[2][:split], digits[3][:split], digits[4][:split]]
    test_digits = [digits[0][split:], digits[1][split:], digits[2][split:], digits[3][split:], digits[4][split:]]

    model = DigitsModel2(shape=(rows,cols,channels))
    model.summary()

    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fitting the model
    model.fit(img_data[:split], train_digits, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
              validation_data=(img_data[split:], test_digits))
    model.save(args.data_dir + 'digits_model_gray_v2.h5')

if __name__ == '__main__':
    main()

# model = DigitsModel2(shape=(96,192,1))
# model.summary()
