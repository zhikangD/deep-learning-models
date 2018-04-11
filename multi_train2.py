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

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
def get_cat(s):
    if s is None:
        res = np_utils.to_categorical(10, 11)
    elif s.isdigit():
        res = np_utils.to_categorical(int(s), 11)
    # elif s.lower() in ['x', '-', '*']:
    #     res = np_utils.to_categorical(11, 12)
    else:
        res=None
        print('error: no category')
    return res

def parse_args(args):
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data_dir', help='Path to dataset directory.')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=20)
    parser.add_argument('--batch_size', default=128)
    return parser.parse_args(args)

def DigitsModel2(weight_file = None):
    data = Input(name='data', shape=(64, 128, 1))
    x = Conv2D(32, (3,3),activation='relu',padding='same', name='conv1')(data)
    # x = Activation('relu')(x)
    x = Conv2D(32, (3, 3),activation='relu', name='conv2')(x)
    # x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    x = Dropout(0.25)(x)
    flatten = Flatten()(x)

    dense = Dense(128, activation='relu')(flatten)
    dense = Dropout(0.25)(dense)

    digit_1 = Dense(11, activation='softmax', name='digit_1')(dense)
    digit_2 = Dense(11, activation='softmax', name='digit_2')(dense)
    digit_3 = Dense(11, activation='softmax', name='digit_3')(dense)
    digit_4 = Dense(11, activation='softmax', name='digit_4')(dense)
    digit_5 = Dense(11, activation='softmax', name='digit_5')(dense)

    model = Model(input=data,output=[digit_1,digit_2,digit_3,digit_4,digit_5])

    return model

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    K.tensorflow_backend.set_session(get_session())

    img_data = pickle.load(open(args.data_dir+"bib_gray.p", "rb"))
    labels = pickle.load(open(args.data_dir+"labels_gray.p", "rb"))

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
            print(1)
            digit_len.append(np_utils.to_categorical(0, 6))
            for i in range(5):
                digits[i].append(get_cat(None))
        else:
            digit_len.append(np_utils.to_categorical(len(bib), 6))
            for i in range(len(bib)):
                digits[i].append(get_cat(bib[i]))
            for j in range(len(bib), 5):
                digits[j].append(get_cat(None))
                print(1)
    for i in range(5):
        digits[i] = np.array(digits[i])
    digit_len = np.array(digit_len)
    digits = np.array(digits)
    train_digits = [digits[0][:8000], digits[1][:8000], digits[2][:8000], digits[3][:8000], digits[4][:8000]]
    test_digits = [digits[0][8000:], digits[1][8000:], digits[2][8000:], digits[3][8000:], digits[4][8000:]]

    model = DigitsModel2()
    model.summary()

    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fitting the model
    model.fit(img_data[:8000], train_digits, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
              validation_data=(img_data[8000:], test_digits))
    model.save(args.data_dir + 'digits_model.h5')

if __name__ == '__main__':
    main()

