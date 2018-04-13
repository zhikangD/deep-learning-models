from keras.models import Model, Sequential
from keras.layers import Flatten,Dense,Input,Conv2D
from keras.layers import GRU, Reshape
from keras.layers import MaxPooling2D,BatchNormalization,Convolution2D,Dropout,Activation
import sys
from keras import backend as K
from keras.layers.merge import add, concatenate
from keras.optimizers import SGD
import tensorflow as tf
from multidigits import DigitsModel
import pickle
import argparse
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle
import os
import cv2
# from Generator import DigitImageGenerator
# from spatial_transformer import SpatialTransformer

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
    parser.add_argument('--model', default='1')
    return parser.parse_args(args)

def DigitsModel2(shape=(96,192,1), weight_file = None):
    data = Input(name='data', shape=shape)
    x = Conv2D(64, (3,3),activation='relu', padding='same', name='conv1')(data)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    x = Conv2D(128, (3, 3),activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
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

def RecurrentModel(shape=(96,192,1), weight_file = None):
    # locnet = Sequential()
    # locnet.add(MaxPooling2D(pool_size=(2, 2), input_shape=shape))
    # locnet.add(Conv2D(20, (5, 5)))
    # locnet.add(MaxPooling2D(pool_size=(2, 2)))
    # locnet.add(Conv2D(20, (5, 5)))
    #
    # locnet.add(Flatten())
    # locnet.add(Dense(50))
    # locnet.add(Activation('relu'))
    # locnet.add(Dense(6))
    data = Input(name='data', shape=shape)
    # trans = SpatialTransformer(localization_net=locnet,
    #                              output_size=(96, 192), input_shape=shape)(data)
    x = Conv2D(64, (3,3),activation='relu', padding='same', name='conv1')(data)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    x = Conv2D(128, (3, 3),activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool4')(x)
    inner = Reshape(target_shape=(5,11*512), name='reshape')(x)
    inner = Dense(128, activation='relu', name='dense1')(inner)
    gru_1 = GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(512, return_sequences=True, kernel_initializer='he_normal',dropout=0.2, name='gru2')(gru1_merged)
    gru_2b = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',dropout=0.2, name='gru2_b')(
        gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(12, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)



    model = Model(input=data,output=y_pred)

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
    filedir = args.data_dir+'renders_v3_rand/'
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

    filedir = args.data_dir+'three_digit_renders/'
    files = os.listdir(filedir)
    for filename in files:
        labels.append(filename[7:10])
        img = cv2.imread(filedir + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (192, 96)).astype('float32')
        img = img / 255
        x = np.expand_dims(img, axis=0)
        x = img.reshape(1, 96, 192, 1)
        img_data_list.append(x)

    data2 = pickle.load(open(args.data_dir+"data2_df.p", "rb"))
    for i in range(data2.shape[0]):
        file = args.data_dir+'bib_test/'+data2['uuids'][i]+'.jpg'
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (192, 96)).astype('float32')
        img = img / 255
        x = np.expand_dims(img, axis=0)
        x = img.reshape(1, 96, 192, 1)
        img_data_list.append(x)
        labels.append(data2['bibs'][i])
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

    if args.model=='1':
        model = DigitsModel2(shape=(rows,cols,channels))
        train_digits = [digits[0][:split], digits[1][:split], digits[2][:split], digits[3][:split], digits[4][:split]]
        test_digits = [digits[0][split:], digits[1][split:], digits[2][split:], digits[3][split:], digits[4][split:]]
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif args.model=='2':
        model = RecurrentModel(shape=(rows,cols,channels))
        train_digits=np.stack((digits[0][:split], digits[1][:split], digits[2][:split], digits[3][:split], digits[4][:split]),axis=1)
        test_digits =np.stack((digits[0][split:], digits[1][split:], digits[2][split:], digits[3][split:], digits[4][split:]),axis=1)
        model.summary()


        # clipnorm seems to speeds up convergence
        # sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        print('model not exist')
        return


    # Fitting the model
    model.fit(img_data[:split], train_digits, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
              validation_data=(img_data[split:], test_digits))
    model.save(args.data_dir + 'digits_model_gray_v2.h5')

if __name__ == '__main__':
    main()

# model = DigitsModel2(shape=(96,192,1))
# model = RecurrentModel(shape=(96,192,1))
# model.summary()
