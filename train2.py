import numpy as np
import os
import time
import sys
import argparse
import _pickle
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD,RMSprop,adam


def parse_args(args):
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data_dir', default=None, help='Path to dataset directory.')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=20)
    parser.add_argument('--usepkldata',action='store_true', help='use data from saved pickle file or create from image.')
    parser.add_argument('--input_size', default=128,help='size of image', type=int)
    parser.add_argument('--model', default='cnn')
    return parser.parse_args(args)


def model2(size):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(size, size, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def cnnmodel(size):

    # number of output classes
    nb_classes = 2


    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    img_rows, img_cols = size,size

    model = Sequential()

    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv),
                            border_mode='valid',
                            input_shape=( img_rows, img_cols,3)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def main(args=None):
    # parse arguments

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    img_data_list = []
    img_data_list = []
    if args.usepkldata == False:
        with open('./data/img_list.pkl', 'rb') as pk:
            img_list = _pickle.load(pk)
        for img in img_list:
            # img_path = data_path + '/' + dataset + '/' + img
            img_path = args.data_dir + '/' + img + '.jpg'
            img = image.load_img(img_path, target_size=(args.input_size, args.input_size))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            print('Input image shape:', x.shape)
            img_data_list.append(x)
        img_data = np.array(img_data_list)
        # img_data = img_data.astype('float32')
        print(img_data.shape)
        img_data = np.rollaxis(img_data, 1, 0)
        print(img_data.shape)
        img_data = img_data[0]
        print(img_data.shape)
        with open('./data/img_data'+str(args.input_size)+'.pkl', 'wb') as pk:
            _pickle.dump(img_data, pk)
    else:
        with open('./data/img_data'+str(args.input_size)+'.pkl', 'rb') as pk:
            img_data = _pickle.load(pk)
            print(img_data.shape)

    num_classes = 2
    num_of_samples = img_data.shape[0]
    with open('./data/focus.pkl', 'rb') as pk:
        labels = _pickle.load(pk)

    names = ['bad', 'good']
    # convert class labels to on-hot encoding
    # Y = np_utils.to_categorical(labels, num_classes)








    ###########################################################################################################################
    # batch_size to train
    batch_size = 32
    # number of epochs to train
    nb_epoch = args.epochs

    # model = cnnmodel(args.input_size)
    if args.model == 'cnn':
        model = cnnmodel(args.input_size)
        Y = np_utils.to_categorical(labels, num_classes)
    else:
        model = model2(args.input_size)
        Y = labels

    # Shuffle the dataset
    x, y = shuffle(img_data, Y, random_state=2)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    t = time.time()
    model.summary()

    filepath = "./checkpoints/cnn-improvement-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                     verbose=1, validation_data=(X_test, y_test), callbacks=callbacks_list)
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = model.evaluate(X_test, y_test, batch_size=10, verbose=1)

    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    model.save()

if __name__ == '__main__':
    main()