import numpy as np
import os
import time
import sys
import argparse
import _pickle
from resnet50 import ResNet50
from vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import metrics
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from blurMapping import KitModel
from keras import backend as K
import tensorflow as tf

# def parse_args(args):
#     parser = argparse.ArgumentParser(description='training script')
#     # parser.add_argument('--data_dir', help='Path to dataset directory.')
#     parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=50)
#     return parser.parse_args(args)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def main(args=None):


    K.tensorflow_backend.set_session(get_session())
    with open('/home/ubuntu/zk/faceimg.pkl', 'rb') as pk:
        faceimg = _pickle.load(pk)
    with open('/home/ubuntu/angles.pkl', 'rb') as pk:
        angles = _pickle.load(pk)
    x, y = shuffle(np.array(faceimg), angles, random_state=2)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    image_input = Input(shape=(224, 224, 3))
    model = ResNet50(input_tensor=image_input, weights='imagenet')
    x = model.get_layer('flatten_1').output
    out = Dense(1, name='output')(x)
    custom_model = Model(inputs=image_input, outputs=out)
    custom_model.summary()
    custom_model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
    custom_model.fit(X_train, y_train, nb_epoch=50, batch_size=2, verbose=1)
    predicted = model.predict(X_test)

    custom_model.save('/home/ubuntu/zk/orientation.h5')
    print(np.array(predicted)-np.array(y_test))

if __name__ == '__main__':
    main()