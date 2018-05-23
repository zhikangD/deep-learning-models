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
from keras.models import load_model
import tensorflow as tf

def parse_args(args):
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data_dir',default='/home/ubuntu/zk/orientation/faceimg/', help='Path to dataset directory.')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--finetuning', action='store_true')
    return parser.parse_args(args)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)


    K.tensorflow_backend.set_session(get_session())
    with open('/home/ubuntu/zk/orientation/facelist.pkl', 'rb') as pk:
        facelist = _pickle.load(pk)
    with open('/home/ubuntu/zk/orientation/angles.pkl', 'rb') as pk:
        angles = _pickle.load(pk)
    t_size=224
    img_data_list = []
    for img in facelist:
        img_path = args.data_dir + img + '.jpg'
        img = image.load_img(img_path, target_size=(t_size, t_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data_list.append(x)
    img_data = np.array(img_data_list)
    print(img_data.shape)
    img_data = np.rollaxis(img_data, 1, 0)
    print(img_data.shape)
    img_data = img_data[0]
    print(img_data.shape)
    x, y = shuffle(img_data, np.array(angles)/180, random_state=2)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    if args.finetuning== True:
        custom_model= load_model('/home/ubuntu/zk/orientation/orient-06.h5')
    else:
        image_input = Input(shape=(224, 224, 3))
        model = ResNet50(input_tensor=image_input, weights='imagenet')
        x = model.get_layer('flatten_1').output
        out = Dense(1, name='output')(x)
        custom_model = Model(inputs=image_input, outputs=out)

    custom_model.summary()
    custom_model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
    filepath='/home/ubuntu/zk/orientation/orient-{epoch:02d}.h5'
    checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=False,period=2)
    callbacks_list = [checkpoint]
    custom_model.fit(np.array(X_train), np.array(y_train), epochs=args.epochs, batch_size=1, verbose=1,
                     validation_data=(np.array(X_test), np.array(y_test)),callbacks=callbacks_list)


if __name__ == '__main__':
    main()