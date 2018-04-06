from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from spatial_transformer import SpatialTransformer
from keras.models import Sequential


def load_weights_from_file(weight_file):
    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def DigitsModel(weight_file = None):
    weights_dict = load_weights_from_file(weight_file) if not weight_file == None else None
    input_shape=(128,224,3)
    # locnet = Sequential()
    # locnet.add(MaxPooling2D(pool_size=(2, 2), input_shape=input_shape))
    # locnet.add(Conv2D(20, (5, 5)))
    # locnet.add(MaxPooling2D(pool_size=(2, 2)))
    # locnet.add(Conv2D(20, (5, 5)))
    #
    # locnet.add(Flatten())
    # locnet.add(Dense(50))
    # locnet.add(Activation('relu'))
    # locnet.add(Dense(6))
    x = Input(name='data', shape=(128, 224, 3,))
    # x = SpatialTransformer(localization_net=locnet,
    #                              output_size=(128, 224), input_shape=input_shape)(data)
    x = Conv2D(48, (5, 5), activation='relu', padding='same', name='conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1),padding='same', name='block2_pool')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block3_pool')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(160, (5, 5), activation='relu', padding='same', name='conv4')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1),padding='same', name='block4_pool')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (5, 5), activation='relu', padding='same', name='conv5')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2  ),padding='same', name='block5_pool')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (5, 5), activation='relu', padding='same', name='conv6')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1),padding='same', name='block6_pool')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (5, 5), activation='relu', padding='same', name='conv7')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block7_pool')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (5, 5), activation='relu', padding='same', name='conv8')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1),padding='same', name='block8_pool')(x)
    x = Dropout(0.25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    digit_length = Dense(6, name='digit_length')(x)
    digit_1 = Dense(12, name='digit_1')(x)
    digit_2 = Dense(12, name='digit_2')(x)
    digit_3 = Dense(12, name='digit_3')(x)
    digit_4 = Dense(12, name='digit_4')(x)
    digit_5 = Dense(12, name='digit_5')(x)
    # digits = K.stack([digit_1, digit_2, digit_3, digit_4, digit_5], axis=1)

    model = Model(inputs=[data], outputs=[digit_length, digit_1, digit_2, digit_3, digit_4, digit_5])
    return model


