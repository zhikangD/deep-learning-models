import keras
from keras.models import Model
from keras import layers
import keras.backend as K
import numpy as np


def load_weights_from_file(weight_file):
    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def set_layer_weights(model, weights_dict):
    for layer in model.layers:
        if layer.name in weights_dict:
            cur_dict = weights_dict[layer.name]
            current_layer_parameters = list()
            if layer.__class__.__name__ == "BatchNormalization":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
                current_layer_parameters.extend([cur_dict['mean'], cur_dict['var']])
            elif layer.__class__.__name__ == "SeparableConv2D":
                current_layer_parameters = [cur_dict['depthwise_filter'], cur_dict['pointwise_filter']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            else:
                # rot weights
                current_layer_parameters = [cur_dict['weights']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            model.get_layer(layer.name).set_weights(current_layer_parameters)

    return model


def KitModel(weight_file = None):
    weights_dict = load_weights_from_file(weight_file) if not weight_file == None else None
        
    data            = layers.Input(name = 'data', shape = (384, 384, 3,) )
    conv1_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(data)
    conv1_1         = convolution(weights_dict, name='conv1_1', input=conv1_1_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu1_1         = layers.Activation(name = 'relu1_1', activation = 'relu')(conv1_1)
    conv1_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu1_1)
    conv1_2         = convolution(weights_dict, name='conv1_2', input=conv1_2_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu1_2         = layers.Activation(name = 'relu1_2', activation = 'relu')(conv1_2)
    pool1_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu1_2)
    pool1           = layers.MaxPooling2D(name = 'pool1', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool1_input)
    conv2_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool1)
    conv2_1         = convolution(weights_dict, name='conv2_1', input=conv2_1_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu2_1         = layers.Activation(name = 'relu2_1', activation = 'relu')(conv2_1)
    conv2_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu2_1)
    conv2_2         = convolution(weights_dict, name='conv2_2', input=conv2_2_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu2_2         = layers.Activation(name = 'relu2_2', activation = 'relu')(conv2_2)
    pool2_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu2_2)
    pool2           = layers.MaxPooling2D(name = 'pool2', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool2_input)
    conv3_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool2)
    conv3_1         = convolution(weights_dict, name='conv3_1', input=conv3_1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu3_1         = layers.Activation(name = 'relu3_1', activation = 'relu')(conv3_1)
    conv3_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu3_1)
    conv3_2         = convolution(weights_dict, name='conv3_2', input=conv3_2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu3_2         = layers.Activation(name = 'relu3_2', activation = 'relu')(conv3_2)
    conv3_3_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu3_2)
    conv3_3         = convolution(weights_dict, name='conv3_3', input=conv3_3_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu3_3         = layers.Activation(name = 'relu3_3', activation = 'relu')(conv3_3)
    pool3_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu3_3)
    pool3           = layers.MaxPooling2D(name = 'pool3', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool3_input)
    conv4_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool3)
    conv4_1         = convolution(weights_dict, name='conv4_1', input=conv4_1_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu4_1         = layers.Activation(name = 'relu4_1', activation = 'relu')(conv4_1)
    conv4_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu4_1)
    conv4_2         = convolution(weights_dict, name='conv4_2', input=conv4_2_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu4_2         = layers.Activation(name = 'relu4_2', activation = 'relu')(conv4_2)
    conv4_3_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu4_2)
    conv4_3         = convolution(weights_dict, name='conv4_3', input=conv4_3_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu4_3         = layers.Activation(name = 'relu4_3', activation = 'relu')(conv4_3)
    pool4_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu4_3)
    pool4           = layers.MaxPooling2D(name = 'pool4', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool4_input)
    conv5_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool4)
    conv5_1         = convolution(weights_dict, name='conv5_1', input=conv5_1_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu5_1         = layers.Activation(name = 'relu5_1', activation = 'relu')(conv5_1)
    conv5_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu5_1)
    conv5_2         = convolution(weights_dict, name='conv5_2', input=conv5_2_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu5_2         = layers.Activation(name = 'relu5_2', activation = 'relu')(conv5_2)
    conv5_3_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu5_2)
    conv5_3         = convolution(weights_dict, name='conv5_3', input=conv5_3_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    relu5_3         = layers.Activation(name = 'relu5_3', activation = 'relu')(conv5_3)
    conv5_blur      = convolution(weights_dict, name='conv5_blur', input=relu5_3, group=1, conv_type='layers.Conv2D', filters=1, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    conv5_blur_up_input = layers.ZeroPadding2D(padding = ((8, 8), (8, 8)))(conv5_blur)
    conv5_blur_up   = convolution(weights_dict, name='conv5_blur_up', input=conv5_blur_up_input, group=1, conv_type='layers.Conv2DTranspose', filters=1, kernel_size=(32, 32), strides=(16, 16), dilation_rate=(1, 1), padding='valid', use_bias=False)
    score           = layers.Activation(name = 'score', activation = 'sigmoid')(conv5_blur_up)
    model           = Model(inputs = [data], outputs = [score])
    set_layer_weights(model, weights_dict)
    return model

def convolution(weights_dict, name, input, group, conv_type, filters=None, **kwargs):
    if not conv_type.startswith('layer'):
        layer = keras.applications.mobilenet.DepthwiseConv2D(name=name, **kwargs)(input)
        return layer

    grouped_channels = int(filters / group)
    group_list = []

    if group == 1:
        func = getattr(layers, conv_type.split('.')[-1])
        layer = func(name = name, filters = filters, **kwargs)(input)
        return layer

    weight_groups = list()
    if not weights_dict == None:
        w = np.array(weights_dict[name]['weights'])
        weight_groups = np.split(w, indices_or_sections=group, axis=-1)

    for c in range(group):
        x = layers.Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
        x = layers.Conv2D(name=name + "_" + str(c), filters=grouped_channels, **kwargs)(x)
        weights_dict[name + "_" + str(c)] = dict()
        weights_dict[name + "_" + str(c)]['weights'] = weight_groups[c]

        group_list.append(x)

    layer = layers.concatenate(group_list, axis = -1)

    if 'bias' in weights_dict[name]:
        b = K.variable(weights_dict[name]['bias'], name = name + "_bias")
        layer = layer + b
    return layer
