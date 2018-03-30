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
from sklearn.preprocessing import LabelEncoder
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator


def parse_args(args):
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data_dir', help='Path to dataset directory.')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=20)
    parser.add_argument('--usepkldata',action='store_true', help='use data from saved pickle file or create from image.')
    parser.add_argument('--model', default='resnet')
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--object', default='focus')
    return parser.parse_args(args)

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def main(args=None):


    K.tensorflow_backend.set_session(get_session())
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    if args.model=='blurmapping':
        t_size=96
    elif args.model=='vgg16':
        t_size=64
    else:
        t_size=224
    # data_path = args.data_dir
    # data_dir_list = os.listdir(data_path)
    img_data_list = []
    if args.usepkldata==False:
        with open('./data/img_list.pkl', 'rb') as pk:
            img_list = _pickle.load(pk)
        for img in img_list:
            # img_path = data_path + '/' + dataset + '/' + img
            img_path = args.data_dir+'/'+img+'.jpg'
            img = image.load_img(img_path, target_size=(t_size,t_size))
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
        with open('./data/img_data'+str(t_size)+'.pkl', 'wb') as pk:
            _pickle.dump(img_data, pk)
    else:
        with open('./data/img_data'+str(t_size)+'.pkl', 'rb') as pk:
            img_data = _pickle.load(pk)
            print(img_data.shape)

    num_classes = 2
    num_of_samples = img_data.shape[0]
    if args.object =='focus':
        with open('./data/focus.pkl', 'rb') as pk:
            labels = _pickle.load(pk)
    elif args.object == 'quality':
        with open('./data/quality.pkl', 'rb') as pk:
            labels = _pickle.load(pk)




    names = ['bad', 'good']
    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)
    # Y = labels
    # encoder = LabelEncoder()
    # encoder.fit(Y)
    # encoded_Y = encoder.transform(Y)
    # Shuffle the dataset
    x, y = shuffle(img_data, Y, random_state=2)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    ###########################################################################################################################
    if args.model=='resnet':
        # Training the classifier alone
        image_input = Input(shape=(224, 224, 3))

        model = ResNet50(input_tensor=image_input, weights='imagenet')
        model.summary()
        last_layer = model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(1000, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        out = Dense(num_classes, activation='softmax', name='output_layer')(x)
        custom_resnet_model = Model(inputs=image_input, outputs=out)
        custom_resnet_model.summary()
        for layer in custom_resnet_model.layers[:-1]:
            layer.trainable = False
        custom_resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        t = time.time()
        filepath = "./data/resnet-"+str(args.object)+"-{epoch:02d}-{val_acc:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
        callbacks_list = [checkpoint]
        hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=args.epochs, verbose=1,
                                       validation_data=(X_test, y_test),
                                       callbacks=callbacks_list)
        Y = np_utils.to_categorical(labels, num_classes)

        print('Training time: %s' % (t - time.time()))
        (loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=args.batch_size, verbose=1)

        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    #########################################################################################
    elif args.model=='vgg16':
        # Custom_vgg_model_1
        # Training the classifier alone
        # image_input = Input(shape=(224, 224, 3))
        # Y = labels
        # encoder = LabelEncoder()
        # encoder.fit(Y)
        # encoded_Y = encoder.transform(Y)
        # Shuffle the dataset
        x, y = shuffle(img_data, Y, random_state=2)
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
        vgg_model = VGG16( include_top=False, weights='imagenet')
        for layer in vgg_model.layers[:-1]:
            layer.trainable = False
        inp = Input(shape=(64, 64, 3), name='input_image')
        output_vgg_conv = vgg_model(inp)
        x_1 = Flatten(name='flatten')(output_vgg_conv)
        x_1 = Dense(512, activation='relu', name='fc1')(x_1)
        x_1 = Dropout(0.5)(x_1)
        x_1 = Dense(128, activation='relu', name='fc2')(x_1)
        x_1 = Dropout(0.25)(x_1)
        x_1 = Dense(8, activation='relu', name='fc3')(x_1)
        x_1 = Dropout(0.125)(x_1)
        x_1 = Dense(2, activation='softmax', name='frontalpred')(x_1)
        custom_vgg_model = Model(input=inp, output=x_1)
        custom_vgg_model.summary()



        filepath = "./data/vgg16-"+str(args.object)+"-{epoch:02d}-{val_acc:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        custom_vgg_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        t = time.time()
        #	t = now()
        hist = custom_vgg_model.fit(X_train, y_train, batch_size=32, epochs=args.epochs, verbose=1, validation_data=(X_test, y_test),
                                    callbacks=callbacks_list)
        print('Training time: %s' % (t - time.time()))
        (loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=args.batch_size, verbose=1)

        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
        ##############################################################################
    elif args.model=='blurmapping':
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        datagen.fit(img_data)

        model = KitModel(weight_file='blurMapping.npy')
        model.summary()
        last_layer = model.get_layer('relu5_3').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(512, activation='relu', name='fc_1')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dropout(0.25)(x)
        out = Dense(num_classes, activation='softmax', name='output_layer')(x)
        custom_model = Model(inputs=model.input, outputs=out)
        custom_model.summary()
        for layer in custom_model.layers[:-1]:
            layer.trainable = False
        filepath = "./data/blurmapping-"+str(args.object)+"-{epoch:02d}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)
        callbacks_list = [checkpoint]

        custom_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        t = time.time()
        #	t = now()
        # hist = custom_model.fit(X_train, y_train, batch_size=32, epochs=args.epochs, verbose=1,
        #                             validation_data=(X_test, y_test),
        #                             callbacks=callbacks_list)
        # Y = np_utils.to_categorical(labels, num_classes)
        hist = custom_model.fit_generator(datagen.flow(img_data, Y, batch_size=32),callbacks=callbacks_list,
                        steps_per_epoch=1000, epochs=50)
        print('Training time: %s' % (t - time.time()))
        (loss, accuracy) = custom_model.evaluate(X_test, y_test, batch_size=args.batch_size, verbose=1)

        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


# python3 train.py --data_dir=./data/img --model=blurmapping


if __name__ == '__main__':
    main()