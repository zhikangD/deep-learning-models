import numpy as np
import os
import time
import sys
import argparse
import _pickle
from resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

def parse_args(args):
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data_dir', help='Path to dataset directory.')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=20)
    parser.add_argument('--usepkldata',action='store_true', help='use data from saved pickle file or create from image.')
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    # data_path = args.data_dir
    # data_dir_list = os.listdir(data_path)
    img_data_list = []
    if args.usepkldata==False:
        with open('./data/img_list.pkl', 'rb') as pk:
            img_list = _pickle.load(pk)
        for img in img_list:
            # img_path = data_path + '/' + dataset + '/' + img
            img_path = args.data_dir+'/'+img+'.jpg'
            img = image.load_img(img_path, target_size=(224, 224))
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
        with open('./data/img_data.pkl', 'wb') as pk:
            _pickle.dump(img_data, pk)
    else:
        with open('./data/img_data.pkl', 'rb') as pk:
            img_data = _pickle.load(pk)
            print(img_data.shape)

    num_classes = 2
    num_of_samples = img_data.shape[0]
    with open('./data/focus.pkl', 'rb') as pk:
        labels = _pickle.load(pk)

    names = ['bad', 'good']
    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)
    # Shuffle the dataset
    x, y = shuffle(img_data, Y, random_state=2)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    ###########################################################################################################################
    # Training the classifier alone
    image_input = Input(shape=(224, 224, 3))

    model = ResNet50(input_tensor=image_input, weights='imagenet')
    model.summary()
    last_layer = model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(num_classes, activation='softmax', name='output_layer')(x)
    custom_resnet_model = Model(inputs=image_input, outputs=out)
    custom_resnet_model.summary()
    for layer in custom_resnet_model.layers[:-1]:
        layer.trainable = False
    custom_resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    t = time.time()
    filepath = "./weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=args.epochs, verbose=1,
                                   validation_data=(X_test, y_test),
                                   callbacks=callbacks_list)
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


if __name__ == '__main__':
    main()