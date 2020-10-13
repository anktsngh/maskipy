# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

import numpy as np
import os
import argparse
from configparser import ConfigParser
import json
import matplotlib.pyplot as plt

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard


def train(cfg):

    image_data = []
    for item in [d for d in os.listdir(cfg['DATASET']['LOC']) if os.path.isdir(os.path.join(cfg['DATASET']['LOC'], d))]:
        for image in os.listdir(os.path.join(cfg['DATASET']['LOC'], item)):
            image_data.append((json.loads(cfg['DATASET']['LABEL_MAP'])[item], preprocess_input(
                img_to_array(load_img(os.path.join(cfg['DATASET']['LOC'], item, image), target_size=(224, 224))))))

    image_data = shuffle(image_data)

    data = np.array([item[1] for item in image_data], dtype="float32")
    labels = np.array([item[0] for item in image_data])
    del image_data

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)
    # one-hot encoding
    labels = to_categorical(labels)

    cfg['TRAINING']['MODEL_OUTPUT_LABELS'] = json.dumps(list(encoder.classes_))

    # split data into test & train sets
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=float(cfg['TRAINING']['TEST_SIZE']),
                                                        stratify=labels)

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    # train only the head
    for layer in base_model.layers:
        layer.trainable = False

    # stack layers on top of the base MobileNetV2 model
    head_layer1 = AveragePooling2D(pool_size=(7, 7))
    head_layer2 = Flatten(name="flatten")
    head_layer3 = Dense(128, activation="relu")
    head_layer4 = Dense(128, activation="relu")
    head_layer5 = Dense(3, activation="softmax")

    model = Model(inputs=base_model.input,
                  outputs=(Sequential([head_layer1, head_layer2, head_layer3, head_layer4, head_layer5]))(
                      base_model.output))

    opt = Adam(lr=float(cfg['TRAINING']['LR']), decay=float(cfg['TRAINING']['LR']) / int(cfg['TRAINING']['EPOCHS']))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # construct the training image generator to increase diversity
    data_aug = ImageDataGenerator(rotation_range=15, zoom_range=0.15, width_shift_range=0.15, height_shift_range=0.15,
                                  shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

    log_dir = r'logs'
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    class_weights = {}
    for key, value in json.loads(cfg['TRAINING']['LABEL_WEIGHTS']).items():
        class_weights[list(encoder.classes_).index(key)] = value

    H = model.fit(data_aug.flow(train_x, train_y, batch_size=int(cfg['TRAINING']['BATCH_SIZE'])),
                  steps_per_epoch=len(train_x) // int(cfg['TRAINING']['BATCH_SIZE']), validation_data=(test_x, test_y),
                  validation_steps=len(test_x) // int(cfg['TRAINING']['BATCH_SIZE']), epochs=int(cfg['TRAINING']['EPOCHS']),
                  class_weight=class_weights, callbacks=[tensorboard_callback])

    # Commented out IPython magic to ensure Python compatibility.
    # %tensorboard --logdir logs

    # find index of label with highest probability & print classification report
    preds = model.predict(test_x, batch_size=int(cfg['TRAINING']['BATCH_SIZE']))
    print(classification_report(test_y.argmax(axis=1), np.argmax(preds, axis=1), target_names=encoder.classes_))

    # serialize model to disk
    os.makedirs('model', exist_ok=True)
    cfg['TRAINING']['MODEL_SAVE_PATH'] = f"model/model_ep{cfg['TRAINING']['EPOCHS']}_bs{cfg['TRAINING']['BATCH_SIZE']}.h5"
    model.save(cfg['TRAINING']['MODEL_SAVE_PATH'], save_format="h5")

    # plot accuracy & loss data
    plt.figure()
    plt.plot(np.arange(0, int(cfg['TRAINING']['EPOCHS'])), H.history["loss"], label="loss")
    plt.plot(np.arange(0, int(cfg['TRAINING']['EPOCHS'])), H.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig('Loss.svg', format='svg', dpi=1200)
    # plt.close()

    plt.figure()
    plt.plot(np.arange(0, int(cfg['TRAINING']['EPOCHS'])), H.history["accuracy"], label="accuracy")
    plt.plot(np.arange(0, int(cfg['TRAINING']['EPOCHS'])), H.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig('Accuracy.svg', format='svg', dpi=1200)
    # plt.close()

    with open(args['config'], 'w') as configfile:
        cfg.write(configfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CONFIG')
    parser.add_argument("-c", "--config", default='config.ini', help="config.ini file")
    args = vars(parser.parse_args())

    cfg = ConfigParser()
    cfg.read(args['config'])
    train(cfg)
