# -*- coding: utf-8 -*-

# %tensorflow_version 2.x
import tensorflow
import config as cfg

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

import os
import numpy as np
import matplotlib.pyplot as plt

image_data = []

# read image files for each class and pre-process them
try:
    for image in os.listdir('dataset/NMFD'):
        image_data.append(('not_masked', preprocess_input(
            img_to_array(load_img(os.path.join('dataset/NMFD', image), target_size=(224, 224))))))

    for image in os.listdir('dataset/IMFD'):
        image_data.append(('improperly_masked', preprocess_input(
            img_to_array(load_img(os.path.join('dataset/IMFD', image), target_size=(224, 224))))))

    for image in os.listdir('dataset/CMFD'):
        image_data.append(('properly_masked', preprocess_input(
            img_to_array(load_img(os.path.join('dataset/CMFD', image), target_size=(224, 224))))))

except FileNotFoundError:
    print('FileNotFoundError: Ensure to download training dataset from links in the README')

image_data = shuffle(image_data)

# convert to numpy array
data = np.array([item[1] for item in image_data], dtype="float32")
labels = np.array([item[0] for item in image_data])

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)

# one-hot encoding
labels = to_categorical(labels)

# split data into test & train sets
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.20, stratify=labels)

# MobileNetV2 without the head
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# stack layers on top of the base MobileNetV2 model
head_layer1 = AveragePooling2D(pool_size=(7, 7))
head_layer2 = Flatten(name="flatten")
head_layer3 = Dense(128, activation="relu")
head_layer4 = Dense(128, activation="relu")
head_layer5 = Dense(3, activation="softmax")

final_model = (Sequential([head_layer1, head_layer2, head_layer3, head_layer4, head_layer5]))(base_model.output)

# train only the head
for layer in base_model.layers:
    layer.trainable = False

model = Model(inputs=base_model.input, outputs=final_model)

opt = Adam(lr=cfg.LR, decay=cfg.LR / cfg.EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the training image generator to increase diversity
data_aug = ImageDataGenerator(rotation_range=15, zoom_range=0.15, width_shift_range=0.15, height_shift_range=0.15,
                              shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

H = model.fit(data_aug.flow(train_x, train_y, batch_size=cfg.BATCH_SIZE),
              steps_per_epoch=len(train_x) // cfg.BATCH_SIZE, validation_data=(test_x, test_y),
              validation_steps=len(test_x) // cfg.BATCH_SIZE, epochs=cfg.EPOCHS, class_weight={0: 1.2, 1: 1, 2: 1})

# find index of label with highest probability & print classification report
preds = model.predict(test_x, batch_size=cfg.BATCH_SIZE)
print(classification_report(test_y.argmax(axis=1), np.argmax(preds, axis=1), target_names=encoder.classes_))

# serialize model to disk
model.save('model.h5', save_format="h5")

# plot accuracy & loss data
plt.figure()
plt.plot(np.arange(0, cfg.EPOCHS), H.history["loss"], label="loss")
plt.plot(np.arange(0, cfg.EPOCHS), H.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Loss.svg', format='svg', dpi=1200)
# plt.close()

plt.figure()
plt.plot(np.arange(0, cfg.EPOCHS), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, cfg.EPOCHS), H.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Accuracy.svg', format='svg', dpi=1200)
# plt.close()
