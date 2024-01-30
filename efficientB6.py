import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import InceptionV3
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from functools import partial
import albumentations as A
import warnings
from tensorflow.keras.applications import EfficientNetB6
warnings.filterwarnings('ignore')
AUTO = tf.data.experimental.AUTOTUNE

# Change this line in your CSV file reading section
df = pd.read_csv(r'C:\Users\vishv\OneDrive\Documents\extracting-attributes-from-fashion-images-jan-2024\train.csv')

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

features = df['file_name']
target = df['label']

X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                  test_size=0.10,
                                                  random_state=10)

transforms_train = A.Compose([
    A.VerticalFlip(p=0.2),
    A.HorizontalFlip(p=0.7),
    A.CoarseDropout(p=0.5),
    A.RandomGamma(p=0.5),
    A.RandomBrightnessContrast(p=1)
])


def aug_fn(img):
    aug_data = transforms_train(image=img)
    aug_img = aug_data['image']

    return aug_img


@tf.function
def process_data(img, label):
    aug_img = tf.numpy_function(aug_fn, [img], Tout=tf.float32)
    return img, label


# Update decode_image function
def decode_image(file_name, label=None):
    # Update the file path to point to the "train" folder
    file_path = 'train/' + file_name

    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [244, 244])
    img = tf.cast(img, tf.float32) / 255.0

    if label is None:
        return img

    return img, tf.one_hot(indices=label, depth=7, dtype=tf.float32)


train_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_train, Y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(partial(process_data), num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_val, Y_val))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

for img, label in train_ds.take(1):
    print(img.shape, label.shape)
from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = EfficientNetB6(
    input_shape=(244, 244, 3),
    weights='imagenet',
    include_top=False
)
for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('top_activation')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
x = layers.Flatten()(last_output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
output = layers.Dense(7, activation='softmax')(x)

model = keras.Model(pre_trained_model.input, output)
model.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['AUC']
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_auc') > 0.99:
            print('\n Validation accuracy has reached upto 90%\
            so, stopping further training.')
            self.model.stop_training = True
es = EarlyStopping(patience=3,
                               monitor='val_auc',
                               restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_loss',
                                   patience=2,
                                   factor=0.5,
                                   verbose=1)
history = model.fit(train_ds,
                                validation_data=val_ds,
                                epochs=10,
                                verbose=1,
                                callbacks=[es, lr, myCallback()])

model.save('fashion_image_classification_modelEB7.h5')