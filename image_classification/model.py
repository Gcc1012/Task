import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_model(num_classes, input_shape=(224, 224, 3), freeze=False):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
