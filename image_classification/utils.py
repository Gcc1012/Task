# def save_model(model, filename):
#     model.save(filename)
#     print(f"Model saved to {model.pt}")

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras

from tensorflow.keras.models import save_model as keras_save_model

def save_model(model, filepath):
    keras_save_model(model, filepath)
    print(f" Model is being saved to this path {filepath}")
