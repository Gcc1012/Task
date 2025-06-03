import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    train_gen = datagen.flow_from_directory("data", target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training')
    val_gen = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation')
    return train_gen, val_gen
