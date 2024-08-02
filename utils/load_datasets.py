from keras.api.utils import image_dataset_from_directory
from keras.api.layers import Rescaling
import tensorflow as tf
import os
import keras

def load_face_dataset(input_shape):

    train_dir = os.path.join('data\face\fer-2013\train')
    test_dir = os.path.join('data\face\fer-2013\test')

    train_dataset = image_dataset_from_directory(
            directory=train_dir,
            label_mode='categorical',
            subset='training',
            seed=123,
            image_size=input_shape[:2],
            color_mode='grayscale',
            batch_size=64
        )
    
    val_dataset = image_dataset_from_directory(
        directory=test_dir,
        label_mode='categorical',
        subset='validation',
        seed=123,
        image_size=input_shape[:2],
        color_mode='grayscale',
        batch_size=64
    )

    # Add Rescaling layer to normalize pixel values
    normalization_layer = Rescaling(1./255)
    
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Prefetch the datasets for better performance
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset
