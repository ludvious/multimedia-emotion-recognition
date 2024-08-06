import tensorflow as tf
import pandas as pd
import cv2, os, librosa
import numpy as np
from keras.api.utils import image_dataset_from_directory
from keras.api.models import Sequential
from keras.api.layers import RandomFlip, RandomRotation, RandomZoom, Rescaling

class DataLoader:
    @staticmethod
    def load_fer2013_data(data_dir):
        data = pd.read_csv(os.path.join(data_dir, 'fer2013.csv'))
        return data

def prepocess_face_dataset(input_shape):

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

    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1)
    ])

    # Add Rescaling layer to normalize pixel values
    normalization_layer = Rescaling(1./255)
    
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y))
    #train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Prefetch the datasets for better performance
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset


def capture_frames_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def extract_spectrogram(file_path):
        y, sr = librosa.load(file_path)
        spect = librosa.feature.melspectrogram(y=y, sr=sr)
        spect_db = librosa.power_to_db(spect, ref=np.max)
        return spect_db
