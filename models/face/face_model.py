import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from utils.load_datasets import load_face_dataset
from keras.api.utils import image_dataset_from_directory
from keras.api.layers import Rescaling
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class FaceModel:
    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._create_face_model()

    def _create_face_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(256, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        print(f"Creating Model ...\n")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model

    def train_face_model(self, batch_size=64, epochs=50, saving=True):
        
        train, validation = load_face_dataset(self.input_shape)
        self.model.fit(train, epochs=epochs, validation_data=validation)

        if saving:
            self.model.save("models/face.keras")

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

model = FaceModel(num_classes=7, input_shape=(48,48,3))
