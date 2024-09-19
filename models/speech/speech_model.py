from keras.api.models import Sequential, Model
from keras.api.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.regularizers import l2
from keras.api import layers
from keras.api.layers import Resizing
import tensorflow as tf
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

class AlexNetCNN:
    def __init__(self, num_classes: int, input_shape, batch_size: int) -> None:
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.model = self._create_model()

    def _create_model(self):

        model = Sequential()
        model.add(Resizing(224, 224, interpolation="bilinear", input_shape=self.input_shape[1:])) #replace with self.input_shape
        model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        model.summary()

        return model
    
    def train_face_model(self, epochs=100, patience=50, verbose = 1):
            
            print(f"Loading and preprocess the data ...\n")
            #TODO
        
            # add callbacks to review for this model
            '''early_stop = EarlyStopping('val_loss', patience=50)
            reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1) # Reduce learning rate when a metric has stopped improving
            checkpoint_models_path = 'models/face/mini_xception_'+'checkpoint.model.keras'
            model_checkpoint = ModelCheckpoint(filepath=checkpoint_models_path, monitor='val_loss', verbose=verbose, save_best_only=True)
            callbacks = [model_checkpoint, early_stop, reduce_lr]
            print(f"add callbacks ...\n")

            print(f"Start training ... \n")
            history = self.model.fit(train, batch_size=self.batch_size, epochs=epochs, validation_data=validation, callbacks=callbacks)

            return history'''

    def plot_training_history(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['acc'], label='Train Accuracy')
        plt.plot(history.history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def _create_model_cnn(self, X_train):

        input_shape = X_train[0].shape
        input_layer = Input(shape=input_shape)
        x = Conv2D(16, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        output_layer = Dense(self.num_classes, activation='softmax')(x)
        model = Model(input_layer, output_layer)

        return model