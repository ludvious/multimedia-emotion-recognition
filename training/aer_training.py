import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.api.models import Model, Sequential
from keras.api.layers import (
    Input, Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dense, Dropout, RandomRotation, RandomZoom, RandomFlip,
    Rescaling, SeparableConv2D, BatchNormalization, Activation, GlobalAveragePooling2D, AveragePooling2D,
    Resizing, RandomTranslation, RandomBrightness, RandomContrast)
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.api.regularizers import l2
from keras.api.optimizers import Adam
import os, cv2
import matplotlib.pyplot as plt
from keras import layers
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.api.applications import VGG16
from preprocessing.utils import unzip
from config import NUM_LABELS, TARGET_RATE, WINDOW_SIZE, HOP_LENGTH, N_MELS_BAND


unzip(filepath="data/speech/spectrogram.zip", outdir="data/speech/")

def process_audio_data_generator(data_path, batch_size, input_shape):

      train_gen = ImageDataGenerator(
          rescale = 1./255,                                  # Rescale pixel values to be between 0 and 1
          validation_split = 0.2                             # Set aside 20% of the data for validation
      )

      val_gen = ImageDataGenerator(
          rescale = 1./255,                                  # Rescale pixel values to be between 0 and 1
          validation_split = 0.2                             # Set aside 20% of the data for validation
      )

      train_generator = train_gen.flow_from_directory(
          directory = data_path,                             # Directory containing the training data
          target_size = input_shape[:2],                     # Resizes all images to 48x48 pixels
          batch_size = batch_size,                           # Number of images per batch
          color_mode = "rgb",                                # Converts the images to grayscale
          class_mode = "categorical",                        # Classifies the images into 7 categories
          subset = "training",                               # Uses the training subset of the data
          shuffle = True,
          seed = 0
      )

      val_generator = val_gen.flow_from_directory(
          directory = data_path,                             # Directory containing the training data
          target_size = input_shape[:2],                     # Resizes all images to 48x48 pixels
          batch_size = batch_size,                           # Number of images per batch
          color_mode = "rgb",                                # Converts the images to grayscale
          class_mode = "categorical",                        # Classifies the images into 7 categories
          subset = "validation",                             # Uses the training subset of the data
          shuffle = True,
          seed = 0
      )

      return train_generator, val_generator

class AudioModel:
    def __init__(self, model_name: str, num_labels: int, input_shape, batch_size: int, epochs: int, vggish_model) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self._build_vggish_model(vggish_model)

    def _create_model(self):

        img_input = Input(self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu')(img_input)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        #x = Conv2D(64, (3, 3), activation='relu')(x)
        #x = MaxPooling2D((2, 2))(x)
        #x = Dropout(0.2)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(24, activation='relu')(x)
        # Output layer
        output = Dense(self.num_labels, activation='softmax')(x)
        model = Model(img_input, output)

        print(f"Creating Model ...\n")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model

    def _build_vggish_model(self, vggish_model):

      # Build custom classification layers on top of VGGish
      x = vggish_model.output
      x = tf.keras.layers.Flatten()(x)
      x = Dense(128, activation='relu')(x)
      x = Dropout(0.3)(x)
      x = Dense(64, activation='relu')(x)
      x = Dropout(0.3)(x)
      output = Dense(self.num_labels, activation='softmax')(x)

      model = Model(inputs=vggish_model.input, outputs=output)
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

      return model

    def create_dataset(self, path_data: str):
        print(f"Loading and preprocess the dataset ...\n")
        # Split data into training and testing sets
        #features, labels = create_audio_dataset(path_data, self.num_labels)
        train, validation = process_audio_data_generator(path_data, self.batch_size, self.input_shape)
        #labels = to_categorical(labels, num_labels=self.num_labels)  # Convert labels to one-hot encoding

        return train, validation

    def train_model(self, train, validation, verbose = 1):

        #X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
        # add callbacks to review for this model
        early_stop = EarlyStopping('val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.2, patience=5, verbose=1) # Reduce learning rate when a metric has stopped improving
        checkpoint_models_path = 'models/speech/audio_modelvgg16'+'checkpoint.model.keras'
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_models_path, monitor='val_loss', verbose=verbose, save_best_only=True)
        callbacks = [model_checkpoint, early_stop, reduce_lr]
        print(f"add callbacks ...\n")

        print(f"Start training ... \n")
        history = self.model.fit(train, batch_size=self.batch_size, epochs=self.epochs, validation_data=validation, callbacks=callbacks)

        return history

del model

model = AudioModel('audiomodel', NUM_LABELS, (N_MELS_BAND, N_MELS_BAND, 3), 32, epochs=30)

del train, val

train, val = model.create_dataset('/content/spectrogram')

del history

history = model.train_model(train, val)

plt.figure(figsize=(7, 3))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(7, 3))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Get the true labels and predicted labels for the validation set
validation_labels = val.classes
#model = load_model('/content/cnn_checkpoint.model.keras')
validation_pred_probs = model.model.predict(val)
validation_pred_labels = np.argmax(validation_pred_probs, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)
class_names = list(train.class_indices.keys())
sns.set()
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

"""###**vgg16**"""

# Load VGGish pretrained model and exclude its top layers
vggish_model = VGG16(weights='imagenet', include_top=False, input_shape=(N_MELS_BAND, N_MELS_BAND, 3))

# Freeze the layers of VGGish
for layer in vggish_model.layers:
    layer.trainable = False

vggmodel = AudioModel('vggomodel', NUM_LABELS, (N_MELS_BAND, N_MELS_BAND, 3), 64, epochs=10, vggish_model=vggish_model)

trainvgg, valvgg = vggmodel.create_dataset('/content/spectrogram')

historyvgg = vggmodel.train_model(trainvgg, valvgg)

for layer in vggish_model.layers[-4:]:
    layer.trainable = True

vggmodel2 = AudioModel('vggomodel', NUM_LABELS, (N_MELS_BAND, N_MELS_BAND, 3), 64, epochs=10, vggish_model=vggish_model)

history2vgg = vggmodel2.train_model(trainvgg, valvgg)