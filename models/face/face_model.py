import os
from keras.api.models import Sequential
from keras.api.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils.emotions import EMOTIONS, NUM_CLASSES, FER_EMOTION_SHAPE
from utils.utils import prepocess_face_dataset
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class CNNFaceModel:
    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._create_model()

    def _create_model(self):

        model = Sequential()

        model.add(Input(shape=self.input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        print(f"Creating Model ...\n")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model

    def train_face_model(self, batch_size: int, epochs: int, patience=50, verbose=1):
        
        train, validation = prepocess_face_dataset(self.input_shape)
        
        # add callbacks
        early_stop = EarlyStopping('val_loss', patience=50)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=verbose) # Reduce learning rate when a metric has stopped improving
        trained_models_path = 'models/face/' + '_cnn'
        model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
        callbacks = [model_checkpoint, early_stop, reduce_lr]

        self.model.fit(train, batch_size=batch_size, epochs=epochs, validation_data=validation, callbacks=callbacks)


# TRAINING

face_model = CNNFaceModel(num_classes=NUM_CLASSES, input_shape=(48,48,1))
face_model.train_face_model(batch_size=64, epochs=50)

'''history = face_model.history
plt.figure(figsize=(10, 5))
plt.plot(history['acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()'''

# The model weights (that are considered the best) can be loaded as -
# model.load_weights(checkpoint_filepath)