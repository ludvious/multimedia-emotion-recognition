import os
from keras.api.models import Sequential
from keras.api.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.api.regularizers import l2
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils.emotions import EMOTIONS, NUM_CLASSES
from utils.utils import prepocess_face_dataset
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class ModelMiniXception:
    def __init__(self, num_classes: int, input_shape, batch_size: int) -> None:
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.model = self._create_model()

    def _create_model(self, l2_regularization=0.01):
        
        regularization = l2(l2_regularization)
        
        model = Sequential()

        # base
        model.add(Input(shape=self.input_shape))
        model.add(Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False, input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # x4 module

        #1
        model.add(SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        #2
        model.add(SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        #3
        model.add(SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        #4
        model.add(SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        #
        model.add(Conv2D(self.num_classes, (3, 3), padding='same'))
        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax', name='predictions'))

        print(f"Creating Model ...\n")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model

    def train_face_model(self, epochs=100, patience=50, verbose = 1):
            
            print(f"Loading and preprocess the data ...\n")
            train, validation = prepocess_face_dataset(self.input_shape, self.batch_size)
        
            # add callbacks
            early_stop = EarlyStopping('val_loss', patience=50)
            reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1) # Reduce learning rate when a metric has stopped improving
            trained_models_path = 'models/face/_mini_xception'
            model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
            model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=verbose, save_best_only=True)
            callbacks = [model_checkpoint, early_stop, reduce_lr]
            print(f"add callbacks ...\n")

            print(f"Start training ... \n")
            self.model.fit(train, batch_size=self.batch_size, epochs=epochs, validation_data=validation, callbacks=callbacks)

    def plot_training_history(self):
        history = self.model.history
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
        plt.show()        

face_model = ModelMiniXception(num_classes=NUM_CLASSES, input_shape=(48,48,1), batch_size=32)
face_model.train_face_model()

# The model weights (that are considered the best) can be loaded as -
# model.load_weights(checkpoint_filepath)
 
