from keras.api.models import Model
from keras.api.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

class AudioModel:
    def __init__(self, num_classes: int, input_shape, batch_size: int) -> None:
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.model = self._create_model()

    def _create_model(self, X_train):

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

        print(f"Creating Model ...\n")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model
    
    def train_audio_model(self, epochs=100, patience=50, verbose = 1):
            
            print(f"Loading and preprocess the data ...\n")
            #TODO mettere metodo per creare il dataset
        
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
