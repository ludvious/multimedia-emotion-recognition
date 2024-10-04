from keras.api.models import Model
from keras.api.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing.dataset import create_audio_dataset
from keras.api.utils import to_categorical

class AudioModel:
    def __init__(self, num_labels: int, input_shape, batch_size: int) -> None:
        self.num_labels = num_labels
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
        output_layer = Dense(self.num_labels, activation='softmax')(x)
        model = Model(input_layer, output_layer)

        print(f"Creating Model ...\n")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model
    
    def create_dataset(self, path_data: str):
        print(f"Loading and preprocess the dataset ...\n")
        # Split data into training and testing sets
        features, labels = create_audio_dataset(path_data, self.num_labels)
        labels = to_categorical(labels, num_labels=self.num_labels)  # Convert labels to one-hot encoding

        return features, labels

    def train_model(self, features, labels, epochs=100, verbose = 1):
        
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
        # add callbacks to review for this model
        early_stop = EarlyStopping('val_loss', patience=10)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=5, verbose=1) # Reduce learning rate when a metric has stopped improving
        checkpoint_models_path = 'models/face/mini_xception_'+'checkpoint.model.keras'
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_models_path, monitor='val_loss', verbose=verbose, save_best_only=True)
        callbacks = [model_checkpoint, early_stop, reduce_lr]
        print(f"add callbacks ...\n")

        print(f"Start training ... \n")
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks)

        return history

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
