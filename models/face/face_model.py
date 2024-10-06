from keras.api.models import Model
from keras.api.layers import Input, Conv2D, MaxPooling2D, MaxPool2D, SeparableConv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.api.regularizers import l2
from keras.api import layers
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from preprocessing.dataset import create_face_dataset
import matplotlib.pyplot as plt
from config import NUM_LABELS

class FaceModel:
    def __init__(self, model_name, input_shape, batch_size: int) -> None:
        self.model_name = model_name
        self.num_labels = NUM_LABELS
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.model = self._create_model()

    def _create_model(self):
        if self.model_name == 'mini_xception':
            model = self._create_model_minixcpetion()
        if self.model_name == 'cnn':
            model = self._create_model_cnn()
        if self.model_name == 'vgg':
            model = self._create_model_vgg()
        
        return model
    
    def _create_model_vgg(self, lr=1e-3):
        
        inputs = Input(shape=self.input_shape)

        # Rescaling layer
        #x = Rescaling(1./255)(inputs)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D()(x)
        x = Dropout(0.5)(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D()(x)
        x = Dropout(0.4)(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D()(x)
        x = Dropout(0.5)(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D()(x)
        x = Dropout(0.4)(x)

        # Flatten and Dense layers
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)

        # Output layer
        outputs = Dense(self.num_labels, activation='softmax')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        print(f"Creating Model ...\n")
        model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model

    def _create_model_minixcpetion(self, l2_regularization=0.01):
        
        regularization = l2(l2_regularization)
        
        # entry layers
        img_input = Input(self.input_shape)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # mini xception blocks

        # block 1
        residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(16, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(16, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # block 2
        residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(32, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # block 3
        residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(64, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # block 4
        residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        
        x = SeparableConv2D(128, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])
        
        # fully connected
        x = Conv2D(self.num_labels, (3, 3), padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        output = Activation('softmax', name='predictions')(x)

        model = Model(img_input, output)

        print(f"Creating Model ...\n")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model
    
    def _create_model_cnn(self):
        # input layer
        inputs = Input(shape=self.input_shape)

        # First convolutional block
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        x = BatchNormalization()(x)

        # Second convolutional block
        x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Third convolutional block
        x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
        x = BatchNormalization()(x)

        # Fourth convolutional block
        x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Fifth convolutional block
        x = Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
        x = BatchNormalization()(x)

        # Sixth convolutional block
        x = Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Flatten and fully connected layers
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # Output layer
        outputs = Dense(self.num_labels, activation='softmax')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        print(f"Creating Model ...\n")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model
    
    def create_datasets(self, data_path):
        print(f"Loading and preprocess the dataset ...\n")
        train, validation = create_face_dataset(data_path, self.input_shape, self.batch_size)

        return train, validation


    def train_model(self, train, validation, epochs=50, verbose = 1):
        # add callbacks
        early_stop = EarlyStopping('val_loss', patience=10)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=5, verbose=1) # Reduce learning rate when a metric has stopped improving
        checkpoint_models_path = f'models/face/{self.model_name}'+'checkpoint.model.keras'
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_models_path, monitor='val_loss', verbose=verbose, save_best_only=True)
        callbacks = [model_checkpoint, early_stop, reduce_lr]
        print(f"add callbacks ...\n")

        print(f"Start training ... \n")
        history = self.model.fit(train, batch_size=self.batch_size, epochs=epochs, validation_data=validation, callbacks=callbacks)

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

# The model weights (that are considered the best) can be loaded as -
# model.load_weights(checkpoint_filepath)
 
