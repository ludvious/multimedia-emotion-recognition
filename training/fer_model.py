from keras.api.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
     SeparableConv2D, BatchNormalization, Activation, GlobalAveragePooling2D)
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.api.regularizers import l2
import matplotlib.pyplot as plt
from keras import layers
from keras.api.models import Model
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.api.applications import ResNet50V2
from config import NUM_LABELS
from preprocessing.utils import unzip
from preprocessing.face_processing import preprocess_face_dataset


unzip("data/face/fer2013.zip", "data/face/")

"""##**Preprocessing e Augmentation**

"""


"""# **FaceModel**"""

class FaceModel:
    def __init__(self, model_name: str, num_labels: int, input_shape, batch_size: int, epochs: int) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self._create_model()

    def _create_model(self):
        if self.model_name == 'mini_xception':
            model = self._create_model_minixception()
        if self.model_name == 'resnet':
            model = self._create_model_resnet50()

        return model

    def create_datasets(self):
        print(f"preprocessing and creating datasets ...\n")
        train_gen, val_gen, test_gen = preprocess_face_dataset(self.input_shape, self.batch_size)

        return train_gen, val_gen, test_gen

    def _create_model_minixception(self, l2_regularization=0.01):

        regularization = l2(l2_regularization)

        # base
        input_img = Input(self.input_shape)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # module 1
        residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(16, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization,use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(16, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # module 2
        residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(32, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # module 3
        residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(64, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # module 4
        residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(128, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        x = Conv2D(self.num_labels, (3, 3), padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        output = Activation('softmax', name='predictions')(x)

        model = Model(input_img, output)

        print(f"Creating Model ...\n")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model
    
    def _create_model_resnet50(self):
        # Load the VGG16 model (without the top layer)
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.summary()

        # Freezing all layers except last 40
        base_model.trainable = True
        for layer in base_model.layers[:-40]:
            layer.trainable = False

        # Add lasts custom layers for emotion classification
        x = base_model.output
        x = Dropout(0.25)(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        # Compile the model
        print(f"Creating Model ...\n")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary : \n")
        model.summary()

        return model


    def train_face_model(self, train_gen, val_gen):

        # add callbacks
        early_stop = EarlyStopping('val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.2, patience=10, verbose=1) # Reduce learning rate when a metric has stopped improving
        checkpoint_models_path = f'models/face/{self.model_name}_'+'checkpoint.model.keras'
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_models_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks = [model_checkpoint, early_stop, reduce_lr]
        print(f"add callbacks ...\n")

        print(f"Start training ... \n")
        history = self.model.fit(train_gen, batch_size=self.batch_size,epochs=self.epochs, validation_data=val_gen, callbacks=callbacks)

        return history

"""##**MINIXCEPTION**"""

minix_face_model = FaceModel('mini_xception', num_classes=NUM_LABELS, input_shape=(48,48,1), batch_size=32, epochs=100)

train_minix, val_minix, test_minix = minix_face_model.create_datasets()

minix_history = minix_face_model.train_face_model(train_minix, val_minix)

plt.figure(figsize=(10, 5))
plt.plot(minix_history.history['accuracy'], label='Train Accuracy')
plt.plot(minix_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(minix_history.history['loss'], label='Train Loss')
plt.plot(minix_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Get the true labels and predicted labels for the validation set
validation_labels = test_minix.classes
#model = load_model('/content/cnn_checkpoint.model.keras')
validation_pred_probs = minix_face_model.model.predict(test_minix)
validation_pred_labels = np.argmax(validation_pred_probs, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)
class_names = list(train_minix.class_indices.keys())
sns.set()
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

"""## **ResNet50**"""

resnet_model = FaceModel('resnet', num_classes=NUM_LABELS, input_shape=(224,224,3), batch_size=64, epochs=15)

train_gen_resnet, val_gen_resnet, test_gen_resnet = minix_face_model.create_datasets()

resnet_history = resnet_model.train_face_model(train_minix, val_minix)

plt.figure(figsize=(10, 5))
plt.plot(resnet_history.history['accuracy'], label='Train Accuracy')
plt.plot(resnet_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(resnet_history.history['loss'], label='Train Loss')
plt.plot(resnet_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Get the true labels and predicted labels for the validation set
validation_labels = val_gen_resnet.classes
validation_pred_probs = resnet_model.model.predict(val_gen_resnet)
validation_pred_labels = np.argmax(validation_pred_probs, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)
class_names = list(train_gen_resnet.class_indices.keys())
sns.set()
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Get the true labels and predicted labels for the test set
test_labels = test_gen_resnet.classes
test_pred_probs = resnet_model.model.predict(test_gen_resnet)
test_pred_labels = np.argmax(test_pred_probs, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(test_labels, test_pred_labels)
class_names = list(train_gen_resnet.class_indices.keys())
sns.set()
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='flare',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the model on the test data
results = resnet_model.model.evaluate(test_gen_resnet, batch_size=64)

print("Test Loss:", results[0])
print("Test Accuracy:", results[1])