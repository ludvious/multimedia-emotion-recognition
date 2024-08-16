import tensorflow as tf
from keras.api.utils import image_dataset_from_directory
from keras.api.models import Sequential
from keras.api.layers import RandomFlip, RandomRotation, RandomZoom, Resizing, Rescaling, RandomTranslation, RandomBrightness, RandomContrast
import os

def prepocess_face_dataset(input_shape, batch_size):

    train_dir = os.path.join('data\face\fer-2013\train')
    val_dir = os.path.join('data\face\fer-2013\test')

    train_dataset = image_dataset_from_directory(
            directory=train_dir,
            label_mode='categorical',
            subset='training',
            seed=123,
            validation_split=0.15,
            image_size=input_shape[:2],
            color_mode='grayscale',
            batch_size=batch_size
    )
    
    val_dataset = image_dataset_from_directory(
        directory=val_dir,
        label_mode='categorical',
        subset='validation',
        seed=123,
        validation_split=0.15,
        image_size=input_shape[:2],
        color_mode='grayscale',
        batch_size=batch_size
    )

    train_dataset = preprocess_data_images(dataset=train_dataset, type_dataset='train', batch_size=batch_size, augment=True)
    val_dataset = preprocess_data_images(dataset=val_dataset, type_dataset='val', batch_size=batch_size, augment=False)

    return train_dataset, val_dataset

def preprocess_data_images(dataset, input_shape, batch_size: int, augment=False):
    """metodo per applicare pre-elaborazione (normalization, rescaling, augmentation, shuffle, prefetch) direttamente sui dati prima di essere data in input al modello
    Args:
        dataset (_type_): _description_
        type_dataset (str): _description_
        batch_size (int): _description_
        augment (bool, optional): _description_. Defaults to False.

    Returns:
        dataset (_type_): _description_
    """
    # Add Rescaling layer to normalize pixel values
    normalization_layer = Rescaling(1./255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))


    if augment==True:
        # applicazione aumento dei dati
        data_augmentation = Sequential([
        RandomFlip("horizontal", input_shape=input_shape[:2]),
        RandomRotation(0.1),
        RandomZoom(0.05),
        RandomContrast(0.1)
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

    # cache mantiene le immagini in memoria dopo che sono state caricate dal disco durante la prima epoca. Ciò garantirà che il set di dati non diventi un collo di bottiglia durante l'addestramento del modello
    # prefetch sovrappone alla preelaborazione dei dati e all'esecuzione del modello durante l'addestramento
    dataset = dataset.shuffle(1000)
    
    return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)