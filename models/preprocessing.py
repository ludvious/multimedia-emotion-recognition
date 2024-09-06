from keras.api.utils import image_dataset_from_directory
from keras.api.models import Sequential
from keras.api.layers import RandomFlip, RandomRotation, RandomZoom, Resizing, Rescaling, RandomTranslation, RandomBrightness, RandomContrast
import tensorflow as tf
import pandas as pd
import cv2, os, librosa, pyaudio, wave
import numpy as np
import moviepy.editor as mp
from pytube import YouTube
from moviepy.editor import *
import matplotlib as plt

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

    train_dataset = normalize_augmentation_images(dataset=train_dataset, type_dataset='train', batch_size=batch_size, augment=True)
    val_dataset = normalize_augmentation_images(dataset=val_dataset, type_dataset='val', batch_size=batch_size, augment=False)

    return train_dataset, val_dataset

def normalize_augmentation_images(dataset, input_shape, batch_size: int, augment=False):
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


def get_audio_from_mp4(filepath):

    files = os.listdir(filepath)

    for file in files:
        if file.endswith(".m4v"):
            fileName = os.path.splitext(file)
            video = mp.VideoFileClip(filepath+file)
            audio = video.audio
            audio.write_audiofile(filepath+fileName[0]+".wav")

def get_audio_from_yt(youtube_url):
    # download a file with only audio, to save space
    # if the final goal is to convert to mp3
    y = YouTube(youtube_url)
    t = y.streams.filter(only_audio=True).all()
    t[0].download(output_path="../VideoFiles")

def audio_to_spectrogram(sample_rate, audio_path, output_image_path=None):
        """
        Converte un file audio in uno spettrogramma e lo salva come immagine.
        
        :param audio_path: Percorso del file audio
        :param output_image_path: Percorso del file immagine in output. Se non specificato, usa lo stesso nome dell'audio.
        :return: Percorso del file immagine salvato
        """
        if output_image_path is None:
            output_image_path = os.path.splitext(audio_path)[0] + "_spectrogram.png"
        
        # Carica l'audio
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Genera lo spettrogramma
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Salva lo spettrogramma come immagine
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()
        
        return output_image_path