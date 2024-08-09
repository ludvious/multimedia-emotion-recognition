import tensorflow as tf
import pandas as pd
import cv2, os, librosa
import numpy as np
from keras.api.utils import image_dataset_from_directory
from keras.api.models import Sequential
from keras.api.layers import RandomFlip, RandomRotation, RandomZoom, Rescaling
import pyaudio
import wave

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
    val_dataset = preprocess_data_images(dataset=val_dataset, type_dataset='val', batch_size=batch_size, augment=True)

    return train_dataset, val_dataset

def preprocess_data_images(dataset, type_dataset: str, input_shape, batch_size: int, augment=False):
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

    if type_dataset == 'train':
        dataset = dataset.shuffle(1000)

    if augment:
        # applicazione aumento dei dati
        data_augmentation = Sequential([
        RandomFlip("horizontal", input_shape=input_shape[:2]),
        RandomRotation(0.1),
        RandomZoom(0.1)
        ])

        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

    # cache mantiene le immagini in memoria dopo che sono state caricate dal disco durante la prima epoca. Ciò garantirà che il set di dati non diventi un collo di bottiglia durante l'addestramento del modello
    # prefetch sovrappone alla preelaborazione dei dati e all'esecuzione del modello durante l'addestramento
    
    return dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def capture_frames_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def extract_spectrogram(file_path):
        y, sr = librosa.load(file_path)
        spect = librosa.feature.melspectrogram(y=y, sr=sr)
        spect_db = librosa.power_to_db(spect, ref=np.max)
        return spect_db

def new_record_audio(chunk_audio=1024, format_audio=pyaudio.paInt16, channels_audio=2, rate_audio=44100):
    """method for add the possibility for record a personal audio and for test it in future

    Args:
        chunk_audio (int, optional): _description_. Defaults to 1024.
        format_audio (_type_, optional): _description_. Defaults to pyaudio.paInt16.
        channels_audio (int, optional): _description_. Defaults to 2.
        rate_audio (int, optional): _description_. Defaults to 44100.
    """
    
    record_seconds = 5
    wave_output_filname = "audio_record_output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=format_audio,
                    channels=channels_audio,
                    rate=rate_audio,
                    input=True,
                    frames_per_buffer=chunk_audio) #buffer

    print("* Start recording ... ")

    frames = []

    for i in range(0, int(rate_audio / chunk_audio * record_seconds)):
        data = stream.read(chunk_audio)
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* Stop recording ... ")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(wave_output_filname, 'wb')
    wf.setnchannels(channels_audio)
    wf.setsampwidth(p.get_sample_size(format_audio))
    wf.setframerate(rate_audio)
    wf.writeframes(b''.join(frames))
    wf.close()