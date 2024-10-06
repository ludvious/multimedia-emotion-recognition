import pyaudio, wave, librosa, os
import numpy as np
import matplotlib as plt
from pytube import YouTube
import moviepy.editor as mp
from audio_processing import AudioProcessing
from config import SAMPLING_RATE

def add_folders(start_path, labels):
    if not os.path.exists(start_path):
        os.makedirs(start_path)
    
    for label in labels:
        label_folder = os.path.join(start_path, label)
        os.makedirs(label_folder)

def plot_spec(audio_data, sr, name):
    """stampa a video lo spettogramma e lo salva come immagine
    """

    spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure(figsize=(12,4))
    librosa.display.specshow(spec_db, sr=sr,
        x_axis='time', y_axis='mel',
        hop_length=sr * 0.01)
    plt.colorbar(format='%+02.0f dB')
    plt.savefig('figs/{}.png'.format(name))
    plt.clf()