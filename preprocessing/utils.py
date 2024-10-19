import pyaudio, wave, librosa, os
import numpy as np
import matplotlib as plt
from config import SAMPLING_RATE

def add_folders(start_path, labels):
    if not os.path.exists(start_path):
        os.makedirs(start_path)
    
    for label in labels:
        label_folder = os.path.join(start_path, label)
        os.makedirs(label_folder)

def count_files(file_path):
    label_folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]
    for label in label_folders:
        label_path = os.path.join(file_path, label)
        # Get all audio files for this label
        audio_files = [f for f in os.listdir(label_path) if f.endswith('.wav')]
        print(f'{label} audio count: {len(audio_files)}')

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