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