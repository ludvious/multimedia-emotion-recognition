import tensorflow as tf
import pandas as pd
import cv2, os, librosa, pyaudio, wave
import numpy as np
import moviepy.editor as mp
from pytube import YouTube
from moviepy.editor import *
import matplotlib as plt

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

def mel_spectrogram(data, sampling_rate, hop_length):
        mel_features = librosa.feature.melspectrogram(y=data, sr=sampling_rate, hop_length=hop_length)
        return librosa.power_to_db(mel_features).flatten()

def audio_to_spectrogram(sample_rate, hop_length, audio_path, output_image_path=None):
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
        spectrogram_db = mel_spectrogram(data=y, sampling_rate=sr, hop_length=hop_length)
        
        # Salva lo spettrogramma come immagine
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()
        
        return output_image_path