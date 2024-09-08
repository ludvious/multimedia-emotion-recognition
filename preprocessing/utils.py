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

def gen_mel_spectrogram_dataset(audio_path: str, spec_path: str):
        """metodo per generare le immagini spettogrammi dei file audio e salvarle(vengono eseguito step 1 e 2 descritti qui):
        1 - carico i file audio e li pre elaboro
        2 - genero i spettogrammi e li salvo in una cartella divisi per label
        3 - passaggio da fare manualmente, controllare i spettogrammi buoni e filtrare quelli non rumorosi e non buoni
        4- una volta fatto 3 passaggio, si carica le immagini e le si preparano per essere date in input al modello (questo é fatto con un altro metodo o classe)
        """
        #spec_path = 'data/speech/spectrogram'
        preproc = AudioProcessing(audio_path=audio_path)
        #TODO: SISTEMARE PATH LIB QUI AL POSTO DI OS LIB
        # check se esiste path di destinazione
        spec_path.mkdir(parents=True, exist_ok=True) # => spectrogram/label/

        for label in os.listdir(audio_path):
            label_folder = os.path.join(audio_path, label)
            if os.path.isdir(label_folder):
                for audio_file in os.listdir(label_folder):
                    audio_path = os.path.join(label_folder, audio_file) #es: audio/happy/file1.wav
                    try:
                        # Load and preprocess audio, Extract Mel spectrogram features and Save the spectrogram as an image with the same name as the audio file
                        preproc.audio_to_spectrogram_img(audio_path, label)
                        print(f"Saved spectrogram for {label}: {audio_file}")
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")

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