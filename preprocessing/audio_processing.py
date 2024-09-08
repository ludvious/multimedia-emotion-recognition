import os, librosa, pyaudio, wave
from librosa.util import fix_length
import numpy as np
from moviepy.editor import *
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from config import SAMPLING_RATE, MIN_AUDIO_LEN, MAX_AUDIO_LEN, N_MELS_BAND, HOP_LENGTH
class AudioProcessing:
    def __init__(self) -> None:
        self.sampling_rate = SAMPLING_RATE
        self.n_mels_band = N_MELS_BAND
        self.hop_length = HOP_LENGTH
        self.min_audio_len = MIN_AUDIO_LEN
        self.max_audio_len = MAX_AUDIO_LEN
        self.max_hz_audio_len = MAX_AUDIO_LEN * SAMPLING_RATE  #usato per fare il padding, misura lunghezza audio in hz

    #TODO: vedere come funzionano e gestire i parametri relativi all audio come sample rate, hop length etc

    def load_and_preprocess_audio(self, audio_path):
        """
        Load and preprocess audio file: applying resampling and pad/trunc to normalize all audio
        
        Args:
            file_path (str): Path to an audio file.
            
        Returns:
            np.ndarray: Audio time-series array.
        """
        audio, sr = librosa.load(audio_path, sr=None) #load audio with original sampling rate
        print(f"Loaded audio min: {audio.min()}, max: {audio.max()}; Sample Rate: {sr}")
        #resample to target rate for normalize all audio
        if sr != self.sampling_rate:
            fix_audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
            print(f"Sample Rate after resample: {sr}")
        #TODO: aggiungere taglio parte silenzio?
        #fix_audio, index = librosa.effects.trim(audio, top_db = 30)

        if len(audio) == 0:
            raise ValueError("Audio data is empty")
        # cut or pad the audio to a fixed length TODO DA TESTARE e vedere se confermare
        if len(audio) < MIN_AUDIO_LEN:
            fix_audio = fix_length(audio, size=MAX_AUDIO_LEN*self.sampling_rate)
        if len(audio) > MAX_AUDIO_LEN:
            fix_audio = fix_length(audio, size=MAX_AUDIO_LEN*self.sampling_rate)
        
        return fix_audio, sr
    
    def extract_feature(self, audio_data):
        """
        Extract Mel spectrogram as feature from audio data.

        Args: audio (np.ndarray): Audio time-series array.

        Returns: np.ndarray: Mel spectrogram.
        """        
        mel_features = librosa.feature.melspectrogram(y=audio_data, sr=self.sampling_rate, hop_length=self.hop_length, n_mels=self.n_mels_band)
        mel_spec_db = librosa.power_to_db(mel_features, ref=np.max) #convert to decibel

        if mel_features.max() == 0:
            raise ValueError("Mel spectrogram contains only zeros.")
        if mel_spec_db.shape[1] == 0:
            raise ValueError("Invalid spectrogram shape")
        
        return mel_spec_db
    
    def audio_to_spectrogram_img(self, audio_path, label):
        """
        Converte un file audio in un spettrogramma e lo salva come img.
        
        :param audio_path: Percorso del file audio
        :param output_image_path: Percorso del file immagine in output. Se non specificato, usa lo stesso nome dell'audio.
        :return: Percorso del file immagine salvato
        """
        audio_file_path = Path(audio_path)
        output_folder = Path('data/speech/spectrogram/')
        spec_label_folder = output_folder / label  # The `/` operator works with pathlib to join paths
        # Create subfolder for the label if it doesn't exist
        spec_label_folder.mkdir(parents=True, exist_ok=True) # => spectrogram/label/

        y, sr = self.load_and_preprocess_audio(audio_file_path)
        file_name = os.path.splitext(os.path.basename(audio_path))[0] # pick the same name file
        
        # Genera lo spettrogramma
        spec_db = self.extract_feature(audio_data=y)
        
        # Salva lo spettrogramma come immagine
        matplotlib.use('TkAgg',force=True)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel')
        #plt.colorbar(format='%+2.0f dB')
        #plt.title(f'Mel Spectrogram ({label})')
        plt.axis('off')
        #plt.tight_layout(pad=0)
        # Save image to label folder with .png extension
        output_path = spec_label_folder / f'{file_name}.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def create_spectrogram_dataset(self, audio_path: str): #TODO: CAPIRE SE USARE QUESTO CHE E OK, OPPURE FARNE UN ALTRO DOVE SI PRENDE IN INPUT LE IMMAGINI DEI SPETTOGRAMMI
        """
        Load all audio files from folders, extract features, and return the dataset with features and labels.

        Returns:
            tuple: Features and corresponding labels.
        """
        X = [] # features
        Y = [] # labels
        target_shape = (self.n_mels_band, self.n_mels_band) # for resize to shape for CNN
        #TODO FIXARE CON PATHLIB 
        for label in os.listdir(audio_path): #each folder name must be the label name
            label_folder = os.path.join(audio_path, label)
            if os.path.isdir(label_folder):
                for audio_file in os.listdir(label_folder):
                    file_path = os.path.join(label_folder, audio_file)
                    try:
                        # Load and preprocess audio
                        audio = self.load_and_preprocess_audio(file_path)
                        # Extract Mel spectrogram features
                        mel_spectrogram = self.extract_feature(audio)
                        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                        # Append the features and label
                        X.append(mel_spectrogram)
                        Y.append(label)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return np.array(X), np.array(Y)