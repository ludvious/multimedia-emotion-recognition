import os, librosa, pyaudio, wave
import numpy as np
from moviepy.editor import *
import matplotlib as plt
from config import SAMPLING_RATE, MIN_AUDIO_LEN, MAX_AUDIO_LEN, N_MELS_BAND, HOP_LENGTH

class AudioProcessing:
    def __init__(self, audio_data_path, frame_length, duration) -> None:
        self.audio_data_path = audio_data_path
        self.sampling_rate = SAMPLING_RATE
        self.n_mels_band = N_MELS_BAND
        self.frame_length = frame_length
        self.hop_length = HOP_LENGTH
        self.duration = duration
        self.min_audio_len = MIN_AUDIO_LEN
        self.max_audio_len = MAX_AUDIO_LEN

    #TODO: vedere come funzionano e gestire i parametri relativi all audio come sample rate, hop length etc

    def load_audio(self, file_path):
        """
        Load and preprocess audio file.
        
        Args:
            file_path (str): Path to an audio file.
            
        Returns:
            np.ndarray: Audio time-series array.
        """
        audio, sr = librosa.load(file_path, sr=self.sampling_rate)
        # cut or pad the audio to a fixed length
        if len(audio) > self.max_audio_len:
            audio = audio[:self.max_audio_len]
        else:
            audio = np.pad(audio, (0, max(0, self.max_audio_len - len(audio))), 'constant')
        
        return audio, sr
    
    def extract_feature(self, audio_data):
        """
        Extract Mel spectrogram as feature from audio data.

        Args: audio (np.ndarray): Audio time-series array.

        Returns: np.ndarray: Mel spectrogram.
        """
        mel_features = librosa.feature.melspectrogram(y=audio_data, sr=self.sampling_rate, hop_length=self.hop_length)
        mel_spec_db = librosa.power_to_db(mel_features).flatten() #convert to decibel
        return mel_spec_db
    
    def audio_to_spectrogram(self, audio_path, label, sample_rate, hop_length, plot=True):
        """
        Converte un file audio in un spettrogramma e lo salva come immagine.
        
        :param audio_path: Percorso del file audio
        :param output_image_path: Percorso del file immagine in output. Se non specificato, usa lo stesso nome dell'audio.
        :return: Percorso del file immagine salvato
        """
        spec_folder = 'spectrogram'

        # Create subfolder for the label if it doesn't exist
        spec_label_folder = os.path.join(spec_folder, label) # => spectrogram/label/
        if not os.path.exists(spec_label_folder):
            os.makedirs(spec_label_folder)

        y, sr = self.load_audio(audio_path, sr=sample_rate)
        file_name = os.path.splitext(audio_path)[0] # mantain the same name file
        
        # Genera lo spettrogramma
        spec_db = self.extract_feature(data=y, sampling_rate=sr, hop_length=hop_length)
        
        # Salva lo spettrogramma come immagine
        if plot:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel')
            #plt.colorbar(format='%+2.0f dB')
            #plt.title(f'Mel Spectrogram ({label})')
            plt.axis('off')
            plt.tight_layout(pad=0)
        # Save image to label folder with .png extension
        plt.savefig(os.path.join(spec_label_folder, f'{file_name}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def generate_mel_spectrogram(self, spec_path='data/speech/spectrogram'):
        """metodo per generare le immagini spettogrammi dei file audio (vengono eseguito step 1 e 2 descritti qui):
        1 - carico i file audio e li pre elaboro
        2 - genero i spettogrammi e li salvo in una cartella divisi per label
        3 - passaggio da fare manualmente, controllare i spettogrammi buoni e filtrare quelli non rumorosi e non buoni
        4- una volta fatto 3 passaggio, si carica le immagini e le si preparano per essere date in input al modello (questo é fatto con un altro metodo o classe)
        """
        # check se esiste path di destinazione
        if not os.path.exists(spec_path):
            os.makedirs(spec_path)

        for label in os.listdir(self.audio_data_path):
            label_folder = os.path.join(self.audio_data_path, label)
            if os.path.isdir(label_folder):
                for audio_file in os.listdir(label_folder):
                    audio_path = os.path.join(label_folder, audio_file) #es: audio/happy/file1.wav
                    try:
                        # Load and preprocess audio, Extract Mel spectrogram features and Save the spectrogram as an image with the same name as the audio file
                        self.audio_to_spectrogram(audio_path, label)
                        print(f"Saved spectrogram for {label}: {audio_file}")
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")
    
    '''def get_feature(self, path, duration, offset):
        data, sr = librosa.load(path, duration=duration, offset=offset)
        features = [self.extract_feature(data, sr)]

        return np.array(features)'''
    
    def create_dataset(self): #TODO: REFACTOR WITH INPUT SPECTROGRAM AND NOT AUDIO FILES
        """
        Load all audio files from folders, extract features, and return the dataset.

        Returns:
            tuple: Features and corresponding labels.
        """
        X = [] # features
        Y = [] # labels

        for label in os.listdir(self.data_path): #each folder name must be the label name
            label_folder = os.path.join(self.data_path, label)
            if os.path.isdir(label_folder):
                for audio_file in os.listdir(label_folder):
                    file_path = os.path.join(label_folder, audio_file)
                    try:
                        # Load and preprocess audio
                        audio = self.load_audio(file_path)
                        # Extract Mel spectrogram features
                        mel_spectrogram = self.extract_feature(audio)
                        # Append the features and label
                        X.append(mel_spectrogram)
                        Y.append(label)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return np.array(X), np.array(Y)