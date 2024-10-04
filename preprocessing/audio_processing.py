import os, librosa
from librosa.util import fix_length
import numpy as np
from moviepy.editor import *
from pathlib import Path
import matplotlib
import tensorflow as tf
from pydub import AudioSegment
from pydub.utils import make_chunks
from config import SAMPLING_RATE, MIN_AUDIO_LEN, MAX_AUDIO_LEN, N_MELS_BAND, HOP_LENGTH


class AudioProcessing:
    def __init__(self) -> None:
        self.sampling_rate = SAMPLING_RATE
        self.n_mels_band = N_MELS_BAND
        self.hop_length = HOP_LENGTH
        self.min_audio_len = MIN_AUDIO_LEN
        self.max_audio_len = MAX_AUDIO_LEN
        self.max_hz_audio_len = MAX_AUDIO_LEN * SAMPLING_RATE  #usato per fare il padding, misura lunghezza audio in hz
        self.target_shape = (N_MELS_BAND, N_MELS_BAND) # for resize to shape for CNN 128x128
    
    def to_wav(self, file_path: str, label):

        file = Path(file_path)
        file_name = file.stem
        audio = AudioSegment.from_file(file)
        chunk_length_ms = 1000  # 1 seconds per chunk
        chunks = make_chunks(audio, chunk_length_ms)
        # Export all of the individual chunks as separate files
        for i, chunk in enumerate(chunks):
            chunk.export(f"data/speech/audio/{label}/{file_name}_{i}.wav", format="wav")

    def load_audio(self, audio_path):
        """
        Load and preprocess audio file: applying resampling and pad/trunc to normalize all audio
        
        Args:
            file_path (str): Path to an audio file.
            
        Returns:
            np.ndarray: Audio time-series array.
        """
        #load audio with original sampling rate
        audio, sr = librosa.load(audio_path, sr=None)
        print(f"Loaded audio min: {audio.min()}, max: {audio.max()}; Sample Rate: {sr}")
        #resample to target rate for normalize all audio
        if sr != self.sampling_rate:
            fix_audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
            print(f"Sample Rate after resample: {sr}")

        if len(audio) == 0:
            raise ValueError("Audio data is empty")
        
        # cut or pad the audio to a fixed length TODO DA TESTARE e vedere se confermare
        #(non serve perche do gia in input audio di lunghezza fissa)
        if len(audio) < MIN_AUDIO_LEN:
            fix_audio = fix_length(audio, size=MAX_AUDIO_LEN*self.sampling_rate)
        if len(audio) > MAX_AUDIO_LEN:
            fix_audio = fix_length(audio, size=MAX_AUDIO_LEN*self.sampling_rate)
        
        return fix_audio, sr
    
    def get_spectrogram(self, audio_data):
        """
        Extract Mel spectrogram as feature from audio data.
        Returns: np.ndarray: Mel spectrogram.
        """        
        mel_features = librosa.feature.melspectrogram(y=audio_data, sr=self.sampling_rate, hop_length=self.hop_length, n_mels=self.n_mels_band)
        mel_spec_db = librosa.power_to_db(mel_features, ref=np.max) #convert to decibel
        normalized_spectrogram = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        resized_spectrogram = tf.image.resize(normalized_spectrogram, self.target_shape, mode='reflect', anti_aliasing=True)
        if mel_features.max() == 0:
            raise ValueError("Mel spectrogram contains only zeros.")
        if mel_spec_db.shape[1] == 0:
            raise ValueError("Invalid spectrogram shape")

        return resized_spectrogram
    
    def audio_to_spectrogram_img(self, audio_path, label, save_img=True):
        """
        Converte un file audio in un spettrogramma e lo salva come img.
        Questa funzione é utile per creare tutti gli spectrogrammi e visualizzare quelli da scartare.

        :param audio_path: Percorso del file audio
        :param output_image_path: Percorso del file immagine in output. Se non specificato, usa lo stesso nome dell'audio.
        :return: Percorso del file immagine salvato
        """
        audio_file_path = Path(audio_path)
        output_folder = Path('data/speech/spectrogram/')
        spec_label_folder = output_folder / label  # The `/` operator works with pathlib to join paths
        # Create subfolder for the label if it doesn't exist
        spec_label_folder.mkdir(parents=True, exist_ok=True) # => spectrogram/label/

        y, sr = self.load_audio(audio_file_path)
        file_name = os.path.splitext(os.path.basename(audio_path))[0] # pick the same name file
        # Genera lo spettrogramma
        spec = self.get_spectrogram(audio_data=y)
        if save_img:
            # Salva lo spettrogramma come immagine
            output_path = spec_label_folder / f'{file_name}.png'
            matplotlib.image.imsave(output_path, spec)

    def gen_mel_spectrogram_dataset(self, audio_file_path: str):
        """metodo per generare le immagini spettogrammi dei file audio e salvarle(vengono eseguito step 1 e 2 descritti qui):
        1 - carico i file audio e li pre elaboro
        2 - genero i spettogrammi e li salvo in una cartella divisi per label
        3 - passaggio da fare manualmente, controllare i spettogrammi buoni e filtrare quelli non rumorosi e non buoni
        """
        audio_path = Path(audio_file_path)
        spec_path = Path('data/speech/spectrogram')
        spec_path.mkdir(parents=True, exist_ok=True) # => spectrogram/label/

        for label in audio_path.iterdir():
            if label.is_dir():
                for audio_file in label.iterdir():
                    audio_path = audio_file
                    try:
                        # Load and preprocess audio, Extract Mel spectrogram features and Save the spectrogram as an image with the same name as the audio file
                        self.audio_to_spectrogram_img(audio_path, label.name)
                        print(f"Saved spectrogram for {label.name}: {audio_file.name}")
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")
