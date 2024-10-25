import os, librosa
from librosa.util import fix_length
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks
from config import SAMPLING_RATE, TARGET_RATE, CHANNELS, MIN_AUDIO_LEN, MAX_AUDIO_LEN, N_MELS_BAND, HOP_LENGTH, OVERLAP_RATIO, FORMAT

class AudioProcessing:
    def __init__(self) -> None:
        self.sampling_rate = SAMPLING_RATE
        self.target_rate = TARGET_RATE
        self.channels = CHANNELS
        self.n_mels_band = N_MELS_BAND
        self.hop_length = HOP_LENGTH
        self.min_audio_len = MIN_AUDIO_LEN
        self.max_audio_len = MAX_AUDIO_LEN
        self.max_hz_audio_len = MAX_AUDIO_LEN * TARGET_RATE  #usato per fare il padding, misura lunghezza audio in hz
        self.target_shape = (N_MELS_BAND, N_MELS_BAND) # for resize to shape for CNN 128x128
        self.overlap_ratio = OVERLAP_RATIO
        self.img_width = 224
        self.img_height = 224
      
    def gen_chunks(self, file_path: str, label: str, chunk_length_ms=1000):
        """
        Split audio downloaded in chunks, save them as a .wav file.
        
        :param file_path: Path to the input audio file.
        :param label: Label for the output directory.
        :param chunk_length_ms: Length of each chunk in milliseconds (default 1000 ms).
        """
        file = Path(file_path)
        file_name = file.stem
        audio = AudioSegment.from_file(file)
        chunks = make_chunks(audio, chunk_length_ms)
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
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        print(f"Loaded audio min: {audio.min()}, max: {audio.max()}; Sample Rate: {sr}")
        #resample to target rate for normalize all audio
        if sr != self.target_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_rate)
            print(f"Sample Rate after resample: {self.target_rate}")

        audio_len = librosa.get_duration(y=audio)
        if audio_len == 0:
            raise ValueError("Audio data is empty")
        
        # cut or pad the audio to a fixed length (non serve perche do gia in input audio di lunghezza fissa)
        if audio_len < MIN_AUDIO_LEN:
            audio = fix_length(audio, size=MAX_AUDIO_LEN*self.target_rate)
        if audio_len > MAX_AUDIO_LEN:
            audio = fix_length(audio, size=MAX_AUDIO_LEN*self.target_rate)
        #normalize audio 
        audio = audio / np.max(np.abs(audio))

        return audio, sr
    
    def get_mel_spectrogram(self, audio_data):
        """
        Extract Mel spectrogram as feature from audio data.
        Returns: np.ndarray: Mel spectrogram.
        """        
        mel_features = librosa.feature.melspectrogram(y=audio_data, sr=self.target_rate, hop_length=self.hop_length, n_mels=self.n_mels_band)
        mel_spec_db = librosa.power_to_db(mel_features, ref=np.max) #convert to decibel
        normalized_spectrogram = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        # Save the Mel spectrogram as an image
        if mel_features.max() == 0:
            raise ValueError("Mel spectrogram contains only zeros.")
        if mel_spec_db.shape[1] == 0:
            raise ValueError("Invalid spectrogram shape")

        return normalized_spectrogram
    
    def audio_to_spectrogram_img(self, audio_path, inference: bool, save_img=True, inference_ouput_folder=None):
        """
        Converte un file audio in un spettrogramma e lo salva come img.
        Questa funzione é utile per creare tutti gli spectrogrammi e visualizzare quelli da scartare.

        :param audio_path: Percorso del file audio
        :param output_image_path: Percorso del file immagine in output. Se non specificato, usa lo stesso nome dell'audio.
        :return: Percorso del file immagine salvato
        """
        matplotlib.use('Agg')
        
        audio_file_path = Path(audio_path)
        
        if inference==False:
            output_folder = Path('data/speech/spectrogram/')
            label = audio_file_path.parent.name
            spec_label_folder = output_folder / label  # The `/` operator works with pathlib to join paths
            spec_label_folder.mkdir(parents=True, exist_ok=True) # => spectrogram/label/
        
        label = audio_file_path.parent.name
        #spec_label_folder = output_folder / label  # The `/` operator works with pathlib to join paths
        # Create subfolder for the label if it doesn't exist
        #spec_label_folder.mkdir(parents=True, exist_ok=True) # => spectrogram/label/
        y, sr = self.load_audio(audio_file_path)
        file_name = os.path.splitext(os.path.basename(audio_path))[0] # pick the same name file
        # Genera lo spettrogramma
        s_dB = self.get_mel_spectrogram(audio_data=y)
        if save_img:
            # Salva lo spettrogramma come immagine nel path specifico; caso di run dell app oppure caso in cui genero il dataset per il training
            if inference==True:
                inference_ouput_folder = Path(inference_ouput_folder)
                inference_ouput_folder.mkdir(parents=True, exist_ok=True)
                output_path = inference_ouput_folder / f'{file_name}.png'
            else:
                output_path = spec_label_folder / f'{file_name}.png'
            # Save the Mel spectrogram as an image
            plt.figure(figsize=(5,5))
            plt.figure(figsize=(self.img_width / 100, self.img_height / 100))
            librosa.display.specshow(s_dB, sr=sr, hop_length=self.hop_length, x_axis='time', y_axis='mel', cmap='viridis')
            plt.axis('off')  # No axes for the image
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
        
        return output_path
    
    def gen_mel_spectrogram_images(self, audio_file_path: str):
        """metodo per generare le immagini spettogrammi dei file audio e salvarle(vengono eseguito step 1 e 2 descritti qui):
        1 - carico i file audio e li pre elaboro
        2 - genero i spettogrammi e li salvo in una cartella divisi per label
        3 - passaggio da fare manualmente, controllare i spettogrammi buoni e filtrare quelli non rumorosi e non buoni
        4 - la cartella con i spectogrammi usata come input ad ImagedataGenerator per creare facilmente i generator per train e validation con le loro labels
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
    
    def gen_overlapped_audio(self, audio1, audio2, overlap_ratio=0.5):
        """
        Create an overlapped audio by mixing two audio samples
        overlap_ratio: Amount of overlap between the two audio samples (0 to 1)
        """
        # Make sure both audios have the same length
        min_length = min(len(audio1), len(audio2))
        audio1 = audio1[:min_length]
        audio2 = audio2[:min_length]
        
        # Mix the audios with equal weights
        mixed_audio = np.array(audio1) * overlap_ratio + np.array(audio2) * overlap_ratio
        
        # Normalize to prevent clipping
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
        
        return mixed_audio