import os, librosa
from librosa.util import fix_length
import numpy as np
from moviepy.editor import *
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io import wavfile
from config import SAMPLING_RATE, MIN_AUDIO_LEN, MAX_AUDIO_LEN, N_MELS_BAND, HOP_LENGTH, OVERLAP_RATIO


class AudioProcessing:
    def __init__(self) -> None:
        self.sampling_rate = SAMPLING_RATE
        self.n_mels_band = N_MELS_BAND
        self.hop_length = HOP_LENGTH
        self.min_audio_len = MIN_AUDIO_LEN
        self.max_audio_len = MAX_AUDIO_LEN
        self.max_hz_audio_len = MAX_AUDIO_LEN * SAMPLING_RATE  #usato per fare il padding, misura lunghezza audio in hz
        self.target_shape = (N_MELS_BAND, N_MELS_BAND) # for resize to shape for CNN 128x128
        self.overlap_ratio = OVERLAP_RATIO
      
    def to_wav(self, file_path: str, label: str, chunk_length_ms=1000):
        """
        Split audio in chunks and augment with overlapping chunks and save each chunk as a .wav file.
        
        :param file_path: Path to the input audio file.
        :param label: Label for the output directory.
        :pram overlapping: bool True or False according to apply overlapping
        :param chunk_length_ms: Length of each chunk in milliseconds (default 1000 ms).
        :param overlap_ms: Amount of overlap between chunks in milliseconds (default 500 ms).
        """
        # Load the audio file
        file = Path(file_path)
        file_name = file.stem
        audio = AudioSegment.from_file(file)
        chunks = make_chunks(audio, chunk_length_ms)
        for i, chunk in enumerate(chunks):
            chunk.export(f"data/speech/audio/{label}/{file_name}_{i}.wav", format="wav")
        
            '''if overlapping==True:
                # Calculate the hop size (how much we shift each chunk)
                hop_size_ms = chunk_length_ms - overlap_ms  # e.g., 1000 ms chunk, 500 ms overlap -> hop_size = 500 ms
                
                # Start slicing the audio into overlapping chunks
                start = 0
                chunk_id = 0  # Track chunk number for naming
                
                while start + chunk_length_ms <= len(audio):  # Ensure we don't exceed audio length
                    # Get the chunk (from 'start' to 'start + chunk_length_ms')
                    overlap_chunk = audio[start:start + chunk_length_ms]
                    
                    # Export chunk as a .wav file
                    overlap_chunk.export(f"data/speech/audio/{label}/{file_name}_aug_overlap_{chunk_id}.wav", format="wav")
                    
                    # Move start to the next position (hop size)
                    start += hop_size_ms
                    chunk_id += 1'''

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
        if sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
            print(f"Sample Rate after resample: {self.sampling_rate}")

        audio_len = librosa.get_duration(y=audio)
        if audio_len == 0:
            raise ValueError("Audio data is empty")
        
        # cut or pad the audio to a fixed length (non serve perche do gia in input audio di lunghezza fissa)
        if audio_len < MIN_AUDIO_LEN:
            audio = fix_length(audio, size=MAX_AUDIO_LEN*self.sampling_rate)
        if audio_len > MAX_AUDIO_LEN:
            audio = fix_length(audio, size=MAX_AUDIO_LEN*self.sampling_rate)
        #normalize audio 
        audio = audio / np.max(np.abs(audio))

        return audio, sr
    
    def get_spectrogram(self, audio_data):
        """
        Extract Mel spectrogram as feature from audio data.
        Returns: np.ndarray: Mel spectrogram.
        """        
        mel_features = librosa.feature.melspectrogram(y=audio_data, sr=self.sampling_rate, hop_length=self.hop_length, n_mels=self.n_mels_band)
        mel_spec_db = librosa.power_to_db(mel_features, ref=np.max) #convert to decibel
        normalized_spectrogram = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        # Save the Mel spectrogram as an image
        if mel_features.max() == 0:
            raise ValueError("Mel spectrogram contains only zeros.")
        if mel_spec_db.shape[1] == 0:
            raise ValueError("Invalid spectrogram shape")

        return normalized_spectrogram
    
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
        s_dB = self.get_spectrogram(audio_data=y)
        if save_img:
            # Salva lo spettrogramma come immagine
            output_path = spec_label_folder / f'{file_name}.png'
            # Save the Mel spectrogram as an image
            plt.figure(figsize=(4, 4))
            librosa.display.specshow(s_dB, sr=sr, hop_length=self.hop_length, x_axis='time', y_axis='mel', cmap='viridis')
            plt.axis('off')  # No axes for the image
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            #matplotlib.image.imsave(output_path, spec)

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
    
    def augment_dataset(self, file_path, num_augmentations: int):
        """
        Augment the dataset by creating overlapped versions of audio files.
        Maintains the label folder structure and augments within each label category.
        
        Usage: after creation wav 1 second audio from a long clip audio for increase the dataset sample
        
        Args:
            input_folder: Root folder containing subfolders for each label
            output_folder: Root folder where augmented files will be saved (maintaining label structure)
            num_augmentations: Number of augmentations to create per label
        """
        label_folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]
        
        for label in label_folders:
            label_path = os.path.join(file_path, label)
            
            # Get all audio files for this label
            audio_files = [f for f in os.listdir(label_path) if f.endswith('.wav')]
            
            # Skip if there are less than 2 files in the label folder
            if len(audio_files) < 2:
                print(f"Skipping label {label}: Not enough files for augmentation")
                continue
                
            for i in range(num_augmentations):
                try:
                    # Randomly select two audio files from the same label
                    file1, file2 = np.random.choice(audio_files, size=2, replace=False)
                    
                    # Load audio files
                    audio1, sr1 = self.load_audio(os.path.join(label_path, file1))
                    audio2, sr2 = self.load_audio(os.path.join(label_path, file2))
                    
                    # Create overlapped audio
                    mixed_audio = self.gen_overlapped_audio(audio1, audio2, self.overlap_ratio)
                    
                    # Generate output filename (including label information)
                    output_filename = f"{file1.split('.')[0]}_{file2.split('.')[0]}_augmented_{i}.wav"
                    
                    # Save the mixed audio
                    wavfile.write(
                        os.path.join(label_path, output_filename),
                        self.sampling_rate,
                        (mixed_audio * 32767).astype(np.int16)
                    )
                    
                    print(f"Created augmentation {i+1}/{num_augmentations} for label {label}")
                    
                except Exception as e:
                    print(f"Error processing augmentation {i} for label {label}: {str(e)}")
                    continue