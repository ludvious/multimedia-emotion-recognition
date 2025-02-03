from preprocessing.face_processing import preprocess_face_dataset, get_landmarks_from_image, preprocess_landmarks
from preprocessing.audio_processing import AudioProcessing
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pathlib import Path
from utils import count_files
import os
from scipy.io import wavfile

def create_img_dataset(data_path: str, input_shape, batch_size):

    train_gen, val_gen, test_gen = preprocess_face_dataset(data_path, input_shape, batch_size)

    return train_gen, val_gen, test_gen

    
def create_audio_spectrogram_dataset(audio_file_path: str):
        """
        Carica tutti i file audio wav dalle cartelle, estrae le features (spettrogrammi)
        e restituisce il set di dati con caratteristiche ed etichette.

        Returns:
            np.array: Features and corresponding labels.
        """
        audio_proc = AudioProcessing()
        X = [] # features
        Y = [] # labels

        audio_path = Path(audio_file_path)
        for label in audio_path.iterdir(): #each folder name must be the label name
            if label.is_dir():
                for audio_file in label.iterdir():
                    file_path = audio_file
                    try:
                        # Load and preprocess audio
                        audio = audio_proc.load_audio(file_path)
                        # Extract Mel spectrogram features
                        mel_spectrogram = audio_proc.get_spectrogram(audio)
                        feature = np.expand_dims(mel_spectrogram, axis=-1)
                        # Append the features and label
                        X.append(feature)
                        Y.append(label)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return np.array(X), np.array(Y)

def augment_audio_dataset(file_path, num_augmentations: int):
        """
        Aumenta il set di dati audio creando versioni sovrapposte di file audio.
        Mantiene la struttura della cartella delle etichette e aumenta all'interno di ogni categoria di etichette.
        
        Uso: dopo la creazione di audio wav di 1 secondo da una clip audio lunga per aumentare il campione del set di dati
        
        Args:
            input_folder: Root folder containing subfolders for each label
            output_folder: Root folder where augmented files will be saved (maintaining label structure)
            num_augmentations: Number of augmentations to create per label
        """
        audio_proc = AudioProcessing()
        count_files(file_path)
        label_folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]

        for label in label_folders:
            label_path = os.path.join(file_path, label)
            
            # Get all audio files for this label
            audio_files = [f for f in os.listdir(label_path) if f.endswith('.wav')]
            print(len(audio_files))
            # Skip if there are less than 2 files in the label folder
            if len(audio_files) < 2:
                print(f"Skipping label {label}: Not enough files for augmentation")
                continue
                
            for i in range(num_augmentations):
                try:
                    # Randomly select two audio files from the same label
                    file1, file2 = np.random.choice(audio_files, size=2, replace=False)
                    
                    # Load audio files
                    audio1, sr1 = audio_proc.load_audio(os.path.join(label_path, file1))
                    audio2, sr2 = audio_proc.load_audio(os.path.join(label_path, file2))
                    
                    # Create overlapped audio
                    mixed_audio = audio_proc.gen_overlapped_audio(audio1, audio2, audio_proc.overlap_ratio)
                    
                    # Generate output filename (including label information)
                    output_filename = f"{file1.split('.')[0]}_{file2.split('.')[0]}_augmented_{i}.wav"
                    
                    # Save the mixed audio
                    wavfile.write(
                        os.path.join(label_path, output_filename),
                        audio_proc.target_rate,
                        (mixed_audio * 32767).astype(np.int16)
                    )
                    
                    print(f"Created augmentation {i+1}/{num_augmentations} for label {label}")
                    
                except Exception as e:
                    print(f"Error processing augmentation {i} for label {label}: {str(e)}")
                    continue

def process_audio_data_generator(data_path, batch_size, input_shape):

        train_gen = ImageDataGenerator(
            rescale = 1./255,                                  # Rescale pixel values to be between 0 and 1
            validation_split = 0.2                             # Set aside 20% of the data for validation
        )

        val_gen = ImageDataGenerator(
            rescale = 1./255,                                  # Rescale pixel values to be between 0 and 1
            validation_split = 0.2                             # Set aside 20% of the data for validation
        )
        
        train_generator = train_gen.flow_from_directory(
            directory = data_path,                             # Directory containing the training data
            target_size = input_shape[:2],                     # Resizes all images to 48x48 pixels
            batch_size = batch_size,                           # Number of images per batch
            color_mode = "rgb",                                # Converts the images to grayscale
            class_mode = "categorical",                        # Classifies the images into 7 categories
            subset = "training",                               # Uses the training subset of the data
            shuffle = True,
            seed = 0
        )

        val_generator = val_gen.flow_from_directory(
            directory = data_path,                             # Directory containing the training data
            target_size = input_shape[:2],                     # Resizes all images to 48x48 pixels
            batch_size = batch_size,                           # Number of images per batch
            color_mode = "rgb",                                # Converts the images to grayscale
            class_mode = "categorical",                        # Classifies the images into 7 categories
            subset = "validation",                             # Uses the training subset of the data
            shuffle = True,
            seed = 0
        )

        return train_generator, val_generator