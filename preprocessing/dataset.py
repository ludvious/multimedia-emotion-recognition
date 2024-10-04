from preprocessing.face_processing import augmentation_face_data, process_split_face_data
from preprocessing.audio_processing import AudioProcessing
import numpy as np
from pathlib import Path


def create_face_dataset(data_path: str, input_shape, batch_size):

    train_datagen, validation_datagen = augmentation_face_data()
    train_gen, val_gen = process_split_face_data(data_path, train_datagen, validation_datagen, input_shape, batch_size)

    return train_gen, val_gen
    
def create_audio_dataset(audio_file_path: str):
        """
        Load all audio wav files from folders, extract features, and return the dataset with features and labels.

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
