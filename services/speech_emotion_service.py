from keras.api.models import load_model
from config import LABELS, LABELS_DICT, SAMPLING_RATE, N_MELS_BAND
from preprocessing.audio_processing import AudioProcessing
import pyaudio, wave
import tensorflow as tf
import numpy as np
from io import BytesIO

class SpeechEmotionService:
    def __init__(self, model_path: str) -> None:
        self.audio_preproc = AudioProcessing()
        self.model = load_model(model_path)
        self.labels = LABELS
        self.n_mels = N_MELS_BAND
        self.chunk_audio = 512
        self.channels = 1
        self.rate_audio = SAMPLING_RATE

    def preprocess_audio(self, audio):

        fix_audio, sr = self.audio_preproc.load_audio(audio_path=audio)
        mel_spectrogram = self.audio_preproc.get_spectrogram(fix_audio)

        target_shape = (self.n_mels, self.n_mels)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

        return mel_spectrogram
    
    def predict(self, audio):
        """metodo usato nell app che permette di fare la predizione dell audio registrato. Prende in input l audio, segue il processo di elaborazione;
        l input del modello saranno spectrogrammi.
        Infine ritorna la label.

        Args:
            audio (_type_): _description_

        Returns:
            str : emotion
        """

        target_shape = (self.n_mels, self.n_mels)
        # Read the audio file as a byte stream, normalize it, and convert it into a numpy array
        fix_audio, sr = self.audio_preproc.load_audio(BytesIO(audio.read()))
        # get spectrogram
        mel_spectrogram = self.audio_preproc.get_spectrogram(fix_audio)
        # Add channel dimension for CNN (height, width, channels)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))
        prediction = self.model.predict(mel_spectrogram)[0]
        emotion = self.labels[np.argmax(prediction)]
        #print(f'Speech emotion detected: {emotion}')
        print(f'Speech emotion detected: OK')

        return emotion