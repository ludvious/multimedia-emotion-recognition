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
    
    def record_audio(self, format_audio=pyaudio.paInt16):
        """method for add taudio and for testing

        Args:
            chunk_audio (int, optional): _description_. Defaults to 1024.
            format_audio (_type_, optional): _description_. Defaults to pyaudio.paInt16.
            channels_audio (int, optional): _description_. Defaults to 2.
            rate_audio (int, optional): _description_. Defaults to 44100.
        """
        #NOTE AL MOMENTO IL RECORDING E GESTITO LATO FRONTEND
        '''duration_record = 5
        wave_output_filname = "audio_record_output.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=format_audio,
                        channels=self.channels,
                        rate=self.rate_audio,
                        input=True,
                        frames_per_buffer=self.chunk_audio) #buffer

        print("* Start recording ... ")

        frames = []

        for i in range(0, int(self.rate_audio / self.chunk_audio * duration_record)):
            data = stream.read(self.chunk_audio)
            frames.append(data) # 2 bytes(16 bits) per channel

        print("* Stop recording ... ")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(wave_output_filname, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(format_audio))
        wf.setframerate(self.rate_audio)
        wf.writeframes(b''.join(frames))
        wf.close()'''

    def preprocess_audio(self, audio):

        fix_audio, sr = self.audio_preproc.load_and_preprocess_audio(audio_path=audio)
        mel_spectrogram = self.audio_preproc.get_feature(fix_audio)

        target_shape = (self.n_mels, self.n_mels)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

        return mel_spectrogram
    
    def predict(self, audio):
        preproc = AudioProcessing()
        target_shape = (self.n_mels, self.n_mels)
        # Read the audio file as a byte stream and convert it into a numpy array
        fix_audio, sr = preproc.load_and_preprocess_audio(BytesIO(audio.read()))
        # get spectrogram
        mel_spectrogram = preproc.get_feature(fix_audio)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))
        prediction = self.model.predict(mel_spectrogram)[0]
        emotion = self.labels[np.argmax(prediction)]
        print(f'Speech emotion detected: {emotion}')

        return emotion