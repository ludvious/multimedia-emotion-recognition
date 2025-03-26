from keras.api.models import load_model
from keras.api.preprocessing.image import load_img, img_to_array
from config import LABELS, SAMPLING_RATE, N_MELS_BAND, FORMAT
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
        self.sampling_rate = SAMPLING_RATE

    def save_wav(self, audio_buffer, audio_path):
        print("saving wav to " + audio_path)
        wav = wave.open(audio_path,'wb')
        wav.setnchannels(1)
        wav.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        # Ensure exactly 1 second of audio
        audio_data = b''.join(audio_buffer)
        samples = len(audio_data) // (2 * self.channels)  # 2 bytes per sample
        if samples > self.sampling_rate:
            audio_data = audio_data[:self.sampling_rate * 2 * self.channels]
        elif samples < self.sampling_rate:
            # Pad with silence if less than 1 second
            padding = b'\x00' * (self.sampling_rate * 2 * self.channels - len(audio_data))
            audio_data += padding
        wav.setframerate(self.sampling_rate)
        wav.writeframes(audio_data)
        wav.close()
    
    def preprocess_audio(self, audio_path):
        """carica e pre-processa gli audio per il modello

        Args:
            audio_path (str): _description_

        Returns:
            _type_: spectrogram with correct tensor shape
        """
        target_shape = (self.n_mels, self.n_mels)

        fix_audio, sr = self.audio_preproc.load_audio(audio_path=audio_path)
        mel_spectrogram = self.audio_preproc.get_mel_spectrogram(fix_audio)

        # Resizing  dimension for CNN (height, width, channels) QUESTO DOVREBBE FUNZIONARE CE DA TESTARLO
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
        resize_spec = tf.image.resize(mel_spectrogram, target_shape)
        #mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        rgb_spectrogram = tf.image.grayscale_to_rgb(resize_spec)
        input_tensor = tf.expand_dims(rgb_spectrogram, axis=0)
        #rgb_spectrogram = tf.reshape(rgb_spectrogram, (1,) + target_shape + (1,))

        return input_tensor
    
    def preprocess_audio_edit(self, audio_path, spec_output_folder):
        """carica e pre-processa gli audio per il modello

        Args:
            audio_path (str): _description_

        Returns:
            _type_: spectrogram with correct tensor shape
        """

        spec_img_path = self.audio_preproc.audio_to_spectrogram_img(audio_path, inference=True, inference_ouput_folder=spec_output_folder)
        img = load_img(spec_img_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        return img_array
    
    def predict_audio(self, audio_path, spec_output_folder):
        """metodo usato nell app che permette di fare la predizione dell audio registrato. Prende in input l audio, segue il processo di elaborazione;
        l input del modello saranno spectrogrammi.
        Infine ritorna la label.
        """
        img_array = self.preprocess_audio_edit(audio_path, spec_output_folder)
        prediction = self.model.predict(img_array)
        emotion = self.labels[np.argmax(prediction)]
        perc = round(float(np.max((prediction))*100), 1)
        emotion_perc = f'{emotion} %{perc}'
        print(f'Speech emotion detected: {emotion_perc}')
        #print(f'Speech emotion detected: OK')
        if perc > 51:
            return emotion_perc
        else:
            return 'Unknow'