import tensorflow as tf
import pandas as pd
import cv2, os, librosa
import numpy as np
from keras.api.utils import image_dataset_from_directory
from keras.api.models import Sequential
from keras.api.layers import RandomFlip, RandomRotation, RandomZoom, Resizing, Rescaling, RandomTranslation, RandomBrightness, RandomContrast
import pyaudio
import wave

def capture_frames_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def extract_spectrogram(file_path):
        y, sr = librosa.load(file_path)
        spect = librosa.feature.melspectrogram(y=y, sr=sr)
        spect_db = librosa.power_to_db(spect, ref=np.max)
        return spect_db

def new_record_audio(chunk_audio=1024, format_audio=pyaudio.paInt16, channels_audio=2, rate_audio=44100):
    """method for add the possibility for record a personal audio and for test it in future

    Args:
        chunk_audio (int, optional): _description_. Defaults to 1024.
        format_audio (_type_, optional): _description_. Defaults to pyaudio.paInt16.
        channels_audio (int, optional): _description_. Defaults to 2.
        rate_audio (int, optional): _description_. Defaults to 44100.
    """
    
    record_seconds = 5
    wave_output_filname = "audio_record_output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=format_audio,
                    channels=channels_audio,
                    rate=rate_audio,
                    input=True,
                    frames_per_buffer=chunk_audio) #buffer

    print("* Start recording ... ")

    frames = []

    for i in range(0, int(rate_audio / chunk_audio * record_seconds)):
        data = stream.read(chunk_audio)
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* Stop recording ... ")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(wave_output_filname, 'wb')
    wf.setnchannels(channels_audio)
    wf.setsampwidth(p.get_sample_size(format_audio))
    wf.setframerate(rate_audio)
    wf.writeframes(b''.join(frames))
    wf.close()