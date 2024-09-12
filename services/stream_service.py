import cv2, time, os
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.image import img_to_array
from config import LABELS, TF_ENABLE_ONEDNN_OPTS
from keras.api.models import load_model
from config import LABELS, LABELS_DICT, SAMPLING_RATE, N_MELS_BAND
from preprocessing.audio_processing import AudioProcessing
import pyaudio, wave
import tensorflow as tf
import numpy as np
from io import BytesIO


class StreamService:
    def __init__(self) -> None:
        pass
    