import os

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
LABELS_DICT = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral'}
NUM_LABELS = 7
TF_ENABLE_ONEDNN_OPTS= os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# AUDIO PARAMETERS

SAMPLING_RATE = 22050
N_MELS_BAND = 128
#FRAME_LENGTH =
HOP_LENGTH = 512 
MIN_AUDIO_LEN = 2.0
MAX_AUDIO_LEN = 4.0