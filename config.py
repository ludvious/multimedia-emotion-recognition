import os

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
LABELS_DICT = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral'}
NUM_LABELS = 7
TF_ENABLE_ONEDNN_OPTS= os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

MIN_VID_LEN = 1.5
VIDEO_ANALYSE_WINDOW_SECS = 4.0