import os

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NUM_LABELS = 7
TF_ENABLE_ONEDNN_OPTS= os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'