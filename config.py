import os, pyaudio

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
LABELS_DICT = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral'}
NUM_LABELS = 7
TF_ENABLE_ONEDNN_OPTS= os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# MODELS PATH
FACE_MODEL_PATH = "models/face/resnet50_checkpoint.model.keras"
SPEECH_MODEL_PATH = "models/speech/audio_modelcheckpoint.model_t4.keras"

# AUDIO PARAMETERS

TARGET_RATE = 22050 #rate for process audio
N_MELS_BAND = 128
WINDOW_SIZE = 2048
HOP_LENGTH = 512 
MIN_AUDIO_LEN = 1 #second
MAX_AUDIO_LEN = 1 #second
OVERLAP_RATIO = 0.5

SAMPLING_RATE = 44100 #rate setting when recording audio
CHANNELS = 1 # is the mono channels
FORMAT = pyaudio.paInt16
CHUNK = SAMPLING_RATE  # Set chunk size to exactly one second of audio
RECORD_SECONDS = 1
# Video configuration
FPS = 30