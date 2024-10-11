import cv2, time, os
from datetime import datetime
import pyaudio, wave
from preprocessing.audio_processing import AudioProcessing
from services.face_emotion_service import FaceEmotionService
from services.speech_emotion_service import SpeechEmotionService
from config import SAMPLING_RATE, CHANNELS, FORMAT, RECORD_SECONDS

class StreamService:
    def __init__(self, face_service: FaceEmotionService, speech_service: SpeechEmotionService):
        self.face_service = face_service
        self.speech_service = speech_service
        self.audio_proc = AudioProcessing()
        self.frame_buffer = []
        self.emotion_buffer = []
        self.audio_buffer = []
        self.chunks_info = []
        self.last_save_time = time.time()
        self.current_chunk_start_time = None
        
    def start_new_chunk(self): #chunk is a section of recording length 1 second
        self.current_chunk_start_time = time.time()
        self.frame_buffer = []
        self.emotion_buffer = []
        self.audio_buffer = []
        print('start chunk')
        
    def save_chunk(self):
        if not self.frame_buffer or not self.audio_buffer or not self.emotion_buffer:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_dir = f'recordings/chunk_{timestamp}'
        os.makedirs(chunk_dir, exist_ok=True)
        
        # List for save temp frames, face emotion and speech emotion
        frames_saved = []
        face_emotion = []

        for i, frame in enumerate(self.frame_buffer):
            frame_path = f'{chunk_dir}/frame_{i:03d}.jpg'
            cv2.imwrite(frame_path, frame)
            frames_saved.append(f'frame_{i:03d}.jpg')
            #TODO SPOSTARE QUI LA LOGICA PER PREDICTION FACE --------------- FATTO , FARE CHECK SE FUNZIONA
        for j, emotion in enumerate(self.emotion_buffer):
            face_emotion.append(emotion)        
        emotion_predicted = max(face_emotion, key=face_emotion.count) #pick the most occurred emotion detected from frame in 1 second 

        # Save audio (exactly 1 second)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = f'{chunk_dir}/audio_{timestamp}.wav'
        self.speech_service.save_wav(self.audio_buffer, audio_path)
        #prediction audio emotion
        audio_emotion, perc = self.speech_service.predict_audio(audio_path)
        #audio_emotion = 'TEST'
        
        # Record chunk information
        chunk_info = {
            'timestamp': timestamp,
            'directory': chunk_dir,
            'frames_count': len(frames_saved),
            'frames': frames_saved,
            'face_emotion_count': len(face_emotion),
            'face_emotion_captured': face_emotion,
            'face_emotion_predicted': emotion_predicted,
            'audio_file': f'audio_{timestamp}.wav',
            'emotion_audio': f'{audio_emotion} {perc}',
            'audio_duration': RECORD_SECONDS,
        }
        self.chunks_info.append(chunk_info)
        
        # Start new chunk
        #time.sleep(0.5)
        self.start_new_chunk()
        
        return chunk_info
    