import cv2, time, os
from datetime import datetime
import numpy as np
from keras.api.preprocessing.image import img_to_array
from preprocessing.audio_processing import AudioProcessing
from services.speech_emotion_service import SpeechEmotionService
import pyaudio, wave
import numpy as np
from io import BytesIO
from config import RATE, CHANNELS, FORMAT, RECORD_SECONDS

class StreamService:
    def __init__(self):
        self.frame_buffer = []
        self.emotion_buffer = []
        self.audio_buffer = []
        self.chunks_info = []
        self.last_save_time = time.time()
        self.current_chunk_start_time = None
        
    def start_new_chunk(self): #chunk is a section of recording length 1 second
        self.current_chunk_start_time = time.time()
        self.frame_buffer = []
        self.audio_buffer = []
        
    def save_chunk(self):
        if not self.frame_buffer or not self.audio_buffer or not self.emotion_buffer:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_dir = f'recordings/chunk_{timestamp}'
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Save frames
        frames_saved = []
        emotion_saved = []
        for i, frame in enumerate(self.frame_buffer):
            frame_path = f'{chunk_dir}/frame_{i:03d}.jpg'
            cv2.imwrite(frame_path, frame)
            frames_saved.append(f'frame_{i:03d}.jpg')
        #TODO aggiungere logica per mandare i frame al modello per prediction
        for j, emotion in enumerate(self.emotion_buffer):
            emotion_saved.append(emotion)
        emotion_predicted = max(emotion_saved,key=emotion_saved.count)

        # Save audio (exactly 1 second)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = f'{chunk_dir}/audio_{timestamp}.wav'
        wave_file = wave.open(audio_path, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        
        # Ensure exactly one second of audio
        audio_data = b''.join(self.audio_buffer)
        samples = len(audio_data) // (2 * CHANNELS)  # 2 bytes per sample
        if samples > RATE:
            audio_data = audio_data[:RATE * 2 * CHANNELS]
        elif samples < RATE:
            # Pad with silence if less than 1 second
            padding = b'\x00' * (RATE * 2 * CHANNELS - len(audio_data))
            audio_data += padding
            
        wave_file.writeframes(audio_data)
        wave_file.close()
        #prediction audio emotion
        model_audio_path = 'path'
        #speech_service = SpeechEmotionService(model_audio_path)
        #audio_emotion = speech_service.predict()
        audio_emotion = 'TEST'
        
        # Record chunk information
        chunk_info = {
            'timestamp': timestamp,
            'directory': chunk_dir,
            'frames_count': len(frames_saved),
            'frames': frames_saved,
            'face_emotion_count': len(emotion_saved),
            'face_emotion_captured': emotion_saved,
            'face_emotion_predicted': emotion_predicted,
            'audio_file': f'audio_{timestamp}.wav',
            'emotion_audio': audio_emotion,
            'audio_duration': RECORD_SECONDS,
            'audio_sample_rate': RATE,
            'audio_channels': CHANNELS
        }
        self.chunks_info.append(chunk_info)
        
        # Start new chunk
        time.sleep(1)
        self.start_new_chunk()
        
        return chunk_info
    