import cv2
import pyaudio
import wave
import threading
from services.face_emotion_service import FaceEmotionPredictor
from services.speech_emotion_service import speechEmotionPredictor

class EmotionRecognitionApp:
    def __init__(self) -> None:
        self.face_predictor = FaceEmotionPredictor('path/to/face_emotion_model.h5')
        self.speech_predictor = speechEmotionPredictor('path/to/speech_emotion_model.h5')
        self.is_running = False
        self.audio_frames = []
    
    def capture_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        while self.is_running:
            data = stream.read(CHUNK)
            self.audio_frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def run(self):
        self.is_running = True
        audio_thread = threading.Thread(target=self.capture_audio)
        audio_thread.start()

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Face emotion recognition
            frame_with_emotion = self.face_predictor.predict(frame)

            cv2.imshow('Emotion Recognition', frame_with_emotion)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.is_running = False
        audio_thread.join()
        cap.release()
        cv2.destroyAllWindows()

        # Speech emotion recognition
        self.save_audio()
        speech_emotion = self.speech_predictor.predict("output.wav")
        print(f"Detected speech emotion: {speech_emotion}")


    def save_audio(self):
        wf = wave.open("output.wav", 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()

    def run(self):
        self.is_running = True
        audio_thread = threading.Thread(target=self.capture_audio)
        audio_thread.start()

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Face emotion recognition
            frame_with_emotion = self.face_predictor.predict(frame)

            cv2.imshow('Emotion Recognition', frame_with_emotion)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.is_running = False
        audio_thread.join()
        cap.release()
        cv2.destroyAllWindows()

        # Speech emotion recognition
        self.save_audio()
        speech_emotion = self.speech_predictor.predict("output.wav")
        print(f"Detected speech emotion: {speech_emotion}")

if __name__ == "__main__":
    app = EmotionRecognitionApp()
    app.run()