import cv2
import pyaudio
import wave
import threading
import tkinter as tk
from tkinter import messagebox
from services.face_emotion_service import FaceEmotionPredictor
from services.speech_emotion_service import speechEmotionPredictor

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Emotion Recognition App")
        
        self.face_predictor_path = FaceEmotionPredictor('path/to/face_emotion_model.h5')
        self.speech_predictor_path = speechEmotionPredictor('path/to/speech_emotion_model.h5')

        self.is_running = False
        self.audio_frames = []
        self.cap = None
        self.audio_thread = None

        self.menu_app()

    def menu_app(self):
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create Start buttons
        start_face_button = tk.Button(self.root, text="Face Emotion Recognition", command=self.start_face_recognition)
        start_speech_button = tk.Button(self.root, text="Speech Emotion Recognition", command=self.start_speech_recognition)
        exit_app_button = tk.Button(self.root, text="Exit", command=self.root.quit)
        
        start_face_button.pack(pady=20)
        start_speech_button.pack(pady=20)
        exit_app_button.pack(pady=20)
    
    def exit_service(self, stop_command):
        for widget in self.root.winfo_children():
            widget.destroy()

        stop_button = tk.Button(self.root, text="Quit", command=stop_command)
        stop_button.pack(pady=20)
    
    def exit_app(self):
        self.is_running=True

    def start_face_recognition(self):
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.exit_service(self.stop_face_recognition)
        self.face_recognition_loop()

    def stop_face_recognition(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.menu_app()

    def face_recognition_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Errore: Impossibile catturare frame dalla webcam.")
                break
            
            frame_with_emotion = self.face_predictor_path.predict(frame) #prediction
            cv2.imshow('Face Emotion Recognition', frame_with_emotion) #show frame with label

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_face_recognition()
            else:
                self.root.after(10, self.face_recognition_loop) #continue every 10 millsec

    def start_speech_recognition(self):
        self.is_running = True
        self.audio_frames = []
        self.audio_thread = threading.Thread(target=self.capture_audio)
        self.audio_thread.start()
        self.show_stop_record_button(self.check_speech_recognition) #stop recording
        self.exit_service(self.)

    def check_speech_recognition(self):
        self.is_running = False
        if self.audio_thread:
            self.audio_thread.join()
        self.save_audio()
        speech_emotion = self.speech_predictor_path.predict("output.wav")
        print(f"Detected: {speech_emotion}")
        messagebox.showinfo(f"Detected: {speech_emotion}")
        #back to menu
        self.menu_app()

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

    def save_audio(self):
        wf = wave.open("output.wav", 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
