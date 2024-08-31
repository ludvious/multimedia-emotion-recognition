import cv2
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.image import img_to_array
import os
import tensorflow as tf

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class FaceEmotionService:
    def __init__(self):
        self.model = load_model('models/face/mini_xception2_checkpoint.model.keras')
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        self.labels = LABELS
    
    def __del__(self):
        self.cap.release()

    def predict(self):
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi_gray = roi_gray.astype('float') / 255.0
                roi_gray = img_to_array(roi_gray)
                roi_gray = np.expand_dims(roi_gray, axis=0)
                #roi_gray = np.expand_dims(roi_gray, axis=-1)

                prediction = self.model.predict(roi_gray)[0]
                emotion = self.labels[np.argmax(prediction)]
                print(f'Face emotion detected: {emotion}')

                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Emotion Detected', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        
        return (jpeg.tobytes())