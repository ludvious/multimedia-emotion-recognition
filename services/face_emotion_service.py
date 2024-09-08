import cv2
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.image import img_to_array
from config import LABELS, TF_ENABLE_ONEDNN_OPTS

class FaceEmotionService:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0) #index relative the webcam (0 if you have only preset cam installed, index can be different if you have more cam plugged on your OS)
        self.labels = LABELS
        self.tf_oneddnn = TF_ENABLE_ONEDNN_OPTS
    
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