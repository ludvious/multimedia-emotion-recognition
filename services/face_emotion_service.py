import cv2
#import dlib
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.image import img_to_array
from config import LABELS, TF_ENABLE_ONEDNN_OPTS
from preprocessing.face_processing import get_landmarks_from_frame, preprocess_landmarks

class FaceEmotionService:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        #self.landmark_detector = dlib.get_frontal_face_detector()
        #self.landmark_predictor = dlib.shape_predictor('models/dlib/shape_predictor_68_face_landmarks.dat')
        self.img_dim = (48,48)
        self.img_dim_resnet = (224,224)
        self.labels = LABELS
        self.tf_oneddnn = TF_ENABLE_ONEDNN_OPTS


    def predict_frame(self, frame, show=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, self.img_dim, interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi_gray = roi_gray.astype('float') / 255.0
                roi_gray = img_to_array(roi_gray)
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1) #channel grayscale

                prediction = self.model.predict(roi_gray)[0]
                emotion = self.labels[np.argmax(prediction)]
                perc = round((max(prediction)*100), 1)
                print(f'Face emotion detected: {emotion} %{perc}')
                cvtext = f'{emotion} %{perc}'
                if perc > 55:
                    #cv2.putText(frame, cvtext, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    return emotion
                else:
                    return 'Unknow'
                    #cv2.putText(frame, 'No Emotion Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX_HERSHEY, 0.9, (0, 255, 0), 2)
    
    def predict_frame_resnet(self, frame, show=False):
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            roi_rgb = rgb[y:y + h, x:x + w]
            roi_rgb = cv2.resize(roi_rgb, self.img_dim_resnet, interpolation=cv2.INTER_AREA)
            if np.sum([roi_rgb]) != 0:
                roi_rgb = roi_rgb.astype('float') / 255.0
                roi_rgb = img_to_array(roi_rgb)
                roi_rgb = np.expand_dims(roi_rgb, axis=0)

                prediction = self.model.predict(roi_rgb)[0]
                emotion = self.labels[np.argmax(prediction)]
                perc = round((max(prediction)*100), 1)
                print(f'Face emotion detected: {emotion} %{perc}')
                cvtext = f'{emotion} %{perc}'
                if perc > 51:
                    #cv2.putText(frame, cvtext, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    return emotion
                else:
                    return 'Unknow'
                    #cv2.putText(frame, 'No Emotion Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX_HERSHEY, 0.9, (0, 255, 0), 2)

    def predict_frame_landmarks(self, frame, show=False):
        
        landmarks = get_landmarks_from_frame(frame, self.landmark_detector, self.landmark_predictor, show)

        if len(landmarks) > 0:
            proc_landmarks = preprocess_landmarks(landmarks)
            if proc_landmarks is not None:
                prediction = self.model.predict(proc_landmarks)[0]
                emotion = self.labels[np.argmax(prediction)]
                perc = round((max(prediction)*100), 1)
                print(f'Face emotion detected: {emotion} %{perc}')
                if perc > 55:
                    return emotion
                else:
                    return 'Unknow'