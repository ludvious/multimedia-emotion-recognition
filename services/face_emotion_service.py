import cv2
import numpy as np
from keras.api.models import load_model
#from utils.emotions import EMOTIONS
import os
import tensorflow as tf

# app base funziona correttamente, implemetare una migliore app; non funziona se lancio da fedora (problemi con Cuda e tensorflow che non riconosce i driver da fedora pdio)
#TODO: migliorare la grafica dell applicazione; provare a migliorare il modello (prima priorita a fare quello della voce)


EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#tf.get_logger().setLevel('ERROR')

class FaceEmotionPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path) # Loads a model saved via model.save()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotions = EMOTIONS

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_gray = roi_gray.astype('float') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            prediction = self.model.predict(roi_gray)[0]
            emotion = self.emotions[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

# # # # # # # # #
#    Testing    #
# # # # # # # # #

def test_face_emotion_predictor_realtime(model_path):
    predictor = FaceEmotionPredictor(model_path)

    # Inizializza la webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Errore: Impossibile aprire la webcam.")
        return

    while True:
        # Cattura frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Errore: Impossibile catturare frame dalla webcam.")
            break

        # Esegui la previsione delle emozioni
        frame_with_predictions = predictor.predict(frame)

        # Mostra il frame con le previsioni delle emozioni
        cv2.imshow('Face Emotion Predictor', frame_with_predictions)

        # Esci dal ciclo premendo 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia la webcam e chiudi le finestre
    cap.release()
    cv2.destroyAllWindows()

face_model_path = 'models/face/mini_xception2_checkpoint.model.keras'
test_face_emotion_predictor_realtime(face_model_path)