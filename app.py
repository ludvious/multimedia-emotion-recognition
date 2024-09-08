from flask import Flask, render_template, Response, request, jsonify
import librosa, os
from io import BytesIO
from services.face_emotion_service import FaceEmotionService
from services.speech_emotion_service import speechEmotionPredictor
from preprocessing.audio_processing import AudioProcessing

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# rotta per il rendering della pagina html per il riconiscimento emozioni facciale, qui é collegato l url per la rotta per lo streaming video e l esecuzione del modello
@app.route('/face-emotion')
def face_emotion():
    return render_template('face_service.html')

@app.route('/speech-emotion')
def speech_emotion():
    return render_template('speech_service.html')

# funzione che genera uno stream di immagini video da una camera
def start_stream(camera):
    while True:
        frame = camera.predict()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# rotta in cui viene eseguito lo streaming e il modello
@app.route('/video-feed')
def video_feed():
    model_path = 'models/face/mini_xception2_checkpoint.model.keras'
    return Response(start_stream(FaceEmotionService(model_path=model_path)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    return

if __name__ == '__main__':
    app.run(debug=True)
