from flask import Flask, render_template, Response
from services.face_emotion_service import FaceEmotionService
from services.speech_emotion_service import speechEmotionPredictor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# rotta per il rendering della pagina html per il riconiscimento emozioni facciale, qui é collegato l url per la rotta per lo streaming video e l esecuzione del modello
@app.route('/face_emotion')
def face_emotion():
    return render_template('face_service.html')

# funzione che genera uno stream di immagini video da una camera
def start_stream(camera):
    while True:
        frame = camera.predict()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# rotta in cui viene eseguito lo streaming e il modello
@app.route('/video_feed')
def video_feed():
    model_path = 'models/face/mini_xception2_checkpoint.model.keras'
    return Response(start_stream(FaceEmotionService(model_path=model_path)), mimetype='multipart/x-mixed-replace; boundary=frame')

'''@app.route('/face_emotion/predict', methods=['POST'])
def predict_face_emotion():
    # Here, you'll receive image data from the client side
    image_data = request.files['image'].read()
    emotion = face_predictor.predict(image_data)
    return jsonify({'emotion': emotion})

@app.route('/speech_emotion', methods=['GET'])
def speech_emotion():
    #TODO
    speech_predictor = speechEmotionPredictor('path/to/speech_emotion_model.h5')
    return

@app.route('/speech_emotion/predict', methods=['POST'])
def predict_speech_emotion():
    # Here, you'll receive audio data from the client side
    audio_data = request.files['audio'].read()
    emotion = speech_predictor.predict(audio_data)
    return jsonify({'emotion': emotion})'''

if __name__ == '__main__':
    app.run(debug=True)
