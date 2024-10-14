from flask import Flask, render_template, Response, request, jsonify
from services.face_emotion_service import FaceEmotionService
from services.speech_emotion_service import SpeechEmotionService
from services.stream_service import StreamService
import os, random, string, time
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
import cv2, pyaudio, os
from datetime import datetime
import numpy as np
from config import FPS, CHANNELS, FORMAT, CHUNK, SAMPLING_RATE
import shutil


app = Flask(__name__, template_folder='templates')

# Global variables
is_recording = False
camera = None
audio_stream = None
p_audio = None
recorder = None

face_service = FaceEmotionService(model_path='models/face/resnet50_checkpoint.model.keras')
speech_service = SpeechEmotionService(model_path='models/speech/audio_modelcheckpoint.model_t4.keras')

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(1)
        camera.set(cv2.CAP_PROP_FPS, FPS)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def audio_callback(in_data, frame_count, time_info, status):
    if is_recording and recorder:
        recorder.audio_buffer.append(in_data)
        # Check if we have collected 1 second of audio
        if len(recorder.audio_buffer) * len(in_data) >= (SAMPLING_RATE * 2 * CHANNELS):
            recorder.save_chunk()
    return (in_data, pyaudio.paContinue)

def generate_frames():
    camera = get_camera()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            if is_recording and recorder:
                recorder.frame_buffer.append(frame.copy())
                emotion = recorder.face_service.predict_frame_resnet(frame)
                #emotion = em_service.predict_frame(frame) # use predict_frame_landmark method 
                recorder.emotion_buffer.append(emotion)

                # Draw recording status
                cv2.putText(frame, "Recording...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                if recorder.chunks_info:
                    last_chunk = recorder.chunks_info[-1]
                    # Draw timestamp
                    cv2.putText(frame, f"Time: {last_chunk['timestamp']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # Draw face emotion
                    cv2.putText(frame, f"Last Face: {last_chunk['face_emotion_predicted']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # Draw audio emotion
                    cv2.putText(frame, f"Speech: {last_chunk['emotion_audio']}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            # Convert frame to jpg for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(1/FPS)
    except GeneratorExit:
        release_camera()

@app.route('/')
def index():
    return render_template('index.html')

# rotta per il rendering della pagina html per il riconiscimento emozioni facciale, qui é collegato l url per la rotta per lo streaming video e l esecuzione del modello
@app.route('/face-service')
def face_emotion():
    return render_template('face_service.html')

@app.route('/speech-service')
def speech_emotion():
    return render_template('speech_service.html')

@app.route('/stream-service')
def stream_service():
    return render_template('stream_service.html')

# rotta in cui viene eseguito lo streaming e il modello
@app.route('/video-feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process_recording')
def process_recording():
    global is_recording, audio_stream, p_audio, recorder
    
    # Stop recording if active
    if is_recording:
        is_recording = False
        
        # Stop and close audio stream
        if audio_stream:
            audio_stream.stop_stream()
            audio_stream.close()
        if p_audio:
            p_audio.terminate()
            
        # Save any remaining data
        final_chunk = None
        if recorder and (recorder.frame_buffer or recorder.audio_buffer):
            final_chunk = recorder.save_chunk()
        
        all_face_emotion_recording = []
        all_speech_emotion_recording = []
        for chunk in recorder.chunks_info:
            emotion_face = chunk['face_emotion_predicted']
            emotion_speech = chunk['emotion_audio']
            all_face_emotion_recording.append(emotion_face)
            all_speech_emotion_recording.append(emotion_speech)
        face_emotion_recording = max(all_face_emotion_recording, key=all_face_emotion_recording.count) #pick the most occurred emotion during all recording
        speech_emotion_recording = max(all_speech_emotion_recording, key=all_speech_emotion_recording.count) #pick the most occurred emotion during all recording
        # Prepare response with all recording information
        response_data = {
            'status': 'success',
            'action': 'stopped',
            'message': 'Recording stopped',
            'timestamp': datetime.now().isoformat(),
            'recording_info': {
                'total_chunks': len(recorder.chunks_info),
                'face_emotion_recording': face_emotion_recording,
                'speech_emotion_recording': speech_emotion_recording,
                'chunks': recorder.chunks_info
            }
        }
        
        return jsonify(response_data)
    
    # Start recording if not active
    else:
        is_recording = True
        recorder = StreamService(face_service, speech_service)
        #time.sleep(1)
        recorder.start_new_chunk()
        
        # Start audio stream
        p_audio = pyaudio.PyAudio()
        audio_stream = p_audio.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=SAMPLING_RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK,
                                  stream_callback=audio_callback)
        audio_stream.start_stream()
        
        return jsonify({
            'status': 'success',
            'action': 'started',
            'message': 'Recording started',
            'timestamp': datetime.now().isoformat()
        })
    
'''@app.teardown_appcontext
def delete_temporary_path(exception=None):
    if os.path.exists('recordings'):
        shutil.rmtree('recordings')  # Deletes the directory and its contents
        print(f"Deleted recordings path on teardown.")'''

if __name__ == '__main__':
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    app.run(debug=True)
