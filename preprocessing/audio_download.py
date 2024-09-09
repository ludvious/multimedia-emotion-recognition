from pydub import AudioSegment
from pydub.utils import make_chunks
from pathlib import Path
from pytube import YouTube
import subprocess, os

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# comando ffmpeg
# ffmpeg -i input_video.mp4 -vn -acodec copy output_audio.aac
# ffmpeg -i "C:\Users\ludov\Downloads\audio\VERY ANGRY VOICE ACTING.mp3" -vn -acodec copy "C:\Users\ludov\Downloads\audio\VERY ANGRY VOICE ACTING.aac"


def extract_with_ffmpeg(file_path: str):
    file_path = Path(file_path)
    file_name = file_name.stem
    output_path = f'videos/{file_name}.acc'
    ffmpeg_command = [
        'ffmpeg', 
        '-i', file_path,  # input file
        '-vn',
        '-acodec',
        'copy',
        output_path        # output file
    ]
    # Run the FFmpeg command

    result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# PARTE IN CUI CONVERTO IL VIDEO DA FFMPEG IN AUDIO WAV CIASCUNO DA 4 SEC

def to_wav(file_path: str, label):

    file_aac = Path(file_path)
    file_name = file_aac.stem
    audio = AudioSegment.from_file(file_aac, format="aac")
    chunk_length_ms = 4000  # 4 seconds per chunk
    chunks = make_chunks(audio, chunk_length_ms)

    # Export all of the individual chunks as separate files
    for i, chunk in enumerate(chunks):
        chunk.export(f"data/speech/audio/{label}/{file_name}_{i}.wav", format="wav")


to_wav("C:/Users/ludov/Downloads/audio/VERY ANGRY VOICE ACTING.aac", LABELS[0])