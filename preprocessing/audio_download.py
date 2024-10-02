from pydub import AudioSegment
from pydub.utils import make_chunks
from pathlib import Path
from pytube import YouTube
import subprocess, os
import youtube_dl

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# comando ffmpeg
# ffmpeg -i input_video.mp4 -vn -acodec copy output_audio.aac
# ffmpeg -i "C:\Users\ludov\Downloads\audio\VERY ANGRY VOICE ACTING.mp3" -vn -acodec copy "C:\Users\ludov\Downloads\audio\angry\VERY ANGRY VOICE ACTING.aac"

def download_youtube_as_mp3(youtube_url, new_filename, out_path):
    try:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # Configura le opzioni di download
        ydl_opts = {
            'format': 'bestaudio/best',  # Scarica il miglior audio disponibile
            'outtmpl': os.path.join(out_path, new_filename + '.mp4'),  # Percorso di salvataggio e nome file
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp4',  # Mantenere il formato MP4 (audio)
            }],
            'quiet': True  # Imposta su False per output più dettagliato
        }

        # Scarica il video come audio
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            print(f"Scaricando l'audio da: {youtube_url}")
            ydl.download([youtube_url])
        file_path = f'{os.path.join(out_path, new_filename)}.mp4'
        print(f"Download completato! File salvato come {os.path.join(out_path, new_filename)}.mp4")
        return file_path
    except Exception as e:
        print(f"Si è verificato un errore: {e}")


# PARTE IN CUI CONVERTO IN AUDIO WAV CIASCUNO DA 1 SEC

def to_wav(file_path: str, label):

    file = Path(file_path)
    file_name = file.stem
    audio = AudioSegment.from_file(file)
    #first_second = audio[:28000]
    chunk_length_ms = 1000  # 1 seconds per chunk
    chunks = make_chunks(audio, chunk_length_ms)

    # Export all of the individual chunks as separate files
    for i, chunk in enumerate(chunks):
        chunk.export(f"data/speech/audio/{label}/{file_name}_{i}.wav", format="wav")


#to_wav("C:/Users/ludov/Downloads/audio/VERY ANGRY VOICE ACTING.aac", LABELS[0])


def download_and_split(youtube_url, new_filename, out_path, label):
    audio = download_youtube_as_mp3(youtube_url, new_filename, out_path)
    to_wav(out_path, label=label)
    