import pyaudio, wave, librosa, os
import numpy as np
import matplotlib as plt
from pytube import YouTube
import moviepy.editor as mp

def add_folders(start_path, labels):
    if not os.path.exists(start_path):
        os.makedirs(start_path)
    
    for label in labels:
        label_folder = os.path.join(start_path, label)
        os.makedirs(label_folder)

def new_record_audio(chunk_audio=1024, format_audio=pyaudio.paInt16, channels_audio=2, rate_audio=44100):
    """method for add the possibility for record a personal audio and for test it in future

    Args:
        chunk_audio (int, optional): _description_. Defaults to 1024.
        format_audio (_type_, optional): _description_. Defaults to pyaudio.paInt16.
        channels_audio (int, optional): _description_. Defaults to 2.
        rate_audio (int, optional): _description_. Defaults to 44100.
    """
    
    record_seconds = 5
    wave_output_filname = "audio_record_output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=format_audio,
                    channels=channels_audio,
                    rate=rate_audio,
                    input=True,
                    frames_per_buffer=chunk_audio) #buffer

    print("* Start recording ... ")

    frames = []

    for i in range(0, int(rate_audio / chunk_audio * record_seconds)):
        data = stream.read(chunk_audio)
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* Stop recording ... ")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(wave_output_filname, 'wb')
    wf.setnchannels(channels_audio)
    wf.setsampwidth(p.get_sample_size(format_audio))
    wf.setframerate(rate_audio)
    wf.writeframes(b''.join(frames))
    wf.close()

def get_audio_from_mp4(filepath):

    files = os.listdir(filepath)

    for file in files:
        if file.endswith(".m4v"):
            fileName = os.path.splitext(file)
            video = mp.VideoFileClip(filepath+file)
            audio = video.audio
            audio.write_audiofile(filepath+fileName[0]+".wav")

def get_audio_from_yt(youtube_url):
    # download a file with only audio, to save space
    # if the final goal is to convert to mp3
    y = YouTube(youtube_url)
    t = y.streams.filter(only_audio=True).all()
    t[0].download(output_path="../VideoFiles")


def plot_spec(audio_data, sr, name):
    """stampa a video lo spettogramma e lo salva come immagine
    """

    spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure(figsize=(12,4))
    librosa.display.specshow(spec_db, sr=sr,
        x_axis='time', y_axis='mel',
        hop_length=sr * 0.01)
    plt.colorbar(format='%+02.0f dB')
    plt.savefig('figs/{}.png'.format(name))
    plt.clf()