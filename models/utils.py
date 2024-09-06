import pyaudio, wave, librosa
import numpy as np
import matplotlib as plt

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

# Plot the Mel spectrogram
def plot_spec(audio_data, sr, name):

    spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure(figsize=(12,4))
    librosa.display.specshow(spec_db, sr=sr,
        x_axis='time', y_axis='mel',
        hop_length=sr * 0.01)
    plt.colorbar(format='%+02.0f dB')
    plt.savefig('figs/{}.png'.format(name))
    plt.clf()