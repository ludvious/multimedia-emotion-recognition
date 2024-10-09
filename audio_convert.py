from preprocessing.audio_processing import AudioProcessing
# Example usage
preproc = AudioProcessing()
preproc.to_wav(file_path="data/speech/download/spaventi.mp3", label="Fear", overlapping=False)