from preprocessing.audio_processing import AudioProcessing

###############
# PER TESTING #
###############

prova = AudioProcessing(audio_data_path='data/speech/audio/Angry/')
prova.audio_to_spectrogram(audio_path='data/speech/audio/Angry/prova.mp3', label='Angry')