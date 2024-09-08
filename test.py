from preprocessing.audio_processing import AudioProcessing

###############
# PER TESTING #
###############

prova = AudioProcessing()
prova.audio_to_spectrogram_img(audio_path='data/speech/audio/Angry/prova.mp3', label='Angry')