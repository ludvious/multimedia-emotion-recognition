from preprocessing.audio_processing import AudioProcessing

###############
# DATASET     #
###############

prepoc = AudioProcessing()
# first step : generate spectogram img for analyse sound
prepoc.gen_mel_spectrogram_dataset('data/speech/audio')