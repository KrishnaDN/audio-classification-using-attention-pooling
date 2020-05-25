# Third Party
import librosa
import numpy as np
# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(audio_filepath, sr, win_length=132300):
    audio_data,fs  = librosa.load(audio_filepath,sr=44100)
    len_file = len(audio_data)
    if len_file <win_length:
        dummy=np.zeros((1,win_length-len_file))
        extened_wav = np.concatenate((audio_data,dummy[0]))
    else:
        extened_wav = audio_data[:win_length]
    return extened_wav
    


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=512):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T

def load_data(filepath, win_length=132300, sr=44100,ham_length=882,hop_length=441, n_fft=1024, spec_len=300):
    wav = load_wav(filepath, sr=sr)
    
    linear_spect = lin_spectogram_from_wav(wav, hop_length, ham_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    spec_mag = mag_T[:,:spec_len]
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    #return spec_mag
    return (spec_mag - mu) / (std + 1e-5)
    #return spec_mag





