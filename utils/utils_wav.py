# Third Party
import librosa
import numpy as np
import torch
# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(audio_filepath, sr, win_length=160000,mode='train'):
    audio_data,fs  = librosa.load(audio_filepath,sr=16000)
    if len(audio_data)<win_length:
        diff = win_length-len(audio_data)
        create_arr = np.zeros([1,diff])
        final_data  = np.concatenate((audio_data,create_arr[0]))
        audio_data = final_data
        ret_data = audio_data
    else:
        ret_data = audio_data[:win_length]
    return ret_data
    

def load_data(path, seg_length=160000, win_length=800, sr=16000, hop_length=400, n_fft=512, spec_len=300, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    
    return wav


def speech_collate(batch):
    label=[]
    specs = []
    for sample in batch:
        specs.append(sample['spec'])
        label.append((sample['labels']))
     
    return specs, label
