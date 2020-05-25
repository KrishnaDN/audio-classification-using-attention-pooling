#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 09:45:22 2020

@author: krishna
"""

import os
import glob
import numpy as np
from shutil import copyfile

root_data ='/media/newhd/sizzle_dump/621668096/clips_621668096/filler'
save_root = '/media/newhd/Valorant_game/part3/filler'


filepath = '/media/newhd/Valorant_game/part3/filler.txt'
read_text = [line.rstrip('\n') for line in open(filepath)]
for filepath in read_text:
    source_path = root_data+'/'+filepath
    dest_path = save_root+'/'+filepath
    copyfile(source_path,dest_path)
    
####################################################################
dataset_root = '/media/newhd/Valorant_game/raw_data'
processed_data = '/media/newhd/Valorant_game/processed_data'

class_id = {'filler':0,'gunfire':1}

all_folders = sorted(glob.glob(dataset_root+'/*/'))
for folder_path in all_folders:
    sub_folders = sorted(glob.glob(folder_path+'/*/'))
    for class_folder in sub_folders:
        create_folder =processed_data+'/'+ folder_path.split('/')[-2]+'/'+class_folder.split('/')[-2]
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)
        all_files = sorted(glob.glob(class_folder+'/*.mp4'))
        for filepath in all_files:
            filename = filepath.split('/')[-1]
            dest_filepath = create_folder+'/'+filename[:-4]+'.wav'
            extract_audio = 'ffmpeg -i '+filepath+' -f wav '+dest_filepath
            os.system(extract_audio)
            
            



#############################################################################
full_list_files = []
all_folders = sorted(glob.glob(processed_data+'/*/'))
for folder_path in all_folders:
    sub_folders = sorted(glob.glob(folder_path+'/*/'))
    for class_folder in sub_folders:
        all_files = sorted(glob.glob(class_folder+'/*.wav'))
        for filepath in all_files:
            to_write = filepath+' ' +str(class_id[filepath.split('/')[-2]])
            full_list_files.append(to_write)


import random
test_files = random.sample(range(len(full_list_files)),1000)

fid_test = open('testing.txt','w')
for item in test_files:
    to_write = full_list_files[item]
    fid_test.write(to_write+'\n')
fid_test.close()

fid_train = open('training.txt','w')

for item in range(len(full_list_files)):
    if item in test_files:
        continue
    else:
        to_write = full_list_files[item]
        fid_train.write(to_write+'\n')
fid_train.close()






###############################################################################











        