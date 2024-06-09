
###
# Download FFmpeg from https://ffmpeg.org/download.html#build-windows
# and add the bin to system path
###

import os
import shutil


path = 'public_dataset'
 
ddir =  'wav'

folders = os.listdir(path)

for fname in folders: #for web is form-web-users
    if '.webm' in fname or '.ogg' in fname:  
        
        cm = 'ffmpeg -i ' + path + '/' + fname + ' ' + ddir + '/' + fname[:-5]+'.wav'
        print(cm)
        os.system(cm)
        