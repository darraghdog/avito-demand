#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://www.kaggle.com/iezepov/get-hash-from-images-slightly-daster/code
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import zipfile
import time
import os, io, gc
from PIL import Image
import pandas as pd
import datetime
import tqdm
start_time = time.time()
import zipfile

def dhash(image,hash_size = 16):
    image = image.convert('LA').resize((hash_size+1, hash_size), Image.ANTIALIAS)
    mat = np.array(
        list(map(lambda x: x[0], image.getdata()))
    ).reshape(hash_size, hash_size+1)
    
    return ''.join(
        map(
            lambda x: hex(x)[2:].rjust(2,'0'),
            np.packbits(np.fliplr(np.diff(mat) < 0))
        )
    )

path = "/home/darragh/avito/data/"
#path = '/Users/dhanley2/Documents/avito/data/'

'''
Get hash from images
'''
img_id_hash = []
counter = 1
for file_ in ['train_jpg', 'test_jpg']:
    imgzipfile = zipfile.ZipFile(path + '%s.zip'%(file_))
    namelist = imgzipfile.namelist()
    for name in tqdm.tqdm(namelist):
        try:
            imgdata = imgzipfile.read(name)
            if len(imgdata) >0:
                img_id = name[:-4]
                stream = io.BytesIO(imgdata)
                img = Image.open(stream)
                img_hash = dhash(img)
                img_id_hash.append([img_id, img_id.split('/')[-1],img_hash])
                counter+=1
        except:
            print ('Could not read ' + str(name) + ' in zip.' )

df = pd.DataFrame(img_id_hash,columns=['image_dir', 'image_id','image_hash'])
gp = df['image_hash'].value_counts().reset_index().rename(columns = {'index':'image_hash', 'image_hash':'image_hash_ct'})
df = df.merge(gp, on = 'image_hash')
df.to_csv(path+'../features/image_dhash.csv.gz', index = False, compression = 'gzip')
