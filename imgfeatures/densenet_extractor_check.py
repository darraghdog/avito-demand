#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import sparse
import os
from collections import OrderedDict
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from os import listdir, makedirs
from os.path import join, exists, expanduser
import numpy as np
import zipfile, os, gc
import pickle, collections
import tensorflow as tf

from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121

from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model


# Params
path = "/home/darragh/avito/data/"
#path = '/Users/dhanley2/Documents/avito/data/'
# path = "/home/ubuntu/avito/data/"
#base_model = VGG19(weights='imagenet')
#base_model = InceptionV3(weights='imagenet')
base_model = DenseNet121(weights='imagenet')
#base_model = MobileNet(weights='imagenet')

base_model.summary()
#model.summary()

# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
model.summary()
end_shape = model.layers[-1].output_shape[-1]

def process_batch(img_ls):
    x = preprocess_input(np.array(img_ls))
    pool_features = model.predict(x)
    pool_features = pool_features.reshape(len(img_ls), end_shape)
    pool_features = pool_features.astype(np.float32)
    return pool_features

batch_size = 512*4
os.chdir(path + '../imgfeatures/tmp')
for file_ in ['train_jpg']: # ,'test_jpg', 
    file_ls   = []
    csr_ls    = [] 
    myzip = zipfile.ZipFile(path + '%s.zip'%(file_)) # zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip')
    files_in_zip = myzip.namelist()[:4]
    img_ls = []
    for idx, file in tqdm(enumerate(files_in_zip), total = len(files_in_zip)):
        if file.endswith('.jpg'):
            try:
                myzip.extract(file, path=file.split('/')[3])
                img_path = './%s/%s'%(((file.split('/')[-1]), file))
                img = image.load_img(img_path, target_size=(224, 224))
                img_ls.append(image.img_to_array(img))
                file_ls.append(file)
            except:
                print('Missing %s'%(file))
            if len(img_ls) == batch_size:
                csr_ls.append(process_batch(img_ls))
                img_ls = []
    if len(img_ls) >0:
        csr_ls.append(process_batch(img_ls))
        img_ls = []
    #myzip.close()

file_ = 'train'
df = pd.read_csv(path + '%s.csv.zip'%(file_), \
                     usecols = ['image', 'item_id'])
# Check a sample image and see if it matches
img = file_ls[0].split('/')[-1].split('.')[0]
csr_ls[0][0]
df[df['image'].isin([img])]
trnmat = np.load(path + '../imgfeatures/densenet_pool_array_%s.npy'%(file_))
trnmat[93146]

batch_size = 512*4
os.chdir(path + '../imgfeatures/tmp')
for file_ in ['test_jpg']: # ,'test_jpg', 
    file_ls   = []
    csr_ls    = [] 
    myzip = zipfile.ZipFile(path + '%s.zip'%(file_)) # zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip')
    files_in_zip = myzip.namelist()[:4]
    img_ls = []
    for idx, file in tqdm(enumerate(files_in_zip), total = len(files_in_zip)):
        if file.endswith('.jpg'):
            try:
                myzip.extract(file, path=file.split('/')[3])
                img_path = './%s/%s'%(((file.split('/')[-1]), file))
                img = image.load_img(img_path, target_size=(224, 224))
                img_ls.append(image.img_to_array(img))
                file_ls.append(file)
            except:
                print('Missing %s'%(file))
            if len(img_ls) == batch_size:
                csr_ls.append(process_batch(img_ls))
                img_ls = []
    if len(img_ls) >0:
        csr_ls.append(process_batch(img_ls))
        img_ls = []
    #myzip.close()

file_ = 'test'
df = pd.read_csv(path + '%s.csv.zip'%(file_), \
                     usecols = ['image', 'item_id'])
# Check a sample image and see if it matches
img = file_ls[1].split('/')[-1].split('.')[0]
csr_ls[0][1]
df[df['image'].isin([img])]
mat = np.load(path + '../imgfeatures/densenet_pool_array_%s.npy'%(file_))
mat[180681]
mat.dtype