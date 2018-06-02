# https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lgbm-clone?scriptVersionId=3540839

from collections import defaultdict, OrderedDict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os, time, io, gc
from subprocess import check_output
import zipfile
from PIL import Image
import datetime
import tqdm
start_time = time.time()
import zipfile


def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

def perform_color_analysis(img, flag):
    path = images_path + img 
    im = IMG.open(path) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None
    
def average_pixel_width(img):
    path = images_path + img 
    im = IMG.open(path)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

def get_average_color(img):
    path = images_path + img 
    img = cv2.imread(path)
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color

def perform_color_analysis2(img, speedup = True):
    path = images_path + img 
    im = IMG.open(path) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    if speedup:
        im = im.resize((256, 256), Image.ANTIALIAS)
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None, None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    return dark_percent, light_percent, size

def get_metas(img, speedup = True):
    path = images_path + img 
    img = cv2.imread(path)
    if speedup:
        img = cv2.resize(img, (256, 256))
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    bimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurness = cv2.Laplacian(bimg, cv2.CV_64F).var()
    img = cv2.resize(img, (96, 96))
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color, average_color, blurness


def image_features(img):
    try:
        c0, c1, sz = perform_color_analysis2(img)
        dom, avg, blur = get_metas(img)
        d0, d1, d2 =  dom / 255.
        a0, a1, a2 = np.array(avg) / 255.
        isz = getSize(img)
        w, h = sz
        # 'name', 'dullness', 'whiteness', 'dominant_red', 'dominant_green', 'dominant_blue', 'average_red', 
        # 'average_green', 'average_blue', 'width', 'height', 'size', 'blurness'
        return img, c0, c1, d0, d1, d2, a0, a1, a2, w, h, isz, blur
    except:
        print('Missing image : %s'%(img))
        return img,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0,   0, 0




def getSize(filename):
    filename = images_path + filename
    st = os.stat(filename)
    return st.st_size

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

'''

img =  '0b53d1fefa09d55c35c0b1d78f376b6c32525e38780701c45caae683118e7d02.jpg'
image_features(img)
'''
#path = '../input/'
path = "/home/darragh/avito/data/"
#path = '/Users/dhanley2/Documents/avito/data/'
path = '/home/ubuntu/avito/data/'

images_path = path + 'imgs/'
images_files = path + 'image_files/'

split_images = False
chunks       = 6
if split_images:
    imgls = os.listdir(images_path)
    len(imgls)
    for t, chunk in enumerate(np.array_split(imgls,chunks)):
        pd.Series(chunk).to_csv(images_files + 'img_list_%s.csv'%(t))
        

# Get features for chunk
process_chunk = 5
infile = images_files  + 'img_list_%s.csv'%(process_chunk)
df = pd.read_csv(infile, header = None, names = ['images'])
imgfeat = [image_features(i) for i in tqdm.tqdm(df['images'].tolist())]

# Write file
header = ['name', 'dullness', 'whiteness', 'dominant_red', 'dominant_green', 'dominant_blue', 'average_red', \
     'average_green', 'average_blue', 'width', 'height', 'size', 'blurness']

imgfeatdf = pd.DataFrame(imgfeat, columns = header)
imgfeatdf.head()
imgfeatdf.to_csv(images_files + 'img_features_%s.csv'%(process_chunk), index = False)
