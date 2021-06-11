#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Sources:
https://scikit-image.org/docs/0.9.x/auto_examples/plot_equalize.html
https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/
https://stackoverflow.com/questions/19239381/pyplot-imsave-saves-image-correctly-but-cv2-imwrite-saved-the-same-image-as
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
https://docs.opencv.org/3.0-beta/modules/photo/doc/denoising.html
https://medium.com/@almutawakel.ali/opencv-filters-arithmetic-operations-2f4ff236d6aa
'''

'''
Sources:
https://towardsdatascience.com/introduction-to-image-segmentation-with-k-means-clustering-83fd0a9e2fc3
https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980
https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/
https://answers.opencv.org/question/94845/python-opencv-extracting-xy-coordinates-of-point-features-on-an-image/
https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
https://datacarpentry.org/image-processing/08-edge-detection/
https://medium.com/@nikatsanka/comparing-edge-detection-methods-638a2919476e
https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#void%20Sobel(InputArray%20src,%20OutputArray%20dst,%20int%20ddepth,%20int%20dx,%20int%20dy,%20int%20ksize,%20double%20scale,%20double%20delta,%20int%20borderType)
https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_ncut.html#sphx-glr-auto-examples-segmentation-plot-ncut-py
https://scikit-image.org/docs/dev/api/skimage.segmentation.html
https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rag_merge.html#sphx-glr-auto-examples-segmentation-plot-rag-merge-py
https://stackoverflow.com/questions/38620129/skimage-region-adjacency-graph-rag-from-quickshift-segmentation
https://github.com/scikit-image/scikit-image/issues/2363
https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_rgb_to_gray.html
https://scikit-image.org/docs/dev/user_guide/getting_started.html
https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imsave
https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_bilateral
https://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise.html
https://medium.com/@almutawakel.ali/opencv-filters-arithmetic-operations-2f4ff236d6aa
https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rag_merge.html#sphx-glr-auto-examples-segmentation-plot-rag-merge-py
https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rag_mean_color.html#sphx-glr-auto-examples-segmentation-plot-rag-mean-color-py
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
https://stackoverflow.com/questions/57618301/convert-image-from-float64-to-uint8-makes-the-image-look-darker
'''


# In[2]:


import cv2

import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float
from skimage import exposure

from sklearn.cluster import KMeans

from skimage.future import graph
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage import img_as_float
from skimage import exposure
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage import io
import skimage.restoration as restore
from skimage import img_as_ubyte

from PIL import Image, ImageFilter, ImageFile

import cpufeature

from fast_slic.avx2 import SlicAvx2 as fastSLIC_AVX2
from fast_slic import Slic as fastSLIC

import os

import time

from Evaluation import evaluateResults

import import_ipynb
from DirectoryGenerator import DirectoryGenerator

# In[3]:

def estimateAOIBoundingBoxes(imgFileName, tempFileName, finalFileName, paramList, debugMode=False):
    def readImage(method, fileName):
        if(method == 'cv2'):
            return cv2.imread(fileName)
        elif(method == 'skimage'):
            return io.imread(fileName)
        elif(method == 'pil'):
            return Image.open(fileName)
        
    def saveImage(method, img, fileName):
        if(method == 'cv2'):
            cv2.imwrite(fileName, img)
        elif(method == 'skimage'):
            io.imsave(fileName, img_as_ubyte(np.uint8(np.floor(img))))
        elif(method == 'pil'):
            img.save(fileName)
    
    def sharpenImage(img, n=1):
        # Create our shapening kernel, it must equal to one eventually
        res = img
        for i in range(0, n):
            res = res.filter(ImageFilter.SHARPEN)
        return res
    
    def histogramEqualization(img):
        return exposure.equalize_hist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) * 255
        
    def kMeansImageSegmentation(img, n_clusters=30, random_state=0):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(img if len(img.shape) == 2 else img.reshape(img.shape[0] * img.shape[1], img.shape[2]))
        pic2show = kmeans.cluster_centers_[kmeans.labels_]
        cluster_pic = pic2show.reshape(img.shape[0], img.shape[1]) if len(img.shape) == 2 else pic2show.reshape(img.shape[0], img.shape[1], img.shape[2])
        return cluster_pic
    
    def bilateralFilter(img, d=9, sigmaColor=500, sigmaSpace=1000):
        return cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    
    def ragThresholding(img, slic_components=9000, slic_compactness=1, rag_cut_threshold=3):
        slic = fastSLIC_AVX2(num_components=slic_components, compactness=slic_compactness, min_size_factor=0) if cpufeature.CPUFeature['AVX2'] else fastSLIC(num_components=slic_components, compactness=slic_compactness, min_size_factor=0)
        labels = slic.iterate(img)
        g = graph.rag_mean_color(img, labels)
        labels2 = graph.cut_threshold(labels, g, rag_cut_threshold)
        return color.label2rgb(labels2, img, kind='avg', bg_label=0)
    
    img = readImage('pil', imgFileName)
    saveImage('pil', img, tempFileName + "_orig.jpg" if debugMode else tempFileName)
    
    img = readImage('pil', tempFileName + "_orig.jpg" if debugMode else tempFileName)
    img_sharp = sharpenImage(img, n=paramList['imgSharpening']['n'])
    saveImage('pil', img_sharp, tempFileName + "_sharp.jpg" if debugMode else tempFileName)
    
    img_sharp = readImage('cv2', tempFileName + "_sharp.jpg" if debugMode else tempFileName)
    img_hist_eq = histogramEqualization(img_sharp)
    saveImage('cv2', img_hist_eq, tempFileName + "_heq.jpg" if debugMode else tempFileName)
    
    img_hist_eq = readImage('skimage', tempFileName + "_heq.jpg" if debugMode else tempFileName)
    img_kmeans = kMeansImageSegmentation(img_hist_eq / 255, n_clusters=paramList['kmeans']['n_clusters'], random_state=paramList['kmeans']['random_state'])
    saveImage('skimage', img_kmeans * 255, tempFileName + "_kmeans.jpg" if debugMode else tempFileName)
    
    img_kmeans = readImage('cv2', tempFileName + "_kmeans.jpg" if debugMode else tempFileName)
    img_blur = bilateralFilter(img_kmeans, d=paramList['bilateralFilter']['d'], sigmaColor=paramList['bilateralFilter']['sigmaColor'], sigmaSpace=paramList['bilateralFilter']['sigmaSpace'])
    saveImage('cv2', img_blur, tempFileName + "_blur.jpg" if debugMode else tempFileName)
    
    img_blur = readImage('skimage', tempFileName + "_blur.jpg" if debugMode else tempFileName)
    img_slic = ragThresholding(img_blur, slic_components=paramList['slic']['components'], slic_compactness=paramList['slic']['compactness'], rag_cut_threshold=paramList['rag']['cut_threshold'])
    saveImage('skimage', img_slic, finalFileName)

# In[4]:

def convertToBorders(sample_data):
    def normalizeAdaptiveThreshold(img):
        # Normalize the grayscale representation of the current frame using cv2.adaptiveThreshold
        # with cv2.ADAPTIVE_THRESH_GAUSSIAN_C and cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,0.01)
    
    for sample in sample_data:
        frameName = (sample.split(DirectoryGenerator().getDelimiter())[-1]).split('.')[0]
        cv2.imwrite(frameName + "_aoi_borders.png", normalizeAdaptiveThreshold(sample))

def performAutomaticAOIEstimationBenchmark(sample_data):
    for sample in sample_data:
        frameName = (sample.split(DirectoryGenerator().getDelimiter())[-1]).split('.')[0]
        tempFileName = frameName + '_temp.jpg'
        finalFileName = frameName + '_aoi.png'

        paramList = {
            'imgSharpening': {
                'n': 1
            },
            'kmeans': {
                'n_clusters': 30,
                'random_state': 0
            },
            'bilateralFilter': {
                'd': 9,
                'sigmaColor': 500,
                'sigmaSpace': 1000
            },
            'slic': {
                'components': 9000,
                'compactness': 1
            },
            'rag': {
                'cut_threshold': 3
            }
        }

        start = time.time()
        estimateAOIBoundingBoxes(sample, tempFileName, finalFileName, paramList)
        end = time.time()
    
        logFile = open('AutomaticAOIBoundingBoxesEstimationBenchmarks_New.txt', "a+")
        logFile.write("Time Taken for " + frameName + ": " + str(end - start) + "\n")
        logFile.close()

    logFile = open('AutomaticAOIBoundingBoxesEstimationBenchmarks_New.txt', "a+")
    logFile.write("\n")
    logFile.close()

def performAutomaticAOIEstimationDetailed(sample_data):
    for sample in sample_data:
        frameName = (sample.split(DirectoryGenerator().getDelimiter())[-1]).split('.')[0]
        tempFileName = frameName + '_temp.jpg'
        finalFileName = frameName + '_aoi.png'

        paramList = {
            'imgSharpening': {
                'n': 1
            },
            'kmeans': {
                'n_clusters': 30,
                'random_state': 0
            },
            'bilateralFilter': {
                'd': 9,
                'sigmaColor': 500,
                'sigmaSpace': 1000
            },
            'slic': {
                'components': 9000,
                'compactness': 1
            },
            'rag': {
                'cut_threshold': 3
            }
        }

        estimateAOIBoundingBoxes(sample, tempFileName, finalFileName, paramList, debugMode=True)