# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 13:07:27 2024

@author: Michal W
"""
# runs only in "tensorflow" environment on my machine

# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle


from scipy.io import wavfile  


# path - specify your path
# in path make a folder called wav - with wav files
# in path make a folder called spectrograms - with spectrograms what will be generated

# and run


path = 'D:/eMUA analysis/big CNN clustering old/' 
calls = os.listdir(path + "/wav")

import cv2 as cv


#%%

sample_rate, snip = wavfile.read(path + "/wav/" + calls[0])


#%%

# get the loudest value

calls = os.listdir(path + "/wav")

# produce tiff plots from wav files
import matplotlib

length = []

spec_max = 0
spec_min = 0

for n,m in enumerate(calls):    
    print(n / len(calls))
    sample_rate, snip = wavfile.read(path + "/wav/" + m)
    length.append(len(snip)/sample_rate)          
    plt.tight_layout()
    spec = plt.specgram(snip, Fs = sample_rate, cmap = "gray", detrend= "linear")[0]
    if spec_max < np.max(spec):
        spec_max = np.max(spec)
        spec_min = np.min(spec)
    plt.close()
    
print("done")

#%%

# scan dir with plots and add the wav files which are not there

path = 'D:/eMUA analysis/big CNN clustering old/' 

calls = os.listdir(path + "/wav")
plots = os.listdir(path + "/symlog plots no margin")

to_process = []

for elem in calls:
    if elem[:-4] + ".png" not in plots:
        to_process.append(elem)
        
if to_process == []:
    print("all wav files plotted")

 
#%%

# maybe normalize after all and add min val and max val of spectrograms as a feature?

spec_max = 8999716188543.578

#calls = os.listdir(path + "/wav")

for n,m in enumerate(to_process):    
    print(m)
    sample_rate, snip = wavfile.read(path + "/wav/" + m)
    plt.tight_layout()
    
    spec = plt.specgram(snip, Fs = sample_rate, cmap = "gray", detrend= "linear")[0]
    plt.close()
    plt.imshow(spec, cmap = "gray", norm = "linear")
    
    plt.clim(0, spec_max)
  
    plt.axis('off') # add hash in front if you want plots with x,y axis
      
    plt.savefig(path + '/symlog plots no margin/' + m[:-4] + ".png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    
print("Preprocessing margins done")

#%%

calls = os.listdir(path + "/wav")
to_process = calls
for n,m in enumerate(to_process):    
    print(m)
    sample_rate, snip = wavfile.read(path + "/wav/" + m)
    plt.tight_layout()
    
    spec = plt.specgram(snip, Fs = sample_rate, cmap = "gray", detrend= "linear")[0]      
    plt.axis('off') # add hash in front if you want plots with x,y axis
      
    plt.savefig(path + '/norm plot/' + m[:-4] + ".png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    
print("Preprocessing margins done")


#%%


#CNN approach

# load model
model = VGG16(weights = "imagenet")
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# load the model first and pass as an argument

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx)
    return features
  

#%%

path = 'D:/eMUA analysis/big CNN clustering old/norm plot'
calls = os.listdir(path)
data = {}

# lop through each image in the dataset
for i,spectrogram in enumerate(calls):
    print(str(i / len(calls)) + " " + spectrogram)
    feat = extract_features(path + "/" + spectrogram, model)
    data[spectrogram] = feat

# get a list of the filenames
filenames = np.array(list(data.keys()))

#%%

import matplotlib.pyplot as plt

# get a list of just the features
feat = np.array(list(data.values()))
feat.shape
feat = feat.reshape(-1,4096)


#%%

# add the length vector 
length = []

for n,call in enumerate(calls):
    print(str(n/len(calls)))
    sample_rate, snip = wavfile.read('D:/eMUA analysis/big CNN clustering old' + "/wav/" + call[:-4] + ".wav")
    length.append(len(snip)/sample_rate)      
    


#%%


np.save('D:/eMUA analysis/big CNN clustering old/norm_plot/norm_plot_imagenet.npy', feat)
np.save('D:/eMUA analysis/big CNN clustering old/norm_plot/length.npy', length)

#%%

feat = np.load('D:/eMUA analysis/big CNN clustering old/norm_plot/norm_plot_imagenet.npy')
length = np.load('D:/eMUA analysis/big CNN clustering old/norm_plot/length.npy')

#%%


from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering


#%%

pca = PCA(n_components=5, random_state=22, whiten = False)
pca.fit(feat)
x = pca.transform(feat)

import seaborn as sns

plt.figure()
sns.scatterplot(x = x[:,0], y = x[:,1], hue = length, s = 5)
#sns.scatterplot(x = x[:,0], y = x[:,1], hue = categories)
plt.title("PCA")
plt.xlabel("PC1")
plt.xlabel("PC2")
plt.title("pca of shape features")


length = np.array(length)
length = length - np.min(length)
length = length / np.max(length)

x = np.column_stack((x, length))

plt.figure()
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()


#%%


def separate_classes(classes, foldername):
    for c in range(len(np.unique(classes.labels_))):
        os.makedirs('D:/eMUA analysis/big CNN clustering old/norm_plot/output classes/' + foldername + "/" + str(c))
        subset = np.array(calls)[np.where(classes.labels_ == c)]

        for wav in subset:
            
            sample_rate, snip = wavfile.read('D:/eMUA analysis/big CNN clustering old' + "/wav/" + wav[:-4] + ".wav")
            
            plt.tight_layout()
            plt.specgram(snip, Fs = sample_rate, rasterized = True, cmap = "gray")
            
            plt.axis("off")
            plt.savefig('D:/eMUA analysis/big CNN clustering old/norm_plot/output classes/' + foldername + "/" + str(c) + "/" + wav )
            plt.close()



#%%
''' raw umap '''
import umap

n = 15
metric = "mahalanobis"

reducer = umap.UMAP(verbose = True,  n_neighbors = n, min_dist = 0, random_state = 44, metric = metric)
umap_reduced = reducer.fit_transform(feat)

# 10 "euclidean"
# 5  "manhattan"
# 15 "minkowski


#%%

clust = SpectralClustering(n_clusters = 10, verbose = True).fit(umap_reduced)

#%%

plt.figure()
sns.scatterplot(x = umap_reduced[:,0], y = umap_reduced[:,1], s = 5, hue = [str(x) for x in clust.labels_], alpha = 0.5)
plt.title("umap n_neighbors = " + str(n) + ", metric = " + metric + ", min_dist = 0 + spectral clustering")


#%%

def getImage(path, zoom = 0.1):
    return OffsetImage(plt.imread(path), zoom = zoom)

idx = np.array(range(len(calls)))[::50]

fig, ax = plt.subplots()
ax.scatter(x = umap_reduced[idx,0], y = umap_reduced[idx,1])

path = 'D:/eMUA analysis/big CNN clustering old/norm plot/'

for x0, y0, file in zip(umap_reduced[idx,0], umap_reduced[idx,1], [calls[i] for i in idx]):
    ab = AnnotationBbox(getImage(path + file[:-4] + ".png"), (x0, y0), frameon = False)
    ax.add_artist(ab)


#%%


