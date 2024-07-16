# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 13:07:27 2024
@author: Michal Wojcik, AG Koch, FU Berlin
"""
# Intended to run on Spyder, cell by cell (shift + enter)
# runs only in "tensorflow" environment on my machine

# runs only in "tensorflow" environment

# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import SpectralClustering
import umap


# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reading wav files
from scipy.io import wavfile  



#%%

# path - specify your path

path = 'D:/eMUA analysis/trial' 

# make necessary folders

os.makedirs(path + "/wav")
os.makedirs(path + "/spectrograms")
os.makedirs(path + "/results")


#%%

# drop wav files into a folder path/wav

calls = os.listdir(path + "/wav")

#%%
# if new files were added
# scan dir with plots and add process the wav files which are not there yet
# ( no save time and not process all of them again)

calls = os.listdir(path + "/wav")
plots = os.listdir(path + "/spectrograms")

to_process = []

for elem in calls:
    if elem[:-4] + ".png" not in plots:
        to_process.append(elem)
        
if to_process == []:
    print("all wav files were plotted")


#%%
# get the length of file and plot the wav files to spectrograms

length = []

for n,m in enumerate(to_process):    
    print(m)
    sample_rate, snip = wavfile.read(path + "/wav/" + m)
    length.append(len(snip) / sample_rate)
    plt.tight_layout()
    plt.specgram(snip, Fs = sample_rate, cmap = "gray", detrend= "linear")[0]
    plt.axis('off') # add hash in front if you want plots with x,y axis
    plt.savefig(path + '/spectrograms/' + m[:-4] + ".png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    
print("Done plotting spectrograms")


#%%


# load pretrained model

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

data = {}

# lop through each image in the dataset
for i,spectrogram in enumerate(calls):
    print(str(i / len(calls)) + " " + spectrogram)
    feat = extract_features(path + "/" + "spectrograms" + "/" + spectrogram[:-4] + ".png", model)
    data[spectrogram] = feat

# get a list of the filenames
filenames = np.array(list(data.keys()))

#%%


# get a list of just the features
feat = np.array(list(data.values()))
feat.shape
feat = feat.reshape(-1,4096)


#%%

# add length vector

feat = np.column_stack((feat, length))

#%%
# save the output of VGG16

np.save(path + "/results/features_VGG16.npy", feat)

#%%

# apply UMAP on the extracted features

n = 5
metric = "manhattan"

reducer = umap.UMAP(verbose = True,  n_neighbors = n, min_dist = 0, random_state = 44, metric = metric)
umap_reduced = reducer.fit_transform(feat)

# These parameters seem to work fine for USVs
# 10 "euclidean"
# 5  "manhattan"
# 15 "minkowski


#%%
# Add spectral clustering with arbitrary number of classes - change if needed (n_clusters)

clust = SpectralClustering(n_clusters = 10, verbose = True).fit(umap_reduced)

#%%
# Plot the results

plt.figure()
sns.scatterplot(x = umap_reduced[:,0], y = umap_reduced[:,1], s = 5, hue = [str(x) for x in clust.labels_], alpha = 0.5)
plt.title("umap n_neighbors = " + str(n) + ", metric = " + metric + ", min_dist = 0 + spectral clustering")


#%%

# Plot the results, but instead of points there are images


def getImage(path, zoom = 0.1):
    return OffsetImage(plt.imread(path), zoom = zoom)

idx = np.array(range(len(calls)))[::50]

fig, ax = plt.subplots()
ax.scatter(x = umap_reduced[idx,0], y = umap_reduced[idx,1])


for x0, y0, file in zip(umap_reduced[idx,0], umap_reduced[idx,1], [calls[i] for i in idx]):
    ab = AnnotationBbox(getImage(path + "/spectrograms/" + file[:-4] + ".png"), (x0, y0), frameon = False)
    ax.add_artist(ab)
