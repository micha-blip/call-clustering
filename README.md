Here, you can find the code for the unsupervised clustering of audio files (.wav) using transfer learning from a VGG16 network.

The data is not available as the work has not been published (yet). 

The approach is image-based and relies on the properties of spectrograms, plotted from wav files
The spectrograms are further fed into a Deep Convolutional Neural Network, which was trained on the Imagenet dataset.
The network is used to extract features from images. The output is further clustered using UMAP.
UMAP is tricky and requires some work to get sound output.

Limitations and considerations:

* The preprocessing requires the reshaping of spectrograms to 224x224 pixels (a format that CNN accepts). This eliminates some information, such as length (to a degree).
  This is easy to solve, and length can be added as an additional feature before clustering.
  
* Reshaping can make small differences in frequency disappear. Using different scales for spectrogram can solve this problem (or cropping the interesting frequency range).
  Padding shorter signals to ensure the same format could also work but was not tested.
  
* Signal-to-noise ratio is an issue. Denoise the signal if possible, beforehand.
  
* If the sounds you want to process are long, composed of shorter segments, and are separated by silence, it may be worth cropping them so that only short pieces of the record are clustered.
  
* Very often UMAP on raw data performs very well (even outperforming this method, as tested using the Audio-MNIST dataset, at least for a single speaker).
  This was not the case with the dataset of mice calls recorded in AG Koch.
