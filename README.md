Here you can find the code for unsupervised clustering of audio files (.wav) using transfer learning from a VGG16 network.

The approach is image-based and relies on the properties of spectrograms, plotted from wav files
The spectrograms are further fed into a Deep Convolutional Neural Network, which was pretrained on imagenet dataset.
The network is used to extract features from images. The output is further clustered using UMAP.
UMAP is tricky and requires some work to get good output.

Limitations:
* The preprocessing requires reshaping of spectrograms to 224x224 pixels (format which CNN likes). This gets rid of some information, such as length (to a degree).
  This is easy to solve and length can be added as additional feature before clustering.
* Reshaping can make small differences in frequency dissapear. Using different scale for spectrogram can solve this problem (or cropping the interesting frequency range)
* Signal to noise ratio is obviously an issue. Denoise the signal if possible beforehand.
  
