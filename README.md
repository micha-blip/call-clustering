Here you can find the code for unsupervised clustering of audio files (.wav) using transfer learning from a VGG16 network.

The approach is image-based and relies on the properties of spectrograms, plotted from wav files
The spectrograms are further fed into a Deep Convolutional Neural Network, which was pretrained on imagenet dataset.
The network is used to extract features from images. The output is further clustered using UMAP.
UMAP is tricky and requires some work to get good output.
