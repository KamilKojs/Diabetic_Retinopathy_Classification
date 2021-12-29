<div id="top"></div>

# Diabetic Retinopathy classification

## About The Project

This is a classification deep learning project. The goal is to correctly classify "Diabetic Retinopathy" disease. The dataset is taken from diabetic retinopathy competition from 2015 that took place on Kaggle. The development of ML models built in this project is described step by step in the "notebooks/" section. If you want to track down the mind process you should follow the notebooks.

### Built With

* [PyTorch](https://pytorch.org/)
* [PyTorch Lightning](https://www.pytorchlightning.ai/)
* [Jupyter](https://jupyter.org/)
* [Poetry](https://python-poetry.org/)

## Results

3 models described in the first notebook:
* baseline:
    * accuracy: 0.786
    * cohen kappa score: 0.528
* auto balancing:
    * accuracy: 0.770
    * cohen kappa score: 0.424
* augmentations:
    * accuracy: 0.800
    * cohen kappa score: 0.626

2 models described in the second notebook:
* cropped dataset:
    * accuracy: 0.802
    * cohen kappa score: 0.632
* MSE loss:
    * accuracy: 0.707
    * cohen kappa score: 0.668

5 models described in the third notebook:
* strenghtened augmentations:
    * accuracy: 0.693
    * cohen kappa score: 0.672
* VGG16:
    * accuracy: 0.771
    * cohen kappa score: 0.716
* Efficientnet b7:
    * accuracy: 0.768
    * cohen kappa score: 0.724
* Densenet201:
    * accuracy: 0.727
    * cohen kappa score: 0.678
* Resnet152:
    * accuracy: 0.735
    * cohen kappa score: 0.703

6 models described in the fourth notebook:
* 384x384:
    * accuracy: 0.774
    * cohen kappa score: 0.755
* 512x512:
    * accuracy: 0.809
    * cohen kappa score: 0.801
* 768x768:
    * accuracy: 
    * cohen kappa score: 
* 896x896:
    * accuracy: 
    * cohen kappa score:
* 1024x1024:
    * accuracy: 
    * cohen kappa score:
* Learning rate scheduler (from 2e-5 to 2e-7 after a couple of epochs):
    * accuracy: 
    * cohen kappa score: