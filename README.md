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
    * accuracy: 
    * cohen kappa score: 

5 models described in the third notebook:
* strenghtened augmentations:
    * accuracy: 
    * cohen kappa score: 
* VGG16:
    * accuracy: 
    * cohen kappa score: 
* Efficientnet b7:
    * accuracy: 
    * cohen kappa score: 
* Densenet201:
    * accuracy: 
    * cohen kappa score: 
* Resnet152:
    * accuracy: 
    * cohen kappa score: 