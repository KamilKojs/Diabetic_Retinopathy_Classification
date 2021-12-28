<div id="top"></div>

## About The Project

This is a classification deep learning project. The goal is to correctly classify "Diabetic Retinopathy" disease. The dataset is taken from diabetic retinopathy competition from 2015 that took place on Kaggle. The development of ML models built in this project is described step by step in the "notebooks/" section. If you want to track down the mind process you should follow the notebooks.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should implement DRY principles to the rest of your life :smile:

### Built With

* [PyTorch](https://pytorch.org/)
* [PyTorch Lightning](https://www.pytorchlightning.ai/)
* [Jupyter](https://jupyter.org/)
* [Poetry](https://python-poetry.org/)

## Results

3 models described in the first notebook:
* base:
    * accuracy: 0.786
    * cohen kappa score: 0.528
* auto balancing:
    * accuracy: 0.770
    * cohen kappa score: 0.424
* augmentations:
    * accuracy: 0.800
    * cohen kappa score: 0.626