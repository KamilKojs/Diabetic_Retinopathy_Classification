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

3 models described in the fourth notebook:
* 384x384:
    * accuracy: 0.774
    * cohen kappa score: 0.755
* 512x512:
    * accuracy: 0.809
    * cohen kappa score: 0.801
* 768x768:
    * accuracy: 0.798
    * cohen kappa score: 0.724

Final result 0.801 Quadratic Weighted Kappa.

Since I have a very limited access to GPU resources I can't implement other ideas but here are some of them to improve the solution even more:
* Implement lr scheduler to lower the lr from 2e-5 to 2e-7 after a couple of epochs
* Use different augmentation parameters and techniques
* Implement Spatial Pyramid Pooling and use different image resolutions during training
* Implement Monte Carlo Dropout during validation/test (for example 10 preds for 1 image and then average the result)
* Implement some kind of uncertainty estimation to return only certain predictions and return the uncertain ones for manual reviewing for doctors
* Try contrast boosting during preprocessing (CLAHE algorithm for histogram equalization?)
* Log false positives and false negatives and analyze them to see what kind of pictures are problematic for the model
* Calculate separate accuracy metric for each label class to see which class is the most problematic -> work on solution to solve it
* Calculate saliency maps for the problematic images and see how to improve the predictions
* Analyze the color histograms for problematic images, see how they differ from the ones used in training
* To make everything easier MLFlow should be implemented. FP and FN could be logged after every validation epoch to see which images are problematic for the model
* Detect potential false labelings
* Analyze the data, look for distorted photos, problematic cases even for humans
* Try to extend model input to more than 3 channels by using additional information that wasn't in the original dataset? Maybe doctors have another techniques of detecting this disease other than RGB images?
* Use ensemble of models
* Implement other techniques apart from Deep Learning and use them in addition to DL in decision trees
* Use Nvidia Dali to improve model learning times and reduce computation costs

Usually visual analysis of problematic data produces interesting ideas for development so that would be my first step in improving this solution.
