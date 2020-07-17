# Pneumonia Classifier:
This is an image classifier attempt to detect pneumonia patients by running CNNs on chest X-rays.
Tensorflow and keras are used as deep learning framework. Additionally a Flask server and basic website pages are added to repo in order to show how to host a basic deep neural network model in case of demo requirement. Please read the important notice below before using it for any purposes.

Two image class sets are used for training: 
  1. Normal chest X-rays 
  2. Chest X-rays of Pneumonia patients (such as Viral Infection, Covid-19) 
  
I ran an CNN network on the following datasets without using transfer learning. 

## Datasets:
  1. https://www.kaggle.com/paultimothymooney/detecting-pneumonia-in-x-ray-images/data?
  2. Covid-19 dataset Chest X-ray dataset from https://www.pyimagesearch.com/ 

## Important Notice:
Please do not use the model in this repo as a medical diagnosis tool by as it is now. Not only Chest X-ray images by themselves are not sufficient for doctors to diagnose a pneumonia case at a patient (Normally patient's history, CRP, leukocyte amount, febrile, cough are also considered as symptoms to make an actual pneumonia diagnosis), but also I didn't have enough time and data resources to test it with chest X-rays from another sources other than these datasets. For example a patient with heart failure was detected positive in one of the test cases. There is a possibility of overfitting too. I'm uploading this repo to remind me later in the future what I was doing during Covid-19 time in April 2020.
