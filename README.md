# Pneumonia Classifier:

**Please do not use this software as a medical diagnosis tool as it lacks clinical study. Please read the important notice below before using it for any purposes.**

After seeing so many pneumonia image classifier attempts flooding the internet at the end of March 2020, I decided to make an attempt to build my own image classifier trying to detect pneumonia patients by running CNNs on chest X-rays. 

## Datasets:
  1. https://www.kaggle.com/paultimothymooney/detecting-pneumonia-in-x-ray-images/data?
  2. Covid-19 dataset Chest X-ray dataset from https://www.pyimagesearch.com/ 

Tensorflow and keras are used as deep learning framework. Additionally a Flask server and basic website pages are added to repo in order to show how to host a basic deep neural network model in case of demo requirement. 

Model is trained with two classes: 
  1. Normal chest X-rays 
  2. Chest X-rays of Pneumonia patients (such as Viral Infection, Covid-19) 
  
Then I ran a simple CNN network without using transfer learning. 

## Important Notice:
The repo is created only for demonstration purposes. Please do not use the model in this repo as a medical diagnosis tool. Not only I learned chest X-ray images by themselves are not sufficient for doctors to diagnose a pneumonia case at a patient (Normally patient's history, CRP, leukocyte amount, febrile, cough are also considered as symptoms to make an actual pneumonia diagnosis), but an actual clinical study was not made on this software. It is already known that there are some false positive problems. For example a patient with heart failure was detected positive in one of the test cases. I'm uploading this repo to remind me later in the future what I was doing during Covid-19 time in April 2020.
