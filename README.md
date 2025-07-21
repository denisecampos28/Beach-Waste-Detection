<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/0584c25c-6ea3-4dca-9a76-715171876dd8" />

# Beach-Waste-Detection

## Overview
This repository contains a computer vision project that uses deep learning to automatically classify beach images as either containing visible waste or being clean. The goal is to support environmental monitoring and cleanup efforts through scalable, AI-assisted detection of litter in beach environments.

## Summary of Workdone
This project classifies beach photos into two categories: Clean and Dirt (i.e., visibly polluted). A dataset of labeled images was used to train and evaluate a binary image classifier based on MobileNetV2, a pre-trained convolutional neural network architecture.

Two versions of the model were developed:
 - **Model v1:** A straightforward MobileNetV2 classifier using a single dense output layer.
 - **Model v2:** An improved version that includes an additional dense layer and dropout to improve generalization.

Model v1 achieved the most balanced and reliable performance. Model v2 had higher validation accuracy but slightly less stable learning behavior.

Final Evaluation For Model v1
Accuracy: 97%
F1 Score: 0.97
Model Stability: No signs of overfitting or underfitting

### Data
Type: Image classification dataset
Name: Plastic on Sand Image Classification

Classes
  -Clean: Beach images without visible trash
  -Dirt: Beach images with visible waste

Total Images Used: 152 (76 clean, 76 dirt)

Train/Validation Split: ~80/20

At the beginning of this project, I used a different beach waste dataset from Kaggle. However, during manual inspection, I noticed that many images labeled as “clean” actually contained visible trash. To address this, I manually reviewed and reorganized the images in both the “clean” and “dirty” folders, assuming only a few were mislabeled. Unfortunately, I discovered that the majority of images were in fact dirty, which resulted in a significant class imbalance.

I considered generating synthetic clean images to rebalance the dataset, but ultimately decided it didn’t make sense to use high-level image generation tools to support a lower-level classification task. Instead, I searched for a better dataset and eventually found the one used in this final version of the project, also hosted on Kaggle.

#### Preprocessing
To prepare the data for modeling, several cleaning and transformation steps were performed: 
  -Image resizing to 224x224 (which is MobileNetV2 input)
  -Rescaling pixel values to [0, 1]
  -Data augmentation (random flips, zoom, rotation)
  -Directory-based loading with image_dataset_from_directory()

#### Training
Model Architecture: MobileNetV2  with a custom classification head
Loss Function: Binary cross-entropy
Optimizer: Adam
Epochs: 20
Early Stopping: Enabled with patience of 4
Checkpoints: Best model saved during training

#### Performance

Model v1:
  -More stable loss/accuracy curves
  -High precision and recall for both classes
  -No signs of overfitting

Model v2:
  -Higher validation accuracy (93.3%)
  -Slight signs of overfitting on training accuracy
  -Not selected as final model due to slightly higher variance















  
