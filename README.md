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


Final Evaluation For Model v1:
 - Accuracy: 97%
 - F1 Score: 0.97
 - Model Stability: No signs of overfitting or underfitting

### Data
Type: Image dataset (.jpg images)

Name: Plastic on Sand Image Classification

Source: Public dataset hosted on Kaggle

Total Images Used: 152 (76 clean, 76 dirt)

Train/Validation Split: ~80/20

Classes:
  - Clean: Beach images without visible trash
  - Dirt: Beach images with visible waste

At the beginning of this project, I used a different beach waste dataset from Kaggle. However, during manual inspection, I noticed that many images labeled as “clean” actually contained visible trash. To address this, I manually reviewed and reorganized the images in both the “clean” and “dirty” folders, assuming only a few were mislabeled. Unfortunately, I discovered that the majority of images were in fact dirty, which resulted in a significant class imbalance.

I considered generating synthetic clean images to rebalance the dataset, but ultimately decided it didn’t make sense to use high-level image generation tools to support a lower-level classification task. Instead, I searched for a better dataset and eventually found the one used in this final version of the project, also hosted on Kaggle.

Below is an image of the dataset that I used with examples of each class. 

<img width="1164" height="390" alt="cvd" src="https://github.com/user-attachments/assets/8d904de1-f6a2-4909-a03e-16e56d8e072e" />


### Preprocessing
To prepare the data for modeling, several cleaning and transformation steps were performed: 
  - Image resizing to 224x224 (which is MobileNetV2 input)
  - Rescaling pixel values to [0, 1]
  - Data augmentation (random flips, zoom, rotation)
  - Directory-based loading with image_dataset_from_directory()

### Training

Software: GoogleCollab, Pandas, Numpy, Matplotlib, Tensorflow, Keras
Hardware: Trained on a windows PC
Model Architecture: MobileNetV2  with a custom classification head
Loss Function: Binary cross-entropy
Optimizer: Adam
Epochs: 20
Early Stopping: Enabled with patience of 4
Checkpoints: Best model saved during training

**Model v1:**
 - no dropout
 - global average pooling
 - dense layer with 1 unit and sigmoid activation
 - trained for about 10 epochs

**Model v2:**
 - dropout layer (0.3)
 - global average pooling
 - dense layer (128 units, ReLU activation)
 - final dense layer with 1 unit and sigmoid activation


### Performance Comparison

**Model v1:**
  - More stable loss/accuracy curves
  - High precision and recall for both classes
  - No signs of overfitting

<img width="1189" height="490" alt="m1" src="https://github.com/user-attachments/assets/58618073-38dc-4cb2-889b-002eed5da0a1" />


<img width="435" height="393" alt="cm1" src="https://github.com/user-attachments/assets/5f537c72-4efa-4bda-af84-5e4f9dd01d37" />


<img width="432" height="167" alt="mc1" src="https://github.com/user-attachments/assets/603d4f48-817a-435b-8c57-1bae60d38c7a" />






**Model v2:**
  - Higher validation accuracy (93.3%)
  - Slight signs of overfitting on training accuracy
  - Not selected as final model due to slightly higher variance


<img width="1189" height="490" alt="m2" src="https://github.com/user-attachments/assets/6042f114-5198-405f-8beb-e255ee9fe8d4" />


<img width="435" height="393" alt="cm2" src="https://github.com/user-attachments/assets/742f096d-0a5a-4821-8852-7110263f8bce" />


<img width="434" height="165" alt="cmm2" src="https://github.com/user-attachments/assets/0d138351-e6d5-4e65-9512-ef9923f228ad" />



## Conclusion 

This project successfully built a binary image classification model that distinguishes between clean and dirty beaches using deep learning. The final model achieved strong performance with over 97% accuracy and balanced precision/recall between the two classes. Through data cleaning,  and experimenting with different model heads, I was able to achieve great performance. The project demonstrated the power of transfer learning for small, real-world datasets and highlighted the importance of careful data curation in computer vision.

## Future Work 

To further improve this model, I’d like to:
  - Collect more high-quality clean beach images to balance the dataset and avoid overfitting.
  - Try different base models (EfficientNet, ResNet50) and compare their performance.
  - Test the model on real-world photos taken from various beaches to evaluate generalizability.

## How to reproduce results 

1.) Download the Data from Kaggle 

2.) Download all the required libraries 

3.) Run the Notebook

4.) Compare Models 


## Overview of files in repository 
 - beachwaste_p2.ipynb: this file contains all the clean, preprocessing, model creation, and evaluation
 - feas_and_proto (1).ipynb: this contains my orignal cleaning, preprocessing, model and evaluation with the old, bad dataset
 - beach-waste-detection-dataset: zip file of the old bad dataset
 - Plastic on beach: dataset used


## Software Setup 

| Package        | Version (or Minimum) |
|----------------|----------------------|
| TensorFlow     | >= 2.11              |
| Keras          | Built-in             |
| Pandas         | >= 1.5               |
| NumPy          | >= 1.22              |
| Matplotlib     | >= 3.5               |
| Google Colab   | (Used for training)  |


## Data Download 

https://www.kaggle.com/datasets/rogeriovaz/plastic-on-sand-image-classification





  
