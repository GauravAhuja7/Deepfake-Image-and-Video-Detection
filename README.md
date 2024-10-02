# Image Deepfake Detection

## Overview
This project aims to implement and improve deepfake detection methods with a high level of generalization. It addresses images generated using General Adversarial Networks (GANs), Diffusion Models, and other emerging generative techniques by identifying perceptual cues left by these methods. The approaches involve using pretrained models for feature extraction and traditional classifiers for detecting fake images.

## Approaches

### Approach 1: Learning Low-Level Cues from Image Generators
In this approach, we hypothesized that image generators leave behind low-level cues such as colors, textures, and brightness. To classify real vs fake images, two pretrained models—ResNet50 (Keras) and ResNet18 (PyTorch)—were fine-tuned with a binary classification layer.

### Key Implementation Details:

- **Dataset:** ImageNet images for training.  
- **Model Architectures:** ResNet50 and ResNet18 with binary classification layers.

### Training Configuration:
- **Batch size:** 128  
- **Learning rate:** 0.00001  
- **Optimizer:** SGD with momentum of 0.9  
- **Loss function:** Cross Entropy Loss  
- **Image Resolution:** 256 x 256  
- **Epochs:** 80  
- **Image Augmentation:** Applied in ResNet50 implementation.  

### Issues Identified:
- The models failed to generalize to unseen image generation methods.  
- Overfitting occurred when trained on a specific technique.  
- Training was time-consuming and computationally expensive.

### Approach 2: Use a generalized backbone for feature maps and then use Classifiers
Given that Approach 1 showed that training a neural network on a single generation technique limits its effectiveness to that specific method, a more flexible strategy was considered. Instead of focusing on training a neural network solely to classify real or fake images based on a specific generation technique, a better approach is to utilize a pretrained backbone to extract feature maps. These feature maps can then be fed into classifiers like linear probes, KNN, SVMs, and others for the classification task. This method helps improve generalization across different image generation methods.

### Section 1: Implementaion details:

### Datasets:
- **Laion vs LDM100:** 1000 real, 1000 fake
- **ImageNet vs LDM200:** 1000 real, 1000 fake
- **BigGAN Real vs BigGAN Fake:** 2000 real, 2000 fake

### Transformations:
1. No change
2. Gaussian Blur
3. Jitter
4. Gaussian Blur + Jitter

### Backbone Models:
- **DINO ViT-B/16**
- **DINO ResNet50**
- **CLIP ViT-B/16**

### Classifiers:
- Bagging Classifier
- Decision Tree
- Random Forest
- Linear Discriminant Analysis (LDA)
- KNN (1 neighbor, 3 neighbors, 5 neighbors)
- Support Vector Machine (SVM)
- Naive Bayes
- Gradient Boosting
- Linear Probe

### Dimensionality Reduction:
- No reduction
- Principal Component Analysis (PCA)
- Autoencoding
