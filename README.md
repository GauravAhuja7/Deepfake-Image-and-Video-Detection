
# Image Deepfake Detection
## Overview
This project aims to implement and improve deepfake detection methods with a high level of generalization. It addresses images generated using General Adversarial Networks (GANs), Diffusion Models, and other emerging generative techniques by identifying perceptual cues left by these methods. The approaches involve using pretrained models for feature extraction and traditional classifiers for detecting fake images.

## Approaches
### Approach 1: Learning Low-Level Cues from Image Generators
In this approach, we hypothesized that image generators leave behind low-level cues such as colors, textures, and brightness. To classify real vs fake images, two pretrained models—ResNet50 (Keras) and ResNet18 (PyTorch)—were fine-tuned with a binary classification layer.

### Key Implementation Details:

**Dataset:** ImageNet images for training.
**Model Architectures:** ResNet50 and ResNet18 with binary classification layers.
### Training Configuration:
**Batch size:** 128
**Learning rate:** 0.00001
**Optimizer:** SGD with momentum of 0.9
**Loss function: **Cross Entropy Loss
**Image Resolution:** 256 x 256
**Epochs:** 80
**Image Augmentation:** Applied in ResNet50 implementation.
**Issues Identified:**
The models failed to generalize to unseen image generation methods.
Overfitting occurred when trained on a specific technique.
Training was time-consuming and computationally expensive.
