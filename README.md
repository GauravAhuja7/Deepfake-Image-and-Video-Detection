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

## Findings:

**1. Impact of using transforms before training and after training:**
![image](https://github.com/user-attachments/assets/e8a9bab6-9809-426b-8857-714da4529f6f)

**NOTE:** Goodness Factor was calculated by subtracting min(accuracy) from all data points for each line. Each line represents a different dataset-backbone model combination. The names have been omitted for clarity

**Inference 1:** Adding No transform while training is better. But adding transform 2 also shows good results. Dealing with Gaussian Blur might cause issues in accuracy (around 0.05 on average)

**Inference 2:** On adding no transform while training but adding transformation while testing it was found that the models were robust to those transforms. This means that if the user were to do some edits like adding jitter or gaussian blur (while compressing it) to the images the models would still be very accurate

**2. Impact of Dimensionality Reduction:**
![image](https://github.com/user-attachments/assets/84b4d0ec-62ba-4944-9d2e-8dc668a076cf)
**Inference:** In many cases no dimensionality reduction is the best choice. But in some cases autoencoding performs better than no reduction or PCA

**3. Impact of classifier used:**
![image](https://github.com/user-attachments/assets/ae917e0c-30d6-45dd-80f0-ac5bb97fdd98)

**Inference:** SVM, Linear Probing and Linear Discriminant Analysis seem to be good classifiers

**4. Impact of backbone model used:**
![image](https://github.com/user-attachments/assets/b0cc47a3-6789-4ac0-bced-ca1a24ad1a68)

**Inference:** CLIP ViT-B/16 is the best backbone over all cases, followed by DINO ResNET50, followed by DINO ViT-b/16

## Section 3: Going Beyond ...

After this I tried to split the load for feature transforms over multiple backbones instead of just one backbone, I made the following combinations:
DINO ViT-B/16 and DINO ResNET50 and CLIP ViT-B/16
DINO ViT-B/16 and DINO ResNET50
CLIP ViT-B/16 and DINO ResNET50
CLIP ViT-B/16 and DINO ViT-B/16
I then trained each model over the entire dataset and took the best classifier for each. I used randomised jitter (p=0.5) and guassian blur (p=0.5).

![image](https://github.com/user-attachments/assets/9246d883-633d-42ba-96cd-7a354a33d9f3)

**Inference:** This showed that using a combination was still not able to beat the previous best model, i.e. CLIP ViT-B/16

Lastly, tried to test the models across datasets (Training on dataset A and testing on dataset B). This was to check for generalization of the models
![image](https://github.com/user-attachments/assets/47665eb0-c7ac-4f82-9d66-5e6ad1ad4932)

## Section 4: Results and Conclusion:

## Results:
The best accuracy achieved was with no transformation, CLIP ViT-B/16 backbone model and Linear Discriminant Analysis with no reduction as the classifier. The test accuracy was 98.1875% with train accuracy 98.75% when trained and tested over the combined datasets.  
On testing across datasets, the model with CLIP ViT-B/16 backbone, Support Vector Machine classifier, autoencoder for dimensionality, and randomized jitter and blur for transformation gave the best results.

### Best Model for GANs:
It was generalized best when trained on **BigGAN** and tested on other datasets like **Laion, ImageNet, LDM100,** and **LDM200**.  
- When trained and tested on **BigGAN**, the accuracy was 98.875%.  
- When tested on **Laion** and **LDM100**, accuracy was 79.1%.  
- When tested on **ImageNet** and **LDM200**, accuracy was 81.1%.

### Best Model for Diffusion Models:
It generalized well when trained on **ImageNet vs LDM200** and tested across other datasets like **Laion, BigGAN,** and **LDM100**.  
- When trained and tested on **ImageNet vs LDM200**, the accuracy was 98.5%.  
- When tested on **Laion** and **LDM100**, accuracy was 94.44%.  
- When tested on **BigGAN Real vs BigGAN Fake**, accuracy was 72.1%.

## Conclusion:
The best results are achieved when using a generalized backbone like **CLIP ViT-B/16** with **SVM** or **LDA** as the classifier. A well-trained autoencoder can also significantly enhance performance.  
While good accuracy is achieved within different diffusion models or various GAN-based generation techniques, obtaining high accuracy for images generated using unknown techniques remains a challenge.

## Future Scope:
- Investigate the use of **text embeddings** to improve classification by leveraging semantic information within images.
- Refine the **autoencoder's hyperparameters** for more effective dimensionality reduction.
- Train models on **larger datasets** to enhance generalization.
- Explore additional **backbone models** to improve performance.
- Experiment with building neural networks that combine multiple **backbones** to extract feature maps for improved classification.


