# Week 3 Report – CIFAR-10 Classification  

  
**Group Members:** Aditya Goel, Boppudi Sai Chaitanya, Priyanshu Pandey 
**Date:** June 9, 2025  

---

## 1. Introduction  
This report summarizes the results of training multiple image cassification models on the CIFAR-10 dataset. The models range from simple fully-connected neural networks to deep convolutional architectures like ResNet-150. The primary objective was to compare their performance, understand their capacity to generalize, and draw insights from the training process.

---

## 2. Dataset  

**Dataset Used:** CIFAR-10 (60,000 32×32 RGB images in 10 classes)  
- **Training Set:** 50,000  
- **Test Set:** 10,000  
- **Validation Split:** 10% from training set (5,000 images)  

**Preprocessing Steps:**  
- Normalization to mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)
- For better performance in VGG16 and VGG19, imagenet mean and variance was used to normalize the dataset
- Data Augmentation(For Training Data):
  - Random Horizontal Flip  
  - Random Crop (with padding)  
  - Random Rotation  

---
# Approach

- The preprocessing technique was shared among the group members, who then individually worked on different models.
- The main idea is to get the best accuracy among different models and using transfer learning for defined architectures




---
## 3. Models Implemented  
| Model Type | Name       | Architecture Details                          |
| ---------- | ---------- | --------------------------------------------- |
| ANN        | 3-Layer FC | Input →  Output(10) *edit this******          |
| Basic CNN  | CNN-3      | Conv → Pool → Conv → Pool → FCs               |
| Classical  | LeNet      | LeNet-5 Architecture                          |
| Classical  | AlexNet    | torchvision.models.alexnet(pretrained=True)   |
| Classical  | VGG16      | torchvision.models.vgg16(pretrained=True)     |
| Classical  | VGG19      | torchvision.models.vgg19(pretrained=True)     |
| Classical  | ResNet-50  | torchvision.models.resnet50(pretrained=True)  |
| Classical  | ResNet-150 | torchvision.models.resnet152(pretrained=True) |

---

## 4. Evaluation Metrics  

Each model was evaluated on the test set using:  
- **Accuracy**  
- **Precision, Recall, F1-Score**  
- **Confusion Matrix**  
- **ROC-AUC (One-vs-Rest strategy)**

---

## 5. Results Summary  

| Model      | Trained Params | Accuracy (%) | F1-Score | Training Time | Generalization Gap |
| ---------- | -------------- | ------------ | -------- | ------------- | ------------------ |
| ANN        | ~1M            |              |          |               |                    |
| Basic CNN  | ~1.2M          |              |          |               |                    |
| LeNet      | ~60K           | 55.3         | 0.59     | 10 mins       | Low                |
| AlexNet    | ~61M           |              |          |               |                    |
| VGG16      | ~138M          | 88.3         | 88.3     | 1 hr          | Moderate           |
| VGG19      | ~144M          | 91.0         | 91.1     | 1 hr          | Low                |
| ResNet-50  | ~25M           |              |          |               | Low                |
| ResNet-150 | ~60M           |              |          |               | Low                |

*Note: Replace the numbers with your actual results.*

---

# Models

## **LeNet**

### Basic Model Overview:

The model was designed for digits recognition and is not expected to provide very good results in our case. Therefore transfer learning here, will only result in lesser accuracy

**![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfI02oJXugfWWFr6kHWIhilIZeZEUu3U-YshkHKpkvXHcDfOcnwaG7QK7SP_8NskXio6SkZuED7nfJQu-A-losRZ_wuYf6uGXDy01YawgtPoQtnMtCwPrIo8Jb3CCs6EuNT6QcFyQ?key=LzzI9D8axshoLbbyeWIidw)**

### Accuracy Achieved: 55.3%

### Confusion Matrix 
 
![[Pasted image 20250609181036.png]]

### AUC Curves



## **VGG16**

### Basic Model Overview:

I used the model with pretrained convolutuion parameters, and trained new parameters for the classifier part of the model. This made the training time very large, but improved accuracy.



### Accuracy Achieved: 88.3%

### Confusion Matrix 

![[Pasted image 20250609181045.png]]

### AUC Curves

## **VGG19**

### Basic Model Overview:

I used the model with pretrained convolutuion parameters, and trained new parameters for the classifier part of the model.

For this model, i also tried the approach of using the pretrained weights of the classifier too and it resulted in an accuracy of 83%

`vgg19 = models.vgg19(pretrained=True)
`for param in vgg19.features.parameters():
`param.requires_grad = False
`for param in vgg19.classifier.parameters():
`param.r`equires_grad = False`
`vgg19.classifier[6] = nn.Linear(4096, 10)
`for param in vgg19.classifier[6].parameters():
`param.requires_grad = True`
### Accuracy Achieved: 91.1%

### Confusion Matrix 

![[Pasted image 20250609181154.png]]

AUC

---

## 7. Discussion  
### Overfitting and Underfitting  
- ANN clearly underfits due to lack of spatial information modeling.  
- Basic CNN shows moderate learning but saturates early.  
- Deeper models (VGG/ResNet) show better generalization but at the cost of high training time.



---

## 8. Challenges and Solutions  


Challenge 1: Long training time for VGG

Solution Applied: Used pretrained weights and fine-tuned last layers.

Challenge 2: GPU Memory Limit
Solution Applied: Reduced batch size and cleared cache during training

---

## 9. Conclusion  
This assignment reinforced practical knowledge of training and evaluating deep learning models. While simpler models offer faster training, deeper pretrained models significantly outperform them on complex datasets like CIFAR-10. The experiment also highlighted the importance of good preprocessing, regularization, and monitoring to build robust models.

---
