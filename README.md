# Facial Emotion Recognition (PyTorch)

A deep learning project for **facial emotion classification** using a custom VGG-style CNN trained on grayscale images. Built with **PyTorch**, this project includes data preprocessing, augmentation, model architecture, and training pipeline.

---
## Dataset - FER-2013

Kaggle dataset link: https://www.kaggle.com/datasets/msambare/fer2013

The data consists of 48x48 pixel grayscale images of faces. 
The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.
The training set consists of 28,709 examples and the public test set consists of 3,589 examples.
(There is one differense between the original dataset and the dataset that was used - images of the "disgust" class were deleted from the test and train dataset, so in the code you can see that the "NUM_CLASSES   = 6", not 7)

---

## Data Augmentation

To improve generalization and reduce overfitting, the training pipeline applies a set of data augmentation techniques using Torchvision transforms.

- **Resize** (48×48): ensures all images have a consistent input size.
- **RandomAffine** (p = 0.5)
  Applies random:
  - Rotation (±10°)
  - Translation (up to 20%)
  - Scaling (0.8 – 1.2)
- **RandomCrop** (40×40):  introduces spatial variation and forces robustness to positioning.
- **RandomHorizontalFlip** (p = 0.5): improves invariance to left-right orientation.
- **ToTensor**: converts images to PyTorch tensors.
- **Normalize** ([0.5], [0.5]): scales pixel values to a standardized range.
- **Random Erasing** (p = 0.5): randomly removes a rectangular region from the image to simulate occlusion.

---

## Model Architecture

This project uses a custom convolutional neural network (CNN) inspired by the VGG architecture, referred to as SmallVGG:

- Conv + BatchNorm + ReLU blocks  
- MaxPooling layers  
- AdaptiveAvgPool  
- Fully connected classifier with dropout

### Classifier
Fully connected layers:
- Flatten
- Linear → ReLU → Dropout
- Linear → ReLU → Dropout
- Final Linear layer (outputs class scores)

### Activation Function

The model uses the ReLU (Rectified Linear Unit) activation function throughout the network.

ReLU - Pytorch 2.10 documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html

### Loss Function

The loss function is CrossEntropyLoss.

CrossEntropyLoss - Pytorch documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

### Optimizer: 

SGD with Nesterov momentum:

### Learning Rate Scheduler

The model uses ReduceLROnPlateau: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

### Input

- Shape: `1 x 40 x 40` (grayscale)

### Output

- `NUM_CLASSES` emotion categories

---

## Results:

<img width="1903" height="738" alt="image" src="https://github.com/user-attachments/assets/fc502f48-ae92-466d-a3ae-ad9f8b8c36d2" />

## Best Training Results

| Metric              | Value   |
|--------------------|--------|
| Train Loss         | 0.7566 |
| Train Accuracy     | 71.63% |
| Validation Loss    | 0.8930 |
| Validation Accuracy| 69.18% |

---

- ## 🗂️ Project Structure

- `dataset.py` → Handles dataset loading and data augmentation  
- `initing.py` → Contains configuration, imports, and global settings  
- `model.py` → Defines the CNN architecture (SmallVGG)  
- `train.py` → Training loop (not included here)  
- `main.py` → Main entry point of the project  
- `outputs/` → Directory for saved models, logs, and results
