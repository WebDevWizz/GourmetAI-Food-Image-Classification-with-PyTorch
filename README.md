# GourmetAI-Food-Image-Classification-with-PyTorch

This project was developed for **GourmetAI Inc.** (as part of an exercise for my Master in AI Engineering with ProfessionAI), a leading company in the food-tech sector, with the goal of building an advanced **food image classification** system using modern **Deep Learning** and **Transfer Learning** techniques.

The entire workflow follows the project specification, including:
- Data augmentation strategies  
- Dataset preparation  
- Model selection  
- Transfer learning and fine-tuning  
- Hyperparameter optimization  
- Validation and regularization  
- Final evaluation on the test set  

---

## 🎯 Project Overview

The objective of this project is to improve the accuracy and efficiency of food image recognition systems by developing a robust and generalizable classification model using this **Food Classification dataset**: https://proai-datasets.s3.eu-west-3.amazonaws.com/dataset_food_classification.zip.

By implementing a deep learning–based solution, GourmetAI aims to:
- Improve user experience through fast and accurate predictions  
- Automate image categorization to optimize operational workflows  
- Promote technological innovation in the food-tech domain  
- Strengthen its competitive position through advanced AI solutions  

---

## 🧰 Technologies Used

- Python  
- PyTorch  
- Torchvision  
- NumPy & Pandas  
- Matplotlib / Seaborn  
- Google Colab (CPU / GPU)

---

## 🍳 Dataset

The project uses the **Food Classification dataset**, organized into three subsets:

- **Training set**  
- **Validation set**  
- **Test set**  

To improve the model's generalization, several **data augmentation techniques** were applied, including:

- Random horizontal flip  
- Random rotations  
- Color jitter  
- Resize and normalization (ImageNet style)

These techniques help enrich the dataset and prevent overfitting.

---

## 🧪 Model Architecture

The core model is a **ResNet50** pre-trained on ImageNet, used through **Transfer Learning**.

### 🔒 Frozen Backbone
All convolutional layers (feature extractor) were frozen to:
- Reduce training time  
- Prevent overfitting  
- Lower memory consumption  

### 🧱 Custom Classifier Head
The original ResNet50 classifier was replaced with:

```python
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 14)
)
```
This head is specifically designed to adapt the model to the 14 food categories.

---

## ⚙️ Training Procedure

The model was trained using the following setup:

- **Loss function:** CrossEntropyLoss  
- **Optimizer:** Adam (lr = 0.001)  
- **Batch size:** 128  
- **Epochs:** 30  
- **Regularization:** Dropout + Early Stopping (patience = 5)

During training, the following metrics were logged:

- Training loss  
- Validation loss  
- Validation accuracy  

All logs are automatically saved into a `.json` file for further analysis.

---

## 🧵 Hyperparameter Tuning & Regularization

To improve performance and avoid overfitting, the following techniques were applied:

- Dropout  
- Early stopping  
- Validation-driven hyperparameter selection  
- Learning rate adjustment (when needed)

---

## 🧾 Results

The final model was evaluated on the **test set**, and you can see the results at the end of the project.

The evaluation includes:

- Final accuracy  
- Confusion matrix  
- Per-class metrics  
- Training curves (loss and accuracy)

---

## 👨‍💻 Author
Developed by **Riccardo Ciullini**.
