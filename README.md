# Defect Detection System for Manufacturing

## Project Overview
This project is a machine learning-based defect detection system for manufacturing. It uses a Convolutional Neural Network (CNN) and the PyTorch framework to automatically classify products as "Good" or "Defective" based on visual inspection. The system was developed as a proof-of-concept to demonstrate a robust pipeline for quality control in manufacturing.

## Problem Solved
The system automatically detects defects (e.g., scratches, dents, misprints) in products, which is a critical task for quality control in manufacturing. The model is trained to be robust in identifying defects even in a highly imbalanced dataset, a common challenge in real-world manufacturing scenarios.

## Key Features
- **Transfer Learning with ResNet-50:** The model leverages a pre-trained ResNet-50 model on the ImageNet dataset, allowing for faster training and superior feature extraction on a limited dataset.
- **Class Imbalance Handling:** The training process uses a weighted cross-entropy loss function to address the significant class imbalance between "Good" and "Defective" products. This ensures the model learns to prioritize the detection of rare defects.
- **Comprehensive Evaluation:** The model's final performance is thoroughly evaluated on a held-out test set using a confusion matrix, precision, recall, and F1-score to provide a clear and honest picture of its effectiveness.
- **Scalability:** The pipeline is designed to be scalable and can be retrained on new datasets to adapt to different product types or defect categories.

## Model Performance
The fine-tuned model achieved the following performance on the test set:

- **Overall Test Accuracy**: 0.7857

- **Confusion Matrix**:

```bash
[13 11]
[ 7 53]
```
**Metrics for Defective Class**:

- **Precision**: 0.6500

- **Recall**: 0.5417

- **F1-score**: 0.5909

## Project Stack
- **Framework:** PyTorch (>= 2.0), torchvision
- **Libraries:** scikit-learn, numpy, Pillow
- **Data:** MVTec Anomaly Detection (AD) Dataset. The project combines multiple categories from this dataset to create a more generalized defect classifier.
- **Hardware:** The project is designed to be trained efficiently on a GPU, with all development conducted on Kaggle's free GPU notebooks.

---

## **Getting Started: Local Setup**

This section guides you through setting up the project on your local machine to replicate the results.

### **1. Prerequisites**
- Python (>= 3.9)
- A GPU with CUDA support is highly recommended for training.

### **2. Virtual Environment and Dependencies**
First, create a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```
### **3. Install the required packages**
Run the following command in the terminal to successfully install the libraries.

```bash
pip install -r requirements.txt
```

### **4. Dataset Setup and Organization**

The project uses a custom-organized dataset derived from the MVTec AD dataset. The `data_organizer.py` script is provided to automatically organize this dataset.

#### 4.1

Download the MVTec AD dataset from the official website `` and place all category folders (e.g., bottle, cable, zipper) inside a single `root` directory.

#### 4.2

Open `data_organizer.py` and modify the `MVTEC_ROOT_DIR` and `CATEGORIES` variables to match your local setup.

#### Example Configuration:

```bash
# --- Configuration ---
MVTEC_ROOT_DIR = 'path/to/your/mvtec_ad_folder'  # e.g., 'C:/Users/YourName/Datasets/mvtec_ad'
CATEGORIES = ['bottle', 'cable', 'capsule', 'hazelnut'] # A list of categories to include
OUTPUT_ROOT_DIR = 'my_multi_category_defect_dataset'
```
#### 4.3
Run the script to organize the dataset. This will create a `my_multi_category_defect_dataset` folder with `train`, `val`, and `test` splits.

```bash
python data_organizer.py
```

### **5. Model Training and Evaluation**
The full training and evaluation pipeline is contained in the mc-defect-classification-using-resnet-50.ipynb Jupyter Notebook.

**Important**: Before running, open the notebook and update the `DATA_DIR` variable to point to your local dataset path:

```bash
# Change this line in the notebook to your local path
DATA_DIR = 'path/to/your/my_multi_category_defect_dataset'
```

### **Contributions**
Contributions are welcome! Feel free to fork this repository, make improvements, and submit pull requests. Help us enhance the project for the benefit of the open-source community.