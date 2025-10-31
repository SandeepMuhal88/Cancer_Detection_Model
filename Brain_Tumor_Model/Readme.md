# 🧠 Brain Tumor Classification Model

This project classifies MRI images into brain tumor categories using **Convolutional Neural Networks (CNNs)** built with **PyTorch**.

---

## 📘 Project Overview

The goal of this project is to automatically detect and classify brain tumors from MRI images.  
The model has been trained on a dataset of MRI scans with multiple classes (e.g., **Glioma**, **Meningioma**, **Pituitary**, **No Tumor**).

---

## ⚙️ Installation & Setup

### 🧩 Requirements
- Python 3.10 (Link: https://www.python.org/downloads/release/python-31011/) 
- Recommended IDE: VS Code / PyCharm  
- GPU support (optional but recommended)

### 📦 Step 1: Clone or Extract
Download and extract the ZIP file:
```bash
Brain_Tumor_Model.zip

```

### 📦 Step 2: Open the folder:
```bash
cd Brain_Tumor_Model
```


### 🧰 Step 3: Create Virtual Environment
Windows:
```bash
python -m venv .venv
```

Mac/Linux:
```bash
python3 -m venv .venv
```
### Activate it:

Windows: .\.venv\Scripts\activate

Mac/Linux: source .venv/bin/activate

### 📥 Step 4: Install Dependencies

Install all required libraries:
```bash
pip install -r requirements.txt
```
### 🧾 File Structure
```bash
Brain_Tumor_Model/
│
├── Model.ipynb               # Model training script
├── model.pth               # Saved trained model
├── Data/ 
        |--Train/Glioma/
        |--Train/Meningioma/
        |--Train/Pituitary/
        |--Train/notumor/
        |--Test/Glioma/
        |--Test/Meningioma/
        |--Test/Pituitary/
        |--Test/notumor/
        |                  # Dataset folder
├── requirements.txt         # List of dependencies
└── README.md                # Documentation
```

### 🧪 Model Details

Architecture: CNN with multiple Conv2D and MaxPool layers

Framework: PyTorch

Optimizer: Adam

Loss Function: CrossEntropyLoss

Accuracy: ~[add your result]% on test data