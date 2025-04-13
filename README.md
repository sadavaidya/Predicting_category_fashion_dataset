# Predicting Category | Fashion Dataset

This repository showcases the application of **Self-Supervised Learning (SSL)** using **Vision Transformers (ViT)** to classify category types in fashion product images. The work demonstrates how powerful pre-trained models like **DINOv2** can be leveraged to extract meaningful visual features without needing labeled data for feature learning.

## 🔍 Project Overview

**Objective:**  
Predict the **category** from product images in the [Fashion Product Images dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) using self-supervised ViT embeddings.

**Key Highlights:**
- Used **DINOv2 ViT-Small** (self-supervised) to generate image embeddings.
- Built a lightweight **MLP classifier** on top of the frozen embeddings.
- Achieved high accuracy with minimal computational cost and no manual annotations for feature extraction.
- Streamlined approach suitable for scalable fashion product tagging systems.

## 📁 Notebook of Interest

👉 [`Category_pred_ssl.ipynb`](./Category_pred_ssl.ipynb)  
This notebook contains the full pipeline:
- Data preprocessing  
- Feature extraction using DINOv2  
- Training & evaluating classifier  
- Visualizing predictions  

## 🧠 Why Self-Supervised Learning?

Self-supervised learning enables models to:
- Leverage **large volumes of unlabelled data** effectively  
- Learn **generalizable representations** that perform well across tasks  
- Reduce dependence on expensive manual labeling  

## 📦 Dependencies

- `torch`, `torchvision`  
- `scikit-learn`  
- `timm` (for DINOv2 models)  
- `PIL`, `matplotlib`, `pandas`, `numpy`  

Install using:

```bash
pip install torch torchvision timm scikit-learn pandas matplotlib
```
## 📌 Credits

- Dataset: Fashion Product Images by Param Aggarwal
- ViT Model: DINOv2 by Meta AI