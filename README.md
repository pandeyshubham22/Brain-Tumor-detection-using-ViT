# 🧠 Brain Tumor MRI Classifier using Vision Transformer (ViT)

This project uses a pretrained Vision Transformer (ViT) model to classify brain MRI scans into different tumor types. It leverages PyTorch, TIMM, and Gradio to provide both a training pipeline and an interactive web interface for real-time predictions.

---

## 📁 Dataset

- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**: Glioma Tumor, Meningioma Tumor, No Tumor, Pituitary Tumor
- **Format**: The dataset is structured into `Training/` and `Testing/` folders.

---

## ⚙️ Features

- ✅ Fine-tuned ViT (`vit_base_patch16_224`) from the TIMM library  
- ✅ Mixed precision training with `torch.cuda.amp`  
- ✅ Gradio-powered web UI for live inference  
- ✅ Auto-download dataset using `kagglehub`  
- ✅ Real-time accuracy, loss tracking with `tqdm`  

---

## 🧪 Installation

```bash
pip install timm kagglehub ipywidgets matplotlib gradio

