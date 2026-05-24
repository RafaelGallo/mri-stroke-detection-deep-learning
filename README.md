# 🧠 MRI Stroke Detection — Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

![](https://github.com/RafaelGallo/mri-stroke-detection-deep-learning/blob/main/img/001.png?raw=true)

</div>

## 🏥 Business Problem

Stroke is a leading cause of death and long-term disability worldwide. Early and accurate diagnosis is critical — **every minute without treatment results in the loss of approximately 1.9 million neurons**. Hospitals depend on radiologists to manually analyze brain MRI scans, a process that is time-consuming and subject to human fatigue in high-demand clinical environments. This project develops an **automated deep learning triage system** capable of classifying brain MRI scans into three categories:

| Class | Description |
|---|---|
| 🔴 Haemorrhagic | Stroke caused by brain bleeding |
| 🟡 Ischemic | Stroke caused by blood vessel blockage |
| 🟢 Normal | No stroke detected |

## 📂 Dataset

| Property | Value |
|---|---|
| Source | [Brain Stroke MRI Images — Kaggle](https://www.kaggle.com/datasets/mitangshu11/brain-stroke-mri-images) |
| Directory | Stroke_classification (preprocessed, no text overlay) |
| Classes | Haemorrhagic, Ischemic, Normal |
| Total Images | ~750 |
| Split | 70% Train / 15% Val / 15% Test |

## 🔧 Solution Pipeline

```
Dataset
   │
   ├── EDA & Class Distribution
   │
   ├── Data Augmentation
   │
   ├── VGG16 Transfer Learning
   │     ├── Phase 1: Frozen Base
   │     └── Phase 2: Fine-tuning Block5
   │
   ├── Hyperparameter Tuning (Keras Tuner Hyperband)
   │
   ├── Evaluation
   │     ├── Classification Report
   │     ├── Confusion Matrix
   │     └── ROC Curve
   │
   ├── Explainability (Grad-CAM)
   │
   └── Generative Model (CGAN)
         ├── Generator
         └── Discriminator
```

## 🖼️ Data Augmentation

Augmented training samples showing different MRI modalities across all three classes.

![Augmented Training Samples](https://raw.githubusercontent.com/RafaelGallo/mri-stroke-detection-deep-learning/main/img/2.png)

## 📈 Training History

VGG16 training curves across Phase 1 (frozen base) and Phase 2 (fine-tuning block5).
The dashed line marks where fine-tuning begins.

![VGG16 Training History](https://raw.githubusercontent.com/RafaelGallo/mri-stroke-detection-deep-learning/main/img/3.png)

## 🤖 Models

### VGG16 Transfer Learning

| Parameter | Value |
|---|---|
| Base Model | VGG16 (ImageNet weights) |
| Input Size | 224x224x3 |
| Phase 1 LR | 1e-4 (frozen base) |
| Phase 2 LR | 1e-5 (fine-tuning block5) |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |

### Keras Tuner Hyperband

| Search Space | Values |
|---|---|
| Dense Units 1 | 256, 512, 1024 |
| Dense Units 2 | 128, 256, 512 |
| Dropout 1 | 0.2 → 0.6 |
| Dropout 2 | 0.2 → 0.5 |
| Learning Rate | 1e-3, 5e-4, 1e-4 |

### Conditional GAN (CGAN)

| Parameter | Value |
|---|---|
| Latent Dimension | 128 |
| Epochs | 3000 |
| Batch Size | 16 |
| Learning Rate | 2e-4 |
| Image Size | 64x64 (grayscale) |
| Optimizer | Adam (beta_1=0.5) |

## 📊 Results

### Confusion Matrix

Raw and normalized confusion matrices on the test set.

![Confusion Matrix](https://raw.githubusercontent.com/RafaelGallo/mri-stroke-detection-deep-learning/main/img/4.png)

### ROC Curve

ROC curves per class using one-vs-rest strategy.

![ROC Curve](https://raw.githubusercontent.com/RafaelGallo/mri-stroke-detection-deep-learning/main/img/5.png)

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Haemorrhagic | 1.00 | 0.86 | 0.92 | 28 |
| Ischemic | 0.83 | 1.00 | 0.91 | 5 |
| Normal | 0.95 | 1.00 | 0.98 | 60 |
| **Accuracy** | | | **0.96** | **93** |
| Macro Avg | 0.93 | 0.95 | 0.94 | 93 |
| Weighted Avg | 0.96 | 0.96 | 0.96 | 93 |

### ROC AUC

| Class | AUC |
|---|---|
| Haemorrhagic | 0.996 |
| Ischemic | 1.000 |
| Normal | 0.997 |

### Clinical Highlights

- ✅ **Zero missed Ischemic cases** (Recall = 1.00)
- ✅ **96% overall accuracy** on test set
- ✅ **Grad-CAM** activations align with anatomically relevant regions
- ✅ **CGAN** successfully generates class-conditioned synthetic MRIs

## 🔍 Batch Predictions

Model predictions on random test set images with confidence scores.

![Batch Predictions](https://raw.githubusercontent.com/RafaelGallo/mri-stroke-detection-deep-learning/main/img/6.png)

## 🔬 Explainability — Grad-CAM

Grad-CAM highlights the regions of the MRI that most influenced the model prediction,
providing visual evidence aligned with clinical knowledge.

![Grad-CAM Explainability](https://raw.githubusercontent.com/RafaelGallo/mri-stroke-detection-deep-learning/main/img/7.png)

- **Haemorrhagic** — activations over bleeding areas in brain tissue
- **Ischemic** — activations over diffusion-weighted imaging patterns
- **Normal** — distributed activation across healthy brain structure

## 🎨 Generative Model — CGAN

Synthetic brain MRI images generated by the CGAN conditioned on each stroke class.

![CGAN Synthetic MRI](https://raw.githubusercontent.com/RafaelGallo/mri-stroke-detection-deep-learning/main/img/8.png)

A dedicated notebook explores synthetic brain MRI generation using a Conditional GAN (CGAN),
producing realistic scans conditioned on the stroke class label to address data scarcity.

🔗 [MRI Brain Stroke — CGAN Notebook](https://www.kaggle.com/code/gallo33henrique/mri-brain-stroke-gan/edit)

## 📁 Project Structure

```
mri-stroke-detection-deep-learning/
│
├── img/
│   ├── 2.png    # augmented training samples
│   ├── 3.png    # training history
│   ├── 4.png    # confusion matrix
│   ├── 5.png    # ROC curve
│   ├── 6.png    # batch predictions
│   ├── 7.png    # Grad-CAM
│   └── 8.png    # CGAN synthetic images
│
├── notebooks/
│   ├── mri_brain_stroke_vgg16.ipynb
│   └── mri_brain_stroke_gan.ipynb
│
├── models/
│   ├── vgg16_brain_stroke.keras
│   ├── vgg16_brain_stroke_tuned.keras
│   └── cgan_generator_brain_mri.keras
│
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### Kaggle Notebooks

| Notebook | Link |
|---|---|
| VGG16 + Keras Tuner + Grad-CAM | [Open on Kaggle](https://www.kaggle.com/code/gallo33henrique/mri-brain-stroke-vgg16) |
| Conditional GAN | [Open on Kaggle](https://www.kaggle.com/code/gallo33henrique/mri-brain-stroke-gan/edit) |

### Local Setup

```bash
# clone repository / clona o repositório
git clone https://github.com/RafaelGallo/mri-stroke-detection-deep-learning.git
cd mri-stroke-detection-deep-learning

# install dependencies / instala dependências
pip install -r requirements.txt

# run notebook / executa o notebook
jupyter notebook notebooks/mri_brain_stroke_vgg16.ipynb
```

### Requirements

```text
tensorflow>=2.12.0
keras-tuner>=1.3.0
scikit-learn>=1.2.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
Pillow>=9.0.0
```

## 📚 References

- Simonyan & Zisserman (2014). Very Deep Convolutional Networks (VGG)
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
- Mirza & Osindero (2014). Conditional Generative Adversarial Nets
- Goodfellow et al. (2014). Generative Adversarial Networks
- Dataset: [Brain Stroke MRI Images — Kaggle](https://www.kaggle.com/datasets/mitangshu11/brain-stroke-mri-images)

## 👨‍💻 Author

**Rafael**  
Data Science & Artificial Intelligence — FIAP

[![Kaggle](https://img.shields.io/badge/Kaggle-gallo33henrique-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/gallo33henrique)
[![GitHub](https://img.shields.io/badge/GitHub-RafaelGallo-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/RafaelGallo)

<div align="center">
⭐ If this project was helpful, please give it a star on GitHub!
</div>
