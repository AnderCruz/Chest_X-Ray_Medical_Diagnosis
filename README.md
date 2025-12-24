# 🩺 Deep Learning–Based Pneumonia Detection from Chest X-Rays

**DenseNet121 vs EfficientNetB0 – A Clinically Optimized Pipeline**

This repository presents a **clinically oriented deep learning pipeline** for automated pneumonia detection from chest X-ray images.
The project was designed with a strong focus on **patient safety, interpretability, and hospital deployment readiness**.

The system compares **DenseNet121** and **EfficientNetB0** under an **identical experimental setup**, incorporating **clinical threshold optimization**, **medical metrics**, and **explainable AI (Grad-CAM)**.


## 🚀 Key Features

### ✅ Automatic Comparison: DenseNet121 vs EfficientNetB0

The notebook includes a **fully standardized comparison** between the two architectures:

* Identical pipeline for both models
* Same strategy for:

  * Data augmentation
  * Clinical threshold selection
  * Evaluation metrics
* Comparative analysis table including:

  * **AUC**
  * **Recall (Sensitivity)**
  * **Computational cost**
  * **Clinical recommendation**

📌 **Embedded technical conclusion**:

> *EfficientNetB0 is better suited for hospital production environments, while DenseNet121 is excellent for research and explainability.*


### 🏥 Explicit Hospital-Oriented Design

The pipeline was explicitly adapted for **real-world clinical environments**, including:

* **Clinical decision threshold ≥ 97% recall**
* Direct emphasis on **false negative reduction**
* Clinical interpretation of trade-offs:

  * More referrals
  * Fewer missed pneumonia cases
* Language and structure suitable for:

  * Medical boards
  * Ethics committees
  * Hospital IT and AI governance teams


## 🧾 Scientific Background (Paper-Ready)

### **Deep Learning–Assisted Pneumonia Diagnosis from Chest X-Ray Images**

#### Abstract

Pneumonia remains one of the leading causes of global morbidity and mortality, particularly among vulnerable populations. Chest X-ray interpretation, while widely used for diagnosis, is subject to inter-observer variability and limited availability of specialized radiologists.
This work proposes a deep learning–based clinical decision support system for automated pneumonia detection from chest X-ray images. DenseNet121 and EfficientNetB0 architectures were evaluated using transfer learning, clinically optimized decision thresholds, and Grad-CAM–based explainability. The results demonstrate high diagnostic sensitivity with a strong emphasis on false negative reduction, making the system suitable for hospital screening and triage scenarios.


## 📚 Methodology Overview

### Dataset

* Public chest X-ray dataset
* Binary classes: **NORMAL** and **PNEUMONIA**
* Split into training, validation, and test sets
* Class imbalance handled via **class weighting**

### Preprocessing & Augmentation

* Image resizing to **224 × 224**
* ImageNet-based normalization
* Training-only augmentation:

  * Rotation
  * Translation
  * Horizontal flipping

### Evaluated Architectures

* **DenseNet121** – strong feature reuse and explainability
* **EfficientNetB0** – high performance with lower computational cost
* Transfer learning with frozen convolutional backbones

### Clinical Decision Strategy

* Decision threshold optimized for **≥ 97% sensitivity**
* Focus on minimizing **false negatives**, reflecting real clinical risk
* Threshold selected using ROC analysis, not default 0.5 probability

### Evaluation Metrics

* AUC (ROC)
* Recall (Sensitivity)
* Confusion Matrix
* False Negative Rate


## 📊 Results Summary

* Both models achieved **AUC > 0.94**
* EfficientNetB0 matched or exceeded DenseNet121 performance
* EfficientNetB0 required fewer parameters and lower computational cost
* Clinical threshold adjustment successfully achieved **>97% recall**
* Significant reduction in false negatives


## 🔍 Explainability (Grad-CAM)

* Grad-CAM was used to visualize model attention
* Activation maps consistently highlighted **pulmonary regions**
* Confirms anatomical and clinical plausibility of predictions
* Supports transparency and trust in clinical decision support


## 🏥 Hospital Deployment Pipeline

The project includes a **production-ready hospital inference pipeline** with:

* Modular inference functions
* Structured, auditable outputs (JSON-like)
* Model versioning and timestamps
* Clinical recommendations based on prediction
* Batch inference support
* EarlyStopping and Recall-based checkpointing
* Custom **clinical loss** penalizing false negatives


## ⚠️ Clinical Disclaimer

> This system is intended as a **clinical decision support tool** only.
> It does **not replace medical judgment** or professional diagnosis.


## 🧠 Final Conclusions

* The pipeline follows principles of **responsible medical AI**
* Decision-making is guided by **clinical impact**, not accuracy alone
* **EfficientNetB0** is recommended for hospital production use
* **DenseNet121** remains valuable for research and interpretability
* Grad-CAM confirms anatomically coherent decision behavior
* Ready for **institutional validation**, research, or hospital deployment


## 📌 Suggested Use Cases

* Hospital triage support
* Radiology workflow prioritization
* Clinical research
* Academic projects (TCC, MSc, PhD)
* Technical or regulatory reports


## 📁 Project Structure

The repository is organized to support **research, clinical validation, and hospital deployment workflows**:

```text
.
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── notebooks/
│   ├── pneumonia_densenet_pipeline.ipynb
│   ├── pneumonia_efficientnet_pipeline.ipynb
│   └── comparison_analysis.ipynb
│
├── models/
│   ├── modelo_pneumonia_densenet.keras
│   ├── modelo_pneumonia_efficientnet.keras
│   └── best_model_clinical.keras
│
├── inference/
│   ├── hospital_inference.py
│   └── batch_inference.py
│
├── explainability/
│   └── gradcam_utils.py
│
├── requirements.txt
├── README.md
└── LICENSE


### Structure Rationale

- **`chest_xray/`** – Dataset organized following TensorFlow best practices  
- **`notebooks/`** – End-to-end experimental pipelines and comparative analysis  
- **`models/`** – Trained and clinically optimized models  
- **`inference/`** – Modular hospital-grade inference functions  
- **`explainability/`** – Grad-CAM utilities for medical explainability  
- **`requirements.txt`** – Reproducible environment setup  
- **`LICENSE`** – Open-source license for research and deployment

This structure supports the full lifecycle of medical AI development, from
research and experimentation to clinical validation and hospital deployment,
ensuring reproducibility, traceability, and regulatory readiness.


