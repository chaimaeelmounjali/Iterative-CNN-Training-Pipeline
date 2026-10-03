# 🧠 Iterative CNN Training Pipeline & Deep Face Representation

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![ArcFace](https://img.shields.io/badge/Loss-ArcFace%20Margin-red.svg)](https://arxiv.org/abs/1801.07698)
[![CBAM](https://img.shields.io/badge/Attention-CBAM%20Module-blueviolet.svg)](https://arxiv.org/abs/1807.06521)
[![AUC](https://img.shields.io/badge/AUC-0.9042-success.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*Bilingual README: [Français](#-version-française) | [English](#-english-version)*

---

## 🇫🇷 Version Française

### 🎯 Objectif
Le projet **Iterative CNN Training Pipeline** propose une démarche expérimentale rigoureuse et reproductible pour l'apprentissage de représentations faciales discriminantes (*deep face representation & verification*). L'objectif est de tracer l'évolution progressive d'un réseau de neurones convolutif (CNN) à travers trois itérations successives : d'un réseau de base (*baseline*) vers une architecture enrichie de mécanismes d'attention spatiale et de canaux (**CBAM**), optimisée avec une fonction de perte angulaire additive à large marge (**ArcFace Loss**).

### 🛠️ Stack Technologique
- **Deep Learning & Frameworks** : PyTorch, Torchvision, CUDA.
- **Architectures & Composants Avancés** :
  - Backbones convolutifs personnalisés.
  - Module d'attention **CBAM** (*Convolutional Block Attention Module*).
  - Fonction de perte **ArcFace** (*Additive Angular Margin Loss*).
- **Vision par Ordinateur & Prétraitement** : OpenCV, PIL, MTCNN / Haar Cascade (détection de visage, alignement des yeux, recadrage ciblé et normalisation).
- **Évaluation Biométrique & Visualisation** : Scikit-Learn, Matplotlib, Seaborn (courbes ROC, matrices de confusion, réduction de dimension t-SNE pour visualiser les clusters d'identités).

### 👩‍💻 Mon Rôle & Contributions
- **Prétraitement & Nettoyage des Visages (`01_Cleaning_Crop.ipynb`)** :
  - Détection automatique, cadrage serré et redimensionnement standardisé des visages pour éliminer les bruits de fond.
  - Augmentation de données (rotations légères, variations de luminosité, retournements horizontaux).
- **Architecture & Itérations d'Entraînement** :
  - **Version 1 (`version 1/cnnproject1.0.ipynb`)** : Établissement du réseau de base (CNN baseline avec Cross-Entropy Loss) et premières métriques de référence.
  - **Version 2 (`version 2/cnnproject2.0.ipynb`)** : Intégration de la marge angulaire ArcFace combinée aux blocs d'attention CBAM pour maximiser la compacité intra-classe et la séparation inter-classe.
  - **Version 3 (`version 3/cnnproject3.0.ipynb`)** : Optimisation des hyperparamètres, exploration du compromis FAR/FRR et projection des représentations latentes par t-SNE.
- **Analyse des Performances Biométriques** :
  - Calcul de l'aire sous la courbe (AUC-ROC), détermination du seuil de décision optimal, et calcul du taux de fausse acceptation (FAR) et de faux rejet (FRR).

### 📊 Résultats & Métriques Clés
- **Excellente séparabilité biométrique** :
  - **Version 2** : Score **AUC de 0.9042** (AUC > 90%), précision de validation de **64.77%**, seuil optimal à `0.1195`, FAR de `0.1879` et FRR de `0.0909`.
  - **Version 3** : Réduction du taux de fausse acceptation à **FAR = 0.1118** pour une sécurité renforcée à un seuil calibré à `0.2111`.
- **Clustering qualitatif** : Les projections t-SNE démontrent une séparation nette et compacte des embeddings faciaux par identité.

---

## 🇬🇧 English Version

### 🎯 Objective
The **Iterative CNN Training Pipeline** documents a systematic, progressive deep learning engineering methodology for facial representation learning and identity verification. It chronicles the evolution from a standard baseline Convolutional Neural Network (CNN) to an attention-enhanced network integrating the **CBAM** (Convolutional Block Attention Module) architecture optimized via large-margin angular loss (**ArcFace**).

### 🛠️ Tech Stack
- **Deep Learning & Frameworks**: PyTorch, Torchvision, CUDA acceleration.
- **Architectural Enhancements**:
  - Custom deep CNN backbone.
  - **CBAM**: Dual spatial and channel attention mechanisms.
  - **ArcFace**: Additive Angular Margin Loss for hyper-spherical feature embedding separation.
- **Computer Vision & Preprocessing**: OpenCV, PIL, MTCNN (face detection, facial landmark alignment, tight cropping, color normalization).
- **Biometric Evaluation**: Scikit-Learn, Matplotlib, Seaborn (ROC Curves, DET analysis, t-SNE high-dimensional embedding projection).

### 👩‍💻 My Role & Key Contributions
- **Face Processing Pipeline (`01_Cleaning_Crop.ipynb`)**:
  - Engineered face detection, landmark normalization, and cropping routines to isolate facial features.
  - Applied targeted data augmentations (affine transforms, illumination jitter).
- **Iterative Experimental Lifecycle**:
  - **Version 1 (`version 1/cnnproject1.0.ipynb`)**: Benchmarked a baseline CNN architecture trained with Softmax/Cross-Entropy.
  - **Version 2 (`version 2/cnnproject2.0.ipynb`)**: Implemented ArcFace angular margin penalty alongside CBAM attention, enforcing intra-class compactness.
  - **Version 3 (`version 3/cnnproject3.0.ipynb`)**: Fine-tuned hyperparameters to calibrate the biometric operating point (FAR vs FRR trade-off).
- **Biometric Performance Profiling**:
  - Measured Area Under the ROC Curve (AUC), identified optimal verification thresholds, and analyzed False Acceptance (FAR) and False Rejection (FRR) error rates.

### 📊 Key Results & Impact
- **High Verification Discrimination**:
  - **Version 2**: Reached an **AUC of 0.9042**, validation accuracy of **64.77%**, with optimal threshold `0.1195` (FAR: `0.1879`, FRR: `0.0909`).
  - **Version 3**: Minimized security vulnerability by driving **FAR down to 0.1118** at threshold `0.2111`.
- **Discriminative Embedding Space**: High-dimensional t-SNE projections validate distinct, tightly-packed identity clusters in feature space.

---

### 📂 Repository Structure / Structure du Projet
```text
Iterative-CNN-Training-Pipeline/
├── 01_Cleaning_Crop.ipynb          # Face detection, cropping & dataset cleaning
├── version 1/                      # Baseline CNN experiments
│   └── cnnproject1.0.ipynb
├── version 2/                      # CNN + CBAM + ArcFace integration
│   ├── cnnproject2.0.ipynb
│   └── results/                    # ROC curve, t-SNE, model metrics (AUC: 0.9042)
├── version 3/                      # Refined hyperparameters & FAR tuning
│   ├── cnnproject3.0.ipynb
│   └── results/                    # Training curves, DET curves, weights
└── README.md                       # Comprehensive bilingual documentation
```