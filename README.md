# PFE_NLP  

**Projet de Fin d'Études – Traitement Automatique du Langage Naturel (NLP)**

## 📌 Description

Ce projet s'inscrit dans le cadre d'un **Projet de Fin d'Études (PFE)** et porte sur l'exploitation de techniques de **Traitement Automatique du Langage Naturel (NLP)** pour la classification de maladies de vignes à partir de descriptions de symptômes.

L'objectif est d'analyser, traiter et classifier des descriptions textuelles de symptômes de maladies de vignes (Mildiou, Black Rot, Oïdium, Esca, etc.) à l'aide de différentes approches de machine learning et deep learning.

Le projet propose trois implémentations distinctes :
- **TextBlob** : Classification basée sur Naive Bayes
- **spaCy** : Classification avec le composant textcat de spaCy
- **HuggingFace** : Classification fine-tunée avec des modèles transformers (CamemBERT)

---

## 🎯 Objectifs

- Prétraiter des données textuelles de descriptions de symptômes (nettoyage, tokenisation, vectorisation)
- Extraire des caractéristiques linguistiques pertinentes
- Entraîner et évaluer des modèles de classification de maladies
- Comparer différentes approches NLP (TextBlob, spaCy, HuggingFace)
- Fournir des résultats exploitables et reproductibles
- Générer des descriptions de symptômes à partir d'images (module expérimental)

---

## ⚙️ Prérequis

- Python ≥ 3.8
- pip
- (Optionnel) CUDA pour l'accélération GPU avec HuggingFace

---

## 📦 Installation

### Cloner le dépôt :

```bash
git clone https://github.com/LouisT1425/PFE_NLP.git
cd PFE_GITHUB
```

### Installer les dépendances :

```bash
pip install -r requirements.txt
```

### Installer les modèles spaCy :

```bash
python -m spacy download fr_core_news_sm
```

---

## 🚀 Utilisation

### 1. Entraîner un modèle

Entraîner un modèle avec un des trois classifieurs disponibles :

```bash
# Avec TextBlob
python scripts/train.py --model textblob --input descriptions.csv --output models/textblob.joblib

# Avec spaCy
python scripts/train.py --model spacy --input descriptions.csv --output models/spacy

# Avec HuggingFace (CamemBERT)
python scripts/train.py --model hf --input descriptions.csv --output models/hf
```

**Options disponibles :**
- `--model` : Choix du modèle (`textblob`, `spacy`, `hf`)
- `--input` : Fichier CSV contenant les colonnes `description` et `disease`
- `--output` : Chemin de sauvegarde du modèle
- `--test-size` : Proportion des données pour la validation (défaut: 0.2)

### 2. Faire des prédictions

Utiliser un modèle entraîné pour prédire les maladies à partir de descriptions :

```bash
# Avec TextBlob
python scripts/predict.py --model textblob --model-path models/textblob.joblib --input test.csv --output predictions.csv

# Avec spaCy
python scripts/predict.py --model spacy --model-path models/spacy --input test.csv --output predictions.csv

# Avec HuggingFace
python scripts/predict.py --model hf --model-path models/hf --input test.csv --output predictions.csv
```

**Options disponibles :**
- `--model` : Type de modèle utilisé
- `--model-path` : Chemin vers le modèle sauvegardé
- `--input` : Fichier CSV contenant les descriptions (colonne `description`)
- `--output` : Fichier CSV de sortie avec les prédictions

### 3. Comparer les modèles

Comparer les performances des trois modèles sur un jeu de test :

```bash
python scripts/compare.py --input test.csv --models-dir models --output-dir predictions
```

**Options disponibles :**
- `--input` : Fichier CSV de test avec colonnes `description` et `disease`
- `--models-dir` : Dossier contenant les modèles entraînés
- `--output-dir` : Dossier de sortie pour les résultats (défaut: `prediction`)

Cette commande génère :
- `comparison_summary.csv` : Tableau récapitulatif des métriques (Accuracy, Precision, Recall, F1-score)
- `confusion_matrix_normalized_*.png` : Matrices de confusion normalisées pour chaque modèle

### 4. Module expérimental : Génération de descriptions à partir d'images

Générer automatiquement des descriptions de symptômes à partir d'images de vignes :

```bash
python experimental/pipeline.py --input images/ --output descriptions_ia.csv --device auto
```

**Options disponibles :**
- `--input` : Dossier contenant les images à analyser
- `--output` : Fichier CSV de sortie (défaut: `descriptions_ia.csv`)
- `--device` : Device à utiliser (`auto`, `cpu`, `cuda`)

---

## 📊 Données

### Format des données

Les fichiers CSV doivent contenir au minimum une colonne `description` avec les descriptions textuelles des symptômes.

Pour l'entraînement, une colonne `disease` est également requise avec les labels de maladies correspondants.

**Exemple de structure :**

```csv
description,disease
"On constate sur les jeunes feuilles...",Mildiou
"Le feuillage dans son ensemble exhibe...",Black Rot
```

### Fichiers de données

- `descriptions.csv` : Jeu de données d'entraînement avec descriptions et maladies
- `test.csv` : Jeu de données de test pour l'évaluation

Les données sont automatiquement prétraitées avant l'apprentissage (normalisation, gestion des encodages, nettoyage).

---

## 📈 Résultats

Les performances des modèles sont évaluées à l'aide de métriques classiques :

- **Accuracy** : Taux de prédictions correctes
- **Precision** : Précision moyenne (macro)
- **Recall** : Rappel moyen (macro)
- **F1-score** : Score F1 moyen (macro)

Les résultats sont sauvegardés dans un fichier CSV et des matrices de confusion sont générées pour chaque modèle afin de visualiser les performances par classe.

---

## 🏗️ Structure du projet

```
PFE_GITHUB/
├── classifiers/          # Implémentations des classifieurs
│   ├── textblob.py      # Modèle TextBlob (Naive Bayes)
│   ├── spacy.py         # Modèle spaCy (textcat)
│   └── huggingface.py   # Modèle HuggingFace (Transformers)
├── scripts/             # Scripts d'exécution principaux
│   ├── train.py         # Entraînement des modèles
│   ├── predict.py       # Prédictions avec modèles entraînés
│   └── compare.py       # Comparaison des modèles
├── utils/               # Utilitaires
│   └── data.py          # Chargement et prétraitement des données
├── experimental/        # Modules expérimentaux
│   ├── pipeline.py      # Pipeline de génération de descriptions
│   ├── image_captioning.py
│   └── symptom_description.py
├── descriptions.csv     # Données d'entraînement
├── test.csv            # Données de test
└── requirements.txt    # Dépendances Python
```

---

## 🧠 Technologies utilisées

- **Python** : Langage de programmation principal
- **Scikit-learn** : Métriques et outils de machine learning
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **TextBlob** : Classification Naive Bayes
- **spaCy** : Traitement NLP avec textcat
- **Transformers (HuggingFace)** : Modèles de deep learning (CamemBERT)
- **PyTorch** : Framework de deep learning
- **Matplotlib** : Visualisation des résultats
- **Joblib** : Sauvegarde/chargement de modèles

---

## 👤 Auteur

**Louis T.**  
Projet réalisé dans le cadre d'un Projet de Fin d'Études.

---

## 📄 Licence

Projet à usage académique.
