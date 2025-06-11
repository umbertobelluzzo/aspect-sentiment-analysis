# Aspect-Based Sentiment Analysis: Amazon Electronics Reviews

---

## 🎯 Project Overview
This project implements a comprehensive **Aspect-Based Sentiment Analysis (ABSA)** system for Amazon Electronics reviews.  
It explores how different product aspects—such as price, battery, and design—affect overall sentiment predictions and investigates whether fine-grained, aspect-level analysis provides deeper insights than traditional document-level sentiment classification.

## 🔍 Key Research Question
> **How effective is aspect-based sentiment analysis compared to traditional sentiment classification, and which product aspects most strongly influence overall review sentiment?**

## 🏆 Key Achievements
- **Hybrid Feature Engineering** – Combined TF-IDF vectorization with aspect-level sentiment scores.  
- **Multi-Model Implementation** – Logistic Regression, Naive Bayes, and LSTM with embeddings.  
- **Aspect Importance Analysis** – Identified **price**, **performance**, and **battery** as the most influential aspects.  
- **Robust Evaluation** – 81.2 % accuracy with LSTM, using 5-fold cross-validation.  
- **Negativity Bias Detection** – Quantified a 6.6 pp drop in positive predictions for mixed-sentiment reviews.

---

## 📊 Dataset
- **Source:** Amazon Product Data – *Electronics* category  
- **Size:** 50 000 reviews  
- **Features:** review text, ratings, verified-purchase flag, timestamps  
- **Target:** binary sentiment (*positive* ≥ 4 stars, *negative* < 4 stars)  
- **Class distribution:** 82 % positive, 18 % negative  

---

## 🛠️ Technical Architecture

### Aspect Extraction System
```python
product_aspects = {
    "performance": ["speed", "processing", "efficiency", "power", "capability"],
    "battery":      ["battery", "charge", "power consumption", "runtime"],
    "design":       ["build", "design", "size", "weight", "appearance"],
    "durability":   ["quality", "durability", "reliability", "robust"],
    "features":     ["feature", "capability", "function", "specification"],
    "price":        ["cost", "price", "value", "affordable", "expensive"],
    "customer_support": ["support", "service", "help", "assistance"],
    "compatibility":    ["compatibility", "integration", "work with"]
}


## 🛠️ Feature-Engineering Pipeline
- **Pre-processing** — HTML entity decoding, lower-casing, whitespace normalization  
- **Aspect Detection** — spaCy + keyword matching  
- **Sentiment Scoring** — NLTK Opinion Lexicon  
- **Feature Fusion** — TF-IDF (5 000 dimensions) + 8 aspect-sentiment scores  

---

### Model Implementations
| Model                | Features        | Accuracy | F1-neg | Notes                           |
|----------------------|-----------------|----------|--------|---------------------------------|
| Logistic Regression  | 5 008 (hybrid)  | 80.1 %   | 0.585  | L2 regularization, C = 0.1      |
| Naive Bayes          | Hybrid          | 76.9 %   | 0.537  | Class-weighted                  |
| LSTM                 | 100-d embeddings| **81.2 %** | 0.569  | 200-token sequences, 20 k vocab |

---

## 📈 Results Summary
| Model                | Precision (Neg) | Recall (Neg) | F1 (Neg) | Precision (Pos) | Recall (Pos) | F1 (Pos) |
|----------------------|-----------------|--------------|----------|-----------------|--------------|----------|
| **LSTM**             | 0.484           | 0.690        | 0.569    | 0.925           | 0.839        | 0.880    |
| Logistic Regression  | 0.467           | 0.784        | **0.585**| **0.945**       | 0.805        | 0.869    |
| Naive Bayes          | 0.420           | 0.746        | 0.537    | 0.933           | 0.775        | 0.847    |

---

## 🎯 Aspect Importance Rankings
1. **Price** — coefficient = 0.344 (strongest)  
2. **Performance** — 0.267  
3. **Battery** — 0.216  
4. **Compatibility** — 0.191  

---

## 🧠 Key Insights

### Negativity Bias
- Mixed-sentiment reviews (negative aspects **+** positive rating) cut positive predictions by **6.6 percentage points**.  
- 95 % CI: **[ 5.9 , 7.4 ]**.  
- **Takeaway:** Negative aspects wield disproportionate influence.

### Model Comparison
- **LSTM** — Highest accuracy; best at capturing sequence context.  
- **Logistic Regression** — Most interpretable; ideal for explaining aspect weights.  
- **Naive Bayes** — Fastest training; solid baseline.

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/aspect-sentiment-analysis.git
cd aspect-sentiment-analysis
pip install -r requirements.txt


### Data Preparation
python src/data_preprocessing.py \
  --input  data/raw/reviews.json.gz \
  --output data/processed/

### Training
Train all models:
python src/train_models.py --config configs/training_config.yaml

Train a specific model:
python src/train_models.py --model lstm --features hybrid

### Evaluation
python src/evaluate.py \
  --model-path models/lstm_model.h5 \
  --test-data data/processed/test.csv

# 📁 Project Structure

aspect-sentiment-analysis/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   ├── trained_models/
│   └── model_artifacts/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_aspect_extraction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── aspect_extractor.py
│   ├── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_model.py
│   │   ├── naive_bayes_model.py
│   │   └── lstm_model.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py
│       └── config.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── results/
│   ├── figures/
│   ├── reports/
│   └── model_performance/
└── docs/
    ├── methodology.md
    ├── results_analysis.md
    └── api_reference.md

# Methodology
## Evaluation Strategy
- 5-fold stratified cross-validation
- Class imbalance handled via sample weighting (≈ 4.6 : 1)
- Metrics: Accuracy · Precision · Recall · F1 for each class
- Intrinsic evaluation of aspect extraction — P = 0.94 · R = 0.59 · F1 = 0.73

## Innovation Highlights
- Hybrid TF-IDF + aspect sentiment features
- Negativity-bias quantification with confidence intervals
- Traditional ML vs. Deep Learning side-by-side comparison

## Research Context
This project advances sentiment analysis by:
- Moving from document-level to aspect-level granularity
- Fusing statistical and semantic features
- Delivering actionable insights for product teams

# Tools used
- NLP: spaCy · NLTK · TensorFlow/Keras
- ML / Data: Scikit-learn · NumPy · Pandas
- Visualization: Matplotlib · Seaborn
- Dev Tools: Jupyter · Git · Python 3.8 +

# Citation
@misc{aspect_sentiment_analysis_2024,
  title        = {Aspect-Based Sentiment Analysis for Amazon Electronics Reviews},
  author       = {Umberto Belluzzo},
  year         = {2024},
  howpublished = {\url{https://github.com/umbertobelluzzo/aspect-sentiment-analysis}}
}

# Contributing
Pull requests are welcome! See CONTRIBUTING.md for guidelines.

# Acknowledgements
- Amazon Product Data - courtesy of https://nijianmo.github.io/amazon/
- NLTK Opinion Lexicon — sentiment lexicon
- spaCy — NLP toolkit


