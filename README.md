# 📈 M5 Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![Prophet](https://img.shields.io/badge/Model-Prophet-orange?style=flat)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models%20%26%20Datasets-yellow?style=flat&logo=huggingface)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat)

> End-to-end sales forecasting pipeline for **30,490 Walmart products** across **10 stores** using Facebook Prophet with custom regressors, built on the [M5 Forecasting Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Status](#-project-status)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Pipeline](#-pipeline)
- [Model](#-model)
- [Dashboard](#-dashboard)
- [Results](#-results)

---

## 🎯 Overview

This project builds a complete time-series forecasting pipeline that:

- Processes raw M5 Walmart sales data (58M+ rows after transformation)
- Trains individual **Prophet models** for each item × store combination
- Evaluates against the official 28-day evaluation window
- Visualizes results through an interactive **Streamlit dashboard**

| | |
|---|---|
| **Dataset** | M5 Forecasting Competition (Walmart) |
| **Items** | 3,049 products × 10 stores = 30,490 series |
| **History** | 1,913 days (Jan 2011 – Apr 2016) |
| **Forecast horizon** | 28 days |
| **Model** | Facebook Prophet with custom regressors |

---

## 🚦 Project Status

```
[✅] Data Ingestion        melt WIDE→LONG, fix dtypes
[✅] Data Cleaning         handle missing values, outliers, encode events
[✅] Merge                 join sales + calendar + sell_prices
[✅] Feature Engineering   snap flags, sell_price fill, event encoding
[✅] Model Training        30,490 Prophet models with resume capability
[✅] Forecasting           28-day predictions with confidence intervals
[ ] Evaluation             metrics computation & comparison
[ ] Dashboard              Streamlit visualization (in progress)
```

---

## 📦 Dataset

### Raw Data — Kaggle

Download the raw dataset from the [M5 Forecasting Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data):

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/raw/
```

| File | Description | Size |
|------|-------------|------|
| `sales_train_validation.csv` | Daily sales per item (wide format) | ~116 MB |
| `sales_train_evaluation.csv` | Same + 28 extra days (ground truth) | ~122 MB |
| `calendar.csv` | Date metadata, events, SNAP flags | ~200 KB |
| `sell_prices.csv` | Weekly prices per item × store | ~141 MB |

### Processed Data — Hugging Face

Pre-processed datasets (cleaned, merged, feature-engineered) are available on Hugging Face:

🤗 **[DeckardLong/m5-sales-forecasting-datasets](https://huggingface.co/datasets/DeckardLong/m5-sales-forecasting-datasets/tree/main)**

```bash
# Requires git-lfs
git lfs install
git clone https://huggingface.co/datasets/DeckardLong/m5-sales-forecasting-datasets
```

| File | Description |
|------|-------------|
| `processed/sales_clean.parquet` | Cleaned sales (long format) |
| `processed/calendar_clean.parquet` | Cleaned calendar |
| `processed/prices_clean.parquet` | Cleaned sell prices |
| `merged/m5_merged.parquet` | All 3 files joined |
| `features/m5_features.parquet` | Final feature-engineered dataset |

### Trained Models — Hugging Face

Pre-trained Prophet models (30,490 `.pkl` files) are available on Hugging Face:

🤗 **[DeckardLong/m5-sales-forecasting-models](https://huggingface.co/DeckardLong/m5-sales-forecasting-models/tree/main)**

```bash
# Requires git-lfs
git lfs install
git clone https://huggingface.co/DeckardLong/m5-sales-forecasting-models
```

---

## 📁 Project Structure

```
sales-forecasting/
│
├── data/
│   ├── raw/                          # Raw files from Kaggle
│   │   ├── sales_train_validation.csv
│   │   ├── sales_train_evaluation.csv
│   │   ├── calendar.csv
│   │   └── sell_prices.csv
│   ├── processed/                    # Cleaned individual files
│   │   ├── sales_clean.parquet
│   │   ├── calendar_clean.parquet
│   │   └── prices_clean.parquet
│   ├── merged/                       # Joined dataset
│   │   └── m5_merged.parquet
│   └── features/                     # Feature-engineered dataset
│       └── m5_features.parquet
│
├── models/
│   ├── items/                        # 30,490 Prophet .pkl files
│   │   ├── HOBBIES_1_001_CA_1.pkl
│   │   ├── HOBBIES_1_002_CA_1.pkl
│   │   └── ...
│   └── forecast/
│       └── forecast_results.parquet  # 28-day predictions for all items
│
├── src/
│   ├── preprocessing.py          # cleaning pipeline
│   ├── normalization.py          # normalize features
│   ├── prophet_features.py       # select and combine features
│   ├── prophet_model.py          # training and save models
│   └── prophet_visualization.py  # visualize data after training  
|
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_data_cleaning.ipynb        # Data cleaning            
│   ├── 03_feature_engineering.ipynb  # Feature engineering verification
│   ├── 04_model_training.ipynb       # Training & resume
│   └── 05_evaluation.ipynb           # Metrics & visualization
│
├── dashboard/
│   ├── app.py                        # Streamlit entry point
│   ├── conftest.py                   # sys.path setup for multipage
│   ├── pages/
│   │   ├── 01_overview.py            # Global metrics & charts
│   │   ├── 02_item_detail.py         # Per-item forecast detail
│   │   └── 03_store_comparison.py    # Cross-store analytics
│   └── components/
│       ├── __init__.py
│       ├── data_loader.py            # Cached data loading
│       ├── charts.py                 # All Plotly chart functions
│       └── metrics.py                # All metric computations
│
├── figures/
│   ├── forecast/ 
│   └── raw/                          # Saved EDA plots
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/deckardLong/sales-forecasting.git
cd sales-forecasting
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the Data

**Option A — Download everything from Hugging Face (recommended):**

```bash
# Install git-lfs first
git lfs install

# Processed datasets
git clone https://huggingface.co/datasets/DeckardLong/m5-sales-forecasting-datasets
cp -r m5-sales-forecasting-datasets/processed data/
cp -r m5-sales-forecasting-datasets/merged   data/
cp -r m5-sales-forecasting-datasets/features data/

# Pre-trained models
git clone https://huggingface.co/DeckardLong/m5-sales-forecasting-models
cp -r m5-sales-forecasting-models/items    models/
cp -r m5-sales-forecasting-models/forecast models/
```

**Option B — Start from raw Kaggle data:**

```bash
pip install kaggle
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/raw/
```

Then run the full pipeline (see [Pipeline](#-pipeline) section).

---

## 🚀 Quick Start

### Run Dashboard (with pre-trained models)

```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`

### Run Training from Scratch

```bash
# 1. Process raw data
python src/preprocessing.py
python src/normalization.py

# 2. Feature engineering
python src/prophet_features.py

# 3. Train models (supports resume on crash)
python src/prophet_model.py
```

---

## 🔄 Pipeline

```
Raw Data (Kaggle)
      │
      ▼
┌─────────────────┐
│   Ingestion     │  melt WIDE→LONG (58M rows), fix dtypes
└─────────────────┘
      │
      ▼
┌─────────────────┐
│   Cleaning      │  missing values, outliers (cap), event encoding
└─────────────────┘
      │
      ▼
┌─────────────────┐
│     Merge       │  sales + calendar + sell_prices → 1 DataFrame
└─────────────────┘
      │
      ▼
┌─────────────────┐
│    Features     │  snap flags, sell_price fill, MinMax scaling,
│  Engineering    │  binary event flags (is_christmas, is_sporting...)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│    Training     │  30,490 Prophet models, checkpoint every 100 items,
│   (Prophet)     │  resume capability via joblib
└─────────────────┘
      │
      ▼
┌─────────────────┐
│   Evaluation    │  MAE, RMSE, MAPE vs sales_train_evaluation.csv
└─────────────────┘
      │
      ▼
┌─────────────────┐
│   Dashboard     │  Streamlit — Overview, Item Detail, Store Comparison
└─────────────────┘
```

---

## 🤖 Model

### Facebook Prophet

Each of the 30,490 item × store combinations gets its own Prophet model trained on the full 1,913-day history.

**Custom regressors used:**

| Regressor | Description |
|-----------|-------------|
| `sell_price` | MinMax-scaled weekly price |
| `snap` | SNAP food stamp flag (state-specific) |
| `has_event_1` | Binary: any event on this day |
| `has_event_2` | Binary: secondary event |
| `is_christmas` | Binary: Christmas day |
| `is_national` | Binary: national holiday |
| `is_religious` | Binary: religious holiday |
| `is_sporting` | Binary: sporting event |

**Training with resume capability:**

```python
# Automatically resumes from last checkpoint if interrupted
models = prophet_model.train_model(df)

# Output:
# ▶ Resume: already have 15,000/30,490 items
# Training Prophet Models: 100%|████| 30490/30490
# ✅ Forecast saved → models/forecast/forecast_results.parquet
```

**Notes:**
- Christmas and other holidays are treated as **signal, not outlier** — kept as-is and flagged via regressors
- `sell_price = 0` (pre-launch period) is handled via `ffill` / `bfill` grouped by `item_id × store_id`
- Models are saved as individual `.pkl` files using `joblib`

---

## 🖥️ Dashboard

Three-page Streamlit dashboard:

### 📊 Overview
- Total actual vs predicted (28-day window)
- KPI cards: MAE, RMSE, MAPE, forecast bias
- Sales breakdown by store and category
- MAPE distribution histogram
- Top 10 best and worst performing items

### 🔍 Item Detail
- 180-day history + 28-day forecast with confidence intervals
- Holiday markers on chart
- Seasonality components: trend, day-of-week, monthly
- Price vs sales correlation scatter
- Residual analysis + auto-generated insights
- Day-by-day actual vs predicted table

### 🏪 Store Comparison
- Store rankings with medals (🥇🥈🥉)
- MAPE and bias comparison across stores
- Sales heatmap: Store × Category
- Performance radar chart (multi-metric)
- Long-term historical trend (30-day MA)

```bash
streamlit run dashboard/app.py
```

---

## 📊 Results

> Evaluation metrics will be updated after full training completes.

| Metric | Value |
|--------|-------|
| MAE | — |
| RMSE | — |
| MAPE | — |
| Items with MAPE < 20% | — |

---

## 🛠️ Requirements

```
python>=3.11
pandas
numpy
pyarrow
prophet
scikit-learn
joblib
tqdm
streamlit
plotly
huggingface_hub
kaggle
```

Install all:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

The M5 dataset is provided by Walmart via Kaggle under the competition's terms of use. Raw data files are **not redistributed** in this repository.

---

*README last updated: June 2026*