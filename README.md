<h1 align="center">RFM Customer Segmentation</h1>

<p align="center">
  Customer segmentation from retail transactions using Recency, Frequency, Monetary analysis, and K-Means clustering.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/RFM-7C3AED?style=for-the-badge" alt="RFM" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Method-Customer_Segmentation-0369A1?style=flat-square" alt="Customer segmentation" />
  <img src="https://img.shields.io/badge/Model-K--Means-F59E0B?style=flat-square" alt="K-Means" />
  <img src="https://img.shields.io/badge/Status-Portfolio_Project-16A34A?style=flat-square" alt="Portfolio project" />
</p>

---

## Overview

This project segments retail customers using **RFM analysis**:

- **Recency:** how long it has been since the customer's latest purchase;
- **Frequency:** how many distinct invoices the customer generated;
- **Monetary:** the total value attributed to the customer's transactions.

The workflow cleans transaction data, engineers the three RFM features, applies logarithmic transformation and standardization, and uses K-Means to assign customers to analytical groups.

> The clusters are exploratory analytical segments. They are not predefined business personas and require interpretation before operational use.

---

## Workflow

```text
Retail transactions
        │
        ▼
Data inspection
        │
        ▼
Customer and invoice cleaning
        │
        ▼
RFM feature engineering
        │
        ▼
Positive monetary-value filter
        │
        ▼
Logarithmic transformation
        │
        ▼
Feature standardization
        │
        ▼
K-Means clustering
        │
        ▼
Customer segment visualization
```

---

## Methodology

### 1. Data preparation

The analysis:

- loads the retail transaction dataset;
- removes records without a customer identifier;
- converts invoice dates to datetime values;
- calculates transaction value from quantity and unit price;
- removes duplicate rows.

### 2. RFM feature engineering

The customer-level dataset is created by grouping transactions by customer and calculating:

| Feature | Calculation represented in the project |
|---|---|
| Recency | Days between the reference date and the latest customer invoice. |
| Frequency | Number of distinct invoices. |
| Monetary | Sum of customer transaction value. |

### 3. Feature transformation

Before clustering, the project:

- keeps customers with positive monetary value;
- applies `log1p` to reduce feature skew;
- standardizes Recency, Frequency, and Monetary with `StandardScaler`.

### 4. Clustering

K-Means assigns each customer to one of four clusters using the standardized RFM feature set.

### 5. Visualization

A scatter plot compares Recency and Frequency while using the assigned cluster as the visual grouping.

---

## Repository Contents

| File | Purpose |
|---|---|
| [`rfm_analysis_code.py`](./rfm_analysis_code.py) | Main cleaning, RFM feature engineering, clustering, and visualization workflow. |
| `online_retail.csv` | Retail transaction dataset consumed by the analysis. |
| `online_retail.xlsx` | Original spreadsheet-format dataset. |
| [`requirements.txt`](./requirements.txt) | Python environment dependencies currently stored in the repository. |
| `readme.txt` | Original dataset-source reference. |

---

## Dataset

The repository references the **Online Retail Dataset** hosted on Kaggle:

[Online Retail Dataset — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)

Review the dataset's original license and usage conditions before redistributing or using it outside this portfolio project.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/vfreis/rfm_analysis-.git
cd rfm_analysis-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The core libraries used directly by the analysis are Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

### 4. Run the analysis

```bash
python rfm_analysis_code.py
```

The source dataset must be available as `online_retail.csv` in the repository root.

---

## Skills Demonstrated

- Transactional-data cleaning.
- Customer-level feature engineering.
- RFM analysis.
- Skew reduction with logarithmic transformation.
- Numerical standardization.
- Unsupervised learning with K-Means.
- Analytical visualization.
- Translating transaction histories into customer-segmentation features.

---

## Limitations

- The number of clusters is fixed at four in the current implementation.
- Cluster quality is not currently evaluated with metrics such as silhouette score.
- Segment labels require business interpretation.
- The workflow is a standalone script rather than a reusable package or automated pipeline.
- The repository does not currently include automated tests or saved chart assets.

---

## Roadmap

- Compare cluster counts with elbow and silhouette analysis.
- Add descriptive profiles for each cluster.
- Assign business-friendly segment names after validating cluster behavior.
- Refactor data preparation and clustering into reusable functions.
- Add automated tests for RFM calculations.
- Save model outputs and visualizations under documented directories.
- Reduce `requirements.txt` to project-specific dependencies.

---

## Author

**Vinicios Falqueiro Reis** — Data Engineer focused on data preparation, analytics pipelines, and reliable data products.

[LinkedIn](https://www.linkedin.com/in/vfalqueiroreis/) · [GitHub](https://github.com/vfreis)