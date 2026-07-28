# 📊 Bank Telemarketing Campaign Analysis & Conversion Insights

## Project Overview

The project analyzes the Bank Marketing dataset, which contains data from direct telemarketing campaigns conducted by a Portuguese bank to promote term deposit subscriptions.

The objective is to understand customer behavior, identify high-performing customer segments, analyze campaign effectiveness, and determine factors associated with successful term deposit subscriptions.

This project uses exploratory analysis, segment-based analysis, and an analytical conversion funnel to uncover actionable business insights.

---

# Business Problem

Banks invest significant resources in telemarketing campaigns, but only a small proportion of contacted customers subscribe to term deposits.

This analysis aims to answer:

- Which customer segments are more likely to subscribe?
- Which customer characteristics influence conversion?
- Which campaign strategies produce better outcomes?
- Where does customer drop-off occur in the conversion funnel?

---

# Dataset

The dataset is sourced from the UCI Bank Marketing dataset.

### Dataset Information

| Attribute | Description |
|-----------|-------------|
| Records | 45,211 customers |
| Features | 16 customer and campaign attributes |
| Target Variable | `y` (term deposit subscription) |

### Raw Data

The `data/raw` folder contains:

- `bank-full.csv` - Main dataset used for analysis
- `bank.csv` - Smaller dataset version
- `bank-additional.csv` - Dataset with additional campaign features
- `bank-additional-full.csv` - Full additional dataset
- `bank-names.txt` - Feature descriptions

### Processed Data

The `data/processed` folder contains:

- `clean.csv` - Cleaned dataset after preprocessing
- `funnel-data.csv` - Aggregated data used for analytical funnel analysis

---

# Analysis Approach

## 1. Data Cleaning & Exploratory Data Analysis

Notebook:

`01-data-cleaning-and-eda.ipynb`

Performed:

- Data quality checks
- Missing value analysis
- Duplicate detection
- Feature understanding
- Target variable analysis
- Customer profile exploration

---

## 2. Bivariate & Correlation Analysis

Notebook:

`02-bivariate-correlation-analysis.ipynb`

Focus:

- Relationship between individual features and subscription outcome
- Conversion rate comparison across customer groups
- Numerical feature correlation analysis
- Identification of important customer characteristics

---

## 3. Multivariate Analysis

Notebook:

`03-multivariate-analysis.ipynb`

Objective:

The analysis focuses on understanding how multiple customer characteristics and campaign attributes influence term deposit subscription likelihood.

Key questions:

- What type of customers are targeted by the bank?
- Which customer profiles are more likely to subscribe?
- Does communication channel influence campaign success?

---

## 4. Funnel Conversion Analysis

Notebook:

`04-funnel-conversion-and-performance-analysis.ipynb`

The funnel presented in this analysis is an **analytical conversion funnel** derived from the Bank Marketing dataset.

The stages represent campaign characteristics and customer outcomes rather than a strict sequential customer journey.

| Analytical Funnel Stage | Customers | % of Total |
|---|---:|---:|
| Stage 1 — Total Customers | 45,211 | 100.0% |
| Stage 2 — Customers Contacted via Cellular | 29,285 | 64.8% |
| Stage 3 — Customers with Previous Contact History | 8,257 | 18.3% |
| Stage 4 — Customers Who Subscribed | 5,289 | 11.7% |

---

## Primary Drop-off Insights

### Drop-off 1 — Stage 1 to Stage 2 (35.2% Reduction)
- 13,020 customers were contacted via an __unknown channel__, with only a __4.1%__ conversion rate.
- Cellular contact achieved a **14.9%** conversion rate versus **4.1%** for the unknown channel (3.6× higher).
- Switching entirely to cellular outreach for these customers could have resulted in approximately 1,400 additional subscriptions.

### Drop-off 2 — Stage 2 to Stage 3 (71.8% Reduction) — Largest Drop-off
- **21,028 customers decline** — single largest drop-off in the funnel.
- 81.7% of customers had no prior relationship with the bank.
- New customers converted at **9.2%**, vs customers with a previously successful campaign outcome converted at **64.7%**(~ 7 times higher).

### Drop-off 3 — Stage 3 to Stage 4 (35.9% Reduction)
- Customers called __6+ times__ convert at only __5.8%__ vs __14.6%__ for a single call.
- Excessive calling reduces customer conversion rates.
- Customer fatigue from repeated calls is a key factor contributing to drop-off at this final stage.

### Baseline ROI vs Improved ROI
| Metric | Baseline Campaign | Improved Campaign |
|--------|------------------:|------------------:|
| Total Customers Contacted | 45,211 | 45,211 |
| Successful Subscriptions | 5,289 | 6,689 |
| Conversion Rate | 11.7% | 14.8% |
| Revenue per Subscription | $1,000 | $1,000 |
| Total Revenue | $5,289,000 | $6,689,000 |
| Campaign Cost (@ $5/contact) | $226,055 | $226,055 |
| Net Profit | $5,062,945 | $6,462,945 |
| ROI | 2,239.7% | 2,859.0% |
| ROI Improvement | — | +619.3 percentage points |

# Analysis Highlights

## Customer Segmentation Insights

- Only **11.7% of customers subscribed**, highlighting the need for focused campaign strategies.
- **Retired and student customers** achieved the highest subscription rates despite smaller population sizes.
- Customers with **tertiary education** showed stronger subscription performance.
- Customers without loans demonstrated higher conversion rates.
- Higher account balances (**2K+**) were associated with increased positive response rates.

## Campaign Performance Insights

- Campaign timing affected conversion performance:
  - March, September, and December showed stronger conversion rates.
  - May had the highest campaign volume but lower efficiency.
- Longer calls (**5+ minutes**) were associated with higher subscription likelihood.
- Contact attempts should be optimized, with approximately three attempts providing better campaign efficiency.

## Customer History & Communication Insights

- Customers with previous successful campaign outcomes converted at approximately **7× the rate of new contacts**.
- Cellular communication achieved better conversion performance compared with telephone contact.

---

# Business Recommendations

Based on the analysis:

- Prioritize customers with previous successful campaign history.
- Focus marketing efforts on high-performing customer segments.
- Increase usage of cellular communication channels.
- Optimize campaign timing based on historical performance.
- Avoid excessive contact attempts to improve customer experience and campaign efficiency.
- Develop targeted campaigns instead of broad customer outreach.

---

# Handling Imbalanced Data
To address class imbalance, we employed several techniques:

- **SMOTE Analysis — OVERSAMPLING APPROACH :** We duplicated the minority class observations in the training dataset to balance it with the majority class.
- **Class Weights :** We assigned higher weights to the minority class during model training.
- **Threshold Tuning :** The probability threshold for determining crisp labels was fine-tuned, rather than using the default threshold of 0.5.
- **Cost-Sensitive Optimization :** The algorithm tests different thresholds and picks the one that minimizes total business cost, not statistical performance. It reflects real-world decisions where business impact matters more than academic metrics.

**Impact on Conversion Rate :**
| Approach | Misses Suscriptions | Lost Revenue | Wasted Marketing | Marketing Cost | Total Cost | Subscribers Found | Detection Rate |
|-------|-------|-------|-------|-------|-------|-------|-------|
| Baseline  | 568  | $113,600  | 254  | $2,540  | $116,140  | 490  | 46.3%  |
| SMOTE  | 376  | $75,200  | 529  | $5,290  | $80,490  | 682  | 64.5%  |
| Class Weighted  | 138  | $27,600  | 1,158  | $11,580  | $39,180  | 920  | 87.0%  |
| Threshold Tuned  | 251  | $50,200  | 672  | $6,720  | $56,920  | 807  | 76.3%  |
| Cost-Optimized | 58      |  $11,600      | 1,897     | $18,970      | $30,570      |   1,000     |   94.5%    |

This table illustrate why business-focused optimization outperforms statistical metrics - the approach with the lowest F1-score (Cost-Optimized) delivers the best business outcome by understanding the true cost structure.

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Jupyter Notebook

---

# Repository Structure

<details>
<summary>Repository Structure</summary>

```text
Bank-TeleMarketing/
│
├── data/
│ |── raw/
│ │ ├── bank-full.csv
│ │ ├── bank.csv
│ │ ├── bank-additional.csv
│ │ ├── bank-additional-full.csv
│ │ └── bank-names.txt
| |
│ │── processed/
│   ├── clean.csv
│   └── funnel-data.csv
│
├── notebooks/
│   ├── 01-data-cleaning-and-eda.ipynb
│   ├── 02-bivariate-correlation-analysis.ipynb
│   ├── 03-multivariate-analysis.ipynb
│   └── 04-funnel-conversion-and-performance-analysis.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── my_file.py
│   └── my_library.py
│
├── images/
├── models/
├── requirements.txt
└── README.md

---


