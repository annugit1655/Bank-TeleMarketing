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

***

### Drop-off 1 — Stage 1 to Stage 2 (35.2% Reduction)
- 13,020 customers were contacted via an __unknown channel__, with only a __4.1%__ conversion rate.
- Cellular contact achieved a **14.9%** conversion rate versus **4.1%** for the unknown channel (3.6× higher).
- Switching entirely to cellular outreach for these customers could have resulted in approximately 1,400 additional subscriptions.

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


