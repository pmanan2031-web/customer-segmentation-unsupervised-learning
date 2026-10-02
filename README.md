# 🧩 Customer Segmentation — RFM + Unsupervised Learning

<p align="center">

**Turning millions of retail transactions into actionable customer segments using RFM Analysis and Unsupervised Machine Learning.**

<br>

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas" />
<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn" />
<img src="https://img.shields.io/badge/Unsupervised-Learning-green?style=for-the-badge" />

</p>

---

## 📌 Project Overview

Customer segmentation is the process of grouping customers based on similar purchasing behaviour.

In this project, an **Online Retail transaction dataset** is transformed from transaction-level data into customer-level behavioural features using **RFM Analysis**:

* **R — Recency:** How recently a customer purchased
* **F — Frequency:** How frequently a customer purchased
* **M — Monetary:** How much a customer spent

After feature engineering and preprocessing, multiple unsupervised learning algorithms are evaluated:

* 🔵 K-Means Clustering
* 🌳 Agglomerative Hierarchical Clustering
* 🔷 DBSCAN

The goal is to identify meaningful customer groups and translate those groups into actionable business insights.

---

## 🎯 Business Problem

Retail businesses have thousands of customers with different purchasing behaviours.

Treating every customer in the same way can lead to inefficient marketing campaigns.

This project answers questions such as:

* Who are the most valuable customers?
* Which customers purchase frequently?
* Which customers have not purchased recently?
* Are there naturally occurring customer groups?
* Which clustering algorithm provides the most useful segmentation?
* How can customer segments support targeted marketing?

---

## 🎯 Project Objectives

* Clean and preprocess large-scale transaction data.
* Restrict analysis to relevant UK customers.
* Remove invalid transactions and missing customer identifiers.
* Create customer-level **RFM features**.
* Detect and control extreme values.
* Apply log transformation and feature scaling.
* Compare multiple clustering algorithms.
* Evaluate clustering quality using internal metrics.
* Analyse cluster behaviour using RFM characteristics.
* Translate clusters into meaningful business insights.
* Save the final scaler and clustering model for reuse.

---

# 🔄 End-to-End Workflow

```text
Raw Online Retail Data
          ↓
Data Cleaning
          ↓
UK Customer Filtering
          ↓
Remove Missing / Invalid Transactions
          ↓
Create TotalPrice
          ↓
Customer-Level Aggregation
          ↓
RFM Feature Engineering
          ↓
Outlier Treatment
          ↓
Log Transformation
          ↓
StandardScaler
          ↓
┌───────────────────────────────┐
│     Unsupervised Models       │
│                               │
│  K-Means                      │
│  Agglomerative Clustering     │
│  DBSCAN                       │
└───────────────────────────────┘
          ↓
Clustering Evaluation
          ↓
Cluster Interpretation
          ↓
Customer Personas
          ↓
Business Recommendations
```

---

# 📊 Dataset

The project uses the **Online Retail Dataset**.

The original dataset contains:

| Attribute             |          Value |
| --------------------- | -------------: |
| Original Transactions |      1,048,575 |
| Columns               |              8 |
| Cleaned Transactions  |        714,245 |
| Unique Customers      |          5,337 |
| Country Focus         | United Kingdom |

### Main Features

| Feature       | Description               |
| ------------- | ------------------------- |
| `Invoice`     | Invoice number            |
| `StockCode`   | Product/item code         |
| `Description` | Product description       |
| `Quantity`    | Number of units purchased |
| `InvoiceDate` | Transaction date and time |
| `Price`       | Unit price                |
| `Customer ID` | Customer identifier       |
| `Country`     | Customer country          |

### Dataset Source

The dataset is available from the UCI Machine Learning Repository:

**Online Retail Dataset**

https://archive.ics.uci.edu/dataset/352/online+retail

---

# 🧹 Data Cleaning

The raw transaction data contains missing values, returns and invalid transaction values.

The following preprocessing steps were performed:

### 1. Country Filtering

Only customers from the **United Kingdom** were retained.

### 2. Missing Customer IDs

Rows without a valid `Customer ID` were removed because customer-level segmentation requires a customer identifier.

### 3. Returns / Cancellations

Transactions with non-positive quantities were removed.

### 4. Invalid Prices

Rows containing zero or negative prices were removed.

### 5. Total Transaction Value

A new feature was created:

```python
TotalPrice = Quantity × Price
```

### 6. Date Conversion

`InvoiceDate` was converted into a proper datetime format.

### Result

```text
1,048,575 raw transactions
          ↓
      Cleaning
          ↓
714,245 valid transactions
          ↓
5,337 unique customers
```

---

# 📈 Exploratory Data Analysis

EDA was performed to understand the transaction and customer behaviour before clustering.

### Analysis Included

* Quantity distribution
* Unit price distribution
* Total transaction value
* Monthly transaction trends
* Customer purchase behaviour
* Revenue analysis
* RFM feature distributions
* Outlier analysis

The analysis reports **November 2011** as the highest transaction month and approximately **£14.34 million** in total revenue.

---

# 🧮 RFM Feature Engineering

RFM is the core feature-engineering step of this project.

| Feature      | Business Meaning                          |
| ------------ | ----------------------------------------- |
| 🕒 Recency   | Days since the customer's latest purchase |
| 🔁 Frequency | Number of unique invoices/orders          |
| 💰 Monetary  | Total amount spent by the customer        |

### RFM Calculation

For each customer:

```text
Customer
   │
   ├── Recency
   ├── Frequency
   └── Monetary
```

The result is a customer-level dataset instead of individual transaction-level records.

---

# 🛠️ RFM Preprocessing

Customer behaviour contains extreme values, especially in Frequency and Monetary spending.

The following preprocessing pipeline was applied:

```text
RFM Features
     ↓
Outlier Detection
     ↓
IQR-Based Capping
     ↓
Log Transformation
     ↓
StandardScaler
     ↓
Clustering Dataset
```

### Why Log Transformation?

Frequency and Monetary values can be highly right-skewed.

Log transformation helps reduce the effect of extreme values and makes the feature distributions more suitable for clustering.

### Why StandardScaler?

Clustering algorithms are distance-based.

Scaling ensures that one feature does not dominate the others simply because it has a larger numerical range.

---

# 🤖 Model 1 — K-Means Clustering

K-Means was evaluated using different values of `k`.

The project tested:

```text
k = 2 → 10
```

### Silhouette Scores

| Number of Clusters | Silhouette Score |
| -----------------: | ---------------: |
|                  2 |       **0.4322** |
|                  3 |           0.4081 |
|                  4 |           0.3737 |
|                  5 |           0.3832 |
|                  6 |           0.3522 |
|                  7 |           0.3440 |
|                  8 |           0.3445 |
|                  9 |           0.3187 |
|                 10 |           0.3167 |

Based on the tested configurations, **k = 2** produced the highest silhouette score.

### Selected Configuration

```python
KMeans(
    n_clusters=2,
    random_state=42
)
```

### Interpretation

A silhouette score of **0.4322** indicates that the selected clusters have meaningful separation, although the segmentation is not perfectly separated.

---

# 🌳 Model 2 — Agglomerative Hierarchical Clustering

Hierarchical clustering was explored using:

* Ward linkage
* Complete linkage
* Average linkage

A dendrogram was used to examine the hierarchical structure of customer groups.

The evaluated configuration reported:

```text
Clusters: 4
Linkage: Ward
Silhouette: 0.3441
```

This provides an alternative view of customer structure compared with K-Means.

---

# 🔷 Model 3 — DBSCAN

DBSCAN was used to identify dense groups and potential noise/outlier customers.

### Configuration

```text
eps = 0.5
min_samples = 5
```

Unlike K-Means, DBSCAN can explicitly identify noise points.

The selected configuration produced approximately:

```text
Noise = 48.72%
```

This indicates that a large proportion of customers were treated as noise under this configuration.

---

# 📊 Clustering Model Comparison

| Algorithm     | Configuration          | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz |  Noise |
| ------------- | ---------------------- | -------: | ---------: | -------------: | ----------------: | -----: |
| **K-Means**   | k=2                    |        2 | **0.4322** |     **0.8689** |       **5825.55** |     0% |
| Agglomerative | k=4, Ward              |        4 |     0.3441 |         0.9024 |           5036.51 |     0% |
| DBSCAN        | eps=0.5, min_samples=5 |        2 |     0.3306 |         1.0381 |           3091.64 | 48.72% |

### Evaluation Metrics

#### Silhouette Score

Measures how well each observation fits within its assigned cluster compared with other clusters.

**Higher is generally better.**

#### Davies-Bouldin Index

Measures average similarity between clusters.

**Lower is generally better.**

#### Calinski-Harabasz Index

Measures the ratio of between-cluster dispersion to within-cluster dispersion.

**Higher is generally better.**

---

# 🔁 K-Means Stability Testing

The selected K-Means configuration was evaluated across multiple random states:

```text
0
7
21
42
99
```

The reported mean silhouette score was:

```text
Mean = 0.4322
Std = 0.0000
```

The notebook therefore reports highly consistent results across these tested random states.

---

# 👥 Customer Segmentation & Personas

The final customer segments should be interpreted using their **RFM profiles**, rather than assigning arbitrary names solely from cluster numbers.

Typical business interpretations include:

### 🏆 High-Value / Loyal Customers

Characteristics may include:

* Low Recency
* High Frequency
* High Monetary value

Potential actions:

* Loyalty rewards
* Exclusive offers
* Early product access
* Cross-selling
* Personalized recommendations

---

### 🌱 Low-Engagement Customers

Characteristics may include:

* Lower purchase frequency
* Lower spending
* Less recent activity

Potential actions:

* Welcome campaigns
* Second-purchase incentives
* Product recommendations
* Engagement campaigns

---

### ⚠️ At-Risk Customers

Characteristics may include:

* High Recency
* Reduced purchase activity

Potential actions:

* Reactivation campaigns
* Limited-time discounts
* Personalized reminders
* Win-back campaigns

> **Note:** Persona names should always be based on the actual RFM statistics of the final cluster profiles. They should not be assumed solely from the cluster number.

---

# 💡 Business Insights

This project demonstrates how raw transaction data can be converted into customer-level intelligence.

### Key Insights

* Large transaction datasets can be reduced to meaningful customer-level features.
* RFM provides a practical framework for understanding customer behaviour.
* Feature preprocessing has a major impact on clustering.
* Different clustering algorithms produce different segmentation structures.
* K-Means produced the strongest internal clustering metrics among the tested configurations.
* DBSCAN identified a substantial number of observations as noise under the selected parameters.
* Customer segments can support targeted marketing and retention strategies.

---

# 🚀 Business Applications

The resulting segmentation can support:

### 🎯 Targeted Marketing

Send different campaigns to different customer groups.

### ❤️ Customer Retention

Identify customers whose purchasing activity has declined.

### 💎 Loyalty Programs

Provide rewards to highly engaged customers.

### 🛍️ Cross-Selling

Recommend complementary products based on purchasing behaviour.

### 🔄 Customer Reactivation

Target customers who have not purchased recently.

### 📊 Customer Analytics

Monitor changes in customer behaviour over time.

---

# 💾 Saved Machine Learning Artifacts

The project saves the preprocessing and clustering components for reuse:

```text
rfm_scaler.pkl
customer_segmentation_model.pkl
```

### `rfm_scaler.pkl`

Stores the fitted feature scaler.

### `customer_segmentation_model.pkl`

Stores the final K-Means clustering model.

The saved K-Means model uses:

```text
n_clusters = 2
random_state = 42
```

---

# 📁 Project Structure

```text
customer-segmentation-unsupervised-learning/
│
├── 📂 assets/
│   └── 📊 Project visualizations
│
├── 📓 customer_segmentation.ipynb
│
├── 🤖 customer_segmentation_model.pkl
│
├── ⚙️ rfm_scaler.pkl
│
├── 📄 requirements.txt
│
├── 📝 summary_report.md
│
└── 📖 README.md
```

---

# 🛠️ Tech Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook
* VS Code
* GitHub

---

# ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/pmanan2031-web/customer-segmentation-unsupervised-learning.git
```

### 2. Navigate to the Project

```bash
cd customer-segmentation-unsupervised-learning
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
customer_segmentation.ipynb
```

### 5. Run the Notebook

Execute the notebook cells from top to bottom to reproduce:

```text
Data Cleaning
      ↓
EDA
      ↓
RFM
      ↓
Preprocessing
      ↓
K-Means
      ↓
Hierarchical Clustering
      ↓
DBSCAN
      ↓
Evaluation
      ↓
Customer Segmentation
```

---

# 📊 Project Highlights

```text
1,048,575
     ↓
Raw Transactions

714,245
     ↓
Cleaned Transactions

5,337
     ↓
Unique Customers

RFM Features
     ↓
3 Clustering Algorithms
     ↓
Model Evaluation
     ↓
Customer Segmentation
     ↓
Business Insights
```

---

# 📌 Limitations

* The analysis is based on historical transaction behaviour.
* Clustering results depend on feature engineering and preprocessing.
* The selected number of clusters is specific to the evaluated dataset and methodology.
* DBSCAN performance is sensitive to `eps` and `min_samples`.
* Customer personas should be validated with actual business/customer data before production use.
* Internal clustering metrics do not necessarily represent direct business impact.

---

# 🔮 Future Improvements

Possible extensions include:

* Add PCA-based 2D/3D cluster visualization.
* Build an interactive Streamlit dashboard.
* Add automatic customer segment prediction.
* Experiment with Gaussian Mixture Models.
* Perform systematic hyperparameter tuning for DBSCAN.
* Track customer segments over time.
* Add CLV (Customer Lifetime Value).
* Build a marketing recommendation engine.
* Deploy the segmentation model as a web application.
* Add automated model monitoring.

---

# ⭐ Key Takeaway

> **This project demonstrates an end-to-end unsupervised machine learning workflow that converts raw retail transactions into actionable customer segments using RFM analysis and clustering.**

It combines:

**Data Cleaning → EDA → Feature Engineering → RFM → Outlier Treatment → Scaling → Clustering → Evaluation → Customer Personas → Business Insights**

---

# 👨‍💻 Author

## Manan Patel

Data Science / Machine Learning Enthusiast

🔗 **GitHub:**
https://github.com/pmanan2031-web

🔗 **Project Repository:**
https://github.com/pmanan2031-web/customer-segmentation-unsupervised-learning

---

<p align="center">

### ⭐ If you found this project useful, consider giving the repository a star!

**Built with Python & Scikit-learn**

</p>
