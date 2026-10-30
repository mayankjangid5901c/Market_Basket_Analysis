# 🛒 Market Basket Analysis & Product Recommendation System Using Python

A complete, end-to-end Data Science and Machine Learning project implementing **Market Basket Analysis (MBA)** using the **Apriori Algorithm** and **Association Rule Mining**. The project features an interactive, modern **Streamlit Dashboard** and an automated **Cross-Selling Product Recommendation Engine**.

Designed for both real-world commercial retail intelligence and academic submissions (B.Tech / MCA / M.Sc / Data Science Portfolio).

---

## 📌 1. Project Overview & Objective

In retail and e-commerce, understanding customer purchasing behavior is critical for driving sales growth. **Market Basket Analysis** is an unsupervised data mining technique that uncovers associations between products by analyzing co-occurrences in transaction baskets.

### Key Objectives:
- **Analyze Customer Transactions:** Mine high-frequency itemsets and detect hidden shopping patterns.
- **Generate Actionable Association Rules:** Calculate mathematical metrics (**Support**, **Confidence**, **Lift**).
- **Deliver Real-Time Product Recommendations:** Enable retailers to recommend complementary products (cross-selling) when a user selects an item.
- **Strategic Product Bundling:** Identify pairs or triplets of items to create bundled promotional discounts.
- **Interactive Decision Dashboard:** Provide business stakeholders with visual analytics and dynamic hyperparameter tuning via Streamlit.

---

## 🛠️ 2. Technology Stack

| Component | Library / Technology | Purpose |
|---|---|---|
| **Programming Language** | Python 3.10+ | Core language |
| **Data Manipulation** | Pandas, NumPy | Data cleaning, filtering, matrix pivoting |
| **Data Visualization** | Matplotlib, Seaborn | Exploratory data analysis charts & plots |
| **Association Mining** | Mlxtend (`apriori`, `association_rules`) | Frequent itemset mining & rule generation |
| **Web Dashboard** | Streamlit | Modern interactive web UI & sliders |
| **Data Storage** | CSV / Excel (`openpyxl`) | High-volume transaction datasets |

---

## 📂 3. Project Directory Structure

```text
Market_basket_analysis/
├── app.py                     # Streamlit multi-page web application & UI
├── preprocessing.py           # Data ingestion, cleaning, sanitization & basket matrix builder
├── analysis.py                # EDA visualizations, Apriori mining & Association Rules logic
├── recommendations.py         # Product recommendation engine & bundle creator
├── requirements.txt           # Project dependencies & versions
├── README.md                  # Comprehensive documentation & presentation guide
├── data/                      # Transaction datasets
│   ├── online_retail.csv      # Full UCI Online Retail dataset (541k records)
│   ├── Online Retail.xlsx     # Original UCI Excel archive
│   └── sample_retail_data.csv # Curated fast-loading sample dataset
└── outputs/                   # Exported charts & mined rules
    ├── top_frequent_products.png
    ├── top_products_quantity.png
    ├── monthly_sales_trend.png
    ├── basket_size_distribution.png
    ├── orders_by_day.png
    ├── top_countries.png
    ├── frequent_itemsets.png
    ├── rules_scatter.png
    ├── top_rules_lift.png
    └── mined_association_rules.csv
```

---

## 📊 4. Dataset Information

The project utilizes the benchmark **Online Retail Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail).

- **Source:** Transnational data set containing all transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based non-store online retail firm.
- **Records:** 541,909 transactions across 38 countries.
- **Features:**
  - `InvoiceNo`: 6-digit integer unique to each transaction. Letters starting with 'C' indicate cancellations.
  - `StockCode`: Distinct 5-digit product code.
  - `Description`: Product/item name.
  - `Quantity`: Number of units purchased per transaction line.
  - `InvoiceDate`: Timestamp of invoice generation.
  - `UnitPrice`: Product price per unit in sterling (£).
  - `CustomerID`: 5-digit integer unique to each customer.
  - `Country`: Customer's country of residence.

> **Offline/Sample Dataset Fallback:** If the full dataset is absent, the system automatically falls back to `data/sample_retail_data.csv`, clearly labeled in the application interface.

---

## 🧹 5. Data Preprocessing Pipeline

The `preprocessing.py` module performs an automated data audit and sanitation workflow:

1. **Duplicate Removal:** Identifies and drops repeated identical transaction entries.
2. **Missing Value Handling:** Drops rows missing critical fields (`Description`, `InvoiceNo`). Fills guest `CustomerID` records.
3. **Cancellation Filtering:** Identifies and strips invoice numbers starting with `'C'` or `'c'`.
4. **Invalid Quantity & Price Sanitization:** Filters records where `Quantity <= 0` or `UnitPrice <= 0`.
5. **Description Cleaning & Non-Merchandise Removal:**
   - Strips leading/trailing whitespaces and converts to uppercase.
   - Filters administrative non-inventory records (`POSTAGE`, `DOTCOM POSTAGE`, `MANUAL`, `BANK CHARGES`, `AMAZON FEE`, `SAMPLES`, `DISCOUNT`).
6. **Feature Engineering:** Computes `TotalAmount = Quantity * UnitPrice`, `YearMonth`, `DayOfWeek`, and `Hour`.
7. **Basket Encoding:** Pivots records into an efficient boolean matrix (`InvoiceNo` $\times$ `Description`), where `True` indicates purchase.

---

## 🧮 6. Mathematical Foundations of Market Basket Analysis

Given a set of transactions $T$ and items $I$:

### 1. Support
Measures how frequently an itemset appears across all transactions.
$$\text{Support}(A \rightarrow B) = P(A \cap B) = \frac{\text{Count}(A \cup B)}{|T|}$$

### 2. Confidence
Measures the conditional probability that a customer buys item $B$ given that they have already purchased item $A$.
$$\text{Confidence}(A \rightarrow B) = P(B \mid A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$

### 3. Lift
Measures how much more likely item $B$ is purchased when item $A$ is purchased, compared to if $B$ was purchased independently at random.
$$\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)} = \frac{P(A \cap B)}{P(A) \cdot P(B)}$$

- **$\text{Lift} > 1$:** Positive correlation (Items boost each other's sales).
- **$\text{Lift} = 1$:** Independence (No association).
- **$\text{Lift} < 1$:** Negative correlation (Substitutes or competing items).

---

## 🚀 7. Installation and Execution

### Step 1: Clone or Navigate to the Project Folder
```bash
cd Market_basket_analysis
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 🖥️ 8. Dashboard Walkthrough & Modules

| Tab / Section | Key Capabilities |
|---|---|
| **🏠 Home** | Executive summary, KPI cards (Revenue, Transactions, Basket Size), dataset status badge, and top bundle cards. |
| **📊 Dataset Overview** | Cleaning audit report (duplicates dropped, cancellations removed), data schema, summary statistics, search filter, and CSV download. |
| **📈 Data Analysis (EDA)** | Interactive charts: Top 10 products by frequency/quantity, monthly sales trends, basket size distribution, orders by day/hour, and country volume. |
| **📦 Frequent Itemsets** | Apriori mining results, length filtering (single items, pairs, triplets), support distribution chart, and CSV export. |
| **🔗 Association Rules** | Multi-dimensional scatter plot (Support vs. Confidence vs. Lift), rule ranking bar chart, keyword filterable rules table, and CSV export. |
| **🎯 Product Recommendations** | Interactive product selector, top $N$ recommendations, metric sorting (Lift / Confidence / Support), business rationale cards, and strategic retail bundles. |

---

## 🎓 9. College Presentation & Viva Voce Guide

### 🎤 2-Minute Project Elevator Pitch
> *"Our project implements an end-to-end Market Basket Analysis and Recommendation System using Python and the Apriori algorithm. By mining hundreds of thousands of retail transactions from the UCI Online Retail repository, our system uncovers non-obvious co-purchasing patterns among customers. We evaluate these rules using statistical metrics—Support, Confidence, and Lift. The insights feed into an interactive Streamlit dashboard that allows retailers to visualize purchasing trends and generate real-time product recommendations. For example, when a shopper selects 'Alarm Clock Bakelike Green', our model recommends 'Alarm Clock Bakelike Red' with an 8.6x Lift ratio and 75% confidence, enabling businesses to boost their Average Order Value through targeted cross-selling."*

### ❓ Top Viva Questions & Model Answers

**Q1: Why did you choose the Apriori algorithm over other methods?**  
*Answer:* The Apriori algorithm utilizes the "Apriori Principle" (all non-empty subsets of a frequent itemset must also be frequent). This allows the algorithm to prune candidate itemsets systematically without computing combinations for infrequent items, making it mathematically transparent, reliable, and interpretable.

**Q2: What is the difference between Confidence and Lift?**  
*Answer:* Confidence measures how often $B$ is bought when $A$ is bought, but it ignores the baseline popularity of $B$. If $B$ is bought by 90% of all shoppers anyway, high confidence is misleading. Lift corrects for this by dividing confidence by $B$'s overall support. A Lift $> 1$ proves a true synergistic relationship beyond random chance.

**Q3: How do you handle non-product items like 'POSTAGE' or 'MANUAL'?**  
*Answer:* During data preprocessing in `preprocessing.py`, we created an administrative blacklist of service codes. Non-merchandise entries distort association rules by appearing artificially in many baskets. Removing them ensures the mined rules represent genuine consumer product affinities.

**Q4: How does your product recommendation system work?**  
*Answer:* When a product is selected, the recommender searches for association rules where the product appears in the antecedent. It retrieves the consequents, ranks them by Lift or Confidence, and formats an explanation. If a rare product has no mined rules above the support threshold, our fallback engine computes empirical basket co-occurrence directly.

---

## 📜 10. License & Acknowledgements

- **Dataset:** UCI Machine Learning Repository (Online Retail Dataset, donated by Daqing Chen, Sai Liang Sain, and Kun Guo).
- **Libraries:** Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Mlxtend.
- **Author:** Developed for Data Science Portfolio & Academic Excellence.
