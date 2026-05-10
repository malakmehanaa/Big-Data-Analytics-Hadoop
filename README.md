# Customer Analytics & Machine Learning Pipeline

## 📖 Overview

This project is an end-to-end data engineering, analytics, and machine learning pipeline. It processes large-scale customer, lead, organization, and product datasets to extract business insights, engineer features, and predict business outcomes (like customer churn and lead conversion). The architecture spans from raw data ingestion and Big Data ETL processing to exploratory data analysis (EDA), predictive modeling, and an interactive business intelligence (BI) dashboard.

## 🗂️ Repository Structure

The repository consists of Python scripts for data merging, a Pig Latin script for big data processing, Jupyter notebooks for EDA and ML, and a Streamlit app for visualization.

### 1. Data Management & Integration

* **`separated_data.py`**: A Python script that loads five 10k-row CSV datasets (`customers`, `leads`, `organizations`, `people`, `products`) and combines them into a single multi-sheet Excel file (`merged_data.xlsx`) for easier distribution and review.
* **`merge_horizontal.py`**: A Python script that performs an outer horizontal merge on the five core datasets using their `Index` column. It appends source suffixes to column names to avoid duplication and outputs a flat, wide dataset (`merged_horizontal.csv`/`.xlsx`).

### 2. Big Data ETL Pipeline

* **`customers_full_pipeline_.pig`**: A robust Apache Pig script designed to handle large-scale data (e.g., 1 million row datasets). The pipeline executes in a single pass to:
  * Load raw data with a strict schema.
  * Clean data by removing header rows and null `customer_id`s, and normalizing strings.
  * Perform aggregations to extract subscription date ranges and top 5 customer countries.
  * Export a streamlined "ML-ready dataset" (~60MB) containing only essential identity and feature fields.

### 3. Exploratory Data Analysis (EDA)

* **`EDA visualization.ipynb`**: A Jupyter Notebook dedicated to data cleaning and visual exploration using Pandas, Matplotlib, and Seaborn. Key operations include:
  * Parsing temporal features (e.g., `Subscription Date`, `Date of birth`).
  * Standardizing numeric fields (`Price`, `Stock`, `Number of employees`).
  * Visualizing demographic distributions, top job titles, and product category counts.

### 4. Predictive Modeling

* **`Final.ipynb`**: The core Machine Learning notebook. It takes the cleaned, merged datasets and prepares them for predictive modeling. Features include:
  * **Strict Feature Engineering:** Calculating actionable metrics like `days_since_subscription` and `age` while dropping noisy columns (IDs, free text, emails).
  * **Model Training:** Training and comparing algorithms like Random Forest and Logistic Regression.
  * **Evaluation:** Debugging data leakage, fixing NaN/binning issues, interpreting feature importance, and generating final insights on temporal segmentation and churn patterns.

### 5. Interactive BI Dashboard

* **`pig_output_viz_app.py`**: A responsive Streamlit web application with a dark-themed UI that visualizes the outputs of the Pig ETL pipeline. It leverages Plotly to display:
  * High-level metric cards.
  * Interactive time-series charts (e.g., New Subscriptions by Month).
  * A searchable customer data table.

## 🚀 Setup and Execution

### Prerequisites

* **Python 3.8+**
* **Hadoop/Apache Pig** (for running the `.pig` script)
* Required Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`, `plotly`, `scikit-learn`, `openpyxl`.

```bash
pip install pandas numpy matplotlib seaborn streamlit plotly scikit-learn openpyxl
```

### Running the Pipeline

1. **Merge Raw Data:** Run the merging scripts to prepare your raw CSVs.
   ```bash
   python separated_data.py
   python merge_horizontal.py
   ```

2. **Run Big Data ETL:** Execute the Pig script on your Hadoop cluster to process the 1M row dataset.
   ```bash
   pig -f customers_full_pipeline_.pig
   ```

3. **Explore & Train Models:** Open the Jupyter notebooks to visualize the data and run the ML models.
   ```bash
   jupyter notebook "EDA visualization.ipynb"
   jupyter notebook "Final.ipynb"
   ```

4. **Launch Dashboard:** Start the interactive Streamlit dashboard to view the processed analytics.
   ```bash
   streamlit run pig_output_viz_app.py
   ```

## 🎯 Key Capabilities Demonstrated

* Handling multiple disparate data sources and merging them cleanly.
* Writing production-ready Big Data ETL scripts.
* Preventing data leakage and optimizing feature importance in Machine Learning.
* Building dynamic, visually appealing front-end dashboards for end-users.
