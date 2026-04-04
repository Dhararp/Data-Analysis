<div align="center">

# 📊 Customer Segmentation Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*An end-to-end Machine Learning web application that segments customers based on their purchasing behavior using K-Means Clustering.*

</div>

---

## 🚀 Overview
Understanding customer behavior is crucial for any business seeking to optimize their marketing strategies. This project implements an unsupervised machine learning model (K-Means Clustering) to group customers into behavioral segments based on two primary features: **Annual Income** and **Spending Score**. 

To make these data insights accessible to non-technical stakeholders, the underlying machine learning pipeline is served through an interactive, clean **Streamlit Web Dashboard**.

## ✨ Key Features
- **Interactive Web Interface**: Seamlessly upload CSV datasets directly into the browser.
- **Dynamic Elbow-Curve Optimization**: Automatically iterate through clusters to determine optimal model constraints.
- **Adjustable Hyperparameters**: Fine-tune the K-Means clustering algorithm interactively using UI sliders.
- **High-Quality Visualizations**: Renders comprehensive dataset previews and behavioral scatterplots utilizing `seaborn` and `matplotlib`.
- **Exportable Data**: Instantly download the segmented results and cluster inferences as a `.csv` for further CRM (Customer Relationship Management) integration.

## 🛠️ Technology Stack
* **Language:** Python
* **Machine Learning:** `scikit-learn` (K-Means Clustering)
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Frontend / Deployment:** `streamlit`

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dhararp/Data-Analysis.git
   cd Data-Analysis
   ```

2. **Install the required dependencies:**
   It's recommended to run this in an isolated Python virtual environment.
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn streamlit
   ```

3. **Launch the application:**
   ```bash
   streamlit run "Customer Segmentation A.py"
   ```

4. **Navigate to the dashboard:**
   Open your browser and navigate to `http://localhost:8501`. 
   
   *(Note: You can use the included `customer_segmentation_data.csv` to test the application immediately.)*

## 📁 Project Structure
```text
Data-Analysis/
├── Customer Segmentation A.py       # Core Streamlit application & ML pipeline
├── generate_test_data.py            # Local script for generating synthetic testing data
├── customer_segmentation_data.csv   # Mock dataset
└── README.md                        # Project documentation
```

## 🧠 Model Insights
The K-Means model partitions the customers into `k` discrete clusters by minimizing the within-cluster sum of squares (WCCS). 

By analyzing the visualization outputs, marketing teams can easily identify target demographics, such as:
* **High Income, High Spenders**: Target with premium, luxury offerings.
* **Low Income, High Spenders**: Target with discounts and high-value credit offerings.
* **High Income, Low Spenders**: Target with compelling, return-on-investment focused campaigns.

---
<div align="center">
  <b>Built by Dhararp</b> — <i>Seeking to extract actionable value from raw data.</i>
</div>
