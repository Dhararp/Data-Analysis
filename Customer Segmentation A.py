"""
Customer Segmentation Analysis Script & Streamlit App
"""

import os
# Prevent K-Means memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st


def run_local_analysis():
    """Runs the procedural standard execution (for debugging or local CLI usage)"""
    print("--- Running Local Analysis ---")
    data_path = 'customer_segmentation_data.csv'
    
    if not os.path.exists(data_path):
        print(f"[Warning] Local data file '{data_path}' not found.")
        print("Please upload a CSV via the Streamlit web dashboard instead.")
        return

    data = pd.read_csv(data_path)
    print("Preview:\n", data.head())
    print("\nMissing values:\n", data.isnull().sum())
    print("\nDescribe:\n", data.describe())

    features = ['Annual_Income', 'Spending_Score']
    
    if not all(feature in data.columns for feature in features):
         print("Dataset missing required columns.")
         return

    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine Optimal Clusters and print (using 3 as an example)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    data['Cluster'] = clusters

    print("\nCluster counts:\n", data['Cluster'].value_counts())

    for cluster in sorted(data['Cluster'].unique()):
        print(f"\nCluster {cluster} Summary:")
        print(data[data['Cluster'] == cluster][features].mean())
        
    output_file = 'customer_segments.csv'
    data.to_csv(output_file, index=False)
    print(f"\nResults saved to '{output_file}'")


def run_streamlit_app():
    """Main Streamlit Web Application execution"""
    st.title("Customer Segmentation Analysis")
    st.write("Upload a CSV file containing `Annual_Income` and `Spending_Score` columns to analyze customer segments via K-Means clustering.")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview", data.head())

            # Feature Selection
            features = ['Annual_Income', 'Spending_Score']
            if all(feature in data.columns for feature in features):
                X = data[features]

                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Optimal number of clusters (Elbow Method)
                st.subheader("Elbow Curve")
                st.write("The elbow curve helps determine the optimal number of clusters.")
                
                inertia = []
                for k in range(1, 11):
                    # random_state keeps results deterministic
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(X_scaled)
                    inertia.append(kmeans.inertia_)

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(1, 11), inertia, marker='o', linestyle='--')
                ax.set_title('Elbow Method')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('Inertia')
                st.pyplot(fig) # Draw the matplotlib plot safely in the web page

                # K-means Clustering UI
                st.subheader("K-means Clustering Configuration")
                optimal_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3)
                
                # Execute K-means
                kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                data['Cluster'] = clusters
                
                st.write("### Clustered Data Preview", data.head())

                # Visualize Clusters
                st.subheader("Cluster Visualization")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                
                # Use Seaborn to gracefully plot a scatter
                sns.scatterplot(
                    x='Annual_Income', y='Spending_Score', 
                    hue='Cluster', data=data, palette='viridis', 
                    ax=ax2, s=80, edgecolor='black', alpha=0.8
                )
                
                ax2.set_title('Customer Segments')
                st.pyplot(fig2)

                # Download Clustered Data
                st.subheader("Download Results")
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download Clustered Data as CSV",
                    data=csv,
                    file_name='customer_segments.csv',
                    mime='text/csv',
                )
            else:
                st.error("The uploaded dataset must contain 'Annual_Income' and 'Spending_Score' columns.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")

if __name__ == "__main__":
    # Check if we are running in a streamlit context
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            # Standard Python CLI execution
            run_local_analysis()
        else:
            # Streamlit execution
            run_streamlit_app()
    except ImportError:
        # Fallback if standard execution or streamlit isn't fully set up this way
        run_streamlit_app()
