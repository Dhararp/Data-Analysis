import pandas as pd
import numpy as np

# Create synthetic data similar to what the script expects
np.random.seed(42)

# Generate 200 random customers
data = {
    'CustomerID': range(1, 201),
    'Gender': np.random.choice(['Male', 'Female'], 200),
    'Age': np.random.randint(18, 70, 200),
    'Annual_Income': np.random.randint(15, 137, 200),
    'Spending_Score': np.random.randint(1, 100, 200)
}

df = pd.DataFrame(data)
df.to_csv('customer_segmentation_data.csv', index=False)
print("Synthetic customer_segmentation_data.csv created successfully!")
