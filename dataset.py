# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column to the DataFrame
iris_df['target'] = iris.target

# Print the first few rows of the DataFrame
print(iris_df.head())
