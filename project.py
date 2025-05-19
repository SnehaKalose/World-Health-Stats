import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt # Import the matplotlib library and assign it to the alias 'plt'

# Load the dataset
df = pd.read_csv('30-70cancerChdEtc.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Calculate summary statistics
mean_val = df['first_tooltip'].mean()
median_val = df['first_tooltip'].median()
mode_val = df['first_tooltip'].mode().iloc[0]  # mode() can return multiple values, we pick the first

# Print results
print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Mode: {mode_val}")

# Select numeric columns for correlation
numeric_df = df.select_dtypes(include='number')

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

# Show correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# --- Visualization 1: Heatmap ---
plt.figure(figsize=(8, 6)) # Now 'plt' is recognized because it has been imported
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()

# --- Visualization 2: Pairplot ---
# You can include more features if needed
selected_features = ['period', 'first_tooltip']
sns.pairplot(df[selected_features])
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()


