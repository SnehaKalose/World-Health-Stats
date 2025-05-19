import pandas as pd
from scipy.stats import ttest_ind

# Load the dataset
df = pd.read_csv('30-70cancerChdEtc.csv')

# Filter male and female data
male_data = df[df['Dim1'] == 'Male']['First Tooltip']
female_data = df[df['Dim1'] == 'Female']['First Tooltip']

# Perform an independent t-test (Welch's t-test assuming unequal variances)
t_stat, p_value = ttest_ind(male_data, female_data, equal_var=False)

# Print results
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Significant difference in mortality between genders.")
else:
    print("Fail to reject the null hypothesis: No significant difference found.")
