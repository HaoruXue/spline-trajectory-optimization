
import pandas as pd

# Replace 'file1.csv' and 'file2.csv' with the actual file paths
file1_path = 'LVMS_A_OPTM_4.5_0.csv'
file2_path = 'LVMS_A_OPTM_2.95_0.csv'

# Read the CSV files, skipping the first row
df1 = pd.read_csv(file1_path, skiprows=1, header=None)
df2 = pd.read_csv(file2_path, skiprows=1, header=None)

# Remove the last three columns from each DataFrame
df1_trimmed = df1.iloc[:, :-3]
df2_trimmed = df2.iloc[:, :-3]

# Save the trimmed DataFrames back to new CSV files
df1_trimmed.to_csv('LVMS_A_OPTM_4.5_0_trimmed.csv', index=False, header=None)
df2_trimmed.to_csv('LVMS_A_OPTM_2.95_0_trimmed.csv', index=False, header=None)