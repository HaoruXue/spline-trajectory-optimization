import pandas as pd
import numpy as np

# Load the CSV file
def load_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    return data

# Ensure the fourth column exists and set it to zeros
def set_fourth_column_to_zeros(data):
    if data.shape[1] < 3:
        # If there are fewer than 4 columns, add columns until there are 4
        for i in range(3 - data.shape[1]):
            data[data.shape[1]] = 0
    else:
        # If there are 4 or more columns, set the 4th column to zeros
        data[2] = 0
    return data

# Save the modified CSV file
def save_csv(data, output_file_path):
    data.to_csv(output_file_path, index=False, header=False)

# Main function
def main(input_file_path, output_file_path):
    # Load the original CSV
    data = load_csv(input_file_path)
    
    # Set the fourth column to zeros
    modified_data = set_fourth_column_to_zeros(data)
    
    # Save the modified data to a new CSV file
    save_csv(modified_data, output_file_path)
    print(f"Modified CSV saved to {output_file_path}")

# Example usage
if __name__ == "__main__":
    input_file_path = "CURVE_OUTSIDE_WALL.csv"  # Replace with your input file path
    output_file_path = "CURVE_OUTSIDE_WALL.csv"  # Replace with your desired output file path
    
    main(input_file_path, output_file_path)
