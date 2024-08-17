import matplotlib.pyplot as plt
import pandas as pd

def plot_coordinates(csv_file, txt_file):
    # Load the CSV file
    csv_data = pd.read_csv(csv_file)
    # Load the text file (assuming it's space-separated or tab-separated)
    txt_data = pd.read_csv(txt_file, delim_whitespace=True, header=None)

    # Extract x and y coordinates from both files
    csv_x = csv_data.iloc[:, 0]
    csv_y = csv_data.iloc[:, 1]
    
    txt_x = txt_data.iloc[:, 0]
    txt_y = txt_data.iloc[:, 1]

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot CSV data
    plt.scatter(csv_x, csv_y, color='blue', label='CSV Data')
    
    # Plot Text file data
    plt.scatter(txt_x, txt_y, color='red', label='Text Data')
    
    plt.title('CSV vs Text File Coordinates')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.legend()
    
    plt.grid(True)
    plt.show()

# Example usage
csv_file = 'LVMS_SVL_ENU_TTL_LEFT_2.csv'
txt_file = '2_lvms_left copy.txt'
plot_coordinates(csv_file, txt_file)
