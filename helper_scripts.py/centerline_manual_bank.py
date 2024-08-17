import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev, interp1d

# Load and parse the CSV file
def load_csv_file(file_path):
    data = pd.read_csv(file_path)
    x_csv = data.iloc[:, 0].values
    y_csv = data.iloc[:, 2].values
    z_csv = data.iloc[:, 1].values
    return x_csv, y_csv, z_csv

# Find the closest point in `inside` to each point in `outside`
def find_closest_points(inside, outside):
    distances = cdist(outside, inside)
    closest_indices = np.argmin(distances, axis=1)
    closest_points = inside[closest_indices]
    return closest_points

# Apply a low-pass filter to data
def low_pass_filter_with_sync(x_data, y_data, z_data, alpha=0.1):
    filtered_x, filtered_y, filtered_z = [x_data[0]], [y_data[0]], [z_data[0]]
    
    for i in range(1, len(z_data)):
        new_z = alpha * z_data[i] + (1 - alpha) * filtered_z[-1]
        if new_z != filtered_z[-1]:  # Only add the point if the Z value changes
            filtered_z.append(new_z)
            filtered_x.append(x_data[i])
            filtered_y.append(y_data[i])
    
    return np.array(filtered_x), np.array(filtered_y), np.array(filtered_z)

# Interpolate points to create a smooth centerline
def interpolate_points(x, y, z, num_points=1000):
    tck, _ = splprep([x, y, z], s=0)
    u_new = np.linspace(0, 1, num_points)
    new_x, new_y, new_z = splev(u_new, tck)
    
    return np.array(new_x), np.array(new_y), np.array(new_z)

# Calculate the centerline coordinates
def calculate_centerline(inside_x, inside_y, inside_z, outside_x, outside_y, outside_z, N):
    inside_points = np.column_stack((inside_x, inside_y, inside_z))
    outside_points = np.column_stack((outside_x, outside_y, outside_z))
    closest_points_inside = find_closest_points(inside_points, outside_points)
    
    # Calculate rolling averages
    def rolling_average(arr, N):
        return np.convolve(arr, np.ones(N)/N, mode='valid')
    
    outside_x_avg = rolling_average(outside_x, N)
    outside_y_avg = rolling_average(outside_y, N)
    outside_z_avg = rolling_average(outside_z, N)
    
    closest_x_avg = rolling_average(closest_points_inside[:, 0], N)
    closest_y_avg = rolling_average(closest_points_inside[:, 1], N)
    closest_z_avg = rolling_average(closest_points_inside[:, 2], N)
    
    # Calculate centerline coordinates using rolling averages
    center_x = (outside_x_avg + closest_x_avg) / 2
    center_y = (outside_y_avg + closest_y_avg) / 2
    center_z = (outside_z_avg + closest_z_avg) / 2
    
    return center_x, center_y, center_z

# Plot the centerline and allow the user to input bank angles
def plot_centerline(center_x, center_y):
    plt.figure(figsize=(10, 6))
    plt.plot(center_x, center_y, color='green', label='Centerline', linestyle='--')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Centerline with Abscissas')
    
    abscissas = np.linspace(0, len(center_x)-1, 100, dtype=int)
    plt.scatter(center_x[abscissas], center_y[abscissas], color='red', s=50)
    for abscissa in abscissas:
        plt.text(center_x[abscissa], center_y[abscissa], str(abscissa), fontsize=12, ha='right')
    
    plt.grid(True)
    plt.show()

    return abscissas

# Interpolate bank angles and map back to the centerline
def interpolate_bank_angles(center_x, center_y, selected_abscissas, selected_bank_angles):
    interp_func = interp1d(selected_abscissas, selected_bank_angles, kind='linear', fill_value='extrapolate')
    all_bank_angles = interp_func(np.arange(len(center_x)))
    
    return all_bank_angles

# Write the results to a CSV file
def write_results_to_csv(file_path, center_x, center_y,center_z, bank_angles):
    results_data = np.column_stack((center_x, center_y, center_z, bank_angles))
    np.savetxt(file_path, results_data, delimiter=',', header='x,y,height, bank_angle', comments='', fmt='%.6f')

# Plot the final results
def plot_final_results(center_x, center_y, bank_angles):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(center_x, center_y, c=bank_angles, cmap='viridis', s=10)
    plt.colorbar(scatter, label='Bank Angle (radians)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Final Bank Angles Along Centerline')
    plt.grid(True)
    plt.show()

# File paths (replace with your actual file paths)
in_file_path = 'IMS INNER LINE_smoothed.csv'
out_file_path = 'IMS OUTER WALL_smoothed.csv'
result_file_path = 'IMS_centerline_with_bank_angles.csv'

# Load the data
x_in, y_in, z_in = load_csv_file(in_file_path)
x_out, y_out, z_out = load_csv_file(out_file_path)

# Apply low-pass filter to the heights
alpha = 0.08
filtered_x_in, filtered_y_in, filtered_z_in = low_pass_filter_with_sync(x_in, y_in, z_in, alpha)
filtered_x_out, filtered_y_out, filtered_z_out = low_pass_filter_with_sync(x_out, y_out, z_out, alpha)

# Interpolate and calculate the centerline
num_interpolated_points = 400
interpolated_x_in, interpolated_y_in, interpolated_z_in = interpolate_points(filtered_x_in, filtered_y_in, filtered_z_in, num_points=num_interpolated_points)
interpolated_x_out, interpolated_y_out, interpolated_z_out = interpolate_points(filtered_x_out, filtered_y_out, filtered_z_out, num_points=num_interpolated_points)

center_x, center_y, center_z = calculate_centerline(interpolated_x_in, interpolated_y_in, interpolated_z_in, interpolated_x_out, interpolated_y_out, interpolated_z_out, N=3)

# Plot the centerline with abscissas
abscissas = plot_centerline(center_x, center_y)

# Input bank angles manually
selected_abscissas = []
selected_bank_angles = []

while True:
    abscissa_input = input("Enter an abscissa (or type 'done' to finish): ")
    if abscissa_input.lower() == 'done':
        break
    try:
        abscissa_input = float(abscissa_input)
        if abscissa_input not in abscissas:
            print(f"Abscissa {abscissa_input} is not in the list of available abscissas.")
            continue
        bank_angle = float(input(f"Enter the bank angle for abscissa {abscissa_input}: "))
        selected_abscissas.append(abscissa_input)
        selected_bank_angles.append(bank_angle)
    except ValueError:
        print("Invalid input. Please enter numerical values only.")


# Interpolate bank angles along the entire centerline
all_bank_angles = interpolate_bank_angles(center_x, center_y, selected_abscissas, selected_bank_angles)

# Write the results to a CSV file
write_results_to_csv(result_file_path, center_x, center_y, center_z, all_bank_angles)

# Plot the final results
plot_final_results(center_x, center_y, all_bank_angles)
