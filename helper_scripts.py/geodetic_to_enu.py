import csv
import pymap3d as pm
import numpy as np

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def filter_outliers(data, threshold):
    filtered_data = [data[0]]
    for i in range(1, len(data) - 1):
        dist_prev = distance(data[i], data[i - 1])
        dist_next = distance(data[i], data[i + 1])
        if dist_prev < threshold and dist_next < threshold:
            filtered_data.append(data[i])
    filtered_data.append(data[-1])
    return filtered_data

input_file = 'CURVE_OUTSIDE_WALL.csv'
output_file = 'CURVE_OUTSIDE_WALL_ENU.csv'
origin_lat = 36.27207268554108
origin_lon = -115.0108130553903
origin_alt = 0.0
ignore_line = 1
distance_threshold = 1000.0  # Adjust the threshold value as needed

input_data = []
output_data = []

with open(input_file, 'r') as f_in:
    csv_reader = csv.reader(f_in)
    for _ in range(ignore_line):
        next(csv_reader)
    for row in csv_reader:
        input_data.append(row)
        lat, lon, alt = pm.geodetic2enu(float(row[0]), float(row[1]), float(row[2]), origin_lat, origin_lon, origin_alt)
        output_data.append([lat, lon, alt] + row[3:])

filtered_output_data = filter_outliers(output_data, distance_threshold)

with open(output_file, 'w') as f_out:
    csv_writer = csv.writer(f_out)
    for row in filtered_output_data:
        csv_writer.writerow([str(item) for item in row])
