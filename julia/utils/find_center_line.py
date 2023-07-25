import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import LineString, Point
from sys import argv

# Read csv files
def read_csv_file(filename):
    df = pd.read_csv(filename)
    points = list(zip(df['x'], df['y'], df['z']))
    return LineString(points)

# Resample a linestring into n evenly spaced points
def resample(linestring, n):
    length = linestring.length
    return [Point(linestring.interpolate(i * length / n)) for i in range(n)]

# Compute the mid points between two sets of points
def compute_mid_points(points1, points2):
    mid_points = [(p1.x*0.5 + p2.x*0.5, p1.y*0.5 + p2.y*0.5, p1.z*0.5 + p2.z*0.5) for p1, p2 in zip(points1, points2)]
    return mid_points

def plot_points(points, color, ax):
    x,y,z = zip(*points)
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(x, y, z, c=color, s=0.1)

def main():
    if len(argv) != 4:
        print('Usage: python find_center_line.py <outer_boundary_csv> <inner_boundary_csv> <output_csv>')
        return
    
    outer_boundary_csv = argv[1]
    inner_boundary_csv = argv[2]
    output_csv = argv[3]

    # Read polygons from csv files
    line_outer = read_csv_file(outer_boundary_csv)
    line_inner = read_csv_file(inner_boundary_csv)

    # Compute the mid points
    resampled_outer_points = resample(line_outer, 1000)  # Adjust number of points as needed
    resampled_inner_points = resample(line_inner, 1000)  # Adjust number of points as needed

    mid_points = compute_mid_points(resampled_outer_points, resampled_inner_points)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot outer, inner, and mid points
    plot_points([(point.x, point.y, point.z) for point in resampled_outer_points], 'red', ax)
    plot_points([(point.x, point.y, point.z) for point in resampled_inner_points], 'blue', ax)
    plot_points(mid_points, 'green', ax)
    ax.legend(['Outer', 'Inner', 'Mid'])
    plt.show()

    # Save the mid points to a csv file
    df = pd.DataFrame(mid_points, columns=['x', 'y', 'z'])
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
