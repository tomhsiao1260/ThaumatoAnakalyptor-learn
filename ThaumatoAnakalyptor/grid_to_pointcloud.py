### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import os
import numpy as np
from scipy.interpolate import interp1d

CFG = {'num_threads': 4, 'GPUs': 1}

def load_xyz_from_file(filename='umbilicus.txt'):
    """
    Load a file with comma-separated xyz coordinates into a 2D numpy array.
    
    :param filename: The path to the file.
    :return: A 2D numpy array of shape (n, 3) where n is the number of lines/coordinates in the file.
    """
    return np.loadtxt(filename, delimiter=',')

def umbilicus(points_array):
    """
    Interpolate between points in the provided 2D array based on z values.

    :param points_array: A 2D numpy array of shape (n, 3) with y, z, and x coordinates.
    :return: A 2D numpy array with interpolated points for each 1 step in the z direction.
    """

    # Separate the coordinates
    y, z, x = points_array.T

    # Create interpolation functions for x and y based on z
    fx = interp1d(z, x, kind='linear', fill_value="extrapolate")
    fy = interp1d(z, y, kind='linear', fill_value="extrapolate")

    # Define new z values for interpolation
    z_new = np.arange(z.min(), z.max(), 1)

    # Calculate interpolated x and y values
    x_new = fx(z_new)
    y_new = fy(z_new)

    # Return the combined y, z, and x values as a 2D array
    return np.column_stack((y_new, z_new, x_new))

def compute(base_path, volume_subpath, pointcloud_subpath, num_threads, gpus):
    CFG['num_threads'] = num_threads
    CFG['GPUs'] = gpus

    umbilicus_path = '../umbilicus.txt'

    # Usage
    umbilicus_raw_points = load_xyz_from_file(umbilicus_path)
    umbilicus_points = umbilicus(umbilicus_raw_points)
    print(umbilicus_raw_points, umbilicus_points.shape)

def main():
    base_path = ""
    volume_subpath = "../2dtifs_8um_grids"
    pointcloud_subpath = "../point_cloud"
    num_threads = CFG['num_threads']
    gpus = CFG['GPUs']

    compute(base_path, volume_subpath, pointcloud_subpath, num_threads, gpus)

if __name__ == "__main__":
    main()