### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import torch
from torch.utils.data import Dataset

import os
import multiprocessing
import numpy as np
import open3d as o3d
from scipy.interpolate import interp1d

CFG = {'num_threads': 4, 'GPUs': 1}

def save_surface_ply(surface_points, normals, filename, color=None):
    try:
        if (len(surface_points)  < 1):
            return
        # Create an Open3D point cloud object and populate it
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_points.astype(np.float32))
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float16))
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(color.astype(np.float16))

        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save to a temporary file first to ensure data integrity
        temp_filename = filename.replace(".ply", "_temp.ply")
        o3d.io.write_point_cloud(temp_filename, pcd)

        # Rename the temp file to the original filename
        os.rename(temp_filename, filename)
    except Exception as e:
        print(f"Error saving surface PLY: {e}")

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

class GridDataset(Dataset):
    def __init__(self, pointcloud_base, start_block, path_template, umbilicus_points, umbilicus_points_old, grid_block_size=200, recompute=False, fix_umbilicus=False, maximum_distance=-1):
        self.grid_block_size = grid_block_size
        self.path_template = path_template
        self.umbilicus_points = umbilicus_points
        self.blocks_to_process = self.init_blocks_to_process(pointcloud_base, start_block, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance)

    def init_blocks_to_process(self, pointcloud_base, start_block, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance):
        # Load the set of computed blocks
        computed_blocks = self.load_computed_blocks(pointcloud_base)

        # Initialize the blocks that need computing
        self.blocks_to_compute(start_block, computed_blocks, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance)

    def load_computed_blocks(self, pointcloud_base):
        computed_blocks = set()
        # Try to load the list of computed blocks
        try:
            with open(os.path.join("/", pointcloud_base, "computed_blocks.txt"), "r") as f:
                # load saved tuples with 3 elements
                computed_blocks = set([eval(line.strip()) for line in f])
        except FileNotFoundError:
            print("[INFO]: No computed blocks found.")
        except Exception as e:
            print(f"Error loading computed blocks: {e}")
        return computed_blocks
    
    def blocks_to_compute(self, start_coord, computed_blocks, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance):
        print('start_coord: ', start_coord)
        print('computed_blocks: ', computed_blocks)
        print('umbilicus_points.shape: ', umbilicus_points.shape)
        print('umbilicus_points_old: ', umbilicus_points_old)
        print('path_template: ', path_template)
        print('grid_block_size: ', grid_block_size)
        print('recompute: ', recompute)
        print('fix_umbilicus: ', fix_umbilicus)
        print('maximum_distance: ', maximum_distance)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

def grid_inference(pointcloud_base, start_block, path_template, umbilicus_points, umbilicus_points_old, grid_block_size=200, recompute=False, fix_umbilicus=False, maximum_distance=-1):
    dataset = GridDataset(pointcloud_base, start_block, path_template, umbilicus_points, umbilicus_points_old, grid_block_size=grid_block_size, recompute=recompute, fix_umbilicus=fix_umbilicus, maximum_distance=maximum_distance)

    return

def compute(base_path, volume_subpath, pointcloud_subpath, maximum_distance, recompute, fix_umbilicus, start_block, num_threads, gpus, skip_surface_blocks):
    CFG['num_threads'] = num_threads
    CFG['GPUs'] = gpus
    
    pointcloud_base = os.path.dirname(pointcloud_subpath)
    umbilicus_path = '../umbilicus.txt'
    save_umbilicus_path = umbilicus_path.replace(".txt", ".ply")

    src_dir = base_path + "/" + volume_subpath + "/"
    path_template = src_dir + "cell_yxz_{:03}_{:03}_{:03}.tif"

    # Usage
    umbilicus_raw_points = load_xyz_from_file(umbilicus_path)
    umbilicus_points = umbilicus(umbilicus_raw_points)
    # Red color for umbilicus
    colors = np.zeros_like(umbilicus_points)
    colors[:,0] = 1.0
    # Save umbilicus as a PLY file, for visualization (CloudCompare)
    save_surface_ply(umbilicus_points, np.zeros_like(umbilicus_points), save_umbilicus_path, color=colors)

    umbilicus_points_old = None

    # Starting grid block at corner (3000, 4000, 2000) to match cell_yxz_006_008_004
    # (2600, 2200, 5000)
    if not skip_surface_blocks:
        # compute_surface_for_block_multiprocessing(start_block, pointcloud_base, path_template, save_template_v, save_template_r, umbilicus_points, grid_block_size=200, recompute=recompute, fix_umbilicus=fix_umbilicus, umbilicus_points_old=umbilicus_points_old, maximum_distance=maximum_distance)
        grid_inference(pointcloud_base, start_block, path_template, umbilicus_points, umbilicus_points_old, grid_block_size=200, recompute=recompute, fix_umbilicus=fix_umbilicus, maximum_distance=maximum_distance)
    else:
        print("Skipping surface block computation.")

def main():
    maximum_distance= -1 #1750 # maximum distance between blocks to compute and the umbilicus (speed up pointcloud generation if only interested in inner part of scrolls)
    recompute=False # whether to completely recompute all already processed blocks or continue (recompute=False). 
    fix_umbilicus = False
    start_block = (3000, 4000, 2000) # scroll1

    base_path = ""
    volume_subpath = "../2dtifs_8um_grids"
    pointcloud_subpath = "../point_cloud"
    num_threads = CFG['num_threads']
    gpus = CFG['GPUs']
    skip_surface_blocks = False

    compute(base_path, volume_subpath, pointcloud_subpath, maximum_distance, recompute, fix_umbilicus, start_block, num_threads, gpus, skip_surface_blocks)

if __name__ == "__main__":
    main()