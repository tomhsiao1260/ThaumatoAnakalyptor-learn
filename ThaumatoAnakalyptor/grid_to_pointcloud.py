### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from surface_detection import surface_detection

import os
import tifffile
import multiprocessing
import numpy as np
import open3d as o3d
from scipy.interpolate import interp1d

CFG = {'num_threads': 4, 'GPUs': 1}

def load_grid(path_template, cords, grid_block_size=500, cell_block_size=500, uint8=True):
    """
    path_template: Template for the path to load individual grid files
    cords: Tuple (x, y, z) representing the corner coordinates of the grid block
    grid_block_size: Size of the grid block
    cell_block_size: Size of the individual grid files
    """
    # make grid_block_size an array with 3 elements
    if isinstance(grid_block_size, int):
        grid_block_size = np.array([grid_block_size, grid_block_size, grid_block_size])
    
    # Convert corner coordinates to file indices and generate the file path
    # Starting indices
    file_x_start, file_y_start, file_z_start = cords[0]//cell_block_size, cords[1]//cell_block_size, cords[2]//cell_block_size
    # Ending indices
    file_x_end, file_y_end, file_z_end = (cords[0] + grid_block_size[0])//cell_block_size, (cords[1] + grid_block_size[1])//cell_block_size, (cords[2] + grid_block_size[2])//cell_block_size

    # Generate the grid block
    if uint8:
        grid_block = np.zeros((grid_block_size[2], grid_block_size[0], grid_block_size[1]), dtype=np.uint8)
    else:
        grid_block = np.zeros((grid_block_size[2], grid_block_size[0], grid_block_size[1]), dtype=np.uint16)

    # Load the grid block from the individual grid files and place it in the larger grid block
    for file_x in range(file_x_start, file_x_end + 1):
        for file_y in range(file_y_start, file_y_end + 1):
            for file_z in range(file_z_start, file_z_end + 1):
                path = path_template.format(file_x, file_y, file_z)

                # Check if the file exists
                if not os.path.exists(path):
                    # print(f"File {path} does not exist.")
                    continue

                # Read the image
                with tifffile.TiffFile(path) as tif:
                    images = tif.asarray()

                if uint8:
                    images = np.uint8(images//256)

                # grid block slice position for the current file
                x_start = max(file_x*cell_block_size, cords[0])
                x_end = min((file_x + 1) * cell_block_size, cords[0] + grid_block_size[0])
                y_start = max(file_y*cell_block_size, cords[1])
                y_end = min((file_y + 1) * cell_block_size, cords[1] + grid_block_size[1])
                z_start = max(file_z*cell_block_size, cords[2])
                z_end = min((file_z + 1) * cell_block_size, cords[2] + grid_block_size[2])

                # Place the current file in the grid block
                try:
                    grid_block[z_start - cords[2]:z_end - cords[2], x_start - cords[0]:x_end - cords[0], y_start - cords[1]:y_end - cords[1]] = images[z_start - file_z*cell_block_size: z_end - file_z*cell_block_size, x_start - file_x*cell_block_size: x_end - file_x*cell_block_size, y_start - file_y*cell_block_size: y_end - file_y*cell_block_size]
                except:
                    print(f"Error in grid block placement for grid block {cords} and file {file_x}, {file_y}, {file_z}")

    return grid_block

def grid_empty(path_template, cords, grid_block_size=500, cell_block_size=500):
    """
    Determines wheter a grid block is empty or not.
        path_template: Template for the path to load individual grid files
        cords: Tuple (y, x, z) representing the corner coordinates of the grid block
        grid_block_size: Size of the grid block
        cell_block_size: Size of the individual grid files
    """
    # make grid_block_size an array with 3 elements
    if isinstance(grid_block_size, int):
        grid_block_size = np.array([grid_block_size, grid_block_size, grid_block_size])
    
    # Convert corner coordinates to file indices and generate the file path
    # Starting indices
    file_y_start, file_x_start, file_z_start = cords[0]//cell_block_size, cords[1]//cell_block_size, cords[2]//cell_block_size
    # Ending indices
    file_y_end, file_x_end, file_z_end = (cords[0] + grid_block_size[0])//cell_block_size, (cords[1] + grid_block_size[1])//cell_block_size, (cords[2] + grid_block_size[2])//cell_block_size

    # Check wheter none of the files exist
    for file_y in range(file_y_start, file_y_end + 1):
        for file_x in range(file_x_start, file_x_end + 1):
            for file_z in range(file_z_start, file_z_end + 1):
                path = path_template.format(file_y, file_x, file_z)

                # Check if the file exists
                if os.path.exists(path):
                    return False
    
    return True

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

def umbilicus_xy_at_z(points_array, z_new):
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

    # Calculate interpolated x and y values
    x_new = fx(z_new)
    y_new = fy(z_new)

    # Return the combined y, z, and x values as a 2D array
    res = np.array([y_new, z_new, x_new])
    return res

def extract_size_tensor(points, normals, grid_block_position_min, grid_block_position_max):
    """
    Extract points and corresponding normals that lie within the given size range.

    Parameters:
        points (torch.Tensor): The point coordinates, shape (n, 3).
        normals (torch.Tensor): The point normals, shape (n, 3).
        grid_block_position_min (int): The minimum block size.
        grid_block_position_max (int): The maximum block size.

    Returns:
        filtered_points (torch.Tensor): The filtered points, shape (m, 3).
        filtered_normals (torch.Tensor): The corresponding filtered normals, shape (m, 3).
    """

    # Convert min and max to tensors for comparison
    min_tensor = torch.tensor([grid_block_position_min] * 3, dtype=points.dtype, device=points.device)
    max_tensor = torch.tensor([grid_block_position_max] * 3, dtype=points.dtype, device=points.device)

    # Create a mask to filter points within the specified range
    mask_min = torch.all(points >= min_tensor, dim=-1)
    mask_max = torch.all(points <= max_tensor, dim=-1)

    # Combine the masks to get the final mask
    mask = torch.logical_and(mask_min, mask_max)

    # Apply the mask to filter points and corresponding normals
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    # Reposition the points to be relative to the grid block
    filtered_points -= min_tensor

    return filtered_points, filtered_normals

# fixing the pointcloud because of computation with too short umbilicus
def skip_computation_block(corner_coords, grid_block_size, umbilicus_points, maximum_distance=2500):
    if maximum_distance <= 0:
        return False

    block_point = np.array(corner_coords) + grid_block_size//2
    umbilicus_point = umbilicus_xy_at_z(umbilicus_points, block_point[2])
    umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords

    umbilicus_point_dist = umbilicus_point - block_point
    umbilicus_point_dist = np.linalg.norm(umbilicus_point_dist)
    return umbilicus_point_dist > maximum_distance

class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, grid_block_size=200):
        super().__init__(write_interval="batch")  # or "epoch" for end of an epoch
        # num_threads = multiprocessing.cpu_count()
        # self.pool = multiprocessing.Pool(processes=num_threads)  # Initialize the pool once

        self.grid_block_size = grid_block_size

    def write_on_predict(self, predictions: list, batch_indices: list, dataloader_idx: int, batch, batch_idx: int, dataloader_len: int):
        # Example: Just print the predictions
        print(predictions)
        print("On predict")

    def write_on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int) -> None:
        if prediction is None:
            # print("Prediction is None")
            return
        # print(f"On batch end, len: {len(prediction)}")
        if len(prediction) == 0:
            # print("Prediction is empty")
            return
        
        (points_r_tensors, normals_r_tensors) = prediction

        grid_volumes = points_r_tensors[0]
        grid_volumes = grid_volumes.numpy()
        grid_volumes = np.uint8(grid_volumes)

        print('batch end & grid size:', grid_volumes.shape)
        tifffile.imwrite('output.tif', grid_volumes) 

class GridDataset(Dataset):
    def __init__(self, pointcloud_base, start_block, path_template, umbilicus_points, umbilicus_points_old, grid_block_size=200, recompute=False, fix_umbilicus=False, maximum_distance=-1):
        self.grid_block_size = grid_block_size
        self.path_template = path_template
        self.umbilicus_points = umbilicus_points
        self.blocks_to_process = self.init_blocks_to_process(pointcloud_base, start_block, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance)
        self.blocks_to_process = self.blocks_to_process[21:22]

        self.writer = MyPredictionWriter(grid_block_size=grid_block_size)

    def init_blocks_to_process(self, pointcloud_base, start_block, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance):
        # Load the set of computed blocks
        computed_blocks = self.load_computed_blocks(pointcloud_base)

        # Initialize the blocks that need computing
        blocks_to_process, blocks_processed = self.blocks_to_compute(start_block, computed_blocks, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance)
        blocks_to_process = sorted(list(blocks_to_process)) # Sort the blocks to process for deterministic behavior
        return blocks_to_process

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
        padding = 50

        all_corner_coords = set() # Set containing all the corner coords that need to be placed into processing/processed set.
        all_corner_coords.add(start_coord) # Add the start coord to the set of all corner coords
        blocks_to_process = set() # Blocks that need to be processed
        blocks_processed = set() # Set to hold the blocks that do not need to be processed. Either have been processed or don't need to be processed.

        while len(all_corner_coords) > 0:
            corner_coords = all_corner_coords.pop()

            # Load the grid block from corner_coords and grid size
            corner_coords_padded = np.array(corner_coords) - padding
            grid_block_size_padded = grid_block_size + 2 * padding

            # Check if the block is empty
            if grid_empty(path_template, corner_coords_padded, grid_block_size=grid_block_size_padded):
                blocks_processed.add(corner_coords)
                # Outside of the scroll, don't add neighbors
                continue
            # this next check comes after the empty check, else under certain umbilicus distances, there might be an infinite loop
            skip_computation_flag = skip_computation_block(corner_coords, grid_block_size, umbilicus_points, maximum_distance=maximum_distance)
            if skip_computation_flag:
                # Block not needed in processing
                blocks_processed.add(corner_coords)
            else:
                # Otherwise add corner coords to the blocks that need processing
                blocks_to_process.add(corner_coords)

            # Compute neighboring blocks
            for dx in [-grid_block_size, 0, grid_block_size]:
                for dy in [-grid_block_size, 0, grid_block_size]:
                    for dz in [-grid_block_size, 0, grid_block_size]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        if abs(dx) + abs(dy) + abs(dz) > grid_block_size:
                            continue
                        neighbor_coords = (corner_coords[0] + dx, corner_coords[1] + dy, corner_coords[2] + dz)

                        # Add the neighbor to the list of blocks to process if it hasn't been processed yet
                        if (neighbor_coords not in blocks_processed) and (neighbor_coords not in blocks_to_process) and (neighbor_coords not in all_corner_coords):
                            all_corner_coords.add(neighbor_coords)

        return blocks_to_process, blocks_processed
    
    def get_writer(self):
        return self.writer
    
    def get_reference_vector(self, corner_coords):
        block_point = np.array(corner_coords) + self.grid_block_size//2
        umbilicus_point = umbilicus_xy_at_z(self.umbilicus_points, block_point[2])
        umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords
        umbilicus_normal = block_point - umbilicus_point
        umbilicus_normal = umbilicus_normal[[2, 0, 1]] # corner coords to tif
        unit_umbilicus_normal = umbilicus_normal / np.linalg.norm(umbilicus_normal)
        return unit_umbilicus_normal

    def __len__(self):
        return len(self.blocks_to_process)

    def __getitem__(self, idx):
        corner_coords = self.blocks_to_process[idx]
        # load the grid block from corner_coords and grid size
        padding = 50
        corner_coords_padded = np.array(corner_coords) - padding
        grid_block_size_padded = self.grid_block_size + 2 * padding
        block = load_grid(self.path_template, corner_coords_padded, grid_block_size=grid_block_size_padded)
        reference_vector = self.get_reference_vector(corner_coords)
        
        # Convert NumPy arrays to PyTorch tensors
        block_tensor = torch.from_numpy(block).float()  # Convert to float32 tensor
        reference_vector_tensor = torch.from_numpy(reference_vector).float()

        return block_tensor, reference_vector_tensor, corner_coords, self.grid_block_size, padding

# Custom collation function
def custom_collate_fn(batches):
    # Initialize containers for the aggregated items
    blocks = []
    reference_vectors = []
    corner_coordss = []
    grid_block_sizes = []
    paddings = []

    # Loop through each batch and aggregate its items
    for batch in batches:
        block, reference_vector, corner_coords, grid_block_size, padding = batch
        blocks.append(block)
        reference_vectors.append(reference_vector)
        corner_coordss.append(corner_coords)
        grid_block_sizes.append(grid_block_size)
        paddings.append(padding)
        
    # Return a single batch containing all aggregated items
    return blocks, reference_vectors, corner_coordss, grid_block_sizes, paddings

class PointCloudModel(pl.LightningModule):
    def __init__(self):
        print("instantiating model")
        super().__init__()

    def forward(self, x):
        # Extract input information
        grid_volumes, reference_vectors, corner_coordss, grid_block_sizes, paddings = x

        points_r_tensors, normals_r_tensors, points_v_tensors, normals_v_tensors = [], [], [], []
        for grid_volume, reference_vector, corner_coords, grid_block_size, padding in zip(grid_volumes, reference_vectors, corner_coordss, grid_block_sizes, paddings):
            tensor_tuple = surface_detection(grid_volume, reference_vector, blur_size=11, window_size=9, stride=1, threshold_der=0.075, threshold_der2=0.002, convert_to_numpy=False)
            points_r_tensor, normals_r_tensor = tensor_tuple

            points_r_tensors.append(points_r_tensor)
            normals_r_tensors.append(normals_r_tensor)

        return (points_r_tensors, normals_r_tensors)

def grid_inference(pointcloud_base, start_block, path_template, umbilicus_points, umbilicus_points_old, grid_block_size=200, recompute=False, fix_umbilicus=False, maximum_distance=-1, batch_size=1):
    dataset = GridDataset(pointcloud_base, start_block, path_template, umbilicus_points, umbilicus_points_old, grid_block_size=grid_block_size, recompute=recompute, fix_umbilicus=fix_umbilicus, maximum_distance=maximum_distance)
    num_threads = multiprocessing.cpu_count() // int(1.5 * int(CFG['GPUs']))
    num_treads_for_gpus = 5
    num_workers = min(num_threads, num_treads_for_gpus)
    num_workers = max(num_workers, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=num_workers, prefetch_factor=3)
    model = PointCloudModel()

    writer = dataset.get_writer()
    trainer = pl.Trainer(callbacks=[writer], strategy="ddp")
    # trainer = pl.Trainer(callbacks=[writer], gpus=int(CFG['GPUs']), strategy="ddp")

    print("Start prediction")
    # Run prediction
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    print("Prediction done")

    return

def compute(base_path, volume_subpath, pointcloud_subpath, maximum_distance, recompute, fix_umbilicus, start_block, num_threads, gpus, skip_surface_blocks):
    CFG['num_threads'] = num_threads
    CFG['GPUs'] = gpus
    
    pointcloud_base = os.path.dirname(pointcloud_subpath)
    umbilicus_path = '../umbilicus.txt'
    save_umbilicus_path = umbilicus_path.replace(".txt", ".ply")

    src_dir = os.path.join(base_path, volume_subpath)
    path_template = os.path.join(src_dir, "cell_yxz_{:03}_{:03}_{:03}.tif")

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
    if not skip_surface_blocks:
        # compute_surface_for_block_multiprocessing(start_block, pointcloud_base, path_template, save_template_v, save_template_r, umbilicus_points, grid_block_size=200, recompute=recompute, fix_umbilicus=fix_umbilicus, umbilicus_points_old=umbilicus_points_old, maximum_distance=maximum_distance)
        grid_inference(pointcloud_base, start_block, path_template, umbilicus_points, umbilicus_points_old, grid_block_size=200, recompute=recompute, fix_umbilicus=fix_umbilicus, maximum_distance=maximum_distance, batch_size=1*int(CFG['GPUs']))
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