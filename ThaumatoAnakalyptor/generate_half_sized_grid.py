### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import os
import tifffile
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from multiprocessing import Pool, cpu_count

def downsample_image(args):
    input_directory, output_directory, filename, downsample_factor = args
    filepath = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, filename)

    # Check if image already exists in the output directory
    if os.path.exists(output_path):
        return f"'{filename}' already exists in the output directory. Skipping."
    # print(f"Downsampling {filename}.")
    
    with tifffile.TiffFile(filepath) as tif:
        image = tif.asarray()
        # Check if the image is not empty
        if image.size == 0:
            return f"Warning: '{filename}' is empty. Skipping."
        # Downsample the image using the specified factor
        downsampled_image = resize(image, 
                                   (image.shape[0] // downsample_factor, image.shape[1] // downsample_factor),
                                   anti_aliasing=False,
                                   preserve_range=True).astype(image.dtype)

        # Save the downsampled image
        tifffile.imwrite(output_path, downsampled_image)
    return f"Downsampled and saved '{filename}'."

def downsample_folder_tifs(input_directory, output_directory, downsample_factor, num_threads):
    if downsample_factor == 1:
        print("Downsample factor is 1, skipping.")
        return
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = [f for f in os.listdir(input_directory) if f.endswith('.tif') and int(f.split('.')[0]) % downsample_factor == 0]
    print(f"Found {len(files)} even tif files to downsample.")

    # Check that all downsample_factor-th tif from min to max are present
    min_file = min([int(f.split('.')[0]) for f in files])
    max_file = max([int(f.split('.')[0]) for f in files])
    for i in range(min_file, max_file + 1, downsample_factor):  # Adjusted the range to step by downsample_factor for efficiency
        if (f"{i:04}.tif" not in files) and (f"{i:05}.tif" not in files):
            raise Exception(f"Missing {i:04}.tif")
        
    # Prepare the arguments for each process
    tasks = [(input_directory, output_directory, f, downsample_factor) for f in files]

    # Initialize a Pool of processes
    with Pool(processes=num_threads) as pool:
        # Process the files in parallel
        for _ in tqdm(pool.imap_unordered(downsample_image, tasks), total=len(tasks)):
            pass

    print('Downsampling complete.')

def process_block(args):
    bz, by, bx, directory_path, block_size, nz, ny, nx, tif_files = args
    block_directory = directory_path + '_grids'
    block = np.zeros((block_size, block_size, block_size), dtype=np.uint16)
    block_filename = f"cell_yxz_{by+1:03}_{bx+1:03}_{bz+1:03}.tif"
    block_path = os.path.join(block_directory, block_filename)

    for z in range(block_size):
        z_index = bz * block_size + z
        if z_index >= nz:
            break
        image_path = os.path.join(directory_path, tif_files[z_index])
        image_slice = tifffile.imread(image_path)
        y_slice, x_slice = (slice(b * block_size, min((b + 1) * block_size, d)) for b, d in ((by, ny), (bx, nx)))
        block[z, :y_slice.stop - y_slice.start, :x_slice.stop - x_slice.start] = image_slice[y_slice, x_slice]

    tifffile.imwrite(block_path, block)

def generate_grid_blocks(directory_path, block_size, num_threads):
    block_directory = directory_path + '_grids'
    os.makedirs(block_directory, exist_ok=True)
    tif_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.tif')])

    sample_image = tifffile.imread(os.path.join(directory_path, tif_files[0]))
    nz, ny, nx = len(tif_files), *sample_image.shape
    blocks_in_x, blocks_in_y, blocks_in_z = (int(np.ceil(d / block_size)) for d in (nx, ny, nz))

    tasks = [(bz, by, bx, directory_path, block_size, nz, ny, nx, tif_files) 
        for bz in range(blocks_in_z) for by in range(blocks_in_y) for bx in range(blocks_in_x)]
    
    # multiprocessing
    num_pools = max(1, num_threads // 3)
    with Pool(processes=num_pools) as pool:
        for _ in tqdm(pool.imap_unordered(process_block, tasks), total=len(tasks)):
            pass

    print('Grid blocks have been generated.')

def compute(input_directory, output_directory, downsample_factor, num_threads):
    downsample_folder_tifs(input_directory, output_directory, downsample_factor, num_threads)
    generate_grid_blocks(output_directory, 500, num_threads)

def main():
    input_directory = '../../full-scrolls/Scroll1.volpkg/volumes/20230205180739'
    output_directory = '../2dtifs_8um'
    downsample_factor = 2
    num_threads = cpu_count()

    compute(input_directory, output_directory, downsample_factor, num_threads)

if __name__ == '__main__':
    main()