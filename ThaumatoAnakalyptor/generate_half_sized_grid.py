### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def downsample_image(tasks):
    print(tasks)

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

def compute(input_directory, output_directory, downsample_factor, num_threads):
    downsample_folder_tifs(input_directory, output_directory, downsample_factor, num_threads)

def main():
    input_directory = '../../full-scrolls/Scroll1.volpkg/volumes/20230205180739'
    output_directory = '../output'
    downsample_factor = 2
    num_threads = cpu_count()

    compute(input_directory, output_directory, downsample_factor, num_threads)

if __name__ == '__main__':
    main()