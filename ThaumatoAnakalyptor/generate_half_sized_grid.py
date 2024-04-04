### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def hello_world(i):
    print(i * i)

def downsample_folder_tifs(input_directory, output_directory, downsample_factor, num_threads):
    if downsample_factor == 1:
        print("Downsample factor is 1, skipping.")
        return
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = [f for f in os.listdir(input_directory) if f.endswith('.tif') and int(f.split('.')[0]) % downsample_factor == 0]
    print(files)
    print(f"Found {len(files)} even tif files to downsample.")

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