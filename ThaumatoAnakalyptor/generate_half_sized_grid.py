### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def hello_world(i):
    print(i * i)

def main():

    num_threads = cpu_count()
    tasks = [1, 2, 3, 4, 5]

    with Pool(processes=num_threads) as pool:
        for _ in tqdm(pool.imap_unordered(hello_world, tasks), total=len(tasks)):
            pass

if __name__ == '__main__':
    main()