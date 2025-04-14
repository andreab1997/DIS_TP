import pathlib
import sys
import time
from multiprocessing import Pool
import numpy as np

from dis_tp import MatchingFunc

here = pathlib.Path(__file__).parent
here.mkdir(exist_ok=True)

MB = 4.92
n_threads = 4

def read_grid(input_file):
    # Open the file in read mode
    with open(input_file, 'r') as file:
        # Read the entire line from the file
        line = file.readline().strip()

    # Split the line into individual numbers and convert them into integers
    numbers = list(map(float, line.split()))

    return numbers

def function_to_exe_in_parallel(pair):
    z, q, nf = pair
    res = MatchingFunc.Mbg_3_reg(z, [MB, q], nf)
    return res

def run(n_threads, x_grid, q_grid, nf):
    grid = []
    for x in x_grid:
        for q in q_grid:
            grid.append((x, q, nf))
    args = (function_to_exe_in_parallel, grid)
    with Pool(n_threads) as pool:
        result = pool.map(*args)
    return result

def produce_grid(nf):
    output_file = f"grids/Mbg_3/Mbg3_nf{nf}.txt"
    x_fname = "x.txt"
    x_grid = read_grid(x_fname)
    q_fname = "Q.txt"
    q_grid = read_grid(q_fname)

    start = time.perf_counter()
    res_vec = np.array(run(n_threads, x_grid, q_grid, nf))
    print("total running time: ", time.perf_counter() - start)

    res_mat = res_vec.reshape(len(q_grid), len(x_grid))

    np.save(output_file, res_mat)


if __name__ == "__main__":
    produce_grid(nf = 5)
