import pathlib
import os
import time
from multiprocessing import Pool
import numpy as np

from dis_tp import MatchingFunc, parameters

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
    res = MatchingFunc.Mbg_3_reg(z, [MB, q], nf + 1, use_analytic=True)
    #print(z, q, res)
    return res

def run(n_threads, x_grid, q_grid, nf):
    grid = []
    for q in q_grid:
        for x in x_grid:
            grid.append((x, q, nf))
    args = (function_to_exe_in_parallel, grid)
    with Pool(n_threads) as pool:
        result = pool.map(*args)
    return result

def produce_grid(nf, debug=False):
    print(f"Producing Mbg_3(nf={nf})")
    parameters.initialize_theory(use_grids=False, masses=[1.51, 4.92, 172.5])
    
    output_dir = f"../External/Mbg_3"
    output_file = output_dir + f"/Mbg3_nf{nf}.txt"
    x_fname = "../External/x.txt"
    x_grid = read_grid(x_fname)
    q_fname = "../External/Q.txt"
    q_grid = read_grid(q_fname)

    if debug:
        x_grid = np.geomspace(1e-6, 1., 10)
        q_grid = np.geomspace(1, 150, 5)

    start = time.perf_counter()
    res_vec = np.array(run(n_threads, x_grid, q_grid, nf))
    print("total running time: ", time.perf_counter() - start, "s")

    res_mat = res_vec.reshape(len(q_grid), len(x_grid))

    os.system(f"mkdir -p {output_dir}")
    np.savetxt(output_file, res_mat)


if __name__ == "__main__":
    produce_grid(3)
    produce_grid(4)
    produce_grid(5)
