import os
import time
from multiprocessing import Pool, set_start_method
import numpy as np
import sys
from tqdm import tqdm

from dis_tp import parameters, MatchingFunc

set_start_method("fork")

if len(sys.argv) == 1:
    debug = False
elif len(sys.argv) == 2:
    debug = bool(sys.argv[1])
else:
    raise ValueError("Too many command line arguments!!")

n_threads = 1

mQ = {4: 1.51, 5: 4.92}

def read_grid(input_file):
    # Open the file in read mode
    with open(input_file, 'r') as file:
        # Read the entire line from the file
        line = file.readline().strip()

    # Split the line into individual numbers and convert them into integers
    numbers = list(map(float, line.split()))

    return numbers

def function_to_exe_in_parallel_g(pair):
    z, q, nf = pair
    res = MatchingFunc.Mbg_3_reg(z, [mQ[nf], q], nf, use_analytic=True)
    #print(z, q, res)
    return res

def function_to_exe_in_parallel_q(pair):
    z, q, nf = pair
    res = MatchingFunc.Mbq_3_reg(z, [mQ[nf], q], nf, use_analytic=True)
    #print(z, q, res)
    return res

def run(n_threads, x_grid, q_grid, nf, channel):
    grid = []
    for q in q_grid:
        for x in x_grid:
            grid.append((x, q, nf))
    if channel == "g":
        args = (function_to_exe_in_parallel_g, grid)
    if channel == "q":
        args = (function_to_exe_in_parallel_q, grid)
    with Pool(n_threads) as pool:
        result = list(tqdm(pool.imap(*args), total=len(grid)))
    return result

def produce_grid(nf, channel, debug=False):
    print(f"Producing Mb{channel}_3(nf={nf})")
    parameters.initialize_theory(use_grids=False, masses=[1.51, 4.92, 172.5])
    
    output_dir = f"./Mb{channel}_3"
    output_file = output_dir + f"/Mb{channel}3_nf{nf}.txt"
    x_fname = "./x.txt"
    x_grid = read_grid(x_fname)
    q_fname = "./Q.txt"
    q_grid = read_grid(q_fname)

    if debug:
        x_grid = np.geomspace(1e-6, 1., 10)
        q_grid = np.geomspace(1, 150, 5)

    start = time.perf_counter()
    res_vec = np.array(run(n_threads, x_grid, q_grid, nf, channel))
    print("total running time: ", time.perf_counter() - start, "s")

    res_mat = res_vec.reshape(len(q_grid), len(x_grid))

    os.system(f"mkdir -p {output_dir}")
    np.savetxt(output_file, res_mat.T)


if __name__ == "__main__":
    for nf in range(4, 5 + 1):
        for channel in ["g", "q"]:
            produce_grid(nf, channel, debug)
