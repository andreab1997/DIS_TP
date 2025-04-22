import os
import time
import sys
from multiprocessing import Pool
import numpy as np
from dis_tp import TildeCoeffFunc, Initialize, parameters

if len(sys.argv) == 1:
    debug = False
elif len(sys.argv) == 2:
    debug = bool(sys.argv[1])
else:
    raise ValueError("Too many command line arguments!!")

MB = 4.92
n_threads = 4
nflist = [4, 5]

def read_grid(input_file):
    # Open the file in read mode
    with open(input_file, 'r') as file:
        # Read the entire line from the file
        line = file.readline().strip()

    # Split the line into individual numbers and convert them into integers
    numbers = list(map(float, line.split()))

    return numbers

def function_to_exe_in_parallel_2g(pair):
    z, q, nf = pair

    res = TildeCoeffFunc.Cg_3_til_reg(z, q, [MB, q], nf, use_analytic=True)
    return res

def function_to_exe_in_parallel_2q(pair):
    z, q, nf = pair

    res = TildeCoeffFunc.Cq_3_til_reg(z, q, [MB, q], nf, use_analytic=True)
    return res

def function_to_exe_in_parallel_Lg(pair):
    z, q, nf = pair

    res = TildeCoeffFunc.CLg_3_til_reg(z, q, [MB, q], nf, use_analytic=True)
    return res

def function_to_exe_in_parallel_Lq(pair):
    z, q, nf = pair

    res = TildeCoeffFunc.CLq_3_til_reg(z, q, [MB, q], nf, use_analytic=True)
    return res

def run(n_threads, x_grid, q_grid, kind, channel, n3lo_var, nf):
    Initialize.InitializeQX()
    Initialize.InitializeHPL()
    Initialize.InitializeMbg_3(nflist)
    Initialize.InitializeMbq_3(nflist)

    Initialize.InitializeCq3_m(nflist, n3lo_var)
    Initialize.InitializeCLq3_m(nflist, n3lo_var)
    Initialize.InitializeCg3_m(nflist, n3lo_var)
    Initialize.InitializeCLg3_m(nflist, n3lo_var)

    grid = []
    for q in q_grid:
        for x in x_grid:
            grid.append((x, q, nf))
    
    if (kind, channel) == ("2", "g"):
        args = (function_to_exe_in_parallel_2g, grid)
    if (kind, channel) == ("L", "g"):
        args = (function_to_exe_in_parallel_Lg, grid)
    if (kind, channel) == ("2", "q"):
        args = (function_to_exe_in_parallel_2q, grid)
    if (kind, channel) == ("L", "q"):
        args = (function_to_exe_in_parallel_Lq, grid)
    
    with Pool(n_threads) as pool:
        result = pool.map(*args)
    return result

def produce_grid(nf, kind, channel, n3lo_var, debug = False):
    print(f"Producing tilde grid for C{kind}{channel}(nf={nf})")
    parameters.initialize_theory(use_grids=True, masses=[1.51, 4.92, 172.5])
    
    x_fname = "./x.txt"
    x_grid = read_grid(x_fname)
    q_fname = "./Q.txt"
    q_grid = read_grid(q_fname)

    if debug:
        x_grid = np.geomspace(1e-6, 1., 10, endpoint=False)
        q_grid = np.geomspace(1, 150, 5)

    start = time.perf_counter()
    res_vec = np.array(run(n_threads, x_grid, q_grid, kind, channel, n3lo_var, nf))
    print("total running time: ", time.perf_counter() - start, "s")

    res_mat = res_vec.reshape(len(q_grid), len(x_grid))

    kind_ = kind if kind == "L" else ""
    output_dir = f"./C{kind_}{channel}_3_til"
    output_file = output_dir + f"/C{kind}{channel}til_nf{nf}_var{n3lo_var}.txt"
    
    os.system(f"mkdir -p {output_dir}")
    np.savetxt(output_file, res_mat.T)


if __name__ == "__main__":
    for nf in [4, 5]:
        for kind in ["2", "L"]:
            for channel in ["g", "q"]:
                for n3lo_var in range(-1, 1+1):
                    produce_grid(nf, kind, channel, n3lo_var, debug)
