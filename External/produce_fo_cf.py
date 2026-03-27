from tqdm import tqdm
import time
import os
from multiprocessing import Pool
import numpy as np
import sys

import adani as ad

if len(sys.argv) == 1:
    debug = False
elif len(sys.argv) == 2:
    debug = bool(sys.argv[1])
else:
    raise ValueError("Too many command line arguments!!")

mQ = {4: 1.51, 5: 4.92}
n_threads = 1

approx2g = ad.ApproximateCoefficientFunction(3, "2", "g")
approx2q = ad.ApproximateCoefficientFunction(3, "2", "q")
approxLg = ad.ApproximateCoefficientFunction(3, "L", "g")
approxLq = ad.ApproximateCoefficientFunction(3, "L", "q")

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
    m2Q2 = mQ[nf]**2 / q**2
    m2mu2 = m2Q2

    res = approx2g.fxBand(z, m2Q2, m2mu2, nf - 1)
    return [res.GetLower(), res.GetCentral(), res.GetHigher()]

def function_to_exe_in_parallel_2q(pair):
    z, q, nf = pair
    m2Q2 = mQ[nf]**2 / q**2
    m2mu2 = m2Q2

    res = approx2q.fxBand(z, m2Q2, m2mu2, nf - 1)
    return [res.GetLower(), res.GetCentral(), res.GetHigher()]

def function_to_exe_in_parallel_Lg(pair):
    z, q, nf = pair
    m2Q2 = mQ[nf]**2 / q**2
    m2mu2 = m2Q2

    res = approxLg.fxBand(z, m2Q2, m2mu2, nf - 1)
    return [res.GetLower(), res.GetCentral(), res.GetHigher()]

def function_to_exe_in_parallel_Lq(pair):
    z, q, nf = pair
    m2Q2 = mQ[nf]**2 / q**2
    m2mu2 = m2Q2

    res = approxLq.fxBand(z, m2Q2, m2mu2, nf - 1)
    return [res.GetLower(), res.GetCentral(), res.GetHigher()]

def run(n_threads, x_grid, q_grid, nf, kind, channel):
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
        result = list(tqdm(
            pool.imap(*args),
            total=len(grid),
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        ))
    return result

def produce_grid(nf, kind, channel, debug = False):
    print(f"Producing FO grid for C{kind}{channel}(nf={nf})")
    x_fname = "../src/dis_tp/grids/x.txt"
    x_grid = read_grid(x_fname)
    q_fname = "../src/dis_tp/grids/Q.txt"
    q_grid = read_grid(q_fname)

    if debug:
        x_grid = np.logspace(1e-6, 1., 10)
        q_grid = np.logspace(1, 150, 5)

    start = time.perf_counter()
    res_vec = np.array(run(n_threads, x_grid, q_grid, nf, kind, channel))
    print("total running time: ", time.perf_counter() - start, "s")

    res_mat = res_vec.reshape(len(q_grid), len(x_grid), 3)

    for i in range(3):
        var = i - 1
        kind_ = kind if kind == "L" else ""
        output_dir = f"../src/dis_tp/grids/C{kind_}{channel}_3_m"
        output_file = output_dir + f"/C{kind}{channel}_nf{nf}_var{var}.txt"
        os.system(f"mkdir -p {output_dir}")
        np.savetxt(output_file, res_mat[:, :, i])


if __name__ == "__main__":
    for nf in [5]:
        for kind in ["2", "L"]:
            for channel in ["g", "q"]:
                produce_grid(nf, kind, channel, debug)
