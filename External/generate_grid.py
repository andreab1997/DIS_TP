from dis_tp import ReadTxt, Integration
from dis_tp.MatchingFunc import Mbg_3_reg_inv, Mbq_3_reg_inv
from dis_tp.TildeCoeffFunc import Cq_3_til_reg, Cg_3_til_reg, CLq_3_til_reg, CLg_3_til_reg
import sys
import pathlib

here = pathlib.Path(__file__).parent


if __name__ == "__main__":

    h_id = int(sys.argv[1])
    flavor = str(sys.argv[2])
    n_cores = int(sys.argv[3])
    grid_type=sys.argv[4]

    Integration.Initialize_all()
    if grid_type == "matching":
        if flavor == "bq":
            func = Mbq_3_reg_inv
        elif flavor == "bg":
            func = Mbg_3_reg_inv
        path = here / f"M{flavor}_3/M{flavor}3_nf{h_id}.txt"
    elif grid_type == "tilde":
        if flavor == "2q":
            func = Cq_3_til_reg
            path= here/ f"C{flavor[-1]}_3_til/C{flavor[-1]}til_nf{h_id}.txt"
        elif flavor == "2g":
            func = Cg_3_til_reg
            path= here/ f"C{flavor[-1]}_3_til/C{flavor[-1]}til_nf{h_id}.txt"
        elif flavor == "Lq":
            func = CLq_3_til_reg
            path= here/ f"C{flavor}_3_til/C{flavor}til_nf{h_id}.txt"
        elif flavor == "Lg":
            func = CLg_3_til_reg
            path= here/ f"C{flavor}_3_til/C{flavor}til_nf{h_id}.txt"
    
    obj = ReadTxt.Construct_Grid(func, h_id=h_id, path=path, grid_type=grid_type, n_pools=n_cores)
    obj.run()

