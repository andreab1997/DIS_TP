from dis_tp import ReadTxt, Integration
from dis_tp.parameters import masses, number_active_flavors
from dis_tp.MatchingFunc import Mbg_3_reg_inv, Mbq_3_reg_inv
import sys
import pathlib

here = pathlib.Path(__file__).parent


if __name__ == "__main__":

    h_id = int(sys.argv[1])
    matching = sys.argv[2]
    n_cores = int(sys.argv[3])
    quark = "c" if h_id == 4 else 'b'
    nf = number_active_flavors(h_id)

    Integration.Initialize_all()
    if matching == "bq":
        obj = ReadTxt.Construct_Grid(Mbq_3_reg_inv, mass=masses(h_id), nf=nf, path= here/ f"M{matching}_3/Mbq3_{quark}.txt", n_pools=n_cores)
    elif matching == "bg":
        obj = ReadTxt.Construct_Grid(Mbg_3_reg_inv, mass=masses(h_id), nf=nf, path= here/ f"M{matching}_3/Mbg3_{quark}.txt", n_pools=n_cores)
    obj.construct_grid_matching()