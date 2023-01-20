import numpy as np

pids = {"g": 21, "c": 4, "b": 5, "t": 6}


def number_active_flavors(h_id):
    return np.abs(h_id)

def number_light_flavors(h_id):
    return np.abs(h_id) - 1

def charges(h_id):
    ch = {
        4: 2.0 / 3.0,
        5: -1.0 / 3.0,
        6: 2.0 / 3.0,
    }
    return np.sign(h_id) * ch[h_id]

def masses(h_id):
    m={
        4: 1.51,
        5: 4.92,
        6: 172.5
    }
    return m[h_id]