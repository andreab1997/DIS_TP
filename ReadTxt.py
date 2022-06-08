#These are utility functions to read txt grids.
import numpy as np
import Initialize as Ini
def read1D(path_to_file):
    """
    Read a space-separated txt file and return a 1-Dimensional list of the values

    Parameters:
        path_to_file : str
            file to open
    Returns:
            : list
        list of values
    """
    for line in open(path_to_file):
        listWord = line.split(" ")
    mylist = [float(item) for item in listWord[:-1]]
    return mylist 

def read1D_Nic(path_to_file):
    """
    Read a space-separated txt file and return a 1-Dimensional list of the values (for triple space separated files)

    Parameters:
        path_to_file : str
            file to open
    Returns:
            : list
        list of values
    """
    for line in open(path_to_file):
        listWord = line.split("   ")
    mylist = [float(item) for item in listWord[:-1]]
    return mylist 

def readND(path_to_file): 
    """
    Read a space-separated txt file and return a N-Dimensional list of the values

    Input:
        path_to_file : str
            file to open
    Returns:
            : list
        list of values
    """  
    mylist = []
    for line in open(path_to_file):
            listWord = line.split(" ")
            mylist.append(listWord)
    list = [[float(item) if item != '-nan' and item != 'nan' else float(mylist[a][mylist[a].index(item)-1]) for item in mylist[a][:-1]]for a in range(len(mylist))] 
    return list 

def readND_Nic(path_to_file): 
    """
    Read a space-separated txt file and return a N-Dimensional list of the values (for double space separated files)

    Input:
        path_to_file : str
            file to open
    Returns:
            : list
        list of values
    """  
    mylist = []
    for line in open(path_to_file):
            listWord = line.split("  ")
            mylist.append(listWord)
    list = [[float(item) if item != '-nan' and item != 'nan' else float(mylist[a][mylist[a].index(item)-1]) for item in mylist[a][:-1]]for a in range(len(mylist))] 
    return list

def readND_python(path_to_file): 
    """
    Read a space-separated txt file and return a N-Dimensional list of the values

    Input:
        path_to_file : str
            file to open
    Returns:
            : list
        list of values
    """  
    mylist = []
    for line in open(path_to_file):
            listWord = line.split(" ")
            mylist.append(listWord)
    list = [[float(item) if item != '-nan' and item != 'nan' else float(mylist[a][mylist[a].index(item)]) for item in mylist[a][:]]for a in range(len(mylist))] 
    return list 

def construct_grid_matching(func, mass, path):
    func_values = []
    p = []
    for z in Ini.HPL_x_array:
        print(float(Ini.HPL_x_array.index(z))/Ini.HPL_x_array.__len__())
        z_func_values = []
        for q in np.array(Ini.QList):
            p = [mass, q]
            z_func_values.append(func(z,p))
        func_values.append(z_func_values)
    np.savetxt(path, func_values)
    return func_values

def construct_grid_tilde(func, mass, path):
    func_values = []
    p = [mass]
    for z in Ini.ZList:
        print(float(Ini.ZList.index(z))/Ini.ZList.__len__())
        z_func_values = []
        for q in np.array(Ini.QList):
            #z_func_values.append(func(z,q,p)[0])
            z_func_values.append(func(z,q,p))
        func_values.append(z_func_values)
    np.savetxt(path, func_values)
    return func_values
    




