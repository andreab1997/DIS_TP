import numpy as np
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



