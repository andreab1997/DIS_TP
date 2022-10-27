#This is an high-level interface to compute and plot structure functions.
from . import Integration as Int

import numpy as np
import lhapdf
import csv
from progress.bar import Bar
from simple_term_menu import TerminalMenu
import time
import os

def main():
    main_menu_title = " Main Menu\n"
    main_menu_items = ["Structure function", "Methods", "Perturbative orders", "Scale_variations?", "Plot?" , "Start_computation"] 
    main_menu_cursor = "> "
    main_menu_cursor_style = ("fg_red", "bold")
    main_menu_style = ("bg_red", "fg_yellow")
    main_menu_exit = False
    main_menu = TerminalMenu(
        menu_entries=main_menu_items,
        title=main_menu_title,
        menu_cursor=main_menu_cursor,
        menu_cursor_style=main_menu_cursor_style,
        menu_highlight_style=main_menu_style,
        cycle_cursor=True,
        clear_screen=True,
    )

    Stru_funcs_menu_title = "  Structure Functions\n"
    Stru_funcs_menu_items = ["F2", "FL"]
    Stru_funcs_menu = TerminalMenu(
        Stru_funcs_menu_items,
        title=Stru_funcs_menu_title,
        menu_cursor=main_menu_cursor,
        menu_cursor_style=main_menu_cursor_style,
        menu_highlight_style=main_menu_style,
        cycle_cursor=True,
        clear_screen=True,
        multi_select=True,
        show_multi_select_hint=True,
    )

    Methods_menu_title = "  Methods\n"
    Methods_menu_items = ["our", "fonll"]
    Methods_menu = TerminalMenu(
        Methods_menu_items,
        title=Methods_menu_title,
        menu_cursor=main_menu_cursor,
        menu_cursor_style=main_menu_cursor_style,
        menu_highlight_style=main_menu_style,
        cycle_cursor=True,
        clear_screen=True,
        multi_select=True,
        show_multi_select_hint=True,
    )

    order_menu_title = "   Perturbative orders\n"
    order_menu_items = ["nlo", "nnlo", "n3lo"]
    order_menu = TerminalMenu(
        order_menu_items,
        title=order_menu_title,
        menu_cursor=main_menu_cursor,
        menu_cursor_style=main_menu_cursor_style,
        menu_highlight_style=main_menu_style,
        cycle_cursor=True,
        clear_screen=True,
        multi_select=True,
        show_multi_select_hint=True,
    )

    PDForder_menu_title = "   "
    PDForder_menu_items = ["nll", "nnll"]
    PDForder_menu = TerminalMenu(
        PDForder_menu_items,
        title=PDForder_menu_title,
        menu_cursor=main_menu_cursor,
        menu_cursor_style=main_menu_cursor_style,
        menu_highlight_style=main_menu_style,
        cycle_cursor=True,
        clear_screen=False,
    )
    scalevar = False
    plot = False
    Stru_func = []
    Methods = []
    orders = []
    while not main_menu_exit:
        main_sel = main_menu.show()
        Stru_funcs_menu_back = False
        Methods_menu_back = False
        order_menu_back = False
        if main_sel == 0:
            while not Stru_funcs_menu_back: 
                Stru_funcs_sel = Stru_funcs_menu.show()
                Stru_func = [Stru_funcs_menu_items[index] for index in Stru_funcs_sel]
                Stru_funcs_menu_back = True
        elif main_sel == 1:
            while not Methods_menu_back: 
                Methods_sel = Methods_menu.show()
                Methods = [Methods_menu_items[index] for index in Methods_sel]
                Methods_menu_back = True
        elif main_sel == 2:
            while not order_menu_back: 
                order_sel = order_menu.show()
                orders = [order_menu_items[index] for index in order_sel]
                map_PDFsorder = {}
                for o in orders:
                    print("Select PDForder for perturbative order " + o +"\n")
                    PDForder_sel = PDForder_menu.show()
                    map_PDFsorder[o] = order_menu_items[PDForder_sel]
                order_menu_back = True
        elif main_sel == 3:
            scalevar = not scalevar
            print("You have set scalevariation to: " + str(scalevar))
            print("If you want to change this, click again on Scale_variation option")
            time.sleep(4)
        elif main_sel == 4:
            plot = not plot
            print("You have set plot to: " + str(plot))
            print("If you want to change this, click again on plot option")
            time.sleep(4)
        elif main_sel == 5:
            print('Check user choices...')
            check = True
            if len(Stru_func) == 0:
                print("At least one Structure function is needed. Returning to the main menu...")
                check = False
            elif len(Methods) == 0:
                print("At least one Method is needed. Returning to the main menu...")
                check = False
            elif len(orders) == 0:
                print("At least one perturbative order is needed. Returning to the main menu...")
                check = False
            if check:
                print("Check was succesfull!")
                if 'fonll' in Methods and 'n3lo' in orders:
                    print("WARNING: Not all the results are available with FONLL method")
                main_menu_exit = True
                print("Computation is starting with:")
                print("Structure functions: ")
                print(*Stru_func, sep = " , ")
                print("Methods: ")
                print(*Methods, sep = " , ")
                print("Perturbative orders (with respective PDF order): ")
                print(map_PDFsorder)
                print("Scalevariation : " + str(scalevar))
                do_the_calculation(Stru_func,Methods,map_PDFsorder,scalevar)
                if plot:
                    print("Plotting new files...")
                    os.system('python3 Plotter.py')

            time.sleep(4)
    print("All done :)")
    return 0

def do_the_calculation(Stru_func,Methods,map_PDFsorder,scalevar):
    ratios = [""]
    if scalevar:
        ratios = ["0.5", "", "2"]
    
    eps = 0.5
    lhapdf.setVerbosity(0)
    mine = lhapdf.mkPDF("MyPDF_mub=mb_nlo")
    thre = [mine.quarkMass(5)]
    if len(ratios)>1:    
        thre = [mine.quarkMass(5)*ratio for ratio in [0.5,1.,2.0]]
    Qlogmin = np.log10(1.)
    Qlogmax = np.log10(150.)
    Qlog = np.linspace(Qlogmin,Qlogmax,200)
    Qcommon = pow(10,Qlog)
    Qsing = [np.linspace(thr-eps,thr+eps,5) for thr in thre]
    Q = np.sort(np.concatenate((Qcommon,Qsing[0])))
    if len(ratios)>1:
        Q = np.sort(np.concatenate((Qcommon,Qsing[0],Qsing[1],Qsing[2])))

    X = [0.1,0.01,0.001,0.0001]

    FO = {}
    R = {}
    M = {}
    orddict = {}
    orddictR = {}
    orddictM = {}
    methdictM = {}
    methdictR = {}
    intorders = ["nlo","nnlo","n3lo"]
    numberofcalc = (len(Stru_func)*len(map_PDFsorder) + len(Stru_func)*len(map_PDFsorder)*len(Methods) )+1
    with Bar('Processing...', max = numberofcalc) as bar:
        print("\nInitialization in progress...")
        #Initializing global data
        Int.Initialize_all()
        bar.next()
        print("\nComputation of Structure Functions is starting...")
        for Sf in Stru_func:
            for ord in map_PDFsorder:
                if Sf == 'F2':
                    res =np.array([[Int.F2_FO(intorders.index(ord)+1,"MyPDF_4F_"+map_PDFsorder[ord], x,q) for q in Q] for x in X])
                    orddict[ord] = res 
                elif Sf == 'FL':
                    res =np.array([[Int.FL_FO(intorders.index(ord)+1,"MyPDF_4F_"+map_PDFsorder[ord], x,q) for q in Q] for x in X])
                    orddict[ord] = res 
                bar.next()
            FO[Sf] = orddict.copy()
            orddict.clear()
        for Sf in Stru_func:
            for ord in map_PDFsorder:
                for met in Methods:
                    if Sf == 'F2':
                        resM = np.array([[[(lambda q: Int.F2_M(intorders.index(ord)+1,met,"MyPDF_mub=" + ratio + "mb_" + map_PDFsorder[ord], x,q) if q > thre[list(ratios).index(ratio)] else FO[Sf][ord][X.index(x)][list(Q).index(q)])(q) for q in Q] for x in X] for ratio in ratios])
                        if met == "our":
                            resR =np.array([[[(lambda q: Int.F2_R(intorders.index(ord)+1,"MyPDF_mub=" + ratio + "mb_" + map_PDFsorder[ord], x,q) if q > thre[list(ratios).index(ratio)] else 0.)(q) for q in Q] for x in X] for ratio in ratios])
                        elif met == "fonll":
                            resR =np.array([[[0. for q in Q] for x in X] for ratio in ratios])
                    elif Sf == 'FL':
                        resM = np.array([[[(lambda q: Int.FL_M(intorders.index(ord)+1,met,"MyPDF_mub=" + ratio + "mb_" + map_PDFsorder[ord], x,q) if q > thre[list(ratios).index(ratio)] else FO[Sf][ord][X.index(x)][list(Q).index(q)])(q) for q in Q] for x in X] for ratio in ratios])
                        if met == "our":
                            resR =np.array([[[(lambda q: Int.FL_R(intorders.index(ord)+1,"MyPDF_mub=" + ratio + "mb_" + map_PDFsorder[ord], x,q) if q > thre[list(ratios).index(ratio)] else 0.)(q) for q in Q] for x in X] for ratio in ratios])
                        elif met == "fonll":
                            resR =np.array([[[0. for q in Q] for x in X] for ratio in ratios])
                    methdictR[met] = resR
                    methdictM[met] = resM
                    bar.next()
                orddictR[ord] = methdictR.copy()
                orddictM[ord] = methdictM.copy()
                methdictR.clear()
                methdictM.clear()
            R[Sf] = orddictR.copy()
            M[Sf] = orddictM.copy()
            orddictR.clear()
            orddictM.clear()
    for Sf in Stru_func:
        for meth in Methods:
            for order in map_PDFsorder:
                for x in X:
                    with open('./data/FO/'+ Sf + '_FO_' + meth + '_' + order +'_x=' + str(x) + '.csv',mode='w+') as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerows(zip(Q,FO[Sf][order][X.index(x)]))
                    for thr in thre:
                        with open('./data/R/'+ Sf + '_R_' + meth + '_' + order +'_x=' + str(x) + '_mub=' + ratios[thre.index(thr)] + 'mb'+ '.csv',mode='w+') as g:
                            writer = csv.writer(g, delimiter='\t')
                            writer.writerows(zip(Q,R[Sf][order][meth][thre.index(thr)][X.index(x)]))
                        with open('./data/M/'+ Sf + '_M_' + meth + '_' + order +'_x=' + str(x) + '_mub=' + ratios[thre.index(thr)] + 'mb'+ '.csv',mode='w+') as h:
                            writer = csv.writer(h, delimiter='\t')
                            writer.writerows(zip(Q,M[Sf][order][meth][thre.index(thr)][X.index(x)]))
    
    

if __name__ == "__main__":
    main()