# Code for computing F2 and FL with all the possibilities.

import csv

import lhapdf
import numpy as np
from progress.bar import Bar

from . import Integration as Int

# Constructing the possibilities
Stru_func = ["F2", "FL"]
Methods = ["our", "fonll"]
pdf_orders = ["nlo", "nnlo", "nnlo"]
orders = ["nlo", "nnlo", "n3lo"]
ratios = ["0.5", "", "2"]
# PDFs
pdf_FO = ["MyPDF_4F_" + order for order in pdf_orders]
pdf_R = [
    ["MyPDF_mub=" + ratio + "mb_" + order for order in pdf_orders] for ratio in ratios
]
# Constructing the Q-array
eps = 0.5
mine = lhapdf.mkPDF("MyPDF_mub=mb_nlo")
thre = [mine.quarkMass(5) * ratio for ratio in [0.5, 1.0, 2.0]]
Qlogmin = np.log10(1.0)
Qlogmax = np.log10(150.0)
Qlog = np.linspace(Qlogmin, Qlogmax, 200)
Qcommon = pow(10, Qlog)
Qsing = [np.linspace(thr - eps, thr + eps, 5) for thr in thre]
Q = np.sort(np.concatenate((Qcommon, Qsing[0], Qsing[1], Qsing[2])))
# X-array
X = [0.1, 0.01, 0.001, 0.0001]
# Doing the calculation
numberofcalc = len(Stru_func) * len(Methods) * len(orders) * 3 + 1
with Bar("Processing...", max=numberofcalc) as bar:
    print("\nInitialization in progress...")
    # Initializing global data
    Int.Initialize_all()
    bar.next()
    print("\nComputation of Structure Functions is starting...")
    # NLO
    ##F2
    ###Our
    F2_FO_nlo = np.array([[Int.F2_FO(1, pdf_FO, x, q) for q in Q] for x in X])
    bar.next()
    F2_R_nlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.F2_R(1, pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else 0.0
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    F2_M_nlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.F2_M(1, "our", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else F2_FO_nlo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    ###FONLL
    F2_FO_fonll_nlo = F2_FO_nlo
    bar.next()
    F2_R_fonll_nlo = [np.array([[0.0 for q in Q] for x in X]) for thr in thre]
    bar.next()
    F2_M_fonll_nlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.F2_M(1, "fonll", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else F2_FO_fonll_nlo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    ##FL
    ###Our
    FL_FO_nlo = np.array([[Int.FL_FO(1, pdf_FO, x, q) for q in Q] for x in X])
    bar.next()
    FL_R_nlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.FL_R(1, pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else 0.0
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    FL_M_nlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.FL_M(1, "our", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else FL_FO_nlo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    ###FONLL
    FL_FO_fonll_nlo = FL_FO_nlo
    bar.next()
    FL_R_fonll_nlo = [np.array([[0.0 for q in Q] for x in X]) for thr in thre]
    bar.next()
    FL_M_fonll_nlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.FL_M(1, "fonll", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else FL_FO_fonll_nlo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()

    # NNLO

    ##F2
    ###Our
    F2_FO_nnlo = np.array([[Int.F2_FO(2, pdf_FO, x, q) for q in Q] for x in X])
    bar.next()
    F2_R_nnlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.F2_R(2, pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else 0.0
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    F2_M_nnlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.F2_M(2, "our", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else F2_FO_nnlo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    ###FONLL
    F2_FO_fonll_nnlo = F2_FO_nnlo
    bar.next()
    F2_R_fonll_nnlo = [np.array([[0.0 for q in Q] for x in X]) for thr in thre]
    bar.next()
    F2_M_fonll_nnlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.F2_M(2, "fonll", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else F2_FO_fonll_nnlo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    ##FL
    ###Our
    FL_FO_nnlo = np.array([[Int.FL_FO(2, pdf_FO, x, q) for q in Q] for x in X])
    bar.next()
    FL_R_nnlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.FL_R(2, pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else 0.0
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    FL_M_nnlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.FL_M(2, "our", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else FL_FO_nnlo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    ###FONLL
    FL_FO_fonll_nnlo = FL_FO_nnlo
    bar.next()
    FL_R_fonll_nnlo = [np.array([[0.0 for q in Q] for x in X]) for thr in thre]
    bar.next()
    FL_M_fonll_nnlo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.FL_M(2, "fonll", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else FL_FO_fonll_nnlo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()

    # N3LO

    ##F2
    ###Our
    F2_FO_n3lo = np.array([[Int.F2_FO(3, pdf_FO, x, q) for q in Q] for x in X])
    bar.next()
    F2_R_n3lo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.F2_R(3, pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else 0.0
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    F2_M_n3lo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.F2_M(3, "our", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else F2_FO_n3lo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    ###FONLL
    F2_FO_fonll_n3lo = F2_FO_n3lo
    bar.next()
    F2_R_fonll_n3lo = [np.array([[0.0 for q in Q] for x in X]) for thr in thre]
    bar.next()
    F2_M_fonll_n3lo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.F2_M(3, "fonll", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else F2_FO_fonll_n3lo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    ##FL
    ###Our
    FL_FO_n3lo = np.array([[Int.FL_FO(3, pdf_FO, x, q) for q in Q] for x in X])
    bar.next()
    FL_R_n3lo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.FL_R(3, pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else 0.0
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    FL_M_n3lo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.FL_M(3, "our", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else FL_FO_n3lo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
    ###FONLL
    FL_FO_fonll_n3lo = FL_FO_n3lo
    bar.next()
    FL_R_fonll_n3lo = [np.array([[0.0 for q in Q] for x in X]) for thr in thre]
    bar.next()
    FL_M_fonll_n3lo = [
        np.array(
            [
                [
                    (
                        lambda q: Int.FL_M(3, "fonll", pdf_R[thre.index(thr)], x, q)
                        if q > thr
                        else FL_FO_fonll_n3lo[list(X).index(x)][list(Q).index(q)]
                    )(q)
                    for q in Q
                ]
                for x in X
            ]
        )
        for thr in thre
    ]
    bar.next()
print("Computation of Structure Functions done :)\n")

# Organizing results in arrays
FO_Results = np.array(
    [
        [
            [F2_FO_nlo, F2_FO_nnlo, F2_FO_n3lo],
            [F2_FO_fonll_nlo, F2_FO_fonll_nnlo, F2_FO_fonll_n3lo],
        ],
        [
            [FL_FO_nlo, FL_FO_nnlo, FL_FO_n3lo],
            [FL_FO_fonll_nlo, FL_FO_fonll_nnlo, FL_FO_fonll_n3lo],
        ],
    ],
    dtype=object,
)
R_Results = np.array(
    [
        [
            [F2_R_nlo, F2_R_nnlo, F2_R_n3lo],
            [F2_R_fonll_nlo, F2_R_fonll_nnlo, F2_R_fonll_n3lo],
        ],
        [
            [FL_R_nlo, FL_R_nnlo, FL_R_n3lo],
            [FL_R_fonll_nlo, FL_R_fonll_nnlo, FL_R_fonll_n3lo],
        ],
    ],
    dtype=object,
)
M_Results = np.array(
    [
        [
            [F2_M_nlo, F2_M_nnlo, F2_M_n3lo],
            [F2_M_fonll_nlo, F2_M_fonll_nnlo, F2_M_fonll_n3lo],
        ],
        [
            [FL_M_nlo, FL_M_nnlo, FL_M_n3lo],
            [FL_M_fonll_nlo, FL_M_fonll_nnlo, FL_M_fonll_n3lo],
        ],
    ],
    dtype=object,
)

# Exporting the results
for Sf in Stru_func:
    for meth in Methods:
        for order in orders:
            for x in X:
                with open(
                    "../../data/FO/"
                    + Sf
                    + "_FO_"
                    + meth
                    + "_"
                    + order
                    + "_x="
                    + str(x)
                    + ".csv",
                    mode="w+",
                ) as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerows(
                        zip(
                            Q,
                            FO_Results[Stru_func.index(Sf)][Methods.index(meth)][
                                orders.index(order)
                            ][X.index(x)],
                        )
                    )
                for thr in thre:
                    with open(
                        "../../data/R/"
                        + Sf
                        + "_R_"
                        + meth
                        + "_"
                        + order
                        + "_x="
                        + str(x)
                        + "_mub="
                        + ratios[thre.index(thr)]
                        + "mb"
                        + ".csv",
                        mode="w+",
                    ) as g:
                        writer = csv.writer(g, delimiter="\t")
                        writer.writerows(
                            zip(
                                Q,
                                R_Results[Stru_func.index(Sf)][Methods.index(meth)][
                                    orders.index(order)
                                ][thre.index(thr)][X.index(x)],
                            )
                        )
                    with open(
                        "../../data/M/"
                        + Sf
                        + "_M_"
                        + meth
                        + "_"
                        + order
                        + "_x="
                        + str(x)
                        + "_mub="
                        + ratios[thre.index(thr)]
                        + "mb"
                        + ".csv",
                        mode="w+",
                    ) as h:
                        writer = csv.writer(h, delimiter="\t")
                        writer.writerows(
                            zip(
                                Q,
                                M_Results[Stru_func.index(Sf)][Methods.index(meth)][
                                    orders.index(order)
                                ][thre.index(thr)][X.index(x)],
                            )
                        )
