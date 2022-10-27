# This is used to plot the final structure functions.
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .Initialize import PATH_TO_GLOBAL

FO = {}
R = {}
M = {}
Q = []
q = []
a = []
b = []
c = []
# Constructing the possibilities
Stru_func = ["F2", "FL"]
Methods = ["our", "fonll"]
X = [0.1, 0.01, 0.001, 0.0001]
ratios = ["0.5", "", "2"]
orders = ["nlo", "nnlo", "n3lo"]
# Importing data
for Sf in Stru_func:
    for meth in Methods:
        for order in orders:
            for x in X:
                (
                    q,
                    FO[
                        Stru_func.index(Sf),
                        Methods.index(meth),
                        orders.index(order),
                        X.index(x),
                    ],
                ) = zip(
                    *(
                        map(float, line.split())
                        for line in open(
                            PATH_TO_GLOBAL
                            + "/data/FO/"
                            + Sf
                            + "_FO_"
                            + meth
                            + "_"
                            + order
                            + "_x="
                            + str(x)
                            + ".csv"
                        )
                    )
                )
                for ratio in ratios:
                    (
                        Q,
                        R[
                            Stru_func.index(Sf),
                            Methods.index(meth),
                            orders.index(order),
                            ratios.index(ratio),
                            X.index(x),
                        ],
                    ) = zip(
                        *(
                            map(float, line.split())
                            for line in open(
                                PATH_TO_GLOBAL
                                + "/data/R/"
                                + Sf
                                + "_R_"
                                + meth
                                + "_"
                                + order
                                + "_x="
                                + str(x)
                                + "_mub="
                                + ratio
                                + "mb.csv"
                            )
                        )
                    )
                    (
                        Q,
                        M[
                            Stru_func.index(Sf),
                            Methods.index(meth),
                            orders.index(order),
                            ratios.index(ratio),
                            X.index(x),
                        ],
                    ) = zip(
                        *(
                            map(float, line.split())
                            for line in open(
                                PATH_TO_GLOBAL
                                + "/data/M/"
                                + Sf
                                + "_M_"
                                + meth
                                + "_"
                                + order
                                + "_x="
                                + str(x)
                                + "_mub="
                                + ratio
                                + "mb.csv"
                            )
                        )
                    )
# Organizing data
FO_array = np.array(
    [
        [
            [
                [
                    FO[
                        Stru_func.index(Sf),
                        Methods.index(meth),
                        orders.index(order),
                        X.index(x),
                    ]
                    for Sf in Stru_func
                ]
                for meth in Methods
            ]
            for order in orders
        ]
        for x in X
    ]
)
R_array = np.array(
    [
        [
            [
                [
                    [
                        R[
                            Stru_func.index(Sf),
                            Methods.index(meth),
                            orders.index(order),
                            ratios.index(ratio),
                            X.index(x),
                        ]
                        for Sf in Stru_func
                    ]
                    for meth in Methods
                ]
                for order in orders
            ]
            for ratio in ratios
        ]
        for x in X
    ]
)
M_array = np.array(
    [
        [
            [
                [
                    [
                        M[
                            Stru_func.index(Sf),
                            Methods.index(meth),
                            orders.index(order),
                            ratios.index(ratio),
                            X.index(x),
                        ]
                        for Sf in Stru_func
                    ]
                    for meth in Methods
                ]
                for order in orders
            ]
            for ratio in ratios
        ]
        for x in X
    ]
)
R_array[R_array == 0.0] = "nan"

# Constructing the uncertainties: symmetrization of mub variation
UncertR = np.array(
    [
        [
            [
                [
                    [
                        (
                            lambda q: abs(
                                R_array[X.index(x)][0][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                                - R_array[X.index(x)][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                            )
                            if abs(
                                R_array[X.index(x)][0][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                                - R_array[X.index(x)][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                            )
                            > abs(
                                R_array[X.index(x)][2][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                                - R_array[X.index(x)][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                            )
                            else abs(
                                R_array[X.index(x)][2][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                                - R_array[X.index(x)][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                            )
                        )(q)
                        for q in Q
                    ]
                    for Sf in Stru_func
                ]
                for meth in Methods
            ]
            for order in orders
        ]
        for x in X
    ]
)
UncertM = np.array(
    [
        [
            [
                [
                    [
                        (
                            lambda q: abs(
                                M_array[X.index(x)][0][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                                - M_array[X.index(x)][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                            )
                            if abs(
                                M_array[X.index(x)][0][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                                - M_array[X.index(x)][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                            )
                            > abs(
                                M_array[X.index(x)][2][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                                - M_array[X.index(x)][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                            )
                            else abs(
                                M_array[X.index(x)][2][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                                - M_array[X.index(x)][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)][Q.index(q)]
                            )
                        )(q)
                        for q in Q
                    ]
                    for Sf in Stru_func
                ]
                for meth in Methods
            ]
            for order in orders
        ]
        for x in X
    ]
)

# Plotting standard plots: FO+R+M + uncert. chart
for Sf in Stru_func:
    for meth in Methods:
        for order in orders:
            with PdfPages(
                PATH_TO_GLOBAL + "/plots/" + Sf + "_" + meth + "_" + order + ".pdf"
            ) as pdf:
                fig = plt.figure(figsize=(10, 8))
                plt.rcParams["text.usetex"] = True
                fig.suptitle(Sf + "_" + meth + "_" + order, fontsize=16)
                outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.3)
                for i in range(4):
                    inner = gridspec.GridSpecFromSubplotSpec(
                        2,
                        1,
                        subplot_spec=outer[i],
                        wspace=0.1,
                        hspace=0.0,
                        height_ratios=[2.5, 1],
                    )
                    for j in range(2):
                        axs = plt.Subplot(fig, inner[j])
                        if j == 0:
                            axs.set_xscale("log")
                            axs.plot(
                                Q,
                                FO_array[i][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ],
                                label=r"\textbf{FO}",
                                color="m",
                                linewidth=2,
                                linestyle="dashed",
                            )
                            axs.plot(
                                Q,
                                R_array[i][1][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ],
                                label=r"\textbf{R}",
                                color="g",
                                linewidth=0.5,
                            )
                            axs.plot(
                                Q,
                                M_array[i][1][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ],
                                label=r"\textbf{M}",
                                color="b",
                                linewidth=1,
                            )
                            axs.fill_between(
                                Q,
                                R_array[i][1][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                - UncertR[i][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ],
                                R_array[i][1][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                + UncertR[i][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ],
                                color="g",
                                alpha=0.2,
                            )
                            axs.fill_between(
                                Q,
                                M_array[i][1][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                - UncertM[i][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ],
                                M_array[i][1][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                + UncertM[i][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ],
                                color="b",
                                alpha=0.2,
                            )
                            axs.legend()
                            axs.title.set_text("x=" + str(X[i]))
                            axs.set_ylabel("x" + Sf)
                            axs.tick_params(
                                "x",
                                which="both",
                                bottom=False,
                                top=False,
                                labelbottom=False,
                            )
                        if j == 1:
                            np.seterr(invalid="ignore")
                            np.seterr(divide="ignore")
                            axs.set_xscale("log")
                            axs.plot(
                                Q,
                                M_array[i][1][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                / M_array[i][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)],
                                color="b",
                                linewidth=1,
                            )
                            axs.plot(
                                Q,
                                M_array[i][0][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                / M_array[i][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)],
                                label=r"M($\mu_{b}=0.5m_{b}$)",
                                color="slateblue",
                                linewidth=1,
                            )
                            axs.plot(
                                Q,
                                M_array[i][2][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                / M_array[i][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)],
                                label=r"M($\mu_{b}=2m_{b}$)",
                                color="cornflowerblue",
                                linewidth=1,
                            )
                            axs.fill_between(
                                Q,
                                M_array[i][0][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                / M_array[i][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)],
                                M_array[i][1][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                / M_array[i][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)],
                                color="slateblue",
                                alpha=0.3,
                            )
                            axs.fill_between(
                                Q,
                                M_array[i][2][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                / M_array[i][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)],
                                M_array[i][1][orders.index(order)][Methods.index(meth)][
                                    Stru_func.index(Sf)
                                ]
                                / M_array[i][1][orders.index(order)][
                                    Methods.index(meth)
                                ][Stru_func.index(Sf)],
                                color="cornflowerblue",
                                alpha=0.3,
                            )
                            axs.set_xlabel(r"Q[\textit{GeV}]")
                            if Sf == "F2":
                                axs.set_ylim([0.5, 1.5])
                            else:
                                axs.set_ylim([0.98, 1.02])
                            axs.legend(loc="lower right", fontsize=6)
                        fig.add_subplot(axs)
                pdf.savefig()
                plt.close()
# Plotting comparison plot: our+fonll
for Sf in Stru_func:
    for order in orders:
        with PdfPages(
            PATH_TO_GLOBAL + "/plots/Compare_" + Sf + "_" + order + ".pdf"
        ) as pdf:
            fig = plt.figure(figsize=(10, 8))
            plt.rcParams["text.usetex"] = True
            fig.suptitle("Compare" + Sf + "_" + order, fontsize=16)
            outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.3)
            for i in range(4):
                inner = gridspec.GridSpecFromSubplotSpec(
                    2,
                    1,
                    subplot_spec=outer[i],
                    wspace=0.1,
                    hspace=0.0,
                    height_ratios=[2.5, 1],
                )
                for j in range(2):
                    axs = plt.Subplot(fig, inner[j])
                    if j == 0:
                        axs.set_xscale("log")
                        axs.plot(
                            Q,
                            FO_array[i][orders.index(order)][0][Stru_func.index(Sf)],
                            label=r"\textbf{FO}",
                            color="m",
                            linewidth=2,
                            linestyle="dashed",
                        )
                        axs.plot(
                            Q,
                            M_array[i][1][orders.index(order)][0][Stru_func.index(Sf)],
                            label=r"\textbf{M}",
                            color="b",
                            linewidth=1,
                        )
                        axs.plot(
                            Q,
                            M_array[i][1][orders.index(order)][1][Stru_func.index(Sf)],
                            label=r"\textbf{M_{fonll}}",
                            color="g",
                            linewidth=1,
                        )
                        axs.fill_between(
                            Q,
                            M_array[i][1][orders.index(order)][0][Stru_func.index(Sf)]
                            - UncertM[i][orders.index(order)][0][Stru_func.index(Sf)],
                            M_array[i][1][orders.index(order)][0][Stru_func.index(Sf)]
                            + UncertM[i][orders.index(order)][0][Stru_func.index(Sf)],
                            color="b",
                            alpha=0.2,
                        )
                        axs.fill_between(
                            Q,
                            M_array[i][1][orders.index(order)][1][Stru_func.index(Sf)]
                            - UncertM[i][orders.index(order)][1][Stru_func.index(Sf)],
                            M_array[i][1][orders.index(order)][1][Stru_func.index(Sf)]
                            + UncertM[i][orders.index(order)][1][Stru_func.index(Sf)],
                            color="g",
                            alpha=0.2,
                        )
                        axs.legend()
                        axs.title.set_text("x=" + str(X[i]))
                        axs.set_ylabel("x" + Sf)
                        axs.tick_params(
                            "x",
                            which="both",
                            bottom=False,
                            top=False,
                            labelbottom=False,
                        )
                    if j == 1:
                        np.seterr(invalid="ignore")
                        np.seterr(divide="ignore")
                        axs.set_xscale("log")
                        axs.plot(
                            Q,
                            M_array[i][1][orders.index(order)][0][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][0][
                                Stru_func.index(Sf)
                            ],
                            color="b",
                            linewidth=1,
                        )
                        axs.plot(
                            Q,
                            M_array[i][0][orders.index(order)][0][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][0][
                                Stru_func.index(Sf)
                            ],
                            label=r"M($\mu_{b}=0.5m_{b}$)",
                            color="slateblue",
                            linewidth=1,
                        )
                        axs.plot(
                            Q,
                            M_array[i][2][orders.index(order)][0][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][0][
                                Stru_func.index(Sf)
                            ],
                            label=r"M($\mu_{b}=2m_{b}$)",
                            color="cornflowerblue",
                            linewidth=1,
                        )
                        axs.fill_between(
                            Q,
                            M_array[i][0][orders.index(order)][0][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][0][
                                Stru_func.index(Sf)
                            ],
                            M_array[i][1][orders.index(order)][0][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][0][
                                Stru_func.index(Sf)
                            ],
                            color="slateblue",
                            alpha=0.3,
                        )
                        axs.fill_between(
                            Q,
                            M_array[i][2][orders.index(order)][0][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][0][
                                Stru_func.index(Sf)
                            ],
                            M_array[i][1][orders.index(order)][0][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][0][
                                Stru_func.index(Sf)
                            ],
                            color="cornflowerblue",
                            alpha=0.3,
                        )

                        axs.plot(
                            Q,
                            M_array[i][1][orders.index(order)][1][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][1][
                                Stru_func.index(Sf)
                            ],
                            color="g",
                            linewidth=1,
                        )
                        axs.plot(
                            Q,
                            M_array[i][0][orders.index(order)][1][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][1][
                                Stru_func.index(Sf)
                            ],
                            label=r"M($\mu_{b}=0.5m_{b}$)",
                            color="forestgreen",
                            linewidth=1,
                        )
                        axs.plot(
                            Q,
                            M_array[i][2][orders.index(order)][1][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][1][
                                Stru_func.index(Sf)
                            ],
                            label=r"M($\mu_{b}=2m_{b}$)",
                            color="lime",
                            linewidth=1,
                        )
                        axs.fill_between(
                            Q,
                            M_array[i][0][orders.index(order)][1][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][1][
                                Stru_func.index(Sf)
                            ],
                            M_array[i][1][orders.index(order)][1][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][1][
                                Stru_func.index(Sf)
                            ],
                            color="forestgreen",
                            alpha=0.3,
                        )
                        axs.fill_between(
                            Q,
                            M_array[i][2][orders.index(order)][1][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][1][
                                Stru_func.index(Sf)
                            ],
                            M_array[i][1][orders.index(order)][1][Stru_func.index(Sf)]
                            / M_array[i][1][orders.index(order)][1][
                                Stru_func.index(Sf)
                            ],
                            color="lime",
                            alpha=0.3,
                        )
                        axs.set_xlabel(r"Q[\textit{GeV}]")
                        if Sf == "F2":
                            axs.set_ylim([0.5, 1.5])
                        else:
                            axs.set_ylim([0.98, 1.02])
                        axs.legend(loc="lower right", fontsize=6)
                    fig.add_subplot(axs)
            pdf.savefig()
            plt.close()
