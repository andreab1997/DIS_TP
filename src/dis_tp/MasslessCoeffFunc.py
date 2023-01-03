# This contains the massless coefficients functions.
import numpy as np

from . import parameters as para


# F2
def Cb_0_loc(z, Q):
    e_b = para.parameters["e_b"]
    return e_b * e_b


def Cb_1_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    CF = 4.0 / 3.0
    return (
        e_b
        * e_b
        * 2
        * CF
        * (-(1 + z) * np.log(1 - z) - (1 + z * z) * np.log(z) / (1 - z) + 3 + 2 * z)
    )


def Cb_1_loc(z, Q):
    e_b = para.parameters["e_b"]
    CF = 4.0 / 3.0
    zeta2 = 1.6449340668482264
    return e_b * e_b * 2 * CF * (-(2 * zeta2 + 9.0 / 2.0))


def Cb_1_sing(z, Q, p):
    e_b = para.parameters["e_b"]
    CF = 4.0 / 3.0
    return e_b * e_b * 2 * CF * (2 * np.log(1 - z) - 3.0 / 2.0) / (1 - z)


def Cb_2_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    z1 = 1 - z
    dl = np.log(z)
    dl1 = np.log(z1)
    dl_2 = dl * dl
    dl_3 = dl_2 * dl
    dl1_2 = dl1 * dl1
    dl1_3 = dl1_2 * dl1
    return (
        e_b
        * e_b
        * (
            -69.59
            - 1008 * z
            - 2.835 * dl_3
            - 17.08 * dl_2
            + 5.986 * dl
            - 17.19 * dl1_3
            + 71.08 * dl1_2
            - 660.7 * dl1
            - 174.8 * dl * dl1_2
            + 95.09 * dl_2 * dl1
            + 5
            * (
                -5.691
                - 37.91 * z
                + 2.244 * dl_2
                + 5.770 * dl
                - 1.707 * dl1_2
                + 22.95 * dl1
                + 3.036 * dl_2 * dl1
                + 17.97 * dl * dl1
            )
        )
    )


def Cb_2_loc(z, Q):
    e_b = para.parameters["e_b"]
    return e_b * e_b * (-338.046 + 5 * (46.8405))


def Cb_2_sing(z, Q, p):
    e_b = para.parameters["e_b"]
    z1 = 1 - z
    dl = np.log(z)
    dl1 = np.log(z1)
    dl_2 = dl * dl
    dl_3 = dl_2 * dl
    dl1_2 = dl1 * dl1
    dl1_3 = dl1_2 * dl1
    return (
        e_b
        * e_b
        * (1.0 / z1)
        * (
            +14.2222 * dl1_3
            - 61.3333 * dl1_2
            - 31.105 * dl1
            + 188.64
            + 5 * (1.77778 * dl1_2 - 8.5926 * dl1 + 6.3489)
        )
    )


def Cg_1_reg(z, Q, p):
    TR = 1.0 / 2.0
    e_b = para.parameters["e_b"]
    return (
        4
        * TR
        * e_b
        * e_b
        * (((1 - z) * (1 - z) + z * z) * np.log((1 - z) / z) - 8 * z * (z - 1) - 1)
    )


def Cg_2_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    z1 = 1 - z
    L0 = np.log(z)
    L1 = np.log(z1)
    return (
        e_b
        * e_b
        * (
            (58.0 / 9.0) * pow(L1, 3)
            - 24 * pow(L1, 2)
            - 34.88 * L1
            + 30.586
            - (25.08 + 760.3 * z + 29.65 * pow(L1, 3)) * z1
            + 1204 * z * pow(L0, 2)
            + L0 * L1 * (293.8 + 711.2 * z + 1043 * L0)
            + 115.6 * L0
            - 7.109 * pow(L0, 2)
            + (70.0 / 9.0) * pow(L0, 3)
            + 11.9033 * (z1 / z)
        )
    )


def Cg_3_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    z1 = 1 - z
    L0 = np.log(z)
    L1 = np.log(z1)
    nf = 5.0
    meane = (-2.0 / 3.0) * 1.0 / nf
    meansque = (2 * (4.0 / 9.0) + 3 * (1.0 / 9.0)) * 1.0 / nf
    fl11g = meane**2 / meansque
    return (
        e_b
        * e_b
        * (
            (966.0 / 81.0) * (L1**5)
            - (1871.0 / 18.0) * (L1**4)
            + 89.31 * (L1**3)
            + 979.2 * (L1**2)
            - 2405 * L1
            + 1372 * z1 * (L1**4)
            - 15729
            - 310510 * z
            + 331570 * (z**2)
            - 244150 * z * (L0**2)
            - 253.3 * z * (L0**5)
            + L0 * L1 * (138230 - 237010 * L0)
            - 11860 * L0
            - 700.8 * (L0**2)
            - 1440 * (L0**3)
            + (4961.0 / 162.0) * (L0**4)
            - (134.0 / 9.0) * (L0**5)
            - (1.0 / z) * (6362.54 + 932.089 * L0)
            + fl11g
            * nf
            * (
                3.211 * (L1**2)
                + 19.04 * z * L1
                + 0.623 * z1 * (L1**3)
                - 64.47 * z
                + 121.6 * (z**2)
                - 45.82 * (z**3)
                - z * L0 * L1 * (31.68 + 37.24 * L0)
                + 11.27 * (z**2) * (L0**3)
                - 82.40 * z * L0
                - 16.08 * z * (L0**2)
                + (520.0 / 81.0) * z * (L0**3)
                + (20.0 / 27.0) * z * (L0**4)
            )
            + nf
            * (
                (131.0 / 81.0) * (L1**4)
                - 14.72 * (L1**3)
                + 3.607 * (L1**2)
                - 226.1 * L1
                + 4.762
                - 190 * z
                - 818.4 * (z**2)
                - 4019 * z * (L0**2)
                - L0 * L1 * (791.5 + 4646 * L0)
                + 739.0 * L0
                + 418.0 * (L0**2)
                + 104.3 * (L0**3)
                + (809.0 / 81.0) * (L0**4)
                + (12.0 / 9.0) * (L0**5)
                + (1.0 / z) * (84.423)
            )
        )
    )


def Cg_3_loc(z, Q):
    e_b = para.parameters["e_b"]
    z1 = 1 - z
    L0 = np.log(z)
    L1 = np.log(z1)
    nf = 5.0
    meane = (-2.0 / 3.0) * 1.0 / nf
    meansque = (2 * (4.0 / 9.0) + 3 * (1.0 / 9.0)) * 1.0 / nf
    fl11g = meane**2 / meansque
    return e_b * e_b * 0.625


def Cq_2_reg(z, Q, p):
    Q2 = Q * Q
    TR = 1.0 / 2.0
    e_b = para.parameters["e_b"]
    z1 = 1 - z
    L0 = np.log(z)
    L1 = np.log(z1)
    return (
        e_b
        * e_b
        * (
            z1 * ((8.0 / 3.0) * pow(L1, 2) - (32.0 / 3.0) * L1 + 9.8937)
            + (9.57 - 13.41 * z + 0.08 * pow(L1, 3)) * pow(z1, 2)
            + 5.667 * z * pow(L0, 3)
            - pow(L0, 2) * L1 * (20.26 - 33.93 * z)
            + 43.36 * z1 * L0
            - 1.053 * pow(L0, 2)
            + (40.0 / 9.0) * pow(L0, 3)
            + 5.2903 * (pow(z1, 2) / z)
        )
    )


def Cq_3_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    z1 = 1 - z
    L0 = np.log(z)
    L1 = np.log(z1)
    nf = 5.0
    meane = (-2.0 / 3.0) * 1.0 / nf
    meansque = (2 * (4.0 / 9.0) + 3 * (1.0 / 9.0)) * 1.0 / nf
    fl11ps = (meane**2 / meansque) - 3 * meane
    return (
        e_b
        * e_b
        * (
            z1
            * (
                (856.0 / 81.0) * (L1**4)
                - (6032.0 / 81.0) * (L1**3)
                + (130.57) * (L1**2)
                - (542) * L1
                + 8501
                - 4714 * z
                + 61.5 * z * z
            )
            + L0 * L1 * (8831 * L0 + 4162 * z1)
            - 15.44 * z * (L0**5)
            + 3333 * z * (L0**2)
            + 1615 * L0
            + 1208 * (L0**2)
            - 333.73 * (L0**3)
            + (4244.0 / 81.0) * (L0**4)
            - (40.0 / 9.0) * (L0**5)
            - (1.0 / z) * (2731.82 * z1 + 414.262 * L0)
            + fl11ps
            * z
            * (
                z1 * (126.42 - 50.29 * z - 50.15 * (z**2))
                - 26.717
                - 9.075 * z * z1 * L1
                - z * (L0**2) * (101.8 + 34.79 * L0 + 3.070 * (L0**2))
                + 59.59 * L0
                - (320.0 / 81.0) * (L0**2) * (5 + L0)
            )
            + nf
            * (
                z1
                * (
                    -(64.0 / 81.0) * (L1**3)
                    + (208.0 / 81.0) * (L1**2)
                    + 23.09 * L1
                    - 220.27
                    + 59.80 * z
                    - 177.6 * (z**2)
                )
                - L0 * L1 * (160.3 * L0 + 135.4 * z1)
                - 24.14 * z * (L0**3)
                - 215.4 * z * (L0**2)
                - 209.8 * L0
                - 90.38 * (L0**2)
                - (3568.0 / 243.0) * (L0**3)
                - (184.0 / 81.0) * (L0**4)
                + 40.2426 * (z1 / z)
            )
        )
    )


def Cq_3_loc(z, Q):
    e_b = para.parameters["e_b"]
    z1 = 1 - z
    L0 = np.log(z)
    L1 = np.log(z1)
    nf = 5.0
    meane = (-2.0 / 3.0) * 1.0 / nf
    meansque = (2 * (4.0 / 9.0) + 3 * (1.0 / 9.0)) * 1.0 / nf
    fl11ps = (meane**2 / meansque) - 3 * meane
    return e_b * e_b * fl11ps * (-11.888)


# FL
def CLg_1_reg(z, Q, p):
    TR = 1.0 / 2.0
    e_b = para.parameters["e_b"]
    return 16 * TR * e_b * e_b * z * (1.0 - z)


def CLg_2_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    omx = 1.0 - z
    dl = np.log(z)
    dl_2 = dl * dl
    dl1 = np.log(omx)
    dl1_2 = dl1 * dl1
    return (
        e_b
        * e_b
        * (
            (94.74 - 49.20 * z) * omx * dl1_2
            + 864.8 * omx * dl1
            + 1161 * z * dl * dl1
            + 60.06 * z * dl_2
            + 39.66 * omx * dl
            - 5.333 * (1.0 / z - 1)
        )
    )


def CLg_3_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    z1 = 1 - z
    L0 = np.log(z)
    L1 = np.log(z1)
    nf = 5.0
    meane = (-2.0 / 3.0) * 1.0 / nf
    meansque = (2 * (4.0 / 9.0) + 3 * (1.0 / 9.0)) * 1.0 / nf
    fl11g = meane**2 / meansque
    return (
        e_b
        * e_b
        * (
            z1
            * (
                144 * (L1**4)
                - (47024.0 / 27.0) * (L1**3)
                + 6319 * (L1**2)
                + 53160 * L1
            )
            + 72549 * L0 * L1
            + 88238 * L1 * (L0**2)
            + z1 * (3709 - 33514 * z - 9533 * (z**2))
            + 66773 * z * (L0**2)
            - 1117 * L0
            + 45.37 * (L0**2)
            - (5360.0 / 27.0) * (L0**3)
            - (1.0 / z) * (2044.70 * z1 + 409.506 * L0)
            + fl11g
            * nf
            * (
                z1
                * (
                    -0.0105 * (L1**3)
                    + 1.550 * (L1**2)
                    + 19.72 * z * L1
                    - 66.745 * z
                    + 0.615 * (z**2)
                )
                + (20.0 / 27.0) * z * (L0**4)
                + z * (L0**3) * ((280.0 / 81.0) + 2.260 * z)
                - z * (L0**2) * (15.40 - 2.201 * z)
                - z * L0 * (71.66 - 0.121 * z)
            )
            + nf
            * (
                z1
                * (
                    (32.0 / 3.0) * (L1**3)
                    - (1216.0 / 9.0) * (L1**2)
                    - 592.3 * L1
                    + 1511 * z * L1
                )
                + 311.3 * L0 * L1
                + 14.24 * L1 * (L0**2)
                + z1 * (577.3 - 729.0 * z)
                + 30.78 * z * (L0**3)
                + 366.0 * L0
                + (1000.0 / 9.0) * (L0**2)
                + (160.0 / 9.0) * (L0**3)
                + 88.5037 * (z1 / z)
            )
        )
    )


def CLb_1_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    CF = 4.0 / 3.0
    return e_b * e_b * 4 * CF * z


def CLb_2_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    z1 = 1.0 - z
    dl = np.log(z)
    dl1 = np.log(z1)
    dl_2 = dl * dl
    dl1_2 = dl1 * dl1
    return (
        e_b
        * e_b
        * (
            -40.41
            + 97.48 * z
            + (26.56 * z - 0.031) * dl_2
            - 14.85 * dl
            + 13.62 * dl1_2
            - 55.79 * dl1
            - 150.5 * dl * dl1
            + 5 * (16.0 / 27.0) * (6 * z * dl1 - 12 * z * dl - 25 * z + 6)
        )
    )


def CLb_2_loc(z, Q):
    e_b = para.parameters["e_b"]
    return e_b * e_b * (-0.164)


def CLq_2_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    omz = 1.0 - z
    dl = np.log(z)
    dl1 = np.log(omz)
    dl_2 = dl * dl
    omz2 = omz * omz
    omz3 = omz2 * omz
    return (
        e_b
        * e_b
        * (
            (15.94 - 5.212 * z) * omz2 * dl1
            + (0.421 + 1.520 * z) * dl_2
            + 28.09 * omz * dl
            - (2.370 / z - 19.27) * omz3
        )
    )


def CLq_3_reg(z, Q, p):
    e_b = para.parameters["e_b"]
    z1 = 1 - z
    L0 = np.log(z)
    L1 = np.log(z1)
    nf = 5.0
    meane = (-2.0 / 3.0) * 1.0 / nf
    meansque = (2 * (4.0 / 9.0) + 3 * (1.0 / 9.0)) * 1.0 / nf
    fl11ps = (meane**2 / meansque) - 3 * meane
    return (
        e_b
        * e_b
        * (
            (z1**2)
            * ((1568.0 / 27.0) * (L1**3) - (3968.0 / 9.0) * (L1**2) + (5124) * L1)
            + L0 * L1 * (2184 * L0 + 6059 * z1)
            - (z1**2) * (795.6 + 1036 * z)
            - 143.6 * z1 * L0
            + (2848.0 / 9.0) * (L0**2)
            - (1600.0 / 27.0) * (L0**3)
            - (z1 / z) * (885.53 * z1 + 182.00 * L0)
            + fl11ps
            * z
            * (
                z1 * (107.0 + 321.05 * z - 54.62 * (z**2))
                - 26.717
                + 9.773 * L0
                + z * L0 * (363.8 + 68.32 * L0)
                - (320.0 / 81.0) * (L0**2) * (2 + L0)
            )
            + nf
            * (
                (z1**2) * (-(32.0 / 9.0) * (L1**2) + 29.52 * L1 - 14.16 + 69.84 * z)
                + L0 * L1 * (35.18 * L0 + 73.06 * z1)
                - 35.24 * z * (L0**2)
                - 69.41 * z1 * L0
                - (128.0 / 9.0) * (L0**2)
                + 40.239 * ((z1**2) / z)
            )
        )
    )
