# Contains the N3LO matching conditions in N space
import numpy as np
from eko import harmonics

def Mbg_3_l1_N(n, nf):
    # TODO : check the harmonic sums
    sx = harmonics.compute_cache(n, 5, True)
    S1, Sm1 = sx[0]
    S2, Sm2 = sx[1]
    S3, S21, _, Sm21, _, Sm3 = sx[2]
    S4, S31, S211, Sm22, Sm211, Sm31, Sm4 = sx[3]
    tmp = (
        0.3333333333333333
        * (
            (
                -0.07407407407407407
                * (
                    -34560.0
                    - 144000.0 * n
                    - 299712.0 * np.power(n, 2)
                    - 453440.0 * np.power(n, 3)
                    - 534656.0 * np.power(n, 4)
                    - 492936.0 * np.power(n, 5)
                    - 435356.0 * np.power(n, 6)
                    - 228072.0 * np.power(n, 7)
                    + 154773.0 * np.power(n, 8)
                    + 398930.0 * np.power(n, 9)
                    + 371423.0 * np.power(n, 10)
                    + 214620.0 * np.power(n, 11)
                    + 80795.0 * np.power(n, 12)
                    + 18018.0 * np.power(n, 13)
                    + 1773.0 * np.power(n, 14)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            + (
                0.5925925925925926
                * (
                    192.0
                    + 592.0 * n
                    + 786.0 * np.power(n, 2)
                    + 163.0 * np.power(n, 3)
                    + 29.0 * np.power(n, 4)
                )
                * S1
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                1.7777777777777777
                * (18.0 + 59.0 * n + 31.0 * np.power(n, 2) + 10.0 * np.power(n, 3))
                * np.power(S1, 2)
            )
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (
                5.333333333333333
                * (
                    -2.0
                    - 5.0 * n
                    + 42.0 * np.power(n, 2)
                    + 39.0 * np.power(n, 3)
                    + 2.0 * np.power(n, 4)
                )
                * S2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (1.3333333333333333 * np.power(S1, 3) + 4.0 * S1 * S2 - 8.0 * S3)
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.3333333333333333
        * nf
        * (
            (
                0.07407407407407407
                * (
                    -34560.0
                    - 158976.0 * n
                    - 355584.0 * np.power(n, 2)
                    - 562304.0 * np.power(n, 3)
                    - 735104.0 * np.power(n, 4)
                    - 694320.0 * np.power(n, 5)
                    - 490544.0 * np.power(n, 6)
                    - 207408.0 * np.power(n, 7)
                    + 82971.0 * np.power(n, 8)
                    + 205070.0 * np.power(n, 9)
                    + 159437.0 * np.power(n, 10)
                    + 77604.0 * np.power(n, 11)
                    + 25877.0 * np.power(n, 12)
                    + 5454.0 * np.power(n, 13)
                    + 531.0 * np.power(n, 14)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            - (
                1.1851851851851851
                * (
                    78.0
                    + 251.0 * n
                    + 303.0 * np.power(n, 2)
                    + 77.0 * np.power(n, 3)
                    + 19.0 * np.power(n, 4)
                )
                * S1
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                3.5555555555555554
                * (6.0 + 22.0 * n + 11.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
                * np.power(S1, 2)
            )
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            + (
                5.333333333333333
                * (
                    -48.0
                    - 72.0 * n
                    - 108.0 * np.power(n, 2)
                    - 128.0 * np.power(n, 3)
                    - 123.0 * np.power(n, 4)
                    + 8.0 * np.power(n, 5)
                    + 62.0 * np.power(n, 6)
                    + 24.0 * np.power(n, 7)
                    + np.power(n, 8)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    0.8888888888888888 * np.power(S1, 3)
                    + 2.6666666666666665 * S1 * S2
                    - 6.222222222222222 * S3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.75
        * (
            (
                0.5925925925925926
                * (
                    288.0
                    + 1008.0 * n
                    + 80.0 * np.power(n, 2)
                    - 4560.0 * np.power(n, 3)
                    - 19122.0 * np.power(n, 4)
                    - 34963.0 * np.power(n, 5)
                    - 37157.0 * np.power(n, 6)
                    - 21724.0 * np.power(n, 7)
                    - 2662.0 * np.power(n, 8)
                    + 5168.0 * np.power(n, 9)
                    + 3634.0 * np.power(n, 10)
                    + 1035.0 * np.power(n, 11)
                    + 111.0 * np.power(n, 12)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            + (
                0.5925925925925926
                * (
                    536.0
                    + 1516.0 * n
                    + 3186.0 * np.power(n, 2)
                    + 3271.0 * np.power(n, 3)
                    + 1692.0 * np.power(n, 4)
                    + 487.0 * np.power(n, 5)
                    + 76.0 * np.power(n, 6)
                )
                * S1
            )
            / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            - (
                3.5555555555555554
                * (
                    26.0
                    + 82.0 * n
                    + 65.0 * np.power(n, 2)
                    + 23.0 * np.power(n, 3)
                    + 5.0 * np.power(n, 4)
                )
                * np.power(S1, 2)
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                3.5555555555555554
                * (
                    48.0
                    + 34.0 * n
                    + 40.0 * np.power(n, 2)
                    + 35.0 * np.power(n, 3)
                    + 84.0 * np.power(n, 4)
                    + 42.0 * np.power(n, 5)
                    + 5.0 * np.power(n, 6)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (42.666666666666664 * (-4.0 - 1.0 * n + np.power(n, 2)) * Sm2)
            / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (7.111111111111111 * (10.0 + 8.0 * n + 5.0 * np.power(n, 2)) * Sm2)
            / (n * (1.0 + n) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -1.3333333333333333 * np.power(S1, 3)
                    - 6.666666666666667 * S1 * S2
                    - 5.333333333333333 * S21
                    - 8.0 * S3
                    - 10.666666666666666 * S1 * Sm2
                    + 10.666666666666666 * Sm21
                    - 10.666666666666666 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.75
        * nf
        * (
            (
                0.2962962962962963
                * (
                    576.0
                    + 4128.0 * n
                    + 12416.0 * np.power(n, 2)
                    + 22080.0 * np.power(n, 3)
                    + 16644.0 * np.power(n, 4)
                    - 2110.0 * np.power(n, 5)
                    - 15710.0 * np.power(n, 6)
                    - 8917.0 * np.power(n, 7)
                    + 7139.0 * np.power(n, 8)
                    + 11990.0 * np.power(n, 9)
                    + 6742.0 * np.power(n, 10)
                    + 1845.0 * np.power(n, 11)
                    + 201.0 * np.power(n, 12)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            + (
                0.5925925925925926
                * (
                    232.0
                    + 776.0 * n
                    + 1878.0 * np.power(n, 2)
                    + 1820.0 * np.power(n, 3)
                    + 777.0 * np.power(n, 4)
                    + 176.0 * np.power(n, 5)
                    + 29.0 * np.power(n, 6)
                )
                * S1
            )
            / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            - (
                1.7777777777777777
                * (
                    46.0
                    + 131.0 * n
                    + 106.0 * np.power(n, 2)
                    + 43.0 * np.power(n, 3)
                    + 10.0 * np.power(n, 4)
                )
                * np.power(S1, 2)
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                1.7777777777777777
                * (
                    48.0
                    + 14.0 * n
                    + 17.0 * np.power(n, 2)
                    + 31.0 * np.power(n, 3)
                    + 105.0 * np.power(n, 4)
                    + 63.0 * np.power(n, 5)
                    + 10.0 * np.power(n, 6)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (21.333333333333332 * (-4.0 - 1.0 * n + np.power(n, 2)) * Sm2)
            / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (7.111111111111111 * (10.0 + 8.0 * n + 5.0 * np.power(n, 2)) * Sm2)
            / (n * (1.0 + n) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -0.8888888888888888 * np.power(S1, 3)
                    - 2.6666666666666665 * S1 * S2
                    - 5.333333333333333 * S21
                    - 4.444444444444445 * S3
                    - 5.333333333333333 * S1 * Sm2
                    + 5.333333333333333 * Sm21
                    - 8.0 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 2.0
        * (
            (
                -7.212341418957566
                * (
                    -248.0
                    + 12.0 * n
                    - 214.0 * np.power(n, 2)
                    - 407.0 * np.power(n, 3)
                    - 91.0 * np.power(n, 4)
                    + 135.0 * np.power(n, 5)
                    + 45.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                0.018518518518518517
                * (
                    27648.0
                    + 143424.0 * n
                    + 372192.0 * np.power(n, 2)
                    + 664624.0 * np.power(n, 3)
                    + 606016.0 * np.power(n, 4)
                    + 189820.0 * np.power(n, 5)
                    + 204230.0 * np.power(n, 6)
                    + 650149.0 * np.power(n, 7)
                    + 820775.0 * np.power(n, 8)
                    + 532170.0 * np.power(n, 9)
                    + 206656.0 * np.power(n, 10)
                    + 53973.0 * np.power(n, 11)
                    + 7299.0 * np.power(n, 12)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                0.2962962962962963
                * (
                    1944.0
                    + 6156.0 * n
                    + 7122.0 * np.power(n, 2)
                    + 13.0 * np.power(n, 3)
                    - 2242.0 * np.power(n, 4)
                    + 4008.0 * np.power(n, 5)
                    + 6764.0 * np.power(n, 6)
                    + 4206.0 * np.power(n, 7)
                    + 1586.0 * np.power(n, 8)
                    + 251.0 * np.power(n, 9)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                0.4444444444444444
                * (
                    864.0
                    + 2712.0 * n
                    + 3116.0 * np.power(n, 2)
                    + 4106.0 * np.power(n, 3)
                    + 2003.0 * np.power(n, 4)
                    - 1202.0 * np.power(n, 5)
                    - 592.0 * np.power(n, 6)
                    + 568.0 * np.power(n, 7)
                    + 441.0 * np.power(n, 8)
                    + 80.0 * np.power(n, 9)
                )
                * np.power(S1, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            + (
                0.8888888888888888
                * (
                    -144.0
                    - 496.0 * n
                    - 866.0 * np.power(n, 2)
                    - 499.0 * np.power(n, 3)
                    - 46.0 * np.power(n, 4)
                    + 11.0 * np.power(n, 5)
                )
                * np.power(S1, 3)
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                1.3333333333333333
                * (
                    480.0
                    + 1944.0 * n
                    + 4020.0 * np.power(n, 2)
                    + 6446.0 * np.power(n, 3)
                    + 7513.0 * np.power(n, 4)
                    + 4345.0 * np.power(n, 5)
                    + 462.0 * np.power(n, 6)
                    - 424.0 * np.power(n, 7)
                    - 43.0 * np.power(n, 8)
                    + 25.0 * np.power(n, 9)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                2.6666666666666665
                * (3.0 + n)
                * (
                    8.0
                    + 8.0 * n
                    + 52.0 * np.power(n, 2)
                    + 93.0 * np.power(n, 3)
                    + 114.0 * np.power(n, 4)
                    + 13.0 * np.power(n, 5)
                )
                * S1
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (48.0 * np.power(2.0 + n + np.power(n, 2), 2) * S21)
            / ((-1.0 + n) * n * (1.0 + n) * np.power(2.0 + n, 2))
            - (
                0.8888888888888888
                * (
                    432.0
                    - 452.0 * n
                    - 128.0 * np.power(n, 2)
                    + 977.0 * np.power(n, 3)
                    + 627.0 * np.power(n, 4)
                    + 195.0 * np.power(n, 5)
                    + 77.0 * np.power(n, 6)
                )
                * S3
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                16.0
                * (-4.0 - 1.0 * n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * Sm2
            )
            / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
            - (
                16.0
                * (
                    12.0
                    - 10.0 * n
                    - 54.0 * np.power(n, 2)
                    - 87.0 * np.power(n, 3)
                    - 40.0 * np.power(n, 4)
                    + 3.0 * np.power(n, 5)
                )
                * Sm2
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                64.0
                * (-2.0 - 14.0 * n - 13.0 * np.power(n, 2) + np.power(n, 3))
                * S1
                * Sm2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                16.0
                * (
                    8.0
                    + 20.0 * n
                    + 46.0 * np.power(n, 2)
                    + 27.0 * np.power(n, 3)
                    + 8.0 * np.power(n, 4)
                    + 3.0 * np.power(n, 5)
                )
                * S1
                * Sm2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                16.0
                * (
                    -12.0
                    + 32.0 * n
                    + 39.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
                * Sm21
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                (2.0 + n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * (-7.212341418957566 - 8.0 * Sm3)
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                16.0
                * (
                    8.0
                    - 6.0 * n
                    - 5.0 * np.power(n, 2)
                    + 8.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
                * Sm3
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    28.849365675830263 * S1
                    - 4.0 * np.power(S1, 4)
                    - 36.0 * np.power(S1, 2) * S2
                    + 8.0 * np.power(S2, 2)
                    - 24.0 * S211
                    - 8.0 * S31
                    - 40.0 * np.power(S1, 2) * Sm2
                    - 12.0 * np.power(Sm2, 2)
                    + S1 * (48.0 * S21 - 32.0 * S3 + 16.0 * Sm21)
                    + 32.0 * Sm211
                    - 8.0 * Sm22
                    - 8.0 * S1 * Sm3
                    - 20.0 * Sm4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.8888888888888888
        * (
            (57.69873135166053 * (-1.0 + n) * (-2.0 + 3.0 * n + 3.0 * np.power(n, 2)))
            / (np.power(n, 2) * np.power(1.0 + n, 2))
            - (
                0.5
                * (
                    96.0
                    + 400.0 * n
                    + 1296.0 * np.power(n, 2)
                    + 3168.0 * np.power(n, 3)
                    + 5062.0 * np.power(n, 4)
                    + 6277.0 * np.power(n, 5)
                    + 6853.0 * np.power(n, 6)
                    + 5026.0 * np.power(n, 7)
                    + 2368.0 * np.power(n, 8)
                    + 793.0 * np.power(n, 9)
                    + 149.0 * np.power(n, 10)
                )
            )
            / (np.power(n, 5) * np.power(1.0 + n, 5) * (2.0 + n))
            + (
                8.0
                * (
                    24.0
                    + 92.0 * n
                    + 172.0 * np.power(n, 2)
                    + 332.0 * np.power(n, 3)
                    + 529.0 * np.power(n, 4)
                    + 441.0 * np.power(n, 5)
                    + 179.0 * np.power(n, 6)
                    + 29.0 * np.power(n, 7)
                    + 2.0 * np.power(n, 8)
                )
                * S1
            )
            / (np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            - (
                8.0
                * (
                    22.0
                    + 79.0 * n
                    + 135.0 * np.power(n, 2)
                    + 160.0 * np.power(n, 3)
                    + 97.0 * np.power(n, 4)
                    + 26.0 * np.power(n, 5)
                    + 5.0 * np.power(n, 6)
                )
                * np.power(S1, 2)
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (64.0 * (1.0 + n) * np.power(S1, 3)) / (np.power(n, 2) * (2.0 + n))
            + (
                4.0
                * (
                    28.0
                    + 70.0 * n
                    + 96.0 * np.power(n, 2)
                    + 157.0 * np.power(n, 3)
                    + 75.0 * np.power(n, 4)
                    + 39.0 * np.power(n, 5)
                    + 23.0 * np.power(n, 6)
                )
                * S2
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                16.0
                * (
                    -4.0
                    - 18.0 * n
                    + 3.0 * np.power(n, 2)
                    + 10.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * S1
                * S2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (64.0 * (2.0 + n + np.power(n, 2)) * S21)
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                8.0
                * (2.0 + 5.0 * n + np.power(n, 2))
                * (2.0 - 1.0 * n + 3.0 * np.power(n, 2))
                * S3
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                32.0
                * (
                    8.0
                    - 2.0 * n
                    + np.power(n, 2)
                    - 1.0 * np.power(n, 3)
                    + 4.0 * np.power(n, 4)
                    + 2.0 * np.power(n, 5)
                )
                * Sm2
            )
            / (np.power(n, 3) * np.power(1.0 + n, 2) * (2.0 + n))
            + ((2.0 - 1.0 * n + np.power(n, 2)) * (-128.0 * S1 * Sm2 + 128.0 * Sm21))
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (64.0 * (-1.0 + n) * Sm3) / (np.power(1.0 + n, 2) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    2.0 * np.power(S1, 4)
                    + 4.0 * np.power(S1, 2) * S2
                    + 6.0 * np.power(S2, 2)
                    + 24.0 * S211
                    + S1 * (-32.0 * S21 + 24.0 * S3)
                    - 8.0 * S31
                    + 20.0 * S4
                    + 16.0 * S2 * Sm2
                    + 8.0 * np.power(Sm2, 2)
                    - 16.0 * Sm22
                    + 32.0 * S1 * Sm3
                    - 32.0 * Sm31
                    + 40.0 * Sm4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 4.5
        * (
            (
                2.4041138063191885
                * (
                    -624.0
                    - 596.0 * n
                    - 956.0 * np.power(n, 2)
                    - 637.0 * np.power(n, 3)
                    - 111.0 * np.power(n, 4)
                    + 249.0 * np.power(n, 5)
                    + 83.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                0.14814814814814814
                * (
                    34560.0
                    + 211392.0 * n
                    + 604032.0 * np.power(n, 2)
                    + 1.099952e6 * np.power(n, 3)
                    + 1.506496e6 * np.power(n, 4)
                    + 1.640548e6 * np.power(n, 5)
                    + 1.596952e6 * np.power(n, 6)
                    + 1.332497e6 * np.power(n, 7)
                    + 702425.0 * np.power(n, 8)
                    - 16829.0 * np.power(n, 9)
                    - 322813.0 * np.power(n, 10)
                    - 155710.0 * np.power(n, 11)
                    + 66350.0 * np.power(n, 12)
                    + 101719.0 * np.power(n, 13)
                    + 47251.0 * np.power(n, 14)
                    + 10527.0 * np.power(n, 15)
                    + 939.0 * np.power(n, 16)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 5)
            )
            - (
                0.2962962962962963
                * (
                    432.0
                    + 7344.0 * n
                    + 4968.0 * np.power(n, 2)
                    - 26584.0 * np.power(n, 3)
                    - 33249.0 * np.power(n, 4)
                    - 44369.0 * np.power(n, 5)
                    - 71221.0 * np.power(n, 6)
                    - 71723.0 * np.power(n, 7)
                    - 31716.0 * np.power(n, 8)
                    + 13564.0 * np.power(n, 9)
                    + 22254.0 * np.power(n, 10)
                    + 9908.0 * np.power(n, 11)
                    + 2368.0 * np.power(n, 12)
                    + 296.0 * np.power(n, 13)
                )
                * S1
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            + (
                0.4444444444444444
                * (
                    -864.0
                    - 2016.0 * n
                    + 1132.0 * np.power(n, 2)
                    + 5904.0 * np.power(n, 3)
                    + 561.0 * np.power(n, 4)
                    - 4745.0 * np.power(n, 5)
                    - 448.0 * np.power(n, 6)
                    + 2710.0 * np.power(n, 7)
                    + 1897.0 * np.power(n, 8)
                    + 883.0 * np.power(n, 9)
                    + 170.0 * np.power(n, 10)
                )
                * np.power(S1, 2)
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                0.8888888888888888
                * (
                    8.0
                    - 218.0 * n
                    - 139.0 * np.power(n, 2)
                    + 26.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * np.power(S1, 3)
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                0.4444444444444444
                * (
                    -2016.0
                    - 6240.0 * n
                    - 5324.0 * np.power(n, 2)
                    - 2352.0 * np.power(n, 3)
                    - 861.0 * np.power(n, 4)
                    - 3047.0 * np.power(n, 5)
                    - 2692.0 * np.power(n, 6)
                    + 2506.0 * np.power(n, 7)
                    + 3091.0 * np.power(n, 8)
                    + 1213.0 * np.power(n, 9)
                    + 170.0 * np.power(n, 10)
                )
                * S2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                2.6666666666666665
                * (
                    -240.0
                    - 368.0 * n
                    - 254.0 * np.power(n, 2)
                    - 319.0 * np.power(n, 3)
                    - 285.0 * np.power(n, 4)
                    + 15.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
                )
                * S1
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (2.0 + n + np.power(n, 2))
                * (
                    24.0
                    + 2.0 * n
                    + 13.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * S21
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                0.8888888888888888
                * (
                    -1188.0
                    - 1336.0 * n
                    - 1561.0 * np.power(n, 2)
                    - 899.0 * np.power(n, 3)
                    - 420.0 * np.power(n, 4)
                    + 165.0 * np.power(n, 5)
                    + 55.0 * np.power(n, 6)
                )
                * S3
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (-4.0 - 1.0 * n + np.power(n, 2))
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * Sm2
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            + (
                1.7777777777777777
                * (
                    648.0
                    + 2916.0 * n
                    + 3802.0 * np.power(n, 2)
                    + 2731.0 * np.power(n, 3)
                    + 1381.0 * np.power(n, 4)
                    + 1517.0 * np.power(n, 5)
                    + 2086.0 * np.power(n, 6)
                    + 1508.0 * np.power(n, 7)
                    + 597.0 * np.power(n, 8)
                    + 94.0 * np.power(n, 9)
                )
                * Sm2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                32.0
                * (
                    -12.0
                    + 4.0 * n
                    + 13.0 * np.power(n, 2)
                    - 32.0 * np.power(n, 3)
                    - 23.0 * np.power(n, 4)
                    + 2.0 * np.power(n, 5)
                )
                * S1
                * Sm2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (
                    -48.0
                    - 68.0 * n
                    - 24.0 * np.power(n, 2)
                    - 49.0 * np.power(n, 3)
                    + 34.0 * np.power(n, 4)
                    + 11.0 * np.power(n, 5)
                )
                * S1
                * Sm2
            )
            / ((-1.0 + n) * np.power(n, 2) * (1.0 + n) * np.power(2.0 + n, 2))
            + (
                5.333333333333333
                * (
                    -72.0
                    - 92.0 * n
                    - 194.0 * np.power(n, 2)
                    - 361.0 * np.power(n, 3)
                    - 189.0 * np.power(n, 4)
                    + 33.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
                )
                * Sm21
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                (2.0 + n + np.power(n, 2))
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * (-2.4041138063191885 - 2.6666666666666665 * Sm3)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (
                    -108.0
                    - 176.0 * n
                    - 263.0 * np.power(n, 2)
                    - 247.0 * np.power(n, 3)
                    - 114.0 * np.power(n, 4)
                    + 33.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
                )
                * Sm3
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -28.849365675830263 * S1
                    + 2.0 * np.power(S1, 4)
                    + 32.0 * np.power(S1, 2) * S2
                    + 2.0 * np.power(S2, 2)
                    + 16.0 * S31
                    + 4.0 * S4
                    + (40.0 * np.power(S1, 2) + 16.0 * S2) * Sm2
                    + 12.0 * np.power(Sm2, 2)
                    + S1 * (-16.0 * S21 + 40.0 * S3 - 80.0 * Sm21)
                    + 96.0 * Sm211
                    - 56.0 * Sm22
                    + 72.0 * S1 * Sm3
                    - 64.0 * Sm31
                    + 44.0 * Sm4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
    )
    return tmp

def Mbg_3_l2_N(n, nf):
    # TODO : check the harmonic sums
    sx = harmonics.compute_cache(n, 5, True)
    S1, _ = sx[0]
    S2, Sm2 = sx[1]
    S3, _, _, Sm21, _, Sm3 = sx[2]
    #S4, S31, S211, Sm22, Sm211, Sm31, Sm4 = sx[3]
    tmp = (
        0.3333333333333333
        * (
            (
                -0.4444444444444444
                * (
                    -1152.0
                    - 3648.0 * n
                    - 6640.0 * np.power(n, 2)
                    - 11680.0 * np.power(n, 3)
                    - 13912.0 * np.power(n, 4)
                    - 9464.0 * np.power(n, 5)
                    + 383.0 * np.power(n, 6)
                    + 6505.0 * np.power(n, 7)
                    + 5730.0 * np.power(n, 8)
                    + 2894.0 * np.power(n, 9)
                    + 903.0 * np.power(n, 10)
                    + 129.0 * np.power(n, 11)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                3.5555555555555554
                * (18.0 + 37.0 * n + 14.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
                * S1
            )
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (4.0 * np.power(S1, 2) - 1.3333333333333333 * S2)
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.3333333333333333
        * nf
        * (
            (
                -0.4444444444444444
                * (
                    1152.0
                    + 3456.0 * n
                    + 5456.0 * np.power(n, 2)
                    + 7328.0 * np.power(n, 3)
                    + 5096.0 * np.power(n, 4)
                    + 2236.0 * np.power(n, 5)
                    + 1463.0 * np.power(n, 6)
                    + 1513.0 * np.power(n, 7)
                    + 1290.0 * np.power(n, 8)
                    + 698.0 * np.power(n, 9)
                    + 231.0 * np.power(n, 10)
                    + 33.0 * np.power(n, 11)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                3.5555555555555554
                * (6.0 + 19.0 * n + 8.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
                * S1
            )
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (1.3333333333333333 * np.power(S1, 2) + 1.3333333333333333 * S2)
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.75
        * (
            (
                -0.8888888888888888
                * (
                    384.0
                    + 1696.0 * n
                    + 2928.0 * np.power(n, 2)
                    + 3484.0 * np.power(n, 3)
                    + 2740.0 * np.power(n, 4)
                    + 1731.0 * np.power(n, 5)
                    + 1262.0 * np.power(n, 6)
                    + 724.0 * np.power(n, 7)
                    + 270.0 * np.power(n, 8)
                    + 45.0 * np.power(n, 9)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            + (
                3.5555555555555554
                * (
                    20.0
                    - 14.0 * n
                    - 1.0 * np.power(n, 2)
                    + 20.0 * np.power(n, 3)
                    + 5.0 * np.power(n, 4)
                )
                * S1
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (-4.0 * np.power(S1, 2) - 4.0 * S2 - 8.0 * Sm2)
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.75
        * nf
        * (
            (
                -0.8888888888888888
                * (
                    -160.0
                    - 672.0 * n
                    - 1012.0 * np.power(n, 2)
                    - 1120.0 * np.power(n, 3)
                    - 717.0 * np.power(n, 4)
                    - 182.0 * np.power(n, 5)
                    + 56.0 * np.power(n, 6)
                    + 54.0 * np.power(n, 7)
                    + 9.0 * np.power(n, 8)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                3.5555555555555554
                * (
                    20.0
                    + 58.0 * n
                    + 47.0 * np.power(n, 2)
                    + 20.0 * np.power(n, 3)
                    + 5.0 * np.power(n, 4)
                )
                * S1
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -1.3333333333333333 * np.power(S1, 2)
                    - 1.3333333333333333 * S2
                    - 2.6666666666666665 * Sm2
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.8888888888888888
        * (
            (
                32.0
                + 128.0 * n
                + 292.0 * np.power(n, 2)
                + 528.0 * np.power(n, 3)
                + 685.0 * np.power(n, 4)
                + 636.0 * np.power(n, 5)
                + 350.0 * np.power(n, 6)
                + 132.0 * np.power(n, 7)
                + 33.0 * np.power(n, 8)
            )
            / (np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            - (
                8.0
                * (
                    12.0
                    + 34.0 * n
                    + 60.0 * np.power(n, 2)
                    + 84.0 * np.power(n, 3)
                    + 51.0 * np.power(n, 4)
                    + 18.0 * np.power(n, 5)
                    + 5.0 * np.power(n, 6)
                )
                * S1
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                4.0
                * (
                    20.0
                    + 48.0 * n
                    + 43.0 * np.power(n, 2)
                    + 14.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
                * np.power(S1, 2)
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                12.0
                * (2.0 + n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * S2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (32.0 * (2.0 + n + np.power(n, 2)) * Sm2)
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    4.0 * np.power(S1, 3)
                    - 12.0 * S1 * S2
                    - 8.0 * S3
                    - 16.0 * S1 * Sm2
                    + 16.0 * Sm21
                    - 8.0 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 2.0
        * (
            (
                0.1111111111111111
                * (
                    2304.0
                    - 2864.0 * n
                    - 16272.0 * np.power(n, 2)
                    - 24608.0 * np.power(n, 3)
                    - 12692.0 * np.power(n, 4)
                    + 6675.0 * np.power(n, 5)
                    + 12206.0 * np.power(n, 6)
                    + 7636.0 * np.power(n, 7)
                    + 2934.0 * np.power(n, 8)
                    + 489.0 * np.power(n, 9)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                0.8888888888888888
                * (
                    144.0
                    + 24.0 * n
                    - 668.0 * np.power(n, 2)
                    - 314.0 * np.power(n, 3)
                    - 2.0 * np.power(n, 4)
                    + 119.0 * np.power(n, 5)
                    + 613.0 * np.power(n, 6)
                    + 635.0 * np.power(n, 7)
                    + 273.0 * np.power(n, 8)
                    + 40.0 * np.power(n, 9)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            + (
                2.6666666666666665
                * (
                    36.0
                    + 56.0 * n
                    + 29.0 * np.power(n, 2)
                    - 137.0 * np.power(n, 3)
                    - 120.0 * np.power(n, 4)
                    - 9.0 * np.power(n, 5)
                    + np.power(n, 6)
                )
                * np.power(S1, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (
                    18.0
                    + 7.0 * n
                    + 8.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -8.0 * np.power(S1, 3)
                    + 4.0 * S3
                    + 6.0 * Sm2
                    - 8.0 * Sm21
                    + 4.0 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 4.5
        * (
            (
                1.7777777777777777
                * (
                    -576.0
                    - 2592.0 * n
                    - 5296.0 * np.power(n, 2)
                    - 6992.0 * np.power(n, 3)
                    - 6012.0 * np.power(n, 4)
                    - 4462.0 * np.power(n, 5)
                    - 3841.0 * np.power(n, 6)
                    - 4619.0 * np.power(n, 7)
                    - 4428.0 * np.power(n, 8)
                    - 2325.0 * np.power(n, 9)
                    - 511.0 * np.power(n, 10)
                    + 101.0 * np.power(n, 11)
                    + 72.0 * np.power(n, 12)
                    + 9.0 * np.power(n, 13)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            + (
                0.8888888888888888
                * (
                    -576.0
                    - 1296.0 * n
                    + 968.0 * np.power(n, 2)
                    + 2568.0 * np.power(n, 3)
                    + 2238.0 * np.power(n, 4)
                    - 325.0 * np.power(n, 5)
                    - 521.0 * np.power(n, 6)
                    + 788.0 * np.power(n, 7)
                    + 830.0 * np.power(n, 8)
                    + 425.0 * np.power(n, 9)
                    + 85.0 * np.power(n, 10)
                )
                * S1
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                1.3333333333333333
                * (
                    -48.0
                    - 116.0 * n
                    + 4.0 * np.power(n, 2)
                    - 85.0 * np.power(n, 3)
                    - 87.0 * np.power(n, 4)
                    + 33.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
                )
                * np.power(S1, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                1.3333333333333333
                * (2.0 + n + np.power(n, 2))
                * (
                    -72.0
                    - 94.0 * n
                    - 83.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (
                    -48.0
                    - 70.0 * n
                    - 59.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * Sm2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    4.0 * np.power(S1, 3)
                    + 12.0 * S1 * S2
                    + 4.0 * S3
                    + 16.0 * S1 * Sm2
                    - 8.0 * Sm21
                    + 4.0 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
    )
    return tmp

def Mbg_3_l3_N(n, nf):
    # TODO : check the harmonic sums
    sx = harmonics.compute_cache(n, 5, True)
    S1, _ = sx[0]
    # S2, Sm2 = sx[1]
    # S3, S21, _, Sm21, _, Sm3 = sx[2]
    # S4, S31, S211, Sm22, Sm211, Sm31, Sm4 = sx[3]
    tmp = (
        (0.8888888888888888 * (2.0 + n + np.power(n, 2))) / (n * (1.0 + n) * (2.0 + n))
        - 0.3333333333333333
        * nf
        * (
            (
                0.8888888888888888
                * (2.0 + n + np.power(n, 2))
                * (
                    -24.0
                    - 28.0 * n
                    - 38.0 * np.power(n, 2)
                    - 17.0 * np.power(n, 3)
                    - 1.0 * np.power(n, 4)
                    + 9.0 * np.power(n, 5)
                    + 3.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (3.5555555555555554 * (2.0 + n + np.power(n, 2)) * S1)
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.75
        * nf
        * (
            (
                -7.111111111111111
                * (1.0 + n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (3.5555555555555554 * (2.0 + n + np.power(n, 2)) * S1)
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.3333333333333333
        * (
            (
                -1.7777777777777777
                * (2.0 + n + np.power(n, 2))
                * (
                    -12.0
                    - 20.0 * n
                    - 31.0 * np.power(n, 2)
                    - 16.0 * np.power(n, 3)
                    + 7.0 * np.power(n, 4)
                    + 18.0 * np.power(n, 5)
                    + 6.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (14.222222222222221 * (2.0 + n + np.power(n, 2)) * S1)
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.75
        * (
            (
                -49.77777777777778
                * (1.0 + n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (24.88888888888889 * (2.0 + n + np.power(n, 2)) * S1)
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.8888888888888888
        * (
            (
                -0.6666666666666666
                * (2.0 + n + np.power(n, 2))
                * np.power(2.0 + 3.0 * n + 3.0 * np.power(n, 2), 2)
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                5.333333333333333
                * (2.0 + n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * S1
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (10.666666666666666 * (2.0 + n + np.power(n, 2)) * np.power(S1, 2))
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 4.5
        * (
            (
                1.7777777777777777
                * (1.0 + n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                0.8888888888888888
                * (2.0 + n + np.power(n, 2))
                * (
                    -48.0
                    - 70.0 * n
                    - 59.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (10.666666666666666 * (2.0 + n + np.power(n, 2)) * np.power(S1, 2))
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 2.0
        * (
            (
                -0.2222222222222222
                * (2.0 + n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * (
                    -48.0
                    - 70.0 * n
                    - 59.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                0.8888888888888888
                * (2.0 + n + np.power(n, 2))
                * (6.0 + n + np.power(n, 2))
                * (4.0 + 7.0 * n + 7.0 * np.power(n, 2))
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (21.333333333333332 * (2.0 + n + np.power(n, 2)) * np.power(S1, 2))
            / (n * (1.0 + n) * (2.0 + n))
        )
    )
    return tmp