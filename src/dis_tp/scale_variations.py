import numpy as np
from eko import beta


class IncorrectNumberOfIngredients(ValueError):
    pass


def beta_qcd(order, nf):
    return beta.beta_qcd((order + 2, 0), nf)


def apply_rensv_kernel(
    order: int, m: int, ingredients: list, mur_ratio: float, nf: int
):
    """Apply the renormalization scale variation kernel."""
    if np.isclose(mur_ratio, 1.0):
        return ingredients[0]
    if len(ingredients) != (order + 1):
        raise IncorrectNumberOfIngredients
    LR = np.log(mur_ratio)
    if order == 0:
        return ingredients[0]
    elif order == 1:
        return ingredients[0] - ingredients[-1] * beta_qcd(0, nf) * LR * m
    elif order == 2:
        return (
            ingredients[0]
            - LR
            * (
                m * beta_qcd(1, nf) * ingredients[-1]
                + (m + 1) * beta_qcd(0, nf) * ingredients[-2]
            )
            + 0.5
            * (LR**2)
            * m
            * (m + 1)
            * beta_qcd(0, nf)
            * beta_qcd(0, nf)
            * ingredients[-1]
        )
    elif order == 3:
        raise NotImplementedError
