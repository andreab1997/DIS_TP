import numpy as np
from eko import beta


class IncorrectNumberOfIngredients(ValueError):
    pass


def apply_sv_kernel(
    order: int,
    m: int,
    ingredients: list,
    z,
    Q,
    p,
    nf,
    mur_ratio: float,
    muf_ratio: float,
):
    if np.isclose(muf_ratio, 1.0) and np.isclose(mur_ratio, 1.0):
        return ingredients[0]
    if len(ingredients) != (order + 1):
        raise IncorrectNumberOfIngredients
    LF = np.log(muf_ratio)
    LR = np.log(mur_ratio)
    if order == 0:
        return ingredients[0]
    elif order == 1:
        # return ingredients[0] - ingredients[-1](z, Q, p, nf) * gamma[0] * LF - ingredients[-1](z, Q, p, nf) * beta[0] * LR * m
        return ingredients[0]
    elif order == 2:
        # return ingredients[0] - LR*(m*beta[1]*ingredients[-1](z, Q, p, nf) + (m+1)*beta[0]*ingredients[-2](z, Q, p, nf)) + 0.5*(LR**2)*m*(m+1)*beta[0]*beta[0]*ingredients[-1](z, Q, p, nf) - LF*gamma[1]*ingredients[-1](z, Q, p, nf) - LF*gamma[0]*ingredients[-2](z, Q, p, nf) +LF*LR*(m+1)*beta[0]*gamma[0]*ingredients[-1](z, Q, p, nf) + 0.5*(LF**2)*(gamma[0]*gamma[0] - beta[0]*gamma[0])*ingredients[-1](z, Q, p, nf)
        return ingredients[0]
    elif order == 3:
        return ingredients[0]
