# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np

from scamlgp.benchmarking.functions.base import Base


def hartmann_function(
    x: np.ndarray, alpha: np.ndarray, A: np.ndarray, P: np.ndarray
) -> np.ndarray:
    """The hartmann function.

    Parameters
    ----------
    x:
        Numerical representation of the points for which the function should be
        evaluated. shape = (n_points, n_features)
    alpha:
        The parameters of the Hartmann function. shape = (4,)
    A:
        The A-matrix, see function documentation. shape = (4, n_features)
    P:
        The P-matrix, see function documentation. shape = (4, n_features)

    Returns
    -------
    y:
        The function value. shape = (n_points,)

    """
    exponent = np.exp(
        -np.sum(A[:, :, None] * (x.T[None, :, :] - P[:, :, None]) ** 2, axis=1)
    )
    y = (-alpha[None, :] @ exponent).reshape(-1, 1)
    return y


class Hartmann3D(Base):
    def __call__(  # type: ignore[override]
        self,
        x1: float,
        x2: float,
        x3: float,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        alpha4: float,
    ) -> float:
        r"""The three-dimensional Hartmann function.

        The function has four local minima, one global minimum, and is given by

        .. math:: f(x, \alpha) = -\sum_{i=1}^{4} \alpha_i \exp \left(
        -\sum_{j=1}^{3} A_{i,j}\left( x_j - P_{i, j} \right)^2 \right)

        where

        .. math::
            \mathbf{A} = \begin{bmatrix}
            3.0 & 10 & 30 \\
            0.1 & 10 & 35 \\
            3.0 & 10 & 30 \\
            0.1 & 10 & 35
            \end{bmatrix}

        .. math::
            \mathbf{P} = 10^{-4} \begin{bmatrix}
            3689 & 1170 & 2673 \\
            4699 & 4387 & 7470 \\
            1091 & 8732 & 5547 \\
            381 & 5743 & 8828
            \end{bmatrix}

        The domain is given by:

        .. math::
            \mathbf{x}_i \in (0, 1)

        Reference: https://www.sfu.ca/~ssurjano/hart3.html

        Parameters:
        -----------
        x1, x2, x3 Values for the searchable parameters.
        alpha1, alpha2, alpha3, alpha4: Values for the parameters of the Hartmann
            function.

        Returns:
        --------
            Observed value at the query point.
        """
        x_dim_num = 3
        alpha = np.array([alpha1, alpha2, alpha3, alpha4])
        x = np.hstack([x1, x2, x3])[None, :]
        assert x.shape == (1, x_dim_num)
        A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = 1e-4 * np.array(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        )

        y = hartmann_function(x, alpha, A, P)

        return float(y.squeeze())


class Hartmann6D(Base):
    def __call__(  # type: ignore[override]
        self,
        x1: float,
        x2: float,
        x3: float,
        x4: float,
        x5: float,
        x6: float,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        alpha4: float,
    ) -> float:
        r"""The six-dimensional Hartmann function.

        The function has six local minima, one global minimum, and is given by

        .. math:: f(x, \alpha) = -\sum_{i=1}^{4} \alpha_i \exp \left(
        -\sum_{j=1}^{3} A_{i,j}\left( x_j - P_{i, j} \right)^2 \right)

        where

        .. math::
            \mathbf{A} = \begin{bmatrix}
            10 & 3 & 17 & 3.5 & 1.7 & 8 \\
            0.05 & 10 & 17 & 0.1 & 8 & 14 \\
            3 & 3.5 & 1.7 & 10 & 17 & 8 \\
            17 & 8 & 0.05 & 10 & 0.1 & 14
            \end{bmatrix}

        .. math::
            \mathbf{P} = 10^{-4} \begin{bmatrix}
            1312 & 1696 & 5569 & 124 & 8283 & 5886 \\
            2329 & 4135 & 8307 & 3736 & 1004 & 9991 \\
            2348 & 1451 & 3522 & 2883 & 3047 & 6650 \\
            4047 & 8828 & 8732 & 5743 & 1091 & 381
            \end{bmatrix}

        The domain is given by:

        .. math::
            \mathbf{x}_i \in (0, 1)

        Reference: https://www.sfu.ca/~ssurjano/hart6.html

        Parameters:
        -----------
        x1, x2, x3, x4, x5, x6: Numerical representation of the points.
        alpha1, alpha2, alpha3, alpha4: The parameters for the Hartmann 6D function.

        Returns:
        --------
        Observed value at the query point.
        """
        x_dim_num = 6
        alpha = np.array([alpha1, alpha2, alpha3, alpha4])
        x = np.hstack([x1, x2, x3, x4, x5, x6])[None, :]
        assert x.shape == (1, x_dim_num)
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        y = hartmann_function(x, alpha, A, P)

        return float(y.squeeze())
