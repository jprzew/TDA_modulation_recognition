from .feature import Feature
import pandas as pd
from typing import Union
import numpy as np
from math import sqrt


def wasserstein_amplitude(diagram: np.array, p: Union[int, float]) -> float:
    """Compute Wasserstein amplitude of a persistence diagram.

    Attributes
    ----------
    diagram : np.array - Persistence diagram.
    p : int or float - p-parameter of the Lp-norm.
    """
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes < float('inf')]

    return np.linalg.norm(lifetimes, p) / sqrt(2)


def persistent_entropy(diagram: np.array) -> float:
    """Compute persistent entropy of a persistence diagram."""
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes < float('inf')]
    L = sum(lifetimes)

    return sum((lifetimes / L) * np.log(lifetimes / L))


def new_entropy(numbers: np.array) -> float:
    """New algorithm to compute entropy. Needs to be tested."""

    numbers = numbers[abs(numbers) < float('inf')]

    if len(numbers) <= 1:
        return 0

    numbers = np.sort(numbers)

    differences = numbers[2:] - numbers[:-2]
    differences = np.insert(differences, 0, numbers[1] - numbers[0])
    differences = np.append(differences, numbers[-1] - numbers[-2])
    differences = differences[differences > 0]

    return (-1 / len(numbers)) * sum(np.log((2/len(numbers)) *
                                            (1 / differences)))


class No(Feature):
    """Number of homology cycles.

    Attributes
    ----------
    n : int - Grading of homology groups to compute.
    dim : int - Dimension of the point cloud.
    step : int or str - Step of the sliding window (number or column name).
    eps : float - Epsilon to control short-living cycles
    kind : str - Kind of the filtration.
    fil : str - Type of the filtration.
    preproc : str - Preprocessing method (None, 'fft').
    """

    def __init__(self,
                 n: int,
                 dim: int = 2,
                 step: Union[int, str] = 1,
                 eps: float = 0.0,
                 kind: str = None,
                 fil: str = None,
                 preproc: str = None):
        self.dim = dim
        self.n = n
        self.step = step
        self.eps = eps
        self.kind = kind
        self.fil = fil
        self.preproc = preproc

    def compute(self):
        homology = self.creator.create_feature('homology',
                                               n=self.n,
                                               dim=self.dim,
                                               step=self.step,
                                               eps=self.eps,
                                               kind=self.kind,
                                               fil=self.fil,
                                               preproc=self.preproc)

        return homology.values().map(lambda x: x.shape[0])


class Mean(Feature):
    """Mean of homology lifetimes

    Attributes
    ----------
    n : int - Grading of homology groups to compute.
    dim : int - Dimension of the point cloud.
    step : int or str - Step of the sliding window (number or column name).
    eps : float - Epsilon to control short-living cycles
    kind : str - Kind of the filtration.
    fil : str - Type of the filtration.
    preproc : str - Preprocessing method (None, 'fft').
    """

    def __init__(self,
                 n: int,
                 dim: int = 2,
                 step: Union[int, str] = 1,
                 eps: float = 0.0,
                 kind: str = None,
                 fil: str = None,
                 preproc: str = None):
        self.dim = dim
        self.n = n
        self.step = step
        self.eps = eps
        self.kind = kind
        self.fil = fil
        self.preproc = preproc

    def compute(self):
        life_time = self.creator.create_feature('lifetime',
                                                n=self.n,
                                                dim=self.dim,
                                                step=self.step,
                                                eps=self.eps,
                                                kind=self.kind,
                                                fil=self.fil,
                                                preproc=self.preproc)
        return life_time.values().np.mean()


class Var(Feature):
    """Variance of homology lifetimes

    Attributes
    ----------
    n : int - Grading of homology groups to compute.
    dim : int - Dimension of the point cloud.
    step : int or str - Step of the sliding window (number or column name).
    eps : float - Epsilon to control short-living cycles
    kind : str - Kind of the filtration.
    fil : str - Type of the filtration.
    preproc : str - Preprocessing method (None, 'fft').
    """

    def __init__(self,
                 n: int,
                 dim: int = 2,
                 step: Union[int, str] = 1,
                 eps: float = 0.0,
                 kind: str = None,
                 fil: str = None,
                 preproc: str = None):
        self.dim = dim
        self.n = n
        self.step = step
        self.eps = eps
        self.kind = kind
        self.fil = fil
        self.preproc = preproc

    def compute(self):
        lifetime = self.creator.create_feature('lifetime',
                                               n=self.n,
                                               dim=self.dim,
                                               step=self.step,
                                               eps=self.eps,
                                               kind=self.kind,
                                               fil=self.fil,
                                               preproc=self.preproc)
        return lifetime.values().np.var()


class KmpFeatures(Feature):
    """Features from 'Chatter Classification in Turning using
    Machine Learning and Topological Data Analysis'

    Attributes
    ----------
    k : int - Number of KMP-feature (one of: {1, 2, 3, 4, 5})
    n : int - Grading of homology groups to compute.
    dim : int - Dimension of the point cloud.
    step : int or str - Step of the sliding window (number or column name).
    eps : float - Epsilon to control short-living cycles
    kind : str - Kind of the filtration.
    fil : str - Type of the filtration.
    preproc : str - Preprocessing method (None, 'fft')."""

    def __init__(self,
                 k: int,
                 n: int,
                 dim: int = 2,
                 step: Union[int, str] = 1,
                 eps: float = 0.0,
                 kind: str = None,
                 fil: str = None,
                 preproc: str = None):
        self.dim = dim
        self.n = n
        self.k = k
        self.step = step
        self.eps = eps
        self.kind = kind
        self.fil = fil
        self.preproc = preproc

    def compute(self):
        def __features(D):
            mean_y = np.array([y for _, y in D if y != float('inf')]).mean()

            return (sum([x * (y - x) for x, y in D if y != float('inf')]),
                    sum([(mean_y - y) * (y - x) for x, y in D
                         if y != float('inf')]),
                    sum([x ** 2 * (y - x) for x, y in D if y != float('inf')]),
                    sum([(mean_y - y) ** 2 * (y - x) ** 4 for x, y in D
                         if y != float('inf')]),
                    max([y - x for x, y in D if y != float('inf')]))

        homology = self.creator.create_feature('homology',
                                               n=self.n,
                                               dim=self.dim,
                                               step=self.step,
                                               eps=self.eps,
                                               kind=self.kind,
                                               fil=self.fil,
                                               preproc=self.preproc)

        return homology.values().map(lambda x:
                                     __features(x)[self.k - 1])


class WasserAmpl(Feature):
    """Wasserstein amplitude of a persistence diagram.

    Attributes
    ----------
    p : int or float - p-parameter of the Lp-norm
    n : int - Grading of homology groups to compute.
    dim : int - Dimension of the point cloud.
    step : int or str - Step of the sliding window (number or column name).
    eps : float - Epsilon to control short-living cycles
    kind : str - Kind of the filtration.
    fil : str - Type of the filtration.
    preproc : str - Preprocessing method (None, 'fft').
    """

    def __init__(self,
                 p: int,
                 n: int,
                 dim: int = 2,
                 step: Union[int, str] = 1,
                 eps: float = 0.0,
                 kind: str = None,
                 fil: str = None,
                 preproc: str = None):
        self.dim = dim
        self.n = n
        self.p = p
        self.step = step
        self.eps = eps
        self.kind = kind
        self.fil = fil
        self.preproc = preproc

    def compute(self):
        homology = self.creator.create_feature('homology',
                                               n=self.n,
                                               dim=self.dim,
                                               step=self.step,
                                               eps=self.eps,
                                               kind=self.kind,
                                               fil=self.fil,
                                               preproc=self.preproc)

        return homology.values().map(lambda x: wasserstein_amplitude(x, self.p))


class Entropy(Feature):
    """Entropy calculated from the persistence diagram.

    Attributes
    ----------
    n : int - Grading of homology groups to compute.
    dim : int - Dimension of the point cloud.
    step : int or str - Step of the sliding window (number or column name).
    eps : float - Epsilon to control short-living cycles
    kind : str - Kind of the filtration.
    fil : str - Type of the filtration.
    input : str - Input of the entropy function (lifetimes, births, deaths).
    preproc : str - Preprocessing method (None, 'fft').
    input : str - Type of input to calculate entropy (lifetimes, births, deaths).
      'lifetimes' - calculates persistent entropy of lifetimes
      'births' - calculates entropy of births (cf. new_entropy)
      'deaths' - calculates entropy of deaths (cf. new_entropy)
    """

    def __init__(self, n, dim=2, step=1, eps=0, kind=None, fil=None,
                 preproc=None, input='lifetimes'):
        self.dim = dim
        self.n = n
        self.step = step
        self.eps = eps
        self.kind = kind
        self.fil = fil
        self.input = input
        self.preproc = preproc

    def compute(self):
        homology = self.creator.create_feature('homology',
                                               n=self.n,
                                               dim=self.dim,
                                               step=self.step,
                                               eps=self.eps,
                                               kind=self.kind,
                                               fil=self.fil,
                                               preproc=self.preproc)

        if self.input == 'lifetimes':
            return homology.values().map(persistent_entropy)
        elif self.input == 'births':
            return homology.values().map(lambda x: new_entropy(x[:, 0]))
        elif self.input == 'deaths':
            return homology.values().map(lambda x: new_entropy(x[:, 1]))
        else:
            raise NotImplemented(f'Input of type {self.input} not implemented.')
