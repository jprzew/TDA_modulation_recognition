from .feature import Feature
from ripser import Rips, ripser
import numpy as np
from scipy import sparse
from typing import List, Union


def trim_diagrams(diagrams: List[np.array], eps: float) -> List[np.array]:
    """Trim diagrams by removing short-living cycles (less than eps)."""
    trimmed_diagrams = []
    for diagram in diagrams:
        lifetimes = diagram[:, 1] - diagram[:, 0]
        trimmed_diagrams.append(diagram[lifetimes > eps])

    return trimmed_diagrams


class Diagram(Feature):
    """Persistence diagram of a point cloud.

    Attributes
    ----------
    dim : int - Dimension of the point cloud.
    step : int or str - Step of the sliding window (number or column name).
    eps : float - Epsilon to control short-living cycles
    kind : str - Kind of the filtration.
    fil : str - Type of the filtration.
    preproc : str - Preprocessing method (None, 'fft').
    """

    def __init__(self, dim: int = 2,
                 step: Union[int, str] = 1,
                 eps: float = 0.0,
                 kind: str = None,
                 fil: str = None,
                 preproc: str = None):
        self.dim = dim
        self.step = step
        self.eps = eps
        self.kind = kind
        self.fil = fil
        self.preproc = preproc

    @staticmethod
    def star_1d_diagram(time_series: np.ndarray):
        """Compute persistence diagram using star filtration."""

        # Add edges between adjacent points in the time series, with the "distance"
        # along the edge equal to the max value of the points it connects
        N = time_series.shape[0]
        I = np.arange(N - 1)
        J = np.arange(1, N)
        W = np.maximum(time_series[0:-1], time_series[1::])

        # Add vertex birth times along the diagonal of the distance matrix
        I = np.concatenate((I, np.arange(N)))
        J = np.concatenate((J, np.arange(N)))
        W = np.concatenate((W, time_series))

        # Create the sparse distance matrix
        D = sparse.coo_matrix((W, (I, J)), shape=(N, N)).tocsr()
        return ripser(D, maxdim=0, distance_matrix=True)['dgms']

    def compute(self, **kwargs):

        if self.fil == 'star':

            if self.kind not in {'abs', 'phi'}:
                raise NotImplementedError('Options fil=star '
                                          'is implemented only with '
                                          'kind in (abs, phi)')

            point_cloud = self.creator.create_feature('point_cloud', dim=1,
                                                      step=self.step,
                                                      kind=self.kind)

            return point_cloud.values().map(self.star_1d_diagram)

        if self.eps == 0:
            point_cloud = self.creator.create_feature('point_cloud',
                                                      dim=self.dim,
                                                      step=self.step,
                                                      kind=self.kind,
                                                      preproc=self.preproc)
            rips = Rips()
            return point_cloud.values().map(rips.fit_transform)
        else:
            full_diagram = self.creator.create_feature('diagram',
                                                       dim=self.dim,
                                                       step=self.step,
                                                       eps=0,
                                                       kind=self.kind,
                                                       preproc=self.preproc)
            return full_diagram.values().map(lambda x:
                                             trim_diagrams(x, self.eps))
