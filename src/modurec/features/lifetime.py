from .feature import Feature
from typing import Union


class Lifetime(Feature):
    """Lifetime of a homology cycles.

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

        return homology.values().np.diff(axis=1)
