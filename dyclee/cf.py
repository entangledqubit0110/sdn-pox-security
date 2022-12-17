import numpy as np
from forget_methods import Decay

class CF:
    """Characteristic feature"""
    def __init__(self, n, ls: np.ndarray, ss: np.ndarray, tl, ts):
        self.n = n          # int, no of objects in the microcluster

        self.LS = ls        # vector, linear sum of each feature over all objects
        self.SS = ss        # vector, Squared sum of each feature over all objects
        
        self.tl = tl        # float, last object assign time
        self.ts = ts        # float, create time of microcluster
    
    def update (self, sample: np.ndarray, time, decay_function:Decay= None):
        """Update cf elements from sample"""
        assert isinstance(sample, np.ndarray), "Sample must be in numpy array format"
        
        if decay_function is None:
            decay = 1
        else:
            decay = decay_function.decay(time, self.tl)
        
        self.n = self.n*decay + 1
        self.LS = self.LS*decay + sample
        self.SS = self.SS*decay + np.square(sample)
        self.tl = time

