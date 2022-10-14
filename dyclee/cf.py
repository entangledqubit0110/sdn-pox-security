import numpy as np

class CF:
    """Characteristic feature"""
    def __init__(self, n, ls, ss, tl, ts):
        self.n = n          # int, no of objects in the microcluster

        assert isinstance(ls, np.ndarray), "LS must be of type numpy array"
        assert isinstance(ss, np.ndarray), "SS must be of type numpy array"
        self.LS = ls        # vector, linear sum of each feature over all objects
        self.SS = ss        # vector, Squared sum of each feature over all objects
        
        self.tl = tl        # float, last object assign time
        self.ts = ts        # float, create time of microcluster
    
    def update (self, sample, time):
        """Update cf elements from sample"""
        assert isinstance(sample, np.ndarray), "Sample must be in numpy array format"
        self.n += 1
        self.LS = self.LS + sample
        self.SS = self.SS + np.square(sample)
        self.tl = time

