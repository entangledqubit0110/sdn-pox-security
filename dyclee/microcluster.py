import time
import numpy as np

from cf import CF

class MicroCluster:
    """
    Microcluster as defined in DyClee Distance-based clsutering
    Containsn the characteristic feature vector
    """
    def __init__(self, first_sample: np.ndarray, hyperboxSizePerFeature: np.ndarray, decay_function= None,label= -1):
        self.initCF(first_sample)
        self.hyperboxSizePerFeature = hyperboxSizePerFeature
        self.label = label
        self.decay_fn = decay_function
    

    def getDensity (self):
        """Return density of microclsuter"""
        hypervolume = np.prod(self.hyperboxSizePerFeature)
        return self.cf.n/hypervolume
    
    def getCenter (self):
        """Return center of microcluster"""
        return self.cf.LS/self.cf.n

    
    def initCF (self, sample: np.ndarray):
        """Add first sample of the microcluster"""
        ls = sample
        ss = np.square(sample)
        now = time.time()
        self.cf = CF(n= 1, ls= ls, ss= ss, tl= now, ts= now)
    
    def isReachable (self, sample: np.ndarray):
        """Check if this microcluster is reachable from sample"""
        assert isinstance(sample, np.ndarray), "Sample must be in numpy array format"
        diff = np.absolute(sample - self.getCenter())   # feature wise difference of center and given sample
        max_idx = np.argmax(diff)                       # feature with max difference
        if diff[max_idx] < (self.hyperboxSizePerFeature[max_idx]/2):
            return True
        else:
            return False
    
    def insertSample (self, sample: np.ndarray):
        """Insert a new sample to the microcluster"""
        assert isinstance(sample, np.ndarray), "Sample must be in numpy array format"
        self.cf.update(sample, time.time(), self.decay_fn)


    def getManhattanDistance (self, sample: np.ndarray):
        """Manhattan distance of microcluster from given sample"""
        assert isinstance(sample, np.ndarray), "Sample must be in numpy array format"
        diff = np.absolute(sample - self.getCenter())
        return np.sum(diff)

    def isDirectlyConnected (self, mcluster, maxNonOverlaps: int):
        """
        Check if directly connected to mcluster
        conditioned over overlaps on all but at most maxNonOverlaps dimensions
        """
        diff = np.absolute(mcluster.getCenter() - self.getCenter())
        nonOverlaps = 0
        for idx, d in enumerate(diff):
            if d >= self.hyperboxSizePerFeature[idx]:   # feature is not overlapping
                nonOverlaps += 1
            
            if nonOverlaps >= maxNonOverlaps:           # patience crossed
                return False
        
        return True

    def unsetLabel (self):
        """Make the label None"""
        self.label = None

 