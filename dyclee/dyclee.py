import numpy as np
from typing import Iterable

from microcluster import MicroCluster

class DyClee:
    """Implementation of DyClee Algorithm"""
    def __init__(self, relativeSize: float, dataContext, forget_method= None):
        self.relativeSize = relativeSize
        self.dataContext = dataContext
        self.hyperboxSizePerFeature = self.getHyperBoxSizePerFeaure()
        self.forget_method = forget_method

        # microcluster details
        self.numMicroClusters = 0
        self.AList = list()
        self.OList = list()

        self.meanDensity = 0
        self.medianDensity = 0
        

    def getHyperBoxSizePerFeature (self):
        """Return the hyperbox sizes from the data context"""
        hb = list()
        for context in self.dataContext:
            hb.append(context.maximum - context.minimum)
        return np.array(hb)


    ## Phase 1
    def distanceBasedClustering (self, sample: np.ndarray, label= -1):
        """first phase of dyclee"""

        if self.numMicroClusters == 0:  # first sample
            # create a new microcluster
            muC = MicroCluster(sample, self.hyperboxSizePerFeature, self.forget_method, label)
            self.numMicroClusters += 1
            self.OList.append(muC)      # append to O-list

        else:
            reachables = self.getReachableMicroClusters(self.AList, sample) # search in A-list
            if len(reachables) > 0:
                closest = self.getClosest(reachables, sample)
                closest.insertSample(sample)
            else:   # if not found
                reachables = self.getReachableMicroClusters(self.OList, sample)     # search in O-list
                if len(reachables) > 0:
                    closest = self.getClosest(reachables, sample)
                    closest.insertSample(sample)
                else:   # if not found
                    muC = MicroCluster(sample, self.hyperboxSizePerFeature, self.forget_method, label)  # create a new microcluster
                    self.numMicroClusters += 1
                    self.OList.append(muC)      # append to O-list


    
    def getReachableMicroClusters (self, muCList: Iterable[MicroCluster], sample:np.ndarray):
        """returns a list of microclsuter from muCList that are reachable from sample"""
        rc = list()
        for muC in muCList:
            if muC.isReachable(sample):
                rc.append(muC)
        return rc

    def getClosest (self, muCList: Iterable[MicroCluster], sample: np.ndarray):
        """given a list of microclusters, find the closest from a sample (in terms of manhattan distance)"""
        minDist = None
        closest = None
        for muC in muCList:
            dist = muC.getManhattanDistance(sample)
            if minDist is None or dist < minDist:
                minDist = dist
                closest = muC
        
        return closest

    ## Phase 2

    def densityBasedClustering (self):
        """Second phase of dyclee"""
        # get density thresholds
        self.meanDensity = self.getMeanDensity()
        self.medianDensity = self.getMedianDensity(0)

        # update A and O lists
        self.updateLists()

        # group all microclusters


    def getMeanDensity (self, muCList: Iterable[MicroCluster]):
        """Return mean of densities of microclusters"""
        return np.mean([muC.getDensity() for muC in muCList])
    
    def getMedianDensity (self, muCList:Iterable[MicroCluster]):
        """Return median of densities of microclusters"""
        return np.median([muC.getDensity() for muC in muCList])

    def isDense (self, muC: MicroCluster):
        d = muC.getDensity()
        return ((d >= self.meanDensity) and (d >= self.medianDensity))
    
    def isSemiDense (self, muC: MicroCluster):
        d = muC.getDensity()
        return ((d >= self.meanDensity) or (d >= self.medianDensity))
    
    def isOutlier (self, muC: MicroCluster):
        d = muC.getDensity()
        return ((d < self.meanDensity) and (d < self.medianDensity))

    def updateLists (self):
        """Update the A-list and O-list"""
        muCList = self.AList + self.OList
        newAList = list()
        newOList = list()
        for muC in muCList:
            if self.isOutlier(muC):
                newOList.append(muC)
            else:
                newAList.append(muC)
        self.AList = newAList
        self.OList = newOList
    
    def getDenseMicroClusters (self):
        """Return all dense microclusters"""

    
    

