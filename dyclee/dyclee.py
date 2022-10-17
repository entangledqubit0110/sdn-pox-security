import numpy as np
from typing import List
from sklearn.neighbors import KDTree

from microcluster import MicroCluster

class DyClee:
    """Implementation of DyClee Algorithm"""
    def __init__(self, relativeSize: float, dataContext, forget_method= None, maxNonOverlaps= 0):
        self.relativeSize = relativeSize
        self.dataContext = dataContext
        self.hyperboxSizePerFeature = self.getHyperBoxSizePerFeaure()
        self.potential_neighbor_radius = np.max(self.hyperboxSizePerFeature)    # search radius for neighbor in density based phase
        self.forget_method = forget_method
        self.maxNonOverlaps = maxNonOverlaps

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


    ## Phase-1
    def distanceBasedClustering (self, sample: np.ndarray):
        """first phase of dyclee"""
        if self.numMicroClusters == 0:  # first sample
            # create a new microcluster
            muC = MicroCluster(sample, self.hyperboxSizePerFeature, self.forget_method)
            self.numMicroClusters += 1
            # append to O-list
            self.OList.append(muC)
        else:
            # search in A-list
            reachables = self.getReachableMicroClusters(self.AList, sample)
            if len(reachables) > 0:
                closest = self.getClosest(reachables, sample)
                closest.insertSample(sample)
            # if not found
            else:
                # search in O-list
                reachables = self.getReachableMicroClusters(self.OList, sample)
                if len(reachables) > 0:
                    closest = self.getClosest(reachables, sample)
                    closest.insertSample(sample)
                # if not found
                else:
                    # create a new microcluster
                    muC = MicroCluster(sample, self.hyperboxSizePerFeature, self.forget_method)
                    self.numMicroClusters += 1
                    # append to O-list
                    self.OList.append(muC)


    
    def getReachableMicroClusters (self, muCList: List[MicroCluster], sample:np.ndarray):
        """returns a list of microclsuter from muCList that are reachable from sample"""
        rc = list()
        for muC in muCList:
            if muC.isReachable(sample):
                rc.append(muC)
        return rc

    def getClosest (self, muCList: List[MicroCluster], sample: np.ndarray):
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
        concatList = self.AList + self.OList
        self.meanDensity = self.getMeanDensity(concatList)
        self.medianDensity = self.getMedianDensity(concatList)

        # update A and O lists
        self.updateLists()

        # reset labels
        for muC in self.AList:
            muC.unsetLabel()
        for muC in self.OList:
            muC.unsetLabel()

        # get dense clusters
        denseMicroClusters = self.getDenseMicroClusters()
        # the KD Tree for searching
        # based on infinte norm distance
        searchTree = KDTree(np.vstack([c.getCenter() for c in self.AList]), p= np.inf)
        
        # loop variables
        seen = list()   # list of already visited micorclsuters
        cid = 0         # cluster id
        # density based clustering loop
        for muC in denseMicroClusters:
            # not already seen
            if muC not in seen:
                seen.append(muC)
                muC.setLabel(cid)
                neighbors = self.getNeighbors(muC, searchTree, seen)
                # process the neighborhood
                while len(neighbors):
                    _muC = neighbors.pop()
                    # not already seen
                    if _muC not in seen:
                        #_muC can only be dense/semidense
                        # label for both
                        seen.append(_muC)
                        _muC.setLabel(cid)
                        # add neighborhood only if dense
                        if self.isDense(_muC):
                            # add its neighbors also
                            nextNeighbors = self.getNeighbors(_muC, searchTree, seen)
                            neighbors = neighbors.union(nextNeighbors)

            # increment cid
            cid += 1
                


    def getMeanDensity (self, muCList: List[MicroCluster]):
        """Return mean of densities of microclusters"""
        return np.mean([muC.getDensity() for muC in muCList])
    
    def getMedianDensity (self, muCList:List[MicroCluster]):
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

    def getDenseMicroClusters (self):
        """Return dense microclsuters from A-List"""
        dense = list()
        for muC in self.AList:
            if self.isDense(muC):
                dense.append(muC)
        return dense

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
        dense = list()
        for muC in self.AList:
            if self.isDense(muC):
                dense.append(muC)
        return dense

    
    def getNeighbors (self, muC: MicroCluster, searchTree: KDTree, alreadySeen: List[MicroCluster]):
        """Return set of neighbors for a given microcluster"""
        # get neighbor indices
        nghIdc = searchTree.query_radius(muC.getCenter().reshape(1, -1), self.potential_neighbor_radius)[0]
        neighbors = list()
        # add the microclusters from A-List except itself
        for idx in nghIdc:
            if  (    muC != self.AList[idx] and                                     # not itself 
                    muC.isDirectlyConnected(self.AList[idx], self.maxNonOverlaps)   # directly connected
                ):
                neighbors.append(self.AList[idx])
        neighbors = set(neighbors)
        neighbors = neighbors - set(alreadySeen)    # exclude already seen microclusters
        return neighbors
        

    
    
    ## Combined methods
    def step (self, sample: np.ndarray):
        """A training step given a sample"""
