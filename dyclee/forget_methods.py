from abc import ABC, abstractmethod
import math

class Decay (ABC):

    @abstractmethod
    def decay(self, t, t1):
        pass
    

class LinearDecay (Decay):
    def __init__(self, tw0):
        self.tw0 = tw0

    def decay (self, t, tl):
        m = 1/self.tw0
        if (t - tl) <= self.tw0:
            return (1 - m(t - tl))

        else:
            return 0


class TrapezoidalDecay (Decay):
    def __init__(self, tw0, ta):
        self.tw0 = tw0
        self.ta = ta

    def decay (self, t, tl):
        m = 1/self.tw0
        if (t - tl) <= self.ta:
            return 1

        elif (t - tl) > self.ta and (t - tl) <= self.tw0:
            return ((m - t)/(m - self.ta))

        else:
            return 0


class ExponentialDecay (Decay):
    def __init__ (self, d):
        self.d = d

    def decay (self, t, tl):
        return math.exp(-self.d * (t - tl))

class HalfLifeDecay (Decay):
    def __init__(self, d, beta):
        self.d = d
        self.beta = beta

    def decay (self, t, tl):
        return math.pow(self.beta, -self.d * (t - tl))

class SigmoidalDecay (Decay):
    def __init__ (self, a, c):
        self.a = a
        self.c = c

    def decay (self, t, tl):
        return 1/(1 + math.exp(-self.a * (t - self.c)))