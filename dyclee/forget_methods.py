import math
def linear_decay (t, tl, tw0):
    m = 1/tw0
    if (t - tl) <= tw0:
        return (1 - m(t - tl))

    else:
        return 0

def trapezoidal_decay (t, tl, tw0, ta):
    m = 1/tw0
    if (t - tl) <= ta:
        return 1

    elif (t - tl) > ta and (t - tl) <= tw0:
        return ((m - t)/(m - ta))

    else:
        return 0

def z_shaped_function (t, tl, tw0, ta):
    if (t - tl) <= ta:
        return 1

    elif (  (t - tl) > ta
        and (t - tl) <= (ta + tw0)/2 ):
        return 1 - 2*((t - ta)/(tw0 - ta))

    elif (  (t - tl) >= (ta + tw0)/2
        and (t - tl) < tw0):
        return 2*((t - ta)/(tw0 - ta))

    else:
        return 0

def exponential_decay (t, tl, decay):
    return math.exp(-decay * (t - tl))

def half_life_decay (t, tl, decay, beta):
    return math.pow(beta, -decay * (t - tl))

def sigmoidal_decay (t, tl, a, c):
    return 1/(1 + math.exp(-a * (t - c)))