import numpy as np
from sklearn.metrics import *
from sklearn.metrics.pairwise import *
from sklearn.preprocessing import normalize

eps = np.finfo(np.float64).eps

#Projecting onto probability simplex
def proj(Y):
    n, m = Y.shape
    X = np.sort(Y, 1)[:, ::-1]
    Xtmp = (np.cumsum(X, 1) - 1) * (1 / (np.arange(m) + 1))
    return np.maximum(Y - np.reshape(Xtmp[np.arange(n), np.sum(X > Xtmp, 1) - 1], (-1, 1)), 0)


def KL_div(Y, Y_hat):
    Y = np.clip(Y, eps, 1)
    Y_hat = np.clip(Y_hat, eps, 1)   
    kl = np.sum(Y * (np.log(Y) - np.log(Y_hat)), 1)
    
    return kl.mean()


def Cheby(Y, Y_hat):
    diff_abs = np.abs(Y - Y_hat)
    cheby = np.max(diff_abs, 1)
    return cheby.mean()


def Clark(Y, Y_hat):
    Y = np.clip(Y, eps, 1)
    Y_hat = np.clip(Y_hat, eps, 1)
    sum_2 = np.power(Y + Y_hat, 2)
    diff_2 = np.power(Y - Y_hat, 2)
    clark = np.sqrt(np.sum(diff_2 / sum_2, 1))
    
    return clark.mean()
    
def Canberra(Y, Y_hat):
    Y = np.clip(Y, eps, 1)
    Y_hat = np.clip(Y_hat, eps, 1)
    
    sum_2 = Y + Y_hat
    diff_abs = np.abs(Y - Y_hat)
    can = np.sum(diff_abs / sum_2, 1)
    
    return can.mean()

def Cosine(Y, Y_hat):
    return 1 - paired_cosine_distances(Y, Y_hat).mean()


def Intersection(Y, Y_hat):
    l1 = np.sum(np.abs(Y - Y_hat), 1)
    return 1 - 0.5 * l1.mean()

def Fidelity(Y, Y_hat):
    sim = np.sqrt(Y * Y_hat)
    fid = np.sum(sim, 1)
    
    return fid.mean()

def Euclidean(Y, Y_hat):
    ecu = paired_euclidean_distances(Y, Y_hat)
    return ecu.mean()
    

def score(Y, Y_hat):

    cheby = Cheby(Y, Y_hat)
    clark = Clark(Y, Y_hat)
    can = Canberra(Y, Y_hat)
    kl = KL_div(Y, Y_hat)
    inter = Intersection(Y, Y_hat)
    
    
    #return (cheby, clark, can, kl, inter)
    cosine = Cosine(Y, Y_hat)
    return (cheby, clark, can, kl, cosine, inter)
    