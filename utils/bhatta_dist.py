"""
The function bhatta_dist() calculates the Bhattacharyya distance between two classes on a single feature.
    The distance is positively correlated to the class separation of this feature. Four different methods are
    provided for calculating the Bhattacharyya coefficient.

Created on 4/14/2018
Author: Eric Williamson (ericpaulwill@gmail.com)

https://github.com/EricPWilliamson/bhattacharyya-distance/blob/master/bhatta_dist.py
"""
import numpy as np
from math import sqrt
from scipy.stats import gaussian_kde

def bhatta_dist(X1, X2, method='continuous'):
    return -np.log(bhatta_coef(X1, X2, method=method))

def bhatta_coef(X1, X2, method='continuous'):
    #Calculate the Bhattacharyya distance between X1 and X2. X1 and X2 should be 1D numpy arrays representing the same
    # feature in two separate classes. 

    def get_density(x, cov_factor=0.1):
        #Produces a continuous density function for the data in 'x'. Some benefit may be gained from adjusting the cov_factor.
        density = gaussian_kde(x)
        density.covariance_factor = lambda:cov_factor
        density._compute_covariance()
        return density

    #Combine X1 and X2, we'll use it later:
    cX = np.concatenate((X1,X2))

    if method == 'noiseless':
        ###This method works well when the feature is qualitative (rather than quantitative). Each unique value is
        ### treated as an individual bin.
        uX = np.unique(cX)
        A1 = len(X1) * (max(cX)-min(cX)) / len(uX)
        A2 = len(X2) * (max(cX)-min(cX)) / len(uX)
        bht = 0
        for x in uX:
            p1 = (X1==x).sum() / A1
            p2 = (X2==x).sum() / A2
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(uX)

    elif method == 'hist':
        ###Bin the values into a hardcoded number of bins (This is sensitive to N_BINS)
        N_BINS = 10
        #Bin the values:
        h1 = np.histogram(X1,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]
        h2 = np.histogram(X2,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]
        #Calc coeff from bin densities:
        bht = 0
        for i in range(N_BINS):
            p1 = h1[i]
            p2 = h2[i]
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/N_BINS

    elif method == 'autohist':
        ###Bin the values into bins automatically set by np.histogram:
        #Create bins from the combined sets:
        # bins = np.histogram(cX, bins='fd')[1]
        bins = np.histogram(cX, bins='doane')[1] #Seems to work best
        # bins = np.histogram(cX, bins='auto')[1]

        h1 = np.histogram(X1,bins=bins, density=True)[0]
        h2 = np.histogram(X2,bins=bins, density=True)[0]

        #Calc coeff from bin densities:
        bht = 0
        for i in range(len(h1)):
            p1 = h1[i]
            p2 = h2[i]
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(h1)

    elif method == 'continuous':
        ###Use a continuous density function to calculate the coefficient (This is the most consistent, but also slightly slow):
        N_STEPS = 200
        #Get density functions:
        d1 = get_density(X1)
        d2 = get_density(X2)
        #Calc coeff:
        xs = np.linspace(min(cX),max(cX),N_STEPS)
        bht = 0
        for x in xs:
            p1 = d1(x)
            p2 = d2(x)
            bht += sqrt(p1*p2)*(max(cX)-min(cX))/N_STEPS

    else:
        raise ValueError("The value of the 'method' parameter does not match any known method")

    ###Lastly, convert the coefficient into distance:
    if bht==0:
        return float('Inf')
    else:
        return bht


def bhatta_dist2(x, Y, Y_selection=None, method='continuous'):
    #Same as bhatta_dist, but takes different inputs. Takes a feature 'x' and separates it by class ('Y').
    if Y_selection is None:
        Y_selection = list(set(Y))
    #Make sure Y_selection is just 2 classes:
    if len(Y_selection) != 2:
        raise ValueError("Use parameter Y_selection to select just 2 classes.")
    #Separate x into X1 and X2:
    X1 = np.array(x,dtype=np.float64)[Y==Y_selection[0]]
    X2 = np.array(x,dtype=np.float64)[Y==Y_selection[1]]
    #Plug X1 and X2 into bhatta_dist():
    return bhatta_dist(X1, X2, method=method)