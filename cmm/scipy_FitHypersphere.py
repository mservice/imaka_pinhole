"""
        FitHypersphere.py
        
        fit_hypersphere(collection of tuples or lists of real numbers)
        will return a hypersphere of the same dimension as the tuples:
                (radius, (center))

        using the Hyper (hyperaccurate) algorithm of 
        Ali Al-Sharadqah and Nikolai Chernov
        Error analysis for circle fitting algorithms
        Electronic Journal of Statistics
        Vol. 3 (2009) 886-911
        DOI: 10.1214/09-EJS419

        generalized to n dimensions

        Sun Apr 22 16:44:58 PDT 2012 Kevin Karplus

        Note: the version using scipy's eigh() code works with Taubin
        and Kasa algorithms as well as Hyper and Pratt, but still
        fails for too-low noise (a singular M matrix).
        
        Creative Commons Attribution-ShareAlike 3.0 Unported License.
        http://creativecommons.org/licenses/by-sa/3.0/
"""             

import numpy as np
# from numpy import linalg 
from scipy import linalg
from sys import stderr
from math import sqrt

def fit_hypersphere(data, method="Hyper"):
    """returns a hypersphere of the same dimension as the 
        collection of input tuples
                (radius, (center))
    
       Methods available for fitting are "algebraic" fitting methods
        Hyper   Al-Sharadqah and Chernov's Hyperfit algorithm
        Pratt   Vaughn Pratt's algorithm
    
       The following methods, though very similar, are not implemented yet,
          because the contraint matrix N would be singular, 
          and so the N.I*M computation is not doable.
       
        Taubin  G. Taubin's algorithm
        Kasa    Kasa's algorithm
    """
    num_points = len(data)
#    print >>stderr, "DEBUG: num_points=", num_points
    
    if num_points==0:
        return (0,None)
    if num_points==1:
        return (0,data[0])
    dimen = len(data[0])        # dimensionality of hypersphere
#    print >>stderr, "DEBUG: dimen=", dimen
    
    if num_points<dimen+1:
        raise ValueError(\
            "Error: fit_hypersphere needs at least {} points to fit {}-dimensional sphere, but only given {}".format(dimen+1,dimen,num_points))
    
    # squared magnitude for each tuple, as a column vector
    square_mag = np.matrix( [sum(a*a for a in d) for d in data] ).transpose()
#    print >>stderr, "DEBUG: square_mag=", square_mag
    
    # central n columns of matrix
    center = np.asmatrix(data, dtype=float)
#    print >>stderr, "DEBUG: center=", center
    
    # matrix of data 
    data_M = np.bmat( [[square_mag, center, np.ones((num_points,1))]])
    
#    print >> stderr, "DEBUG: data_M=",data_M
    
    # matrix of 2nd moments:
    M = data_M.transpose() * data_M
    Means = np.mean(data_M, axis=0)
    
#    print >> stderr, "DEBUG: M=",M
#    print >> stderr, "DEBUG: Means=",Means
    
    # construct constraint matrix
    N = np.asmatrix(np.identity(dimen+2, dtype=float))
    N[0,0] = 0
    N[-1,-1]=0
    if method=="Hyper":
        row = 4*Means
        row[0,-1]=2
        N[0,:] = row
        N[:,0] += row.transpose()
    elif method=="Pratt":
        # constraint is sum of squares for coordinates minus
        # 4 * 
        N[0,-1]= -2
        N[-1,0]=-2
    elif method=="Taubin":
        row = 2*Means
        row[0,-1]=0
        N[0,:] = row
        N[:,0] += row.transpose()
    elif method=="Kasa":
        N[0,0]=1
        for i in range(1,dimen+1):
            N[i,i]=0
    else:
        raise ValueError("Error: unknown method: {} should be one of 'Hyper', 'Pratt', or 'Taubin'")
#    print >> stderr, "DEBUG:", method, "N=",N
    
    try:
        eigen_vals,eigen_vects = linalg.eigh(N,M)
        # M is positive definite, so needs to go in the "b" position
        #       but this means that our eigenvalues are inverted
        #       so we want the largest eigenvalue, not the smallest positive one
    except np.linalg.linalg.LinAlgError:
        # probably M singular (noise so low that we have perfect hypersphere)
        #       How do I get an eigenvector corresponding to a zero eigenvalue?
        raise ValueError("OOPS: eigh error in {}, haven't figured out what the workaround is yet".format(method))
    
    # I'm assuming that the eigenvalues are all real, as they should be for a Hermitian matrix
    
#    print >> stderr, "DEBUG: eigen_vals=", eigen_vals
#    print >> stderr, "DEBUG: eigen_vects=", eigen_vects
#    print >> stderr, "DEBUG: list(eigen_vals)=", list(eigen_vals)
  

    param_vect =eigen_vects[:,list(eigen_vals).index(max(eigen_vals))].transpose()
#    print >> stderr, "DEBUG: param_vect=", param_vect
#    params = np.asarray(param_vect)[0]         # numpy gets 1xn matrix for param_vect, scipy gets array of n
    params = np.asarray(param_vect)
#    print >> stderr, "DEBUG: params=", params
    radius = 0.5* sqrt( sum(a*a for a in params[1:-1])- 4*params[0]*params[-1])/abs(params[0])
    center = tuple(-0.5*params[1:-1]/params[0])
    return (radius,center)
    
