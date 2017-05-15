
import numpy as np
from numpy.linalg import svd
from scipy.optimize import leastsq



def fit_point(cyl_tab, plane_tab):
    '''
    '''

    points_cyl = reformat_tab(cyl_tab)
    points_plane = reformat_tab(plane_tab)
    p0, n = planeFit(points_plane)
    import pdb;pdb.set_trace()
    cyl, outpoint_n = fit_cyl(points_plane, n, p0)
    zout = plane_eval(p0, n, cyl[0][1], cyl[0][2])
    return outpoint_n
    

def plane_eval(p0, n, x, y):
    d = np.sum(-1.0 * p0 * n )
    z = (n[0] * x + n[1] * y + d) / (n[2] * -1.0)
    return z 
def reformat_tab(tabin):

    points = np.zeros((3 , len(tabin)))
    
    for i in range(len(tabin)):
        points[0,i] = tabin['col3'][i]
        points[1,i] = tabin['col4'][i]
        points[2,i] = tabin['col5'][i]
    return points

def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    import numpy as np
    from numpy.linalg import svd
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:,-1]




def fit_cyl(points, n, p0):
    '''
    takes normal vector N, and rotates the points frame such that the z axis aligns with n
    '''

    #define rotation matrix to tranform cylinder coordinates into frame where z directin is aligned to n
    cosTx = n[2] / (n[1]**2+n[2]**2)**0.5
    sinTx = n[1] / (n[1]**2+n[2]**2)**0.5
    Rx = np.matrix([[1,0,0],[0,cosTx, sinTx], [0, -sinTx, cosTx]])

    cosTy = n[2] / (n[0]**2+n[2]**2)**0.5
    sinTy = n[0] / (n[0]**2+n[2]**2)**0.5
    Ry = np.matrix([[cosTy,0,-sinTy],[0,1,0],[sinTy, 0, cosTy]])
    
    R = Rx * Ry
    p0n = p0 * R

    points_n = points.T * R
    points_n = points_n.T

    
    cyl_param = leastsq(errfunc_cyl, [8, 0, 0], args=(np.array(points_n.T)))
    outpoint_n = np.array([cyl_param[0][1], cyl_param[0][2], np.array(p0n)[0][2]])
    outpoint =  outpoint_n* np.linalg.inv(R)
    import pdb;pdb.set_trace()
    

    #now just find the point on the plane at x0, y0 (which are cyl_param[1], cyl_param[2])
    return cyl_param, outpoint_n

def fit_cicle(x, y, z):
    #first define guesses
    params = np.zeros(6)
    params[0] = np.std(x)
    params[1] = np.mean(x)
    params[2] = np.mean(y)
    params[3] = np.mean(z)
    

    res = np.minimize(circ_err, params, args=(x,y,z))

def circ_err(params, x, y, z):

    
    return 

def errfunc_cyl(p0, points):
    points = points.T
    err = p0[0]**2 - (np.array(points[0,:]) - p0[1])**2 - (np.array(points[1,:]) - p0[2])**2
    return err
