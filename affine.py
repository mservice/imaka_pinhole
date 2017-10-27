from matplotlib import transforms
from scipy import optimize
import numpy as np
import math

def fit_rot_trans(points_in, points_out):
    """
    Find the optimal rotation, scale, and translation given a set of points
    (and optional error bars). 

    The resulting transformation is:

    points_out = S * (R * points_in) + trans
    
    where R is the rotation matrix and is applied first, S is the scale matrix
    and is applied second, and trans is the translation vector applied last. The
    appropriate code to transform points using the output solution would be:

    t = matplotlib.transforms.Affine2D()
    t.rotate(theta)
    t.scale(scale)
    t.transform(tx, ty)

    point_out = t.transform_point(point_in)

    The format of the input should be:
    points_in -- numpy array of shape N x 2, where N is the number of points
    points_out -- numpy array of shape N x 2
    err_points_in -- numpy array of shape N x 2 (optional)
    err_points_out -- numpy array of shape N x 2 (optional)

    Modeified from original to only compute the rotation and translation, assumes scale is 1.0
    """
    def fit_func(params, points_in, points_out):
        theta = params[0]
        #scale = params[1]
        tx = params[1]
        ty = params[2]

        # Transform the input data points
        t = transforms.Affine2D()
        t.rotate_deg(theta)
        #t.scale(scale)
        t.translate(tx, ty)

        points_test = t.transform(points_in)

       
       
        # Compute the deltas (squared)
        diffXY = points_out - points_test

      
        # Turn XY deltas into R deltas
        diffR = np.hypot(diffXY[:,0], diffXY[:,1])
        #errR = np.hypot(diffXY[:,0] * errXY[:,0], diffXY[:,1] * errXY[:,1]) / diffR
        #diffR /= errR

        return diffR

    params0 = np.array([0., 0., 0.])
    data = (points_in, points_out)

    out = optimize.leastsq(fit_func, params0, args=data, full_output=1)

    pfinal = out[0]
    #covar = out[1]

    params = {'angle': pfinal[0],
              'transX': pfinal[1],
              'transY': pfinal[2]}

    

    return params
