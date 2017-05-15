from numpy import *
from math import sqrt

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B, center=True):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    if center:
        centroid_A = mean(A, axis=0)
        centroid_B = mean(B, axis=0)
    else:
        centroid_A = mean(A, axis=0)
        centroid_A = centroid_A - centroid_A
        centroid_B =  mean(B, axis=0) -  mean(B, axis=0)
    
    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    #print t

    return R, t


