import numpy as np
from astropy.modeling import models

def model(xdat, ydat, *args, num_sec=9, order=4):
    '''
    does the modeling for polynomials
    uses the length of the arguements to decide what order of polynomials this is 
    xdat/ydat are list of lists of positions for each mask location [[xpos1, ypos1, xerr1, yerr1], [xpos2, ypos2, xerr2, yerr2], ...]
    '''


    arg_dict = {1:6, 2:12, 3:20, 4:30, 5:42}

    #check that we put in the right number of arguements
    assert len(args) == num_sec * 4 + arg_dict[order] 

    #first creaate
    xn = []
    yn = []

    #now that we have corrected the individual positions for 4 parameters apply the distortion solution
    tdist = models.Legendre2D(order, order)
    tdistX.parameters = args[4*(len(xdat)-1)+5:arg_dict[order]+4*(len(xdat)-1)+5]
    tdistY.parameters = args[4*(len(xdat)-1)+5+arg_dict[order]:]

    #applies both the distortion map and the 4 parameter tranformation
    for i in range(len(xdat)):
        xn.append(args[4*i]*(np.cos(args[1+4*i])*xdat[i] - np.sin(args[1+4*i])*ydat[i]) + args[2+4*i] + tdistX.evaluate(xdat[i], ydat[i]) )
        yn.append(args[4*i]*(np.sin(args[1+4*i])*xdat[i] + np.cos(args[1+4*i])*ydat[i]) + args[3+4*i] + tdistY.evaluate(xdat[i], ydat[i]) )


    return xn, yn
    


def priors(cube , ndim, nparams, offsets=[(),(),()], order=2):
    '''
    '''
    co = []
    #expar = {2:6, 3:14, 4:24}
    arg_dict = {1:6, 2:12, 3:20, 4:30, 5:42}
    for i in range(len(offsets)):
        co.append(1.0)
        cube[i] = (cube[i] - 0.5) * 0.20 + 1.0
        cube[i+1] = (cube[i+1] - 0.5) * 0.20
        cube[i+2] = (cube[i+2] - 0.5) - offsets[i][0] * 5.0
        cube[i+3] = (cube[i+3] - 0.5) - offsets[i][1] * 5.0
        
    _num_co = i
       
   
    for i in range(arg_dict[order]):
        cube[_num_co + i] = (cube[_num_co + i] - 0.5) * 10**-4
   
    return

def Loglike(cube, ndim, nparams):
    '''
    '''

    xmod, ymod = model(
    



