from imaka_pinhole import fit_all
import shutil, os, pickle
from astropy.table import Table
from flystar.flystar import transforms
import scipy
import numpy as np
import matplotlib.pyplot as plt 

def test(err=0, mknewcat=True, pattern_error=100):
    '''
    Test function for disotriotn fitting routine that includes altering the refernece positions 
    The fit_all.simul_wref() takes a list of xpositoins, ypositions and offsets, compares them to a perfect square reference and then fits for both the distortion and the pattern deviation

    err is the size of measurment error in pixels
    pattern error is the size of the (random) deviations applied ot each pinhole

    To simulate:
      -First create a perfect square reference 
      -For each mask position:
          -Apply a pattern deviation (function of reference pixel location as a single second order polynomial)  TEST:redo using arbitrary deviations 
          -Apply a translation offset (~1000 pixels)
          -Apply a distortion map (function of translated pixels)
          -Add noise if desired
      Note:Deviations (distortion and pattern) must be smaller than 30 pixels in total, as that is the matching radius in the code.  To match measured data the distortion should be ~0.3 pixels and the pattern deviation should be ~ 0.016 pixels.

    Do the tests in terms of the coeffiicients not just the residuals.
    Walk up distoriton solutions from fits order in separate functions
    '''
    wd = os.getcwd()
    if not 'tmp' in wd:
        os.mkdir('tmp')
        os.chdir('./tmp')
    if mknewcat:
        fit_all.mkrefcat(ang=0)
    ref = Table.read('reference.txt', format='ascii.basic')

    #need to declare a pattern deviation
    t = transforms.PolyTransform(np.ones(10), np.ones(10), np.ones(10), np.ones(10), 2)
    #change this to directly call the astropy.models directly 
    #now we can declare the polynomial coefficients
    #translation = 0
    t.px.c0_0 = 0
    #x scale
    t.px.c1_0 = 0 #-0.00002
    #linear cross term
    t.px.c0_1 = 0 #0.00004
    #quadratic terms
    #t.px.c2_0 = 2 * 10**-8 
    #t.px.c0_2 = -4 * 10**-8
    #t.px.c1_1 =  5 * 10**-9
    t.px.c2_0 =  2 * 10**-9 
    t.px.c0_2 =  -4 * 10**-8
    t.px.c1_1 =   5 * 10**-9

    t.py.c0_0 =  0
    t.py.c1_0 =  0 # 0.00009
    t.py.c0_1 =  0 #-0.00005

    t.py.c2_0 =  7 * 10**-8
    t.py.c1_1 =  6  * 10**-8
    t.py.c0_2 =  -5 * 10**-8

    #now apply this as a function  of 

   
    dx, dy = t.evaluate(ref['x'], ref['y'])

#    xpin = ref['x'] + dx
#    ypin = ref['y'] + dy

    #instead just apply gaussian offsets at the 100 nm level
    xpin = ref['x'] + 100.0/6000.0 * scipy.stats.norm(loc=0, scale=pattern_error)
    

    #now we need to apply translation offsets and create the "average stacked catalogs"
    #use the offsets from reduction s6 
    #offsets = [[1083.,394.],[166.8,390.2],[815 - 40.6 , 5689.4 - 7055]]#, [903.7, -1161], [916, 295],[901.4, -1207.0]]#,[1083.,394.],[166.8,390.2],[815 - 40.6 , 5689.4 - 7055], [903.7, -1161], [916, 295],[901.4, -1207.0]
    #these are the offsets from s8, used in paper
    offsets = [[753, 333],[356,336],[7388-7096,5705-7015],[6203-7096,5707-7015],[6173-7055,300--40.6],[1192-40.6,5758-7055.0], [ 1500, 330], [1500, -1300], [0,0]]
    rot_ang = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    xlis = []
    ylis = []

    #add offsets the "measured" positions
    _norm = scipy.stats.norm(loc=0 , scale=err)
    for _off in offsets:
        xlis.append(xpin  + _off[0])
        ylis.append(ypin  + _off[1])

    #now we create and apply the distortion to the measured coordinates
    #this is a 4th order Legendre Polynomial -- measured for single stack s6/pos_1/
    #t = pickle.load(open('/Users/service/code/python/test_dist_2nd.txt', 'r'))
    td = transforms.PolyTransform(np.ones(10), np.ones(10), np.ones(10), np.ones(10), 3)
    #here we use coeffiecients fit to distortion free model per s8, with order=3
    td.px.c0_0 = 0.0
    td.px.c1_0 = 0.0
    td.px.c0_1 = 0.0

    td.px.c2_0 = 0 # 3*10**-8
    td.px.c1_1 = 0 #5*10**-8
    td.px.c0_2 = 0 #6*10**-8

    td.px.c3_0 = -2*10**-13
    td.px.c2_1 = 4*10**-13
    td.px.c1_2 = -1.3*10**-11
    td.px.c0_3 = 7.7*10**-13
    
    td.py.c0_0 = 0.0
    td.py.c1_0 = 0.0
    td.py.c0_1 = 0.0

    td.py.c2_0 =  0 #3.6*10**-8
    td.py.c1_1 =  0 #1.5*10**-7
    td.py.c0_2 = 0 #-1.8*10**-9

    td.py.c3_0 = -2.1 * 10**-13
    td.py.c2_1 = -1.2 * 10**-11
    td.py.c1_2 =  1.6 * 10**-12
    td.py.c0_3 = -3.5 * 10**-12

    #this distortion is only defined over a small region of detector pixel space, we keep this data and discard the rest
    
    xmax = 8000
    xmin = 0

    ymax = 6000
    ymin = 0

    xln = []
    yln = []
    for i in range(len(xlis)):
        cbool = (xlis[i] < xmax) * (xlis[i] > xmin) * (ylis[i] < ymax) * (ylis[i] > ymin)
        dxd, dyd = td.evaluate(xlis[i][cbool], ylis[i][cbool])
        #dx = dx / np.std(dx) * .5
        #dy = dy / np.std(dy) * .5
        #dxd = np.zeros(len(dyd))
        #dyd = np.zeros(len(dxd))
        xln.append(xlis[i][cbool]+dxd+ _norm.rvs(np.sum(cbool)))
        yln.append(ylis[i][cbool]+dyd+ _norm.rvs(np.sum(cbool)))

    #now we are ready to run the fitting routine
    #we iterate to allow the reference positions to converge 
    fit_all.simul_ave_fit(xln,yln,offsets)
    res = fit_all.simul_wref_simple(xln, yln, offsets, nmin=1,  order=3, rot_ang=rot_ang, trim_cam=True,trim_pin=False, renorm=False,sig_clip=False, debug=True, dev_pat=True)

    #if the fitting procedure has updated reference.txt with fixed coordiantes -> these should match xpin and ypin
    refn = Table.read('reference.txt', format='ascii.basic')

    #now we compare coefficients to the fit

    co = Table.read('fit.txt', format='ascii.basic')
    #have to compate like to like coefficients

    
    in_co = []
    #first add in the linear terms .. for now input 0/1 plus offset
    for i in range(len(offsets)):
        in_co.append(offsets[i][0])
        in_co.append(-1.0)
        in_co.append(0.0)
        in_co.append(offsets[i][1])
        in_co.append(0.0)
        in_co.append(-1.0)

    #now the distortion terms
    #first the x terms
    in_co.append(td.px.c2_0.value)
    in_co.append(td.px.c1_1.value)
    in_co.append(td.px.c0_2.value)
    
    in_co.append(td.px.c3_0.value)
    in_co.append(td.px.c2_1.value)
    in_co.append(td.px.c1_2.value)
    in_co.append(td.px.c0_3.value)

    #then the y terms
    in_co.append(td.py.c2_0.value)
    in_co.append(td.py.c1_1.value)
    in_co.append(td.py.c0_2.value)
    
    in_co.append(td.py.c3_0.value)
    in_co.append(td.py.c2_1.value)
    in_co.append(td.py.c1_2.value)
    in_co.append(td.py.c0_3.value)
    
    #now add pattern deviation terms
    in_co.append(-1.0*t.px.c2_0.value)
    in_co.append(-1.0*t.px.c1_1.value)
    in_co.append(-1.0*t.px.c0_2.value)
    #leave space for cubic terms
    in_co.append(0.0)
    in_co.append(0.0)
    in_co.append(0.0)
    in_co.append(0.0)

    #now add pattern deviation terms
    in_co.append(-1.0*t.py.c2_0.value)
    in_co.append(-1.0*t.py.c1_1.value)
    in_co.append(-1.0*t.py.c0_2.value)
    #leave space for cubic terms
    in_co.append(0.0)
    in_co.append(0.0)
    in_co.append(0.0)
    in_co.append(0.0)

    in_co = np.array(in_co) * -1.0

    plt.figure(9)
    plt.clf()
    
    plt.semilogy(range(len(in_co)), (in_co - co['col0'])/in_co, 'o')
    plt.title('Difference in computed terms')
    plt.xlabel('coefficient number')
    plt.ylabel('Coefficient value')
    plt.show()

    for i in range(len(in_co)):
        print('assert that we recover all parameters within 1%')
        assert np.abs((in_co[i] - co['col0'][i])/in_co[i]) < 0.01



