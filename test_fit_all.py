from imaka_pinhole import fit_all
import shutil, os, pickle
from astropy.table import Table
from flystar_old.flystar_old import transforms
import scipy
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize, leastsq

def test(err=0, mknewcat=True, pattern_error=100, dev_pat=True, distort=True, debug_plots=False, order_pattern_error=False, trim_cam=False, fix_trans=False):
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
    t = transforms.PolyTransform(np.ones(10), np.ones(10), np.ones(10), np.ones(10), 3)
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
    t.px.c0_2 = 0 #-4 * 10**-9
    t.px.c1_1 =  0# 5 * 10**-10

    t.px.c3_0 = 10**-12
    t.px.c2_1 = 0
    t.px.c1_2 = 0
    t.px.c0_3 = 0

    t.py.c0_0 =  0
    t.py.c1_0 =  0 # 0.00009
    t.py.c0_1 =  0 #-0.00005

    t.py.c2_0 = 0 # 7 * 10**-10
    t.py.c1_1 =  0 #6  * 10**-9
    t.py.c0_2 =  0 #2* * 10**-9

    t.py.c3_0 = 0 #10**-12
    t.py.c2_1 = 0
    t.py.c1_2 = 0
    t.py.c0_3 = 0

    #now apply this as a function  of 

   
    if order_pattern_error:
        dx, dy = t.evaluate(ref['x']-np.median(ref['x']), ref['y']-np.median(ref['y']))
    else:
        dx = np.zeros(len(ref['x']))
        dy = np.zeros(len(ref['y']))
   # import pdb;pdb.set_trace()
    print(np.std(dx)*6000.0,np.std(dy)*6000.0)
    print(np.min(dx)*6000.0, np.max(dx)*6000.0, np.min(dy)*6000.0, np.max(dy)*6000.0)

    _norm_pattern = scipy.stats.norm(loc=0, scale=pattern_error/6000.0)
    _xerr = _norm_pattern.rvs(len(ref['x']))
    _yerr = _norm_pattern.rvs(len(ref['x']))
    xpin = ref['x'] + _xerr + dx
    ypin = ref['y'] + _yerr + dy
    plt.close('all')
    if debug_plots:
        

        fig, axes = plt.subplots(3, 2, figsize=(8, 10))
        
        im = axes.flatten()[0].scatter(ref['x'], ref['y'], c=dx*6000.0)
        fig.colorbar(im, ax=axes.flatten()[0])
        axes.flatten()[0].set_xlabel('X reference (pix)')
        axes.flatten()[0].set_ylabel('Y reference (pix)')
        axes.flatten()[0].set_title('X Ordered pattern deviation')
        axes.flatten()[0].set_aspect('equal')
        #plt.axes().set_aspect('equal')

        
        #plt.subplot(3, 2, 2)
        im = axes.flatten()[1].scatter(ref['x'], ref['y'], c=dy*6000.0)
        fig.colorbar(im, ax=axes.flatten()[1])
        axes.flatten()[1].set_xlabel('X reference (pix)')
        axes.flatten()[1].set_ylabel('Y reference (pix)')
        axes.flatten()[1].set_title('Y ordered pattern deviation')
        axes.flatten()[1].set_aspect('equal')
        #plt.axes().set_aspect('equal')

        #plt.subplot(3, 2, 3)
        im = axes.flatten()[2].scatter(ref['x'], ref['y'], c=_xerr*6000.0)
        fig.colorbar(im, ax=axes.flatten()[2])
        axes.flatten()[2].set_xlabel('X reference (pix)')
        axes.flatten()[2].set_ylabel('Y reference (pix)')
        axes.flatten()[2].set_title('X random pattern deviation')
        axes.flatten()[2].set_aspect('equal')
        #plt.axes().set_aspect('equal')

        #plt.subplot(3, 2, 4)
        im = axes.flatten()[3].scatter(ref['x'], ref['y'], c=_yerr*6000.0)
        fig.colorbar(im, ax=axes.flatten()[3])
        axes.flatten()[3].set_xlabel('X reference (pix)' )
        axes.flatten()[3].set_ylabel('Y reference (pix)' )
        axes.flatten()[3].set_title('Y random pattern deviation')
        axes.flatten()[3].set_aspect('equal')
        #plt.axes().set_aspect('equal')

        #plt.subplot(3, 2, 5)
        im = axes.flatten()[4].scatter(ref['x'], ref['y'], c=6000.0*(xpin-ref['x']))
        fig.colorbar(im, ax=axes.flatten()[4])
        axes.flatten()[4].set_xlabel('X reference')
        axes.flatten()[4].set_ylabel('Y reference')
        axes.flatten()[4].set_title('X Total Deviation')
        axes.flatten()[4].set_aspect('equal')
        #plt.axes().set_aspect('equal')

        #plt.subplot(3, 2, 6)
        im = axes.flatten()[5].scatter(ref['x'], ref['y'], c=6000*(ypin-ref['y']))
        fig.colorbar(im, ax=axes.flatten()[5])
        axes.flatten()[5].set_xlabel('X reference')
        axes.flatten()[5].set_ylabel('Y reference')
        axes.flatten()[5].set_title('Y Total deviation')
        axes.flatten()[5].set_aspect('equal')
        #plt.axes().set_aspect('equal')

        plt.tight_layout()
        
    #del t, dx, dy
#    xpin = ref['x'] + dx
#    ypin = ref['y'] + dy


    #instead just apply gaussian offsets at the 100 nm level
   
    

    #now we need to apply translation offsets and create the "average stacked catalogs"
    #use the offsets from reduction s6 
    #offsets = [[1083.,394.],[166.8,390.2],[815 - 40.6 , 5689.4 - 7055]]#, [903.7, -1161], [916, 295],[901.4, -1207.0]]#,[1083.,394.],[166.8,390.2],[815 - 40.6 , 5689.4 - 7055], [903.7, -1161], [916, 295],[901.4, -1207.0]
    #these are the offsets from s8, used in paper
    #offsets = [[753, 333],[356,336],[7388-7096,5705-7015],[6203-7096,5707-7015],[6173-7055,300--40.6],[1192-40.6,5758-7055.0], [ 1500, 330], [1500, -1300], [0,0]]


    #offsets = [[447-40.6, 5894-7055], [770-40.6, 5895-7055], [945-40.6, 5888-7055], [944-40.6, 5644-7055], [697-40.6, 5620-7055], [488-40.6, 5619-7055], [486-40.6, 5376-7055], [745-40.6, 5381-7055], [982-40.6, 5371-7055]]
    offsets = []
    rot_ang = []
    for i in range(2):
        for j in range(2):
            offsets.append([i * 100, j * 100])
            rot_ang.append(0)
    for i in range(2):
        for j in range(2):
            offsets.append([i * 100, j * 100])
            rot_ang.append(90)
    #add in data at 90 degree rotation
    xlis = []
    ylis = []

    plt.figure(2)
    plt.clf()
    #add offsets the "measured" positions
    offsets_in = []
    _norm = scipy.stats.norm(loc=0 , scale=err)
    for _iii, _off in enumerate(offsets):
        _ang = -1.0*np.deg2rad(rot_ang[_iii])
        _xtmp = xpin*np.cos(_ang) - ypin*np.sin(_ang)
        _ytmp = ypin*np.cos(_ang) + xpin*np.sin(_ang)
        #now we need to move the measured coordiantes such that the lowest value points (lower left) is at offsets[i][0,1]
            
        _dx = _off[0] - np.min(_xtmp)
        _dy = _off[1] - np.min(_ytmp)
        xlis.append(_xtmp + _dx)
        ylis.append(_ytmp + _dy)

        offsets_in.append((xlis[-1][0],ylis[-1][0]))

        #if _ang != 0:
        #    import pdb;pdb.set_trace()
                
        print(offsets_in[-1])

        if debug_plots:
            plt.figure(2)
            plt.scatter(xlis[-1], ylis[-1], label='catalog '+str(_iii))
            plt.title('Measured Position with no distortion')
            plt.legend(loc='upper right')
            
    #now we create and apply the distortion to the measured coordinates
    #this is a 4th order Legendre Polynomial -- measured for single stack s6/pos_1/
    #t = pickle.load(open('/Users/service/code/python/test_dist_2nd.txt', 'r'))
    td = transforms.PolyTransform(np.ones(10), np.ones(10), np.ones(10), np.ones(10), 3)
    #here we use coeffiecients fit to distortion free model per s8, with order=3
    td.px.c0_0 = 0.0
    td.px.c1_0 = 0.0
    td.px.c0_1 = 0.0

    td.px.c2_0 =  3.9*10**-8
    td.px.c1_1 = 7.8*10**-8
    td.px.c0_2 = 3.2*10**-8

    td.px.c3_0 = 7*10**-12
    td.px.c2_1 = 4.13*10**-12
    td.px.c1_2 = -4.8*10**-12
    td.px.c0_3 = 1.8*10**-12
    
    td.py.c0_0 = 0.0
    td.py.c1_0 = 0.0
    td.py.c0_1 = 0.0

    td.py.c2_0 =  3.09*10**-8
    td.py.c1_1 =  1.4*10**-7
    td.py.c0_2 =  4.23*10**-8

    td.py.c3_0 =  2.3 * 10**-13
    td.py.c2_1 = -8.4 * 10**-12
    td.py.c1_2 =  5.8 * 10**-12
    td.py.c0_3 =  6.09 * 10**-13

    
    #this distortion is only defined over a small region of detector pixel space, we keep this data and discard the rest
    
    xmax = 8000
    xmin = 0

    ymax = 6000
    ymin = 0

    xln = []
    yln = []
    xrn = []
    yrn = []
    
    plt.figure(100)
    plt.clf()
    #import pdb;pdb.set_trace()
    for i in range(len(xlis)):
        #cbool = (xlis[i] < xmax) * (xlis[i] > xmin) * (ylis[i] < ymax) * (ylis[i] > ymin)
        cbool = np.ones(len(xlis[i]), dtype='bool')
        dxd, dyd = td.evaluate(xlis[i][cbool], ylis[i][cbool])
        if not distort:
            dxd = np.zeros(np.sum(cbool))
            dyd = np.zeros(np.sum(cbool))
        #dx = dx / np.std(dx) * .5
        #dy = dy / np.std(dy) * .5
        #dxd = np.zeros(len(dyd))
        #dyd = np.zeros(len(dxd))
        xln.append(xlis[i][cbool]+dxd+ _norm.rvs(np.sum(cbool)))
        yln.append(ylis[i][cbool]+dyd+ _norm.rvs(np.sum(cbool)))

        tl = transforms.PolyTransform(ref['x'], ref['y'], xln[-1], yln[-1], 1)
        __xn, __yn = tl.evaluate(ref['x'], ref['y'])
        xrn.append(__xn)
        yrn.append(__yn)
        if debug_plots:
            plt.figure(1234)
            plt.subplot(2, 1, 1)
            #import pdb;pdb.set_trace()
            plt.scatter(xln[-1], ( xln[-1] - ref['x'][cbool] - offsets[i][0])*6000.0, label='cat '+str(i))
            plt.title("Deviation from Square")
            plt.xlabel('X camera (pix)')
            plt.ylabel("X Difference (nm)")
            plt.vlines([7263, 965], -50, 600, color='red')
            plt.legend(loc='lower right')

            plt.subplot(2, 1, 2)
            plt.scatter(yln[-1], ( yln[-1] - ref['y'][cbool] - offsets[i][1])*6000.0, label='cat '+str(i))
            plt.title("Deviation from Square")
            plt.xlabel('Y camera (pix)')
            plt.ylabel("Y Difference (nm)")
            plt.vlines([490, 5218], -600, 50, color='red')
            plt.legend(loc='lower right')

            plt.tight_layout()
            plt.figure(10+i)
            q = plt.quiver(xln[-1], yln[-1], xln[-1] - ref['x'][cbool] - offsets[i][0], yln[-1] - ref['y'][cbool] - offsets[i][1], scale =1)
            
            plt.quiverkey(q, -50, -50, 0.1, '600 nm', coordinates='data', color='red')
            plt.axes().set_aspect('equal')
            plt.xlabel("X camera (pix)")
            plt.ylabel("Y camera (pix)")
            plt.title('Total Deviation from Square Catalog '+str(i))

            plt.figure(130+i)
            plt.subplot(2, 1,  1)
            plt.scatter(xln[-1], yln[-1], c=(xln[-1] - ref['x'][cbool] - offsets[i][0]))
            plt.title('Total Deviation from Square Catalog '+str(i))
            plt.xlabel('X camera (pix)')
            plt.ylabel('Y camera (pix)')
            plt.colorbar()

            plt.subplot(2,1,  2)
            plt.scatter(xln[-1], yln[-1], c=(yln[-1] - ref['y'][cbool] - offsets[i][1]))
            plt.title('Total Deviation from Square Catalog '+str(i))
            plt.xlabel('X camera (pix)')
            plt.ylabel('Y camera (pix)')
            plt.colorbar()
            

            plt.figure(100)
            colors = ['black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray']
            
            q = plt.quiver(xln[-1], yln[-1], xln[-1] - ref['x'][cbool] - offsets[i][0], yln[-1] - ref['y'][cbool] - offsets[i][1], scale =1, color=colors[i], label='cat '+str(i))
            
            plt.quiverkey(q, -50, -50, 0.1, '600 nm', coordinates='data', color='red')
            plt.axes().set_aspect('equal')
            plt.xlabel("X camera (pix)")
            plt.ylabel("Y camera (pix)")
            plt.title('Total Deviation from Square')
            plt.legend(loc='upper left')

            
            #fig, axes = plt.subplots(1, 2, 1)
            
            #plt.subplot(1, 2, 1)

    #import pdb;pdb.set_trace()
    #now we are ready to run the fitting routine
    #we iterate to allow the reference positions to converge 
    #fit_all.simul_wref(xln,yln,offsets)
    #if debug_plots:
        #return
    fit_all.simul_wref(xln, yln, offsets_in,  order=4, rot_ang=rot_ang, Niter=5, dev_pat=dev_pat, Nmissing=20, sig_clip=False, fourp=False, trim_cam=trim_cam, fix_trans=fix_trans)

    #import pdb;pdb.set_trace()
    #return 
    #if the fitting procedure has updated reference.txt with fixed coordiantes -> these should match xpin and ypin
    refn = Table.read('reference_new.txt', format='ascii.fixed_width')
    cb = (refn['x'] - refn['xorig']) != 0.0
    __dx = (refn['x'] - xpin)
    import pdb;pdb.set_trace()
    print(np.mean(__dx[cb])*6000.0, np.std(__dx[cb])*6000.0, np.std(__dx[cb] - np.mean(__dx[cb]))*6000.0)
    __dy = refn['y'] - ypin
    print(np.mean(__dy[cb])*6000.0, np.std(__dy[cb])*6000.0, np.std(__dy[cb] - np.mean(__dy[cb]))*6000.0)
    #now we compare coefficients to the fit

    import pdb;pdb.set_trace()
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
    #in_co.append(-1.0*t.px.c2_0.value)
    #in_co.append(-1.0*t.px.c1_1.value)
    #in_co.append(-1.0*t.px.c0_2.value)
    #leave space for cubic terms
    #in_co.append(0.0)
    #in_co.append(0.0)
    #in_co.append(0.0)
    #in_co.append(0.0)

    #now add pattern deviation terms
    #in_co.append(-1.0*t.py.c2_0.value)
    #in_co.append(-1.0*t.py.c1_1.value)
    #in_co.append(-1.0*t.py.c0_2.value)
    #leave space for cubic terms
    #in_co.append(0.0)
    #in_co.append(0.0)
    #in_co.append(0.0)
    #in_co.append(0.0)

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





def test1d(iterate=False, offset_max=4000,Maxcount=10, trim_dat=False, random=False, pattern_error=100, fix_scale=False, fix_trans=False):
    '''
fix scale
    '''


    xpin = np.linspace(0, 15000, num=15001)
    if not random:
        xpin_dev = xpin + (xpin-np.median(xpin))**2 * 10**-9
        
    else:
        _norm_pattern = scipy.stats.norm(loc=0, scale=pattern_error/6000.0)
        _xerr = _norm_pattern.rvs(len(xpin))
        xpin_dev = xpin + _xerr
        
    offsets = np.linspace(0, offset_max, num=10)

    xmeas = []
    xref = []
    refbool=[]
    for i in range(len(offsets)):
        _xmeas = xpin_dev +offsets[i]
        if trim_dat:
            _inbool = (_xmeas > offset_max) * (_xmeas < 15000)  
        else:
            _inbool = np.ones(len(_xmeas),dtype='bool')
        xmeas.append(_xmeas[_inbool])
        #_xref = xpin + offsets[i]
        xref.append(xpin[_inbool])
        refbool.append(_inbool)
    xref = np.array(xref)
    xmeas = np.array(xmeas)
    init_guess = []
    for jj in range(len(offsets)):
        init_guess.append(1.0)
        init_guess.append(-1.0*offsets[jj])
    init_guess.append(0.0)
    contin = True
    xref_new = np.zeros((len(xref), len(offsets)))
    xrefn = xref
    count = 0
    while contin:
        res = leastsq(com_mod, init_guess, args=(xmeas, xrefn, fix_scale, fix_trans, init_guess))
        resid = com_mod(res[0],xmeas, xrefn,fix_scale, fix_trans,init_guess, evaluate=True)
        if iterate == False  or count == Maxcount :
            contin = False
        else:
            #compute new refernce coordinates
            contin = True
            dx = np.median(resid, axis=0)
            xrefn = xrefn + dx
            count +=1
            #import pdb;pdb.set_trace()
            print(count)

    if iterate:
        plt.figure(50)
        plt.clf()
        plt.scatter(xref, 6000.0*(xrefn-xref), s=2)
        plt.xlabel('Reference Position')
        plt.ylabel('Measured Model Deviation (nm)')

    print('Did I recover the positions offsets')
    _tally = []
    plt.figure(51)
    plt.clf()
    
    for i in range(len(offsets)):
        _xmeas = xpin_dev +offsets[i]
        if trim_dat:
            _inbool = (_xmeas > offset_max) * (_xmeas < 15000)  
        else:
            _inbool = np.ones(len(_xmeas),dtype='bool')
        plt.scatter(xrefn[i], (xrefn[i] - xpin_dev[_inbool])*6000.0, s=2)
        plt.xlabel("Reference Position (pix)")
        plt.ylabel("Error in measurment of pattern deviation (nm)")
        
        for kk in range(len(xrefn[i])):
            
            _tally.append(xrefn[i][kk] - xpin_dev[_inbool][kk])
    _tally = np.array(_tally)
    print(np.std(_tally)*6000.0)
    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()
    plt.figure(3)
    plt.clf()

    for i in range(len(xref)):
        plt.figure(1)
        plt.subplot(4, 1, 3)
        plt.scatter(xmeas[i], resid[i]*6000.0, s=2, label=str(i))
        #plt.figure(2)
        plt.subplot(4, 1, 1)
        plt.scatter(xmeas[i], (xmeas[i]-xref[i]-offsets[i])*6000.0, s=2, label=str(i))
        plt.subplot(4, 1, 2)
        plt.scatter(xmeas[i], -1.0*(xrefn[i] - xmeas[i] * res[0][2*i] - res[0][2*i+1])*6000.0, s=2, label=str(i))
        plt.subplot(4, 1, 4)
        plt.scatter(xref[i], resid[i]*6000.0, label=str(i))
        

    _out = Table()
    _out['co'] = res[0]
    _out.write('1D_co.txt', format='ascii.basic')
    #plt.figure(1)
    plt.subplot(4, 1, 3)
    plt.title(' Total Model Residual')
    plt.xlabel("Measured Position (pix)")
    plt.ylabel("Residual (nm)")
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 4)
    plt.title(' Total Model Residual')
    plt.xlabel("Refernce Position (pix)")
    plt.ylabel("Residual (nm)")
    plt.legend(loc='upper right')

    
    plt.subplot(4, 1, 1)
    plt.title("Input Pattern Error")
    plt.xlabel("Measured Position (pix)")
    plt.ylabel("Input Deviation (nm)")
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 2)
    plt.title("Linear Model Residuals")
    plt.xlabel("Measured Position (pix)")
    plt.ylabel("Residual (nm)")
    plt.legend(loc='upper right')

    plt.tight_layout()

    plt.figure(5)
    plt.clf()
    scale = []
    trans = []
    
    for i in range((len(res[0])-1)/2):
        scale.append(res[0][2*i])
        trans.append(res[0][2*i+1])
    #import pdb;pdb.set_trace()
    dist = res[0][-1]
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(scale)), np.array(scale)-1.0)
    plt.xlabel('Catalog')
    plt.ylabel("Model scale - Input scale")

    trans = np.array(trans)
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(trans)), (trans+offsets)*6000.0)
    plt.xlabel('Catalog')
    plt.ylabel('Model Translation - Input Translation (nm)')
    
    return resid, xref
    
def com_mod(inco, xmeas, xref,fix_scale, fix_trans,init_guess, mdist=True, evaluate=False):
    '''
    '''

    resid = []
    diff = []
    #import pdb;pdb.set_trace()
    for i in range(len(xmeas)):
        if mdist:
            if fix_scale and not fix_trans:
                _xtrans = init_guess[2*i]*xmeas[i] + inco[2*i+1] + inco[-1]*xmeas[i]**2
            elif fix_scale and fix_trans:
                _xtrans = init_guess[2*i]*xmeas[i] + init_guess[2*i+1] + inco[-1]*xmeas[i]**2
            elif not fix_scale and fix_trans:
                _xtrans = inco[2*i]*xmeas[i] + init_guess[2*i+1] + inco[-1]*xmeas[i]**2
                
            elif not fix_scale and not fix_trans:
                _xtrans = inco[2*i] * xmeas[i] + inco[2*i+1] + inco[-1]*xmeas[i]**2
        else:
            _xtrans = inco[2*i] * xmeas[i] + inco[2*i+1] 
        diff.append(_xtrans - xref[i])
        for ii in range(len(_xtrans)):
            resid.append(_xtrans[ii] - xref[i][ii])
    if not evaluate:
        return np.array(resid)
    else:
        return np.array(diff)

    


def special_fft(x, y, dx):
    '''
    need to normalize x/y before entering to run from 0 to 1 (according to wikipedia math definitions
    '''

    #we will keep the spatial frequencies as integers
    wn = range(len(x) - 1)
    wl = range(len(y) - 1)
        
