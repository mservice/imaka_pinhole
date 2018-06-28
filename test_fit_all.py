from imaka_pinhole import fit_all
import shutil, os, pickle
from astropy.table import Table
from flystar_old.flystar_old import transforms
import scipy
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize, leastsq

def test(err=0, mknewcat=True, pattern_error=100, dev_pat=True, distort=True, debug_plots=False, order_pattern_error=False, trim_cam=False, fix_trans=False, fit=True, npos=3, rot_ang_l=[0, 90, 180, 270], step_size=250, Niter=5, plot_in=False, order=4, correct_ref=False, lab_offsets=False, plot_dist=True):
    '''
    Test function for disotriotn fitting routine that includes altering the refernece positions 
    The fit_all.simul_wref() takes a list of xpositoins, ypositions and offsets, compares them to a perfect square reference and then fits for both the distortion and the pattern deviation

    err is the size of measurment error in pixels
    pattern error is the size of the (random) deviations applied ot each pinhole
    correct_ref(bool): If True, the reference coordinates used in the fit are chaged to the correct deviated values.  This validates that the fit works if you give it the right answer

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
    colors = ['black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray']
            

    wd = os.getcwd()
    if not 'tmp' in wd:
        os.mkdir('tmp')
        os.chdir('./tmp')
    if mknewcat:
        #make the reference catalog that the fitting code will use
        fit_all.mkrefcat(ang=0)
    #make a starting reference catalog that we will deviate and use to create simulated measurements.
    #fit_all.mkrefcat(outf='ref_test.txt', ang=0)
        
    ref = Table.read('reference.txt', format='ascii.basic')

    #need to declare a pattern deviation
    t = transforms.PolyTransform(np.ones(10), np.ones(10), np.ones(10), np.ones(10), 4)
    #change this to directly call the astropy.models directly 
    #now we can declare the polynomial coefficients
    #translation = 0
    t.px.c0_0 = 0
    #x scale
    t.px.c1_0 = 0 #-0.00002
    #linear cross term
    t.px.c0_1 = 0 #0.00004
    #quadratic terms
    t.px.c2_0 = 0# 2 * 10**-8
    t.px.c0_2 = 0 #-4 * 10**-9
    t.px.c1_1 =  0# 5 * 10**-10

    t.px.c3_0 = 0 #10**-12
    t.px.c2_1 = 0
    t.px.c1_2 = 0
    t.px.c0_3 = 0

    t.px.c4_0 = 0 #10**-14
    t.px.c3_1 = 0
    t.px.c2_2 = 0
    t.px.c1_3 = 0
    t.px.c0_4 = 0

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

    t.py.c4_0 = 10**-14
    t.py.c3_1 = 0
    t.py.c2_2 = 0
    t.py.c1_3 = 0
    t.py.c0_4 = 0

    
    #now apply this as a function  of reference position
    xmin = 1000
    xmax = 6000
    ymin = 1000
    ymax = 5500
    cb = (ref['x'] < xmax) * (ref['x'] > xmin) * (ref['y'] < ymax) * (ref['y'] > ymin)
    if order_pattern_error:
        dx, dy = t.evaluate(ref['x']-np.median(ref['x']), ref['y']-np.median(ref['y']))
    
    else:
        dx = np.zeros(len(ref['x']))
        dy = np.zeros(len(ref['y']))
   # import pdb;pdb.set_trace()
    print(np.std(dx[cb])*6000.0,np.std(dy[cb])*6000.0)
    print(np.min(dx[cb])*6000.0, np.max(dx[cb])*6000.0, np.min(dy[cb])*6000.0, np.max(dy[cb])*6000.0)

    _norm_pattern = scipy.stats.norm(loc=0, scale=pattern_error/6000.0)
    _xerr = _norm_pattern.rvs(len(ref['x']))
    _yerr = _norm_pattern.rvs(len(ref['x']))
    xpin = ref['x'] + _xerr + dx
    ypin = ref['y'] + _yerr + dy
    plt.close('all')

    if correct_ref:
        _out = ref
        _out['x'] = xpin
        _out['y'] = ypin
        _out.write('reference.txt', format='ascii.basic')

    if debug_plots:
        mkplots_1(ref, dx, dy, _xerr, _yerr, xpin, ypin)
 
    #now we need to apply translation offsets and create the "average stacked catalogs"
    #these are the offsets from s8, used in paper
    #offsets = [[753, 333],[356,336],[7388-7096,5705-7015],[6203-7096,5707-7015],[6173-7055,300--40.6],[1192-40.6,5758-7055.0], [ 1500, 330], [1500, -1300], [0,0]]
    rot_ang = []
    if lab_offsets:
        offsets_s = [[447-40.6, 5894-7055], [770-40.6, 5895-7055], [945-40.6, 5888-7055], [944-40.6, 5644-7055], [697-40.6, 5620-7055], [488-40.6, 5619-7055], [486-40.6, 5376-7055], [745-40.6, 5381-7055], [982-40.6, 5371-7055]]
    #create a square gird (npos x npos) of obsevations at each rot_ang note these are now input keyword arguements
    
    offsets = []
    #rot_ang_l = [0, 90, 180, 270]
    #import pdb;pdb.set_trace()
    for _ang in rot_ang_l:
        if not lab_offsets:
            for i in range(npos):
                for j in range(npos):
                    offsets.append([i * step_size, j * step_size])
                    rot_ang.append(_ang)
        else:
            for k in range(len(offsets_s)):
                
                offsets.append(offsets_s[k])
                rot_ang.append(_ang)
    xlis = []
    ylis = []

    plt.figure(2)
    plt.clf()
    offsets_in = []
    _norm = scipy.stats.norm(loc=0 , scale=err)
    for _iii, _off in enumerate(offsets):
        _ang =  -1.0*np.deg2rad(rot_ang[_iii])
        _xtmp = (xpin-4000)*np.cos(_ang) - (ypin-3000)*np.sin(_ang)
        _ytmp = (ypin-3000)*np.cos(_ang) + (xpin-4000)*np.sin(_ang)
        #now we need to move the measured coordiantes such that the lowest value points (lower left) is at offsets[i][0,1]
        #import pdb;pdb.set_trace()    
        _dx = _off[0]  + 4000
        _dy = _off[1]  + 3000
        
        xlis.append(_xtmp + _dx)
        ylis.append(_ytmp + _dy)

        offsets_in.append(((xlis[-1][0]),ylis[-1][0]) )

        #if _ang != 0:
        #    import pdb;pdb.set_trace()
                
        print(offsets_in[-1])

        if debug_plots:
            plt.figure(2)
            plt.scatter(xlis[-1], ylis[-1], label='catalog '+str(_iii))
            plt.title('Measured Position with no distortion')
            plt.legend(loc='upper right')
    
    #this is a 4th order Legendre Polynomial -- measured for single stack s6/pos_1/    
    td = mkdist() #look in def mkdist for details of the current distortion applied
    
    xmax = 8000
    xmin = -1

    ymax = 8000
    ymin = -1

    xln = []
    yln = []
    xrn = []
    yrn = []
    xrnC = []
    yrnC = []
    
    plt.figure(100)
    plt.clf()
    #import pdb;pdb.set_trace()
    for i in range(len(xlis)):
        cbool = (xlis[i] < xmax) * (xlis[i] > xmin) * (ylis[i] < ymax) * (ylis[i] > ymin)
        cbool = np.ones(len(xlis[i]), dtype='bool')
        dxd, dyd = td.evaluate(xlis[i][cbool]-4000.0, ylis[i][cbool]-3000.0)
        print('RMS size of distortion', np.std(dxd)*6000, np.std(dyd)*6000.0, np.median(dxd)*6000, np.median(dyd)*6000)
        #import pdb;pdb.set_trace()
        if not distort:
            dxd = np.zeros(np.sum(cbool))
            dyd = np.zeros(np.sum(cbool))
        if plot_dist:
            
            plt.figure(2200)
            q = plt.quiver(xlis[i][cbool], ylis[i][cbool],dxd, dyd, scale =10, color=colors[i])
            
            plt.quiverkey(q, -50, -50, 1, '6000 nm', coordinates='data', color='red')
                
        #add the distortion term to the measurments
        xln.append(xlis[i][cbool]+dxd+ _norm.rvs(np.sum(cbool)))
        yln.append(ylis[i][cbool]+dyd+ _norm.rvs(np.sum(cbool)))
        #create the array of reference positions that is matched to the measurements
        xrnC.append(xpin[cbool])
        yrnC.append(ypin[cbool])

        #create transformstion for the debugging plots
        tl = transforms.PolyTransform(ref['x'], ref['y'], xln[-1], yln[-1], 1)
        __xn, __yn = tl.evaluate(ref['x'], ref['y'])
        xrn.append(__xn)
        yrn.append(__yn)
        
        if debug_plots:
            #need
            mk_debug_plots(ref, xln, yln) 
        
    #import pdb;pdb.set_trace()
    #now we are ready to run the fitting routine
    #we iterate to allow the reference positions to converge 
    #fit_all.simul_wref(xln,yln,offsets)
    if not fit:
        import pdb;pdb.set_trace()
        init_guess = fit_all.guess_co(xln, yln, xrnC, yrnC, order=order)
        res = leastsq(fit_all.com_mod, init_guess, args=(xln, yln, xrnC, yrnC, fix_trans, init_guess, order))
        outres = fit_all.com_mod(res[0], xln, yln, xrnC, yrnC,fix_trans=fix_trans, order=order, evaluate=False, init_guess=init_guess)
        return outres, res
    res = fit_all.simul_wref(xln, yln, offsets_in,  order=order, rot_ang=rot_ang, Niter=Niter, dev_pat=dev_pat, Nmissing=len(xln)+1, sig_clip=False, fourp=False, trim_cam=trim_cam, fix_trans=fix_trans, debug=True, plot_ind=plot_in, trim_pin=False)
   
    
    #import pdb;pdb.set_trace()
    #return 
    #if the fitting procedure has updated reference.txt with fixed coordiantes -> these should match xpin and ypin
    refn = Table.read('reference_new.txt', format='ascii.basic')
    cb = (refn['x'] - refn['xorig']) != 0.0
    __dx = (refn['x'] - xpin) #this is the input deviation
    __dy = (refn['y'] - ypin)
   

    #make scatter plots of the recovered pinhole deviation errors
    plt.figure(2000)
    mkplots_2(refn, xpin, ypin, cb, __dx, __dy)
    #now we compare coefficients to the fit



    fit_all.mksamp(outf='dist_test.txt', order=order)
    _dist = Table.read('dist_test.txt', format='ascii.basic')

    if distort:
        _dxin, _dyin = td.evaluate(_dist['x']-4000, _dist['y']-3000)
    else:
        _dxin = np.zeros(len(_dist['x']))
        _dyin = np.zeros(len(_dist['y']))
    #to do this comparrison you need to eliminate the linear terms in the input distortion
    plt.figure(2005)
    
    #_tl = transforms.PolyTransform(_dist['x']+_dxin, _dist['y']+_dyin, _dist['dx']+_dist['x'], _dist['dy']+_dist['y'], 1)
    _tl = transforms.PolyTransform(_dist['x'], _dist['y'],_dxin, _dyin,  1)
    _xmd, _ymd = _tl.evaluate(_dist['x'], _dist['y'])
    diffx = _dist['dx'] + _dxin
    diffy = _dist['dy'] + _dyin
    import pdb;pdb.set_trace()
    #diffx = _dist['dx']+_dxin
    #diffx = diffx - np.mean(diffx)
    #diffy = _dist['dy']+_dyin
    #diffy = diffy - np.mean(diffy)
    plt.figure(2005)

    plt.scatter(_dist['x'], _dist['y'], c=diffx*6000.0)
    plt.title('X Camera Distortion Meaurement Mistake (model - input)')
    plt.xlabel("X reference (pix)")
    plt.ylabel("Y reference (pix)")
    plt.colorbar()
    plt.tight_layout()

    plt.figure(2006)
    plt.scatter(_dist['x'], _dist['y'], c=diffy*6000.0)
    plt.title('Y Camera Distortion Measurement Mistake (model - input)')
    plt.xlabel("X reference (pix)")
    plt.ylabel("Y reference (pix)")
    plt.colorbar()
    plt.tight_layout()

    plt.figure(2007)
    plt.hist((diffx)*6000.0, histtype='step', label='x', lw=3)
    plt.hist((diffy)*6000.0, histtype='step', label='y', lw=3)
    plt.xlabel("Camera Distortion Meaurement Mistake (model - input)")
    plt.ylabel("N")
    plt.legend(loc='upper left')
    co = Table.read('fit.txt', format='ascii.basic')
    #have to compate like to like coefficients


def test1d(dev_pat=False, offset_max=4000,Maxcount=10, trim_dat=False, random=False, pattern_error=100, fix_scale=False, fix_trans=False):
    '''
fix scale
    '''

    xpin = np.linspace(0, 15000, num=15001)
    
    if not random:
        xpin_dev = xpin + (xpin-np.median(xpin))**2 *3* 10**-9
        
    else:
        _norm_pattern = scipy.stats.norm(loc=0, scale=pattern_error/6000.0)
        _xerr = _norm_pattern.rvs(len(xpin))
        xpin_dev = xpin + _xerr
        
    offsets = np.linspace(0, offset_max, num=10)
    #np.random.shuffle(offsets)
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
    #add in distortion terms
    init_guess.append(0.0)
    #init_guess.append(0.0)
    #init_guess.append(0.0)
    contin = True
    xref_new = np.zeros((len(xref), len(offsets)))
    xrefn = xref
    count = 0
    while contin:
        #import pdb;pdb.set_trace()
        res = leastsq(com_mod, init_guess, args=(xmeas, xrefn, fix_scale, fix_trans, init_guess))
        resid = com_mod(res[0],xmeas, xrefn,fix_scale, fix_trans,init_guess, evaluate=True)
        if dev_pat == False  or count == Maxcount :
            contin = False
        else:
            #compute new refernce coordinates
            contin = True
            dx = np.median(resid, axis=0)
            xrefn = xrefn + dx
            count +=1
            #import pdb;pdb.set_trace()
            print(count)

    if dev_pat:
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
        plt.title("Reference Position Mistake (Model - input)")
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
        plt.subplot(3, 1, 2)
        plt.scatter(xmeas[i], resid[i]*6000.0, s=2, label=str(i))
        #plt.figure(2)
        plt.subplot(3, 1, 1)
        plt.scatter(xmeas[i], (xmeas[i]-xref[i]-offsets[i])*6000.0, s=2, label=str(i))
        #plt.subplot(5, 1, 2)
        #plt.scatter(xmeas[i], -1.0*(xrefn[i] - xmeas[i] * res[0][2*i] - res[0][2*i+1])*6000.0, s=2, label=str(i))
        #plt.subplot(5, 1, 4)
        #plt.scatter(xref[i], resid[i]*6000.0, label=str(i))
        plt.subplot(3, 1, 3)
        plt.scatter(xmeas[i],( -1.0*(xmeas[i]-np.median(xpin))**2*res[0][-1])*6000.0)
        

    _out = Table()
    _out['co'] = res[0]
    _out.write('1D_co.txt', format='ascii.basic')
    #plt.figure(1)
    plt.subplot(3, 1, 2)
    plt.title(' Total Model Residual')
    plt.xlabel("Measured Position (pix)")
    plt.ylabel("Residual (nm)")
    #plt.legend(loc='upper right')
    plt.tight_layout()

    
    #plt.subplot(5, 1, 4)
    #plt.title(' Total Model Residual')
    #plt.xlabel("Reference Position (pix)")
    #plt.ylabel("Residual (nm)")
    #plt.legend(loc='upper right')
    #plt.tight_layout()

    
    plt.subplot(3, 1, 1)
    plt.title("Input Pattern Error")
    plt.xlabel("Measured Position (pix)")
    plt.ylabel("Input Deviation (nm)")
    #plt.legend(loc='upper right')
    plt.tight_layout()

    #plt.subplot(5, 1, 2)
    #plt.title("Linear Model Residuals")
    #plt.xlabel("Measured Position (pix)")
    #plt.ylabel("Residual (nm)")
    #plt.legend(loc='upper right')
    #plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.title('Model Camera Distortion')
    plt.xlabel("Measured position (pix)")
    plt.ylabel('Camaera Distoriton (nm)')
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
                _xtrans = init_guess[2*i]*xmeas[i] + init_guess[2*i+1]  + inco[-1]*xmeas[i]**2
            elif not fix_scale and fix_trans:
                _xtrans = inco[2*i]*xmeas[i] + init_guess[2*i+1] +  inco[-1]*xmeas[i]**2
                
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

    

def mkplots_1(ref, dx, dy, _xerr, _yerr, xpin, ypin):
    '''
    makes debugging plots for the 2d test
    '''
    

        

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
    
def mkplots_2(refn, xpin, ypin, cb, __dx, __dy):


    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    _dx_mod = refn['x'][cb] - refn['xorig'][cb]
    _dx_mod = _dx_mod - np.mean(_dx_mod)
    _dy_mod = refn['y'][cb] - refn['yorig'][cb]
    _dy_mod = _dy_mod - np.mean(_dy_mod)

    _dx_input = xpin[cb] - refn['xorig'][cb]
    _dx_input = _dx_input - np.mean(_dx_input)
    _dy_input = ypin[cb] - refn['yorig'][cb]
    _dy_imput = _dy_input - np.mean(_dy_input)

    

    im = axes.flatten()[2].scatter(refn['xorig'][cb], refn['yorig'][cb], c=(_dx_mod)*6000.0)
    fig.colorbar(im, ax=axes.flatten()[2])
    axes.flatten()[2].set_xlabel('X reference (pix)')
    axes.flatten()[2].set_ylabel('Y reference (pix)')
    axes.flatten()[2].set_title('X Pattern deviation (Model)')
    axes.flatten()[2].set_aspect('equal')
    #plt.axes().set_aspect('equal')

        
    #plt.subplot(3, 2, 2)
    im = axes.flatten()[3].scatter(refn['xorig'][cb], refn['yorig'][cb], c=(_dy_mod)*6000.0)
    fig.colorbar(im, ax=axes.flatten()[3])
    axes.flatten()[3].set_xlabel('X reference (pix)')
    axes.flatten()[3].set_ylabel('Y reference (pix)')
    axes.flatten()[3].set_title('Y Pattern deviation (Model)')
    axes.flatten()[3].set_aspect('equal')
    #plt.axes().set_aspect('equal')

    #plt.subplot(3, 2, 3)
    im = axes.flatten()[1].scatter(refn['xorig'][cb], refn['yorig'][cb], c=(_dy_input)*6000.0)
    fig.colorbar(im, ax=axes.flatten()[1])
    axes.flatten()[1].set_xlabel('X reference (pix)')
    axes.flatten()[1].set_ylabel('Y reference (pix)')
    axes.flatten()[1].set_title('Y Pattern Deviation (input)')
    axes.flatten()[1].set_aspect('equal')
    #plt.axes().set_aspect('equal')
    #plt.subplot(3, 2, 4)
    im = axes.flatten()[0].scatter(refn['xorig'][cb], refn['yorig'][cb], c=(_dx_input)*6000.0)
    fig.colorbar(im, ax=axes.flatten()[0])
    axes.flatten()[0].set_xlabel('X reference (pix)' )
    axes.flatten()[0].set_ylabel('Y reference (pix)' )
    axes.flatten()[0].set_title('X Pattern Deviation (input)')
    axes.flatten()[0].set_aspect('equal')
    #plt.axes().set_aspect('equal')
    
    #plt.subplot(3, 2, 5)
    im = axes.flatten()[4].scatter(refn['x'][cb], refn['y'][cb], c=(_dx_mod-_dx_input)*6000.0)
    fig.colorbar(im, ax=axes.flatten()[4])
    axes.flatten()[4].set_xlabel('X reference')
    axes.flatten()[4].set_ylabel('Y reference')
    axes.flatten()[4].set_title('X Mistake in Pattern Deviation (Model - Input)')
    axes.flatten()[4].set_aspect('equal')
    #plt.axes().set_aspect('equal')
    #plt.subplot(3, 2, 6)
    
    im = axes.flatten()[5].scatter(refn['x'][cb], refn['y'][cb], c=(_dy_mod-_dy_input)*6000.0)
    fig.colorbar(im, ax=axes.flatten()[5])
    axes.flatten()[5].set_xlabel('X reference')
    axes.flatten()[5].set_ylabel('Y reference')
    axes.flatten()[5].set_title('Y Mistake in Pattern Deviation (Model - Input)')
    axes.flatten()[5].set_aspect('equal')
    #plt.axes().set_aspect('equal')
    print('RMS Pattern deviation mistake X:', np.std(_dx_mod-_dx_input)*6000.0, np.std(_dy_mod-_dy_input)*6000.0)
    plt.tight_layout()

def mk_debug_plots(ref, xln, yln):
        t4p = transforms.four_paramNW(ref['x'][cbool], ref['y'][cbool], xln[-1], yln[-1])
        _xev, _yev = t4p.evaluate(ref['x'][cbool], ref['y'][cbool])
        
        plt.figure(1234)
        plt.subplot(2, 1, 1)
        #import pdb;pdb.set_trace()
        plt.scatter(xln[-1], ( xln[-1] - _xev)*6000.0, label='cat '+str(i))
        plt.title("Total Deviation from Square (4p)")
        plt.xlabel('X camera (pix)')
        plt.ylabel("X Difference (nm)")
        plt.vlines([7263, 965], -50, 600, color='red')
        plt.legend(loc='lower right')

        plt.subplot(2, 1, 2)
        plt.scatter(yln[-1], ( yln[-1] - _yev)*6000.0, label='cat '+str(i))
        plt.title("Total Deviation from Square (4p)")
        plt.xlabel('Y camera (pix)')
        plt.ylabel("Y Difference (nm)")
        plt.vlines([490, 5218], -600, 50, color='red')
        plt.legend(loc='lower right')

        plt.tight_layout()
        if plot_in:
            plt.figure(10+i)
            q = plt.quiver(xln[-1], yln[-1], xln[-1] - _xev, yln[-1] - _yev, scale =10)
        
            plt.quiverkey(q, -50, -50, 0.1, '600 nm', coordinates='data', color='red')
            plt.axes().set_aspect('equal')
            plt.xlabel("X camera (pix)")
            plt.ylabel("Y camera (pix)")
            plt.title('Total Deviation from Square Catalog '+str(i))

            plt.figure(130+i)
            plt.subplot(2, 1,  1)
            plt.scatter(xln[-1], yln[-1], c=(xln[-1] - _xev)*6000)
            plt.title('Total Deviation from Square Catalog '+str(i))
            plt.xlabel('X camera (pix)')
            plt.ylabel('Y camera (pix)')
            plt.colorbar()

            plt.subplot(2,1,  2)
            plt.scatter(xln[-1], yln[-1], c=(yln[-1] - _yev)*6000.0)
            plt.title('Total Deviation from Square Catalog '+str(i))
            plt.xlabel('X camera (pix)')
            plt.ylabel('Y camera (pix)')
            plt.colorbar()
        

        plt.figure(100)
        colors = ['black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray']
            
        q = plt.quiver(xln[-1], yln[-1], xln[-1] - _xev, yln[-1] - _yev, scale =20, color=colors[i], label='cat '+str(i))
        
        plt.quiverkey(q, -50, -50, 0.1, '600 nm', coordinates='data', color='red')
        plt.axes().set_aspect('equal')
        plt.xlabel("X camera (pix)")
        plt.ylabel("Y camera (pix)")
        plt.title('Total Deviation from Square')
        plt.legend(loc='upper left')

        
        #fig, axes = plt.subplots(1, 2, 1)
        
        #plt.subplot(1, 2, 1)
        return

def mkdist(all=True):

    if all:
        td = transforms.PolyTransform(np.ones(10), np.ones(10), np.ones(10), np.ones(10), 3)
        #here we use coeffiecients fit to distortion free model per s8, with order=3
        td.px.c0_0 = 0.0 #-1613/6000.0
        td.px.c1_0 = 0.0
        td.px.c0_1 = 0.0
        
        td.px.c2_0 =  7.1*10**-9
        td.px.c1_1 = 5.5*10**-9
        td.px.c0_2 = 5.53*10**-9

        td.px.c3_0 = -9*10**-13
        td.px.c2_1 = 3.5*10**-12
        td.px.c1_2 = -2.4*10**-13
        td.px.c0_3 = 0.7*10**-13
        
        td.py.c0_0 = 0.0 #438 / 6000.0
        td.py.c1_0 = 0.0
        td.py.c0_1 = 0.0
        
        td.py.c2_0 =  4.04*10**-9
        td.py.c1_1 =  1.6*10**-9
        td.py.c0_2 =  1.3*10**-9
        
        td.py.c3_0 =  -1.46 * 10**-13
        td.py.c2_1 =  -1.3 * 10**-12
        td.py.c1_2 =  1.35 * 10**-12
        td.py.c0_3 =  -5.09 * 10**-12
    else:
        td = transforms.PolyTransform(np.ones(10), np.ones(10), np.ones(10), np.ones(10), 2)
        td.px.c0_0 = 0
        td.px.c1_0 = 0
        td.px.c0_1 = 0

        td.px.c2_0 = 4 * 10**-8
        td.px.c1_1 = 0
        td.px.c0_2 = 0

        td.py.c0_0 = 0
        td.py.c1_0 = 0
        td.py.c0_1 = 0

        td.py.c2_0 = 0
        td.py.c1_1 = 0
        td.py.c0_2 = 0

    
    return td
    
def prove_model():
    inco = [10, 0, 1, -20, 1, 0, 12, 1, 0, 25, 0, 1,  4*10**-9, 0, 0, 0, 0, 2*10**-8]
    coo = np.meshgrid(np.linspace(0, 5000, num=45), np.linspace(0, 5000, num=45))
    xin = coo[0].flatten()
    yin = coo[1].flatten()

    #note, reference shoud make no difference when evaluating instead of fitting
    xr = np.ones(len(xin))
    yr = np.ones(len(yin))

    xN = xin * inco[1] + yin * inco[2] + (xin-4000)**2 *4 * 10**-9 + inco[0]
    yN = yin * inco[4] + yin * inco[5] + (yin-3000)**2 * 2*10**-8 + inco[3]

    xN1 = xin * inco[7] + yin * inco[8] + (xin-4000)**2 *4 * 10**-9 +inco[6]
    yN1 = yin * inco[10] + xin * inco[11] + (yin-3000)**2 * 2*10**-8+inco[9]

    
    

    xn, yn, xrn, yrn = fit_all.com_mod(inco, [xin, xin], [yin, yin], [xr, xr], [yr, yr], evaluate=True, order=2)
    print( np.std(xr - xrn), np.std(xN - xn))
    assert np.std(xr - xrn) < 10**-9 
    assert np.std(xN - xn[0]) < 10**-5
    init_guess = fit_all.guess_co([xin, xin], [yin, yin], [xN, xN1], [yN , yN1], order=2)
    res = leastsq(fit_all.com_mod, init_guess, args=([xin, xin], [yin, yin], [xN, xN1], [yN, yN1],False, inco, 2))
    print(res[0])
    outres = fit_all.com_mod(res[0], [xin, xin], [yin, yin], [xN, xN1], [yN, yN1],fix_trans=False, order=2, evaluate=False, init_guess=inco)
    return outres
        



def test_cross(dev_pat=True, Niter=10,trim_cam=True, order=4, debug_plots=True, distort=False, order_pattern=1 ):
    '''
    The first version of this test is easy.  what if  we put in the best fit model of the distoriton as 100% pattern deviation
    Ideally the output should see no power in the distortion solution and a lot of power in the pattern deviation
    order_pattern == 1 keeps all orders of the distoriton as pattern deviation
    '''

    fit_all.mkrefcat(ang=0)
    ref = Table.read('reference.txt', format='ascii.basic')
    inco = Table.read('best_fit.txt', format='ascii.basic')
    #only keep the linear terms
    if order_pattern == 2:
        #only keep the quadratic trerms from the distortion as pattern deviation
        inco[-9:] = 0.0
        inco[-21:-12] = 0.0
        
    if order_pattern == 3:
        #4th order Y terms
        inco[-5:] = 0
        #2nd order Y terms
        inco[-12:-10] = 0
        #4th order x terms
        inco[-17:-12] = 0
        #2nd order x terms
        inco[-24:-21] = 0

    if order_pattern == 4:
        #3rd order Y terms
        inco[-9:-5] = 0
        #2nd order Y terms
        inco[-12:-9] = 0
        #3rd order x terms
        inco[-21:-18] = 0
        #2nd order x terms
        inco[-24:-21] = 0
    #chop out the linear parameters at the begining of this array
    inco = inco[-30:]

    xmin = 0
    xmax = np.inf
    ymin = 0
    ymax = np.inf
    
    xmin = 965.148
    
    xmax = 7263.57
    
    ymin = 490.87
    ymax = 5218.0615
    rb = (ref['x'] > xmin) * (ref['x'] < xmax) * (ref['y'] < ymax) * (ref['y'] > ymin)
    print('input RMS deviations')
    
    #import pdb;pdb.set_trace()
    dx, dy = fit_all.com_mod(inco['col0'], [ref['x']], [ref['y']], [ref['x']], [ref['y']], order=4, evaluate=True, ret_dist=True)
    print(np.std(dx[0][rb])*6000, np.std(dy[0][rb])*6000)
    
    #sets the uncorreltated pattern deviations
    _xerr = 0
    _yerr = 0
    xpin = ref['x'] + _xerr + dx
    ypin = ref['y'] + _yerr + dy
    xpin = np.array(xpin)
    ypin = np.array(ypin)
    
    offsets_s = [[447-40.6, 5894-7055], [770-40.6, 5895-7055], [945-40.6, 5888-7055], [944-40.6, 5644-7055], [697-40.6, 5620-7055], [488-40.6, 5619-7055], [486-40.6, 5376-7055], [745-40.6, 5381-7055], [982-40.6, 5371-7055]]

    colors = ['black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray','black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray', 'black', 'blue', 'green', 'purple', 'brown', 'magenta', 'teal', 'yellow', 'gray']

    offsets = []
    rot_ang = []
    offsets_in = []
    for k in range(len(offsets_s)):
        offsets.append(offsets_s[k])
        rot_ang.append(0)

    xlis = []
    ylis = []

    for _iii, _off in enumerate(offsets):
        _ang =  -1.0*np.deg2rad(rot_ang[_iii])
        _xtmp = (xpin-4000)*np.cos(_ang) - (ypin-3000)*np.sin(_ang)
        _ytmp = (ypin-3000)*np.cos(_ang) + (xpin-4000)*np.sin(_ang)
        #now we need to move the measured coordiantes such that the lowest value points (lower left) is at offsets[i][0,1]
        #import pdb;pdb.set_trace()    
        _dx = _off[0]  + 4000
        _dy = _off[1]  + 3000
        
        xlis.append(_xtmp + _dx)
        ylis.append(_ytmp + _dy)

        offsets_in.append(((xlis[-1][0]),ylis[-1][0]) )

        #if _ang != 0:
        #    import pdb;pdb.set_trace()
                
        print(offsets_in[-1])

        if debug_plots:
            plt.figure(2)
            plt.scatter(xlis[-1], ylis[-1], label='catalog '+str(_iii))
            plt.title('Measured Position with no distortion')
            plt.legend(loc='upper right')
            
    if distort:
        xln = []
        yln = []
        dx, dy =  fit_all.com_mod(inco['col0'], xlis, ylis, xlis, ylis, order=4, evaluate=True, ret_dist=True)
        import pdb;pdb.set_trace()
        for i in range(len(xlis)):
            xln.append(xlis[i] + dx[i])
            yln.append(ylis[i] + dy[i])
        xlis = xln
        ylis = yln
    res = fit_all.simul_wref(xlis, ylis, offsets,  order=order, rot_ang=rot_ang, Niter=Niter, dev_pat=dev_pat, Nmissing=2, sig_clip=False, fourp=False, trim_cam=trim_cam, fix_trans=False, debug=True, plot_ind=False, trim_pin=False)
   
    
    #import pdb;pdb.set_trace()
    #return 
    #if the fitting procedure has updated reference.txt with fixed coordiantes -> these should match xpin and ypin
    refn = Table.read('reference_new.txt', format='ascii.basic')
    cb = (refn['x'] - refn['xorig']) != 0.0
    __dx = (refn['x'] - xpin) #this is the input deviation
    __dy = (refn['y'] - ypin)
   

    #make scatter plots of the recovered pinhole deviation errors
    plt.figure(2000)
    mkplots_2(refn, xpin[0], ypin[0], cb, __dx, __dy)
    #now we compare coefficients to the fit



    fit_all.mksamp(outf='dist_test.txt', order=order)
    _dist = Table.read('dist_test.txt', format='ascii.basic')

  
    _dxin = np.zeros(len(_dist['x']))
    _dyin = np.zeros(len(_dist['y']))
    #to do this comparrison you need to eliminate the linear terms in the input distortion
    plt.figure(2005)
    
    #_tl = transforms.PolyTransform(_dist['x']+_dxin, _dist['y']+_dyin, _dist['dx']+_dist['x'], _dist['dy']+_dist['y'], 1)
    _tl = transforms.PolyTransform(_dist['x'], _dist['y'],_dxin, _dyin,  1)
    _xmd, _ymd = _tl.evaluate(_dist['x'], _dist['y'])
    diffx = _dist['dx'] + _dxin
    diffy = _dist['dy'] + _dyin
    
    #diffx = _dist['dx']+_dxin
    #diffx = diffx - np.mean(diffx)
    #diffy = _dist['dy']+_dyin
    #diffy = diffy - np.mean(diffy)
    plt.figure(2005)

    plt.scatter(_dist['x'], _dist['y'], c=diffx*6000.0)
    plt.title('X Camera Distortion Meaurement Mistake (model - input)')
    plt.xlabel("X reference (pix)")
    plt.ylabel("Y reference (pix)")
    plt.colorbar()
    plt.tight_layout()

    plt.figure(2006)
    plt.scatter(_dist['x'], _dist['y'], c=diffy*6000.0)
    plt.title('Y Camera Distortion Measurement Mistake (model - input)')
    plt.xlabel("X reference (pix)")
    plt.ylabel("Y reference (pix)")
    plt.colorbar()
    plt.tight_layout()

    plt.figure(2007)
    plt.hist((diffx)*6000.0, histtype='step', label='x', lw=3)
    plt.hist((diffy)*6000.0, histtype='step', label='y', lw=3)
    plt.xlabel("Camera Distortion Meaurement Mistake (model - input)")
    plt.ylabel("N")
    plt.legend(loc='upper left')
    co = Table.read('fit.txt', format='ascii.basic')
    #have to compate like to like coefficients
