from . import match_pin
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt 
from flystar import align , match, transforms
from jlu.astrometry import Transform2D as trans
import matplotlib.animation as manimation
from scipy.misc import imread
from jlu.util import statsIter
from astropy.io import fits
import os 
from imaka_pinhole import red_pin
import pickle

def mkave(lisf = 'lis.lis', xl=150, xh=5850, yl=800, yh=6032, retmask=False, retoffsets=False ):
    '''
    creates the average coordinates and trims down the data to the "good" section. 
    Note there are strange reflections at the edge of the lens FOV
    '''

    xall, yall , fall= match_pin.match_all(lisf)
    #create mask for eventual averaging
    mask = (xall == 0.0)
    #instead of just masking out missed measurments, want to create a mask to eliminate those from the averaged coordinates (then no need for xl/yl?)
    g_ave = np.ones(mask.shape[0], dtype='bool')
    for i in range(mask.shape[0]):
        if np.all(mask[i,:] == False):
            g_ave[i] = True
    #import pdb;pdb.set_trace()
            
    
    xs, ys, dx, dy = match_pin.shift_and_ave_coo(xall, yall)

    #import pdb;pdb.set_trace()
    xm = np.ma.array(xs, mask=mask)
    ym = np.ma.array(ys, mask=mask)

    xave = np.mean(xm, axis=1)
    yave = np.mean(ym, axis=1)

    xerr = np.std(xm, axis=1)
    yerr = np.std(ym, axis=1)

    cbool = ( xave > xl) *(xave < xh) * (yave < yh) * (yave > yl) * g_ave
    #instead, simpley eliminate pinholes not detected in all images
    
    if retmask:
        if retoffsets:
            return xave[cbool], yave[cbool], xerr[cbool], yerr[cbool], cbool, dx, dy
        else:
            return xave[cbool], yave[cbool], xerr[cbool], yerr[cbool], cbool
    else:
        if retoffsets:
            xave[cbool], yave[cbool], xerr[cbool], yerr[cbool], dy, dy
        else:
            return xave[cbool], yave[cbool], xerr[cbool], yerr[cbool]
        

        
def mkref(xin, yin, fittype='four',  xl=150, xh=5850, yl=150, yh=6032):
    '''
    '''

    inbool = (xin > xl) * (xin < xh) * (yin > yl) * (yin < yh)
    xin = xin[inbool]
    yin = yin[inbool]

    xmed = np.median(xin)
    ymed = np.median(yin)
    
    origin_arg = np.argmin((xin-xmed)**2 + (yin-ymed)**2)
    
    gspace = 180
    xref = np.array(range(300)) * gspace
    yref = np.array(range(300)) * gspace

    xref = xref - xref[150] + xin[origin_arg]
    yref = yref - yref[150] + yin[origin_arg]

    coo = np.meshgrid(xref, yref)
    xr = coo[0].flatten()
    yr = coo[1].flatten()

    idx1, idx2, dr, dm = match.match(xr, yr, np.ones(len(xr)) , xin , yin, np.ones(len(xin)) , 40)

    assert len(idx1) > 100
    #create linear tranformation of reference to input coordiantes
    #trim refeence coordiantes down
    if fittype == 'four':
        t = trans.fourparam(xr[idx1], yr[idx1], xin[idx2], yin[idx2])
    elif fittype=='linear':
        t = transforms.PolyTransform(xr[idx1], yr[idx1], xin[idx2], yin[idx2],1) 


    #coo_r = np.sqrt(xref**2 + yref**2)
    #refcut = coo_r < 8000
    #xrefn = xref[refcut]
    #yrefn = yref[refcut]
    
    xn, yn = t.evaluate(xr, yr)
    #import pdb;pdb.set_trace()
    return xin[idx2], yin[idx2], xn[idx1], yn[idx1], xr[idx1], yr[idx1], t
        
def compare2square(xin, yin, fit_ord=1,printcat=False):
    '''
   takes in coordinats of pinhole images 
    creates a square grid, matches the square grid to the measured positions using a linear tranformation (6 parameter)
    returns reference positions that are matched to the input positions.
    Note -- All input positions should have a match, may require precleaning on input catalogs.

    subfunction return refernce locations
    '''

    
    #get the point closest to the median of x and y
    xnin, ynin, xr, yr = mkref(xin, yin)

    #now we have the reference coordinates, the next choice is to fit the distortion....
    if printcat:
        outab = Table()
        outab['x'] = xnin
        outab['y'] = ynin
        outab['xr'] = xr[idx1]
        outab['yr'] = yr[idx1]
        outab.write('matched.txt', format='ascii.fixed_width')
        
    xres = xnin - xn
    yres = ynin - yn

    return xn, yn, xnin, ynin, xres, yres, t 



def mkquiver(xnin, ynin, xres, yres, title_s='Residual Distortion', save_f='residual_distortion.png', fig_n=1, scale=20, scale_size=1, frac_plot=3):
    '''
    '''
    plt.figure(fig_n)
    plt.clf()
    plt.title(title_s)
    q = plt.quiver(xnin[::frac_plot], ynin[::frac_plot], xres[::frac_plot], yres[::frac_plot], scale=scale, width=0.0022)
    qk = plt.quiverkey(q, 4400, 6000, scale_size, str(round(6.*scale_size, 3))+' microns', coordinates='data', color='green')
    plt.xlim(0, 8000)
    plt.ylim(0, 6500)
    plt.axes().set_aspect('equal')
    plt.savefig(save_f)


def plot_rms(xnin, ynin, xerr, yerr, title_s='Measurement Precision', save_f='precision.png', fig_n=1):
    '''
    '''
    plt.figure(fig_n)
    plt.clf()
    plt.title(title_s)
    q = plt.quiver(xnin, ynin, xerr, yerr, scale=1)
    qk = plt.quiverkey(q, 4400, 6000, 1.0/60.0, '100 nm', coordinates='data', color='green')
    plt.xlim(0, 8000)
    plt.ylim(0, 6500)
    plt.axes().set_aspect('equal')
    plt.savefig(save_f)

    plt.figure(fig_n+1)
    plt.clf()
    plt.hist(xerr * 6, bins=25, histtype='step', color='red', label='x')
    plt.hist(yerr * 6, bins=25, histtype='step', color='yellow', label='y')
    plt.legend(loc='upper right')
    plt.xlabel('RMS Error (microns)')
    plt.ylabel('N')
    plt.title(title_s)
    


def solve_mult(lisf='lisO.lis'):

    intab = Table.read(lisf, format='ascii.no_header')

    xresT = []
    yresT = []
    xmT = []
    ymT = []
    xpre = []
    ypre = []
    for i, _ff in enumerate(intab['col1']):
        plot_stack(_ff, fn='offsets_'+str(i).zfill(2)+'.png')
        xin, yin, xerr, yerr = mkave(_ff)
        xn, yn, xnin, ynin, xres, yres, t =  compare2square(xin, yin, fit_ord=1)
        
        for ii in range(len(xn)):
            xresT.append(xres[ii])
            yresT.append(yres[ii])
            xmT.append(xn[ii])
            ymT.append(yn[ii])
            xpre.append(xerr[ii])
            ypre.append(yerr[ii])

    xresT = np.array(xresT)
    yresT = np.array(yresT)
    xmT = np.array(xmT)
    ymT = np.array(ymT)

    #fit to the residuals for all the pinholes
    t3 = transforms.LegTransform(xmT,ymT,xresT, yresT, 3)
    t4 = transforms.LegTransform(xmT,ymT,xresT, yresT, 4)
    #eval_fit(t, fn='model_high20par.fits', subr=False)

    xmRes, ymRes = t3.evaluate(xmT, ymT)
    xmRes4, ymRes4 = t4.evaluate(xmT, ymT)
    xfiterr = xresT - xmRes
    yfiterr = yresT - ymRes

    xfiterr4 = xresT - xmRes4
    yfiterr4 = yresT - ymRes4

    print('Number of measurements'+str(len(xresT)))
    print('Mean precision x '+ str(np.mean(xerr)*6)+' microns')
    print('Mean precision y '+ str(np.mean(yerr)*6)+' microns')
    

    print('Mean Residual 20 parameter x '+ str(np.std(xfiterr)*6)+' microns')
    print('Mean Residual 20 parameter y '+ str(np.std(yfiterr)*6)+' microns')

    print('Mean Residual 30 parameter x '+ str(np.std(xfiterr4)*6)+' microns')
    print('Mean Residual 30 parameter y '+ str(np.std(yfiterr4)*6)+' microns')

    print('Mean Residual linear x '+ str(np.std(xresT)*6)+' microns')
    print('Mean Residual linear y '+ str(np.std(yresT)*6)+' microns')

    plt.figure(10)
    plt.clf()
    plt.hist(xresT*6, bins=25, histtype='step', color='red', lw=3, label='x linear')
    plt.hist(yresT*6, bins=25, histtype='step', color='yellow',lw=3, label='y linear')
    plt.hist(xfiterr*6, bins=25, histtype='step', color='blue', lw=3, label='x 20 par')
    plt.hist(yfiterr*6, bins=25, histtype='step', color='black',lw=3, label='y 20 par')
    plt.hist(xfiterr4*6, bins=25, histtype='step', color='green', lw=3, label='x 30 par')
    plt.hist(yfiterr4*6, bins=25, histtype='step', color='purple',lw=3, label='y 30 par')
    #plt.hist(xres*6, bins=25, histtype='step', color='green', lw=3, label='x linear')
    #plt.hist(yres*6, bins=25, histtype='step', color='purple', lw=3, label='y linear')
    plt.legend(loc='upper right')
    plt.title('Distortion Residual')
    plt.xlabel('Residual (microns)')
    plt.ylabel('N')

    plt.savefig('residual_hist.png')

    mkquiver(xmT, ymT, xresT, yresT, title_s='Residual Distortion', save_f='res_dist_lin.png', fig_n=1, scale=10,frac_plot=5 )
    mkquiver(xmT, ymT, xmRes-xresT, ymRes-yresT, title_s='Residual Distortion', save_f='res_dist20par.png', fig_n=2, scale=1, scale_size=.1, frac_plot=5)
    #mkquiver(xmT, ymT, xmRes4, ymRes4, title_s='Residual Distortion', save_f='res_dist30par.png', fig_n=3)
    
        
def mkplots():
    plot_var()
    plot_stack()
    xin, yin, xerr, yerr = mkave()
    xn, yn, xnin, ynin, xres, yres, t =  compare2square(xin, yin, fit_ord=1, printcat=True)
    eval_fit(t, fn='model_dist_6par.fits')

    mkquiver(xnin, ynin, xres, yres, title_s='Residual Distortion (linear)', save_f='res_dist_lin.png', fig_n=1)


    xn2, yn2, xnin2, ynin2, xres2, yres2, t2 =  compare2square(xin, yin, fit_ord=2)

    eval_fit(t2, fn='model_dist_12par.fits')

    mkquiver(xnin2, ynin2, xres2, yres2, title_s='Residual Distortion (12 parameter)', save_f='res_dist_12param.png', fig_n=3, scale=5, scale_size=.1)

    xn3, yn3, xnin3, ynin3, xres3, yres3, t3 =  compare2square(xin, yin, fit_ord=3)

    mkquiver(xnin3, ynin3, xres3, yres3, title_s='Residual Distortion ( 20 parameter)', save_f='res_dist_20param.png', fig_n=5, scale=1, scale_size=.1)
    eval_fit(t3, fn='model_dist_20par.fits')

    eval_fit(t3, fn='model_high_only.fits', tsub=t)
    plt.figure(10)
    plt.clf()
    plt.hist(xres2*6, bins=25, histtype='step', color='red', lw=3, label='x 12 par')
    plt.hist(yres2*6, bins=25, histtype='step', color='yellow',lw=3, label='y 12 par')
    plt.hist(xres3*6, bins=25, histtype='step', color='blue', lw=3, label='x 20 par')
    plt.hist(yres3*6, bins=25, histtype='step', color='black',lw=3, label='y 20 par')
    plt.hist(xres*6, bins=25, histtype='step', color='green', lw=3, label='x linear')
    plt.hist(yres*6, bins=25, histtype='step', color='purple', lw=3, label='y linear')
    plt.legend(loc='upper right')
    plt.title('Distortion Residual')
    plt.xlabel('Residual (microns)')
    plt.ylabel('N')

    plt.savefig('residual_hist.png')
    #read in the individiula fits
    ind_fits = Table.read('ind_fits.txt', format='ascii.fixed_width')
    a0m = np.mean(ind_fits['a0'])
    a1m = np.mean(ind_fits['a1'])
    a2m = np.mean(ind_fits['a2'])
        
    b0m = np.mean(ind_fits['b0'])
    b1m = np.mean(ind_fits['b1'])
    b2m = np.mean(ind_fits['b2'])

    a0s = np.std(ind_fits['a0'])
    a1s = np.std(ind_fits['a1'])
    a2s = np.std(ind_fits['a2'])

    b0s = np.std(ind_fits['b0'])
    b1s = np.std(ind_fits['b1'])
    b2s = np.std(ind_fits['b2'])
    
    #mean_ar = np.mean(ind_fits.data, axis=1)
    #sig_ar = np.std(ind_fits.dat, axis=1)

    oxerr = np.mean(xerr)*6
    oyerr = np.mean(yerr)*6
    xerr6 = np.std(xres)*6
    yerr6=np.std(yres)*6
    xerr12 = np.std(xres2)*6
    yerr12 = np.std(yres2)*6
    xerr20 = np.std(xres3)*6
    yerr20 = np.std(yres3)*6
    
    print('Mean precision x '+ str(oxerr)+' microns')
    print('Mean precision y '+ str(oyerr)+' microns')
    
    print('Mean Residual 12 parameter x '+ str(xerr12)+' microns')
    print('Mean Residual 12 parameter y '+ str(yerr12)+' microns')

    print('Mean Residual 20 parameter x '+ str(xerr20)+' microns')
    print('Mean Residual 20 parameter y '+ str(yerr20)+' microns')

    print('Mean Residual linear x '+ str(xerr6)+' microns')
    print('Mean Residual linear y '+ str(yerr6)+' microns')

    _out = Table(names=['a0', 'a0err', 'a1', 'a1err', 'a2', 'a2err', 'b0', 'b0err', 'b1', 'b1err', 'b2', 'b2err', 'xpre', 'ypre','x6res', 'y6re', 'x12res', 'y12res','x20res', 'y20res'], data=[[a0m], [a0s], [a1m], [a1s], [a2m], [a2s], [b0m], [b0s], [b1m], [b1s], [b2m], [b2s], [oxerr], [oyerr], [xerr6], [yerr6], [xerr12], [yerr12], [xerr20], [yerr20]])
    _out.write('summary.txt', format='ascii.fixed_width')

    

    plt.show()
def mkplots_cat(xkey='xlinr', ykey='ylinr'):
    reflisf = 'ref.lis'
    x, y, dx, dy, dxm, dym, t = fit_from_cat(reflisf, xrefkey=xkey, yrefkey=ykey, fitord=1)
    eval_fit(t, fn='model_dist_6par.fits')
    xres = -1.0 * dx +dxm
    yres = -1.0 * dy + dym
    mkquiver(x, y, xres, yres, title_s='Residual Distortion (linear)', save_f='res_dist_lin.png', fig_n=1, frac_plot=200)
    mkquiver(x, y, dx, dy, title_s='Measured Distortion (linear removed)', save_f='res_dist_lin.png', fig_n=100, frac_plot=200, scale=35, scale_size=5)

    x2, y2, dx2, dy2, dxm2, dym2, t2 = fit_from_cat(reflisf, xrefkey=xkey, yrefkey=ykey, fitord=2)
    xres2 = -1.0 * dx2 + dxm2
    yres2 = -1.0 * dy2 + dym2
    #xn2, yn2, x2, y2, xres2, yres2, t2 =  compare2square(xin, yin, fit_ord=2)

    eval_fit(t2, fn='model_dist_12par.fits')

    mkquiver(x2, y2, xres2, yres2, title_s='Residual Distortion (12 parameter)', save_f='res_dist_12param.png', fig_n=3, scale=3, scale_size=.1,frac_plot=200)

    x3, y3, dx3, dy3, dxm3, dym3, t3 = fit_from_cat(reflisf, xrefkey=xkey, yrefkey=ykey, fitord=3)
    xres3 = -1.0 *dx3 + dxm3
    yres3 = -1.0 *dy3 + dym3
    
    #xn3, yn3, x3, y3, xres3, yres3, t3 =  compare2square(xin, yin, fit_ord=3)

    mkquiver(x3, y3, xres3, yres3, title_s='Residual Distortion ( 20 parameter)', save_f='res_dist_20param.png', fig_n=5, scale=1, scale_size=.1,frac_plot=200)
    eval_fit(t3, fn='model_dist_20par.fits')

    eval_fit(t3, fn='model_high_only.fits', tsub=t)
    plt.figure(10)
    plt.clf()
    plt.hist(xres2*6, bins=25, histtype='step', color='red', lw=3, label='x 12 par')
    plt.hist(yres2*6, bins=25, histtype='step', color='yellow',lw=3, label='y 12 par')
    plt.hist(xres3*6, bins=25, histtype='step', color='blue', lw=3, label='x 20 par')
    plt.hist(yres3*6, bins=25, histtype='step', color='black',lw=3, label='y 20 par')
    plt.hist(xres*6, bins=25, histtype='step', color='green', lw=3, label='x linear')
    plt.hist(yres*6, bins=25, histtype='step', color='purple', lw=3, label='y linear')
    plt.legend(loc='upper right')
    plt.title('Distortion Residual')
    plt.xlabel('Residual (microns)')
    plt.ylabel('N')

    plt.savefig('residual_hist.png')
    #read in the individiula fits
    ind_fits = Table.read('ind_fits.txt', format='ascii.fixed_width')
    a0m = np.mean(ind_fits['a0'])
    a1m = np.mean(ind_fits['a1'])
    a2m = np.mean(ind_fits['a2'])
        
    b0m = np.mean(ind_fits['b0'])
    b1m = np.mean(ind_fits['b1'])
    b2m = np.mean(ind_fits['b2'])

    a0s = np.std(ind_fits['a0'])
    a1s = np.std(ind_fits['a1'])
    a2s = np.std(ind_fits['a2'])

    b0s = np.std(ind_fits['b0'])
    b1s = np.std(ind_fits['b1'])
    b2s = np.std(ind_fits['b2'])
    
    #mean_ar = np.mean(ind_fits.data, axis=1)
    #sig_ar = np.std(ind_fits.dat, axis=1)

    #oxerr = np.mean(xerr)*6
    #oyerr = np.mean(yerr)*6
    xerr6 = np.std(xres)*6
    yerr6=np.std(yres)*6
    xerr12 = np.std(xres2)*6
    yerr12 = np.std(yres2)*6
    xerr20 = np.std(xres3)*6
    yerr20 = np.std(yres3)*6
    
    #print('Mean precision x '+ str(oxerr)+' microns')
    #print('Mean precision y '+ str(oyerr)+' microns')
    
    print('Mean Residual 12 parameter x '+ str(xerr12)+' microns')
    print('Mean Residual 12 parameter y '+ str(yerr12)+' microns')

    print('Mean Residual 20 parameter x '+ str(xerr20)+' microns')
    print('Mean Residual 20 parameter y '+ str(yerr20)+' microns')

    print('Mean Residual linear x '+ str(xerr6)+' microns')
    print('Mean Residual linear y '+ str(yerr6)+' microns')

    _out = Table(names=['a0', 'a0err', 'a1', 'a1err', 'a2', 'a2err', 'b0', 'b0err', 'b1', 'b1err', 'b2', 'b2err','x6res', 'y6re', 'x12res', 'y12res','x20res', 'y20res'], data=[[a0m], [a0s], [a1m], [a1s], [a2m], [a2s], [b0m], [b0s], [b1m], [b1s], [b2m], [b2s], [xerr6], [yerr6], [xerr12], [yerr12], [xerr20], [yerr20]])
    _out.write('cat_sum.txt', format='ascii.fixed_width')

    

    plt.show()
    
def plot_stack(lis='lis.lis', fn='offsets.png'):
    xin, yin, xerr, yerr,mask,  dx, dy = mkave(lisf=lis, retmask=True, retoffsets=True)

    #simple first plot, just look at the offset values
    plt.figure(15)
    plt.clf()
    plt.plot(np.array(dx)*6, label='x')
    plt.plot(np.array(dy)*6, label='y')
    plt.title('Offsets')
    plt.xlabel('Frame Number')
    plt.ylabel('Offset (microns)')
    plt.legend(loc='upper right')
    plt.savefig(fn)
    plt.show()

def eval_fit(t,xl=100, xh=5850, yl=100, yh=6032, fn='model_dist.fits', tsub=None, subr=True ):


    xr = np.linspace(xl,xh,num=5000)
    yr = np.linspace(yl,yh,num=5000)

    coo = np.meshgrid(xr, yr)
    xr = coo[0].flatten()
    yr = coo[1].flatten()

    if tsub == None:
        xsub = np.zeros(xr.shape)
        ysub = np.zeros(yr.shape)
    else:
        xsub, ysub = tsub.evaluate(xr,yr)
    
    xn, yn = t.evaluate(xr, yr)
    xn = xn - xsub
    yn = yn - ysub

    #import pdb;pdb.set_trace()
    if subr:
        xn = np.reshape(xn-xr, (5000,5000))
        yn = np.reshape(yn-yr, (5000,5000))
    else:
        xn = np.reshape(xn, (5000,5000))
        yn = np.reshape(yn, (5000,5000))
        

    fits.writeto(fn.replace('.fits', '_X.fits'), xn, clobber=True)
    fits.writeto(fn.replace('.fits', '_Y.fits'), yn, clobber=True)
    
def calc_var_n(ar_in, stars_index=None):
    '''
    takes array of star positions dimensions (nframes, nstars)
    splits the data into random groups ranginf from N=1 to N = Nstars/3
    
    '''
    
    nmin_im = 3
    nmax_im = int((ar_in.shape[0] / 3.0))
    
    nim = range(nmin_im, nmax_im)

    if stars_index == None:
        stars_index = np.array(range(ar_in.shape[1]))
    avartot = []
    avartot_err = []
    stdtot = []
    stdtot_err = []
        
    for nn in nim:
        
        #at each step we want to split the data into groups of images with nn
        #to do this, we start with a boolean index for all images
        imbool = np.ones(ar_in.shape[0], dtype='bool')
        index_ar = np.indices((ar_in.shape[0],))[0]

        #as long as there are still nn images that have not been put in a group, keep pulling random groups
        _avar = []
        _std = []
        while np.sum(imbool) >= nn:
            #first draw the random group
            #g_index = np.random.choice(index_ar[imbool], size=nn, replace=False)
            #we want to take sequential frames...
            g_index = index_ar[imbool][:nn]
            imbool[g_index] = False
            #now compute the average positions from this group
            _data = ar_in[g_index,:]
            _data = _data[:,stars_index]
            #_rms = np.mean(np.std(_data, axis=0, ddof=1)) #/ np.sqrt(nn-1)
            _rms, _rms_err = statsIter.mean_std_clip(np.std(_data, axis=0, ddof=1)) #/ np.sqrt(nn-1)
            _allan_ar = np.zeros(_data.shape[1])
            #_rms_manual = np.sqrt(1./(nn-1.)*np.sum((np.mean(_data,axis=0)-_data)**2, axis=0))
            
            #_rms_manual = np.mean(_rms_manual)
            #import pdb;pdb.set_trace()
            for ii in range(_data.shape[0]-1):
                _allan_ar += (_data[ii,:] - _data[ii+1, :])**2
                #import pdb;pdb.set_trace()
                
            _allan_ar = np.sqrt(1./(2.*(nn-1.)) * _allan_ar)
            _allan, _allan_err = statsIter.mean_std_clip(_allan_ar)
            #_allan = np.mean(_allan_ar)
            #import pdb;pdb.set_trace()
            #assert np.abs(_rms - _rms_manual) < .0001
            _avar.append(_allan)
            _std.append(_rms)
            
        avartot.append(np.mean(_avar))
        avartot_err.append(np.std(_avar))
        stdtot.append(np.mean(_std))
        stdtot_err.append(np.std(_std))

    return nim, avartot, avartot_err, stdtot, stdtot_err


def plot_var(inlis='lis.lis', stars_index=None):
    xall, yall, fall = match_pin.match_all('lis.lis')
    xshift, yshift, dx, dy = match_pin.shift_and_ave_coo(xall, yall)
    x, y, xerr, yerr, mask = mkave(retmask=True )
    xcut = xshift[mask,:].T
    ycut = yshift[mask,:].T
    #import pdb;pdb.set_trace()
    xnim, xavar, xavarerr, xrms, xrmserr = calc_var_n(xcut[:,:], stars_index=stars_index)
    ynim, yavar, yavarerr, yrms, yrmserr = calc_var_n(ycut[:,:], stars_index=stars_index)

    plt.figure(18)
    plt.clf()
    plt.loglog(xnim, xavar, label='allan x')
    plt.loglog(xnim, xrms, label='rms x')
    plt.loglog(ynim, yavar, label='allan y')
    plt.loglog(ynim, yrms, label='rms y')
    
    plt.loglog(xnim, 1.0/np.array(xnim)/10, color='black')
    plt.legend(loc='upper right')
    plt.title('Deviation Test')
    plt.xlabel('N')
    plt.ylabel('Deviation (pixels)')
    plt.savefig('var.png')
    
def plot_var_cat(inlis='lis.lis', stars_index=None, xkey='x', ykey='y'):
    #xall, yall, fall = match_pin.match_all(inlis, xkey=xkey, ykey=ykey)
    #xshift, yshift, dx, dy = match_pin.shift_and_ave_coo(xall, yall)
    #need home baked stacking routiine
    x, y, xerr, yerr, mask = mkave(retmask=True )
    xcut = xshift[mask,:].T
    ycut = yshift[mask,:].T
    #import pdb;pdb.set_trace()
    xnim, xavar, xavarerr, xrms, xrmserr = calc_var_n(xcut[:,:], stars_index=stars_index)
    ynim, yavar, yavarerr, yrms, yrmserr = calc_var_n(ycut[:,:], stars_index=stars_index)

    plt.figure(18)
    plt.clf()
    plt.loglog(xnim, xavar, label='allan x')
    plt.loglog(xnim, xrms, label='rms x')
    plt.loglog(ynim, yavar, label='allan y')
    plt.loglog(ynim, yrms, label='rms y')
    
    plt.loglog(xnim, 1.0/np.array(xnim)/10, color='black')
    plt.legend(loc='upper right')
    plt.title('Deviation Test')
    plt.xlabel('N')
    plt.ylabel('Deviation (pixels)')
    plt.savefig('var.png')


def stack_cat(inlis):
    lis_lis = Table.read(lis_f, format='ascii.no_header')['col1']
    _ref = Table.read(lis_lis[0], format='ascii.fixed_width')
    t = transforms.PolyTransform(_ref['xm'], _ref['ym'], _ref[''], _ref[''])
    xr = _ref['xm']
    yr = _ref['ym']
    if 'flux' in _ref.keys():
        fr = _ref['flux']
    else:
        fr = np.zeros(len(_ref))

    x = np.zeros((len(xr), len(lis_lis)))
    y = np.zeros((len(xr), len(lis_lis)))
    flux = np.zeros((len(xr), len(lis_lis)))
    for i, lfile in enumerate(lis_lis):
        _tab = Table.read(lfile, format='ascii.fixed_width')
        if 'flux' in _tab.keys():
            _flux = _tab['flux']
        else:
            _flux = np.zeros(len(_ref))
        idx1 , idx2 , dm, dr = align.match(_tab[xkey],_tab[ykey] ,_flux,  xr, yr, fr, 10)
        x[idx2,i] = _tab['x'][idx1]
        y[idx2,i] = _tab['y'][idx1]
        flux[idx2,i] = _tab['flux'][idx1]

def fit_ind_four(inlis='lis.lis'):

    intab = Table.read(inlis, format='ascii.no_header')

    t = []
    params = [[],[],[],[]]
    a = [[],[],[]]
    b = [[],[],[]]
    
    for i, _ff in enumerate(intab['col1']):
        _tab = Table.read(_ff, format='ascii.fixed_width')
        _xin, _yin, _xr, _yr, _xsquare, _ysquare, _t = mkref(_tab['x'], _tab['y'])
        _xin, _yin, _xlinr, _ylinr, _xsquare, _ysquare, _tlin = mkref(_tab['x'], _tab['y'], fittype='linear')
        #_xs, _ys,_xin, _yin,  _xres, _yres,_t = mkref(_tab['x'], _tab['y'])
        t.append(_t)
        params[0].append(_t.params['transX'])
        params[1].append(_t.params['transY'])
        params[2].append(_t.params['angle'])
        params[3].append(_t.params['scale'])
        #xres.append(_xres)
        #yres.append(_yres)
        #eval_fit(_t, fn='model_im'+str(i).zfill(4)+'.fits')
        for i in range(len(a)):
            a[i].append(_tlin.px.parameters[i])
            b[i].append(_tlin.py.parameters[i])
        
        _outtab = Table(names=['xm', 'ym', 'xr','yr', 'xlinr', 'ylinr','xsquare', 'ysquare'], data=[_xin, _yin, _xr, _yr, _xlinr, _ylinr, _xsquare, _ysquare])
        assert '.lis' in _ff
        _outtab.write(_ff.replace('.lis', '_ref.lis'), format='ascii.fixed_width')

    
    #instead of making a movie, convert the tranform cooef to 

    #need to write some text files...
    _outtab = Table(names=['xoff', 'yoff', 'angle', 'scale'], data=[params[0], params[1], params[2], params[3]])
    #for i in range(len(a)):
    #    _outtab['a'+str(i)] = a[i]
    #for i in range(len(b)):
    #    _outtab['b'+str(i)] = b[i]
    _outtab.write('ind_fits_four.txt', format='ascii.fixed_width')

    _outtab = Table()
    for i in range(len(a)):
        _outtab['a'+str(i)] = a[i]
    for i in range(len(b)):
        _outtab['b'+str(i)] = b[i]
    _outtab.write('ind_fits.txt', format='ascii.fixed_width')
        
    
    return a, b


def fit_from_cat(reflis='ref.lis', fitord=3, xrefkey='xr', yrefkey='yr', spatfilt=True):
    '''
    '''
    cattab = Table.read(reflis, format='ascii.no_header')

    x = []
    y = []
    xref = []
    yref = []

    for _ff in cattab['col1']:
        _tab = Table.read(_ff, format='ascii.fixed_width')
        for i in range(len(_tab)):
            x.append(_tab['xm'][i])
            y.append(_tab['ym'][i])
            xref.append(_tab[xrefkey][i])
            yref.append(_tab[yrefkey][i])

    #now we have the data, make the fit
    x = np.array(x)
    y = np.array(y)
    yref = np.array(yref)
    xref = np.array(xref)

    dx = xref - x
    dy = yref - y

    x, y, dx, dy = spat_filt(x, y, dx, dy)
    #do quick, simple filtering
    #cfac = 2
    #sigx = np.std(dx)
    #sigy = np.std(dy)
    #gbool = (dx < dx + cfac*sigx)*(dx > dx - cfac*sigx)*(dy > dy - cfac*sigy)*(dy < dy + cfac*sigy)
    gbool = np.ones(len(dx), dtype='bool')
    print 'Number of sources trimmed: ', len(gbool)-np.sum(gbool), ' of ', len(gbool)
    t = transforms.LegTransform(x[gbool], y[gbool], dx[gbool], dy[gbool], fitord)
    dxm, dym = t.evaluate(x, y)
    return x, y, dx, dy, dxm, dym, t


def spat_filt(x, y, dx, dy, num_section=8, sig_fac=5):
    xout = []
    yout = []
    dxout = []
    dyout = []
    binsx = np.linspace(np.min(x),np.max(x), num=num_section + 1 )
    binsy = np.linspace(np.min(y),np.max(y), num=num_section + 1 )
    

    for i in range(len(binsx)-1):
        for j in range(len(binsy)-1):
            #first find all the stars that are in the current bin
            sbool = (x > binsx[i])*(x<binsx[i+1])*(y>binsy[j])*(y<binsy[j+1])
            print 'number of stars in this section are ', np.sum(sbool)
            #find the mean delta and sigma in x and y, using iterative sigma clipping
            ave_x, sig_x, nclipx = statsIter.mean_std_clip(dx[sbool], clipsig=3.0, return_nclip=True)
            ave_y, sig_y, nclipy = statsIter.mean_std_clip(dy[sbool], clipsig=3.0, return_nclip=True)
            #creates boolean 
            good_bool = (dx < ave_x + sig_fac * sig_x)*(dx > ave_x - sig_fac * sig_x)*(dy < ave_y + sig_fac *sig_x)*(dy > ave_y - sig_fac * sig_x) * sbool
            print 'number of stars cut ', np.sum(sbool) - np.sum(good_bool) 
            
            for ii in range(np.sum(good_bool)):
                xout.append(x[good_bool][ii])
                yout.append(y[good_bool][ii])
                dxout.append(dx[good_bool][ii])
                dyout.append(dx[good_bool][ii])

    import pdb;pdb.set_trace()
    return np.array(xout), np.array(yout), np.array(dxout), np.array(dyout)
    
    
def mkmp4(imlis, mp4n='out.mp4'):
    imtab = Table.read(imlis, format='ascii.no_header')

    ims = []
    for _im in imtab['col1']:
        ims.append(imread(_im))
    an_ims = []
    plt.clf()
    fig = plt.figure(25)
    plt.clf()
    for i in range(len(ims)):
        im = plt.imshow(ims[i])
        plt.axis('off')
        an_ims.append([im])
    ani = manimation.ArtistAnimation(fig, an_ims, interval=300, blit=True)
    ani.save(mp4n)


def red_dir():
    bdir = os.getcwd() + '/' 
    dirlis = 'dir.lis'
    dirtab = Table.read(dirlis, format='ascii.no_header')

    for dir in dirtab['col1']:
        os.chdir(bdir+dir)
        red_pin.runstf('stf.lis')

def plot_dirs(dirf='dir.lis'):
    bdir = os.getcwd()+'/'
    dirtab = Table.read(dirf, format='ascii.no_header')

    for dir in dirtab['col1']:
        os.chdir(bdir+dir)
        os.system('cp ../lis.lis .')
        a, b = fit_ind_cat()
        mkplots()

def combine_summary(dirf='dir.lis'):
    bdir = os.getcwd()+'/'
    dirtab = Table.read(dirf, format='ascii.no_header')

    outdat = {}
    first_time= True
    dirnames =[]
    for dir in dirtab['col1']:
        os.chdir(bdir+dir)
        dirnames.append(dir.replace('/',''))
        _tab = Table.read('summary.txt', format='ascii.fixed_width')
        if first_time:
            first_time = False
            for _keys in _tab.keys():
                outdat[_keys] = []
                
        for _keys in _tab.keys():
            outdat[_keys].append(_tab[_keys])

    os.chdir(bdir)
    outnames = ['dataset']
    allcols = [dirnames]
    for _names in outdat.keys():
        outnames.append(_names)
        allcols.append(np.round(outdat[_names], 7))
    _outtab = Table(names=outnames, data=allcols)
    #for _key in _outtab.keys():
    #    _outtab[_key].format = "%.2f"
    _outtab.write('total_summary.txt', format='ascii.fixed_width')


def plot_rot(intab=''):
    tsum = Table.read(intab, format='ascii.fixed_width')
    angx = np.arctan(-1.0*tsum['a2']/tsum['a1'])
    angy = np.arctan(tsum['b1']/tsum['b2'])
    #TOTDOSOSOSOSO
        
def compsol(path1, path2, pngname='comp'):
    _x1 = fits.open(path1+'model_dist_20par_X.fits')[0].data
    _x1 = _x1 - np.mean(_x1)
    _x2 = fits.open(path2+'model_dist_20par_X.fits')[0].data
    _x2 = _x2 - np.mean(_x2)

    _y1 = fits.open(path1+'model_dist_20par_Y.fits')[0].data
    _y1 = _y1 - np.mean(_y1)
    _y2 = fits.open(path2+'model_dist_20par_Y.fits')[0].data 
    _y2 = _y2 - np.mean(_y2)

    xdiff = _x2 - _x1
    ydiff = _y2 - _y1

    xrms = np.std(xdiff)
    yrms = np.std(ydiff)

    return xdiff, ydiff , xrms, yrms 
    #plt.figure(30)
    #plt.hist(xdiff.flatten(), lw=2, histtype='step', label='x')
    #plt.hist(ydiff.flatten(), lw=2, histtype='step', label='y')
    #plt.xlabel('Residual( pixels)')
    #plt.ylabel('N')
    #plt.legend(loc='upper left')
    #plt.savefig(pngname+'hist.png')

    mkquiver(_x1, _y1, xdiff, ydiff, title_s='Distortion Change', save_f=pngname+'quiver.png', fig_n=1, scale=20, scale_size=1)

    print('x residual '+str(xrms*6)+ 'microns')
    print('y residual '+str(yrms*6)+ 'microns')
    
    
    
    return diff



def plot4p(incat='ind_fits_four.txt'):
    '''
    '''

    cat = Table.read(incat, format='ascii.fixed_width')

    plt.figure(1)
    plt.clf()
    plt.subplot(121)
    plt.title('Angle')
    plt.xlabel('Frame Number')
    plt.ylabel('Rotation (degrees)')
    plt.plot(cat['angle'])
    plt.subplot(122)
    plt.xlabel('Rotation (degrees)')
    plt.ylabel('N')
    plt.hist(cat['angle'], bins=25)
    
    plt.savefig('fourp_angle.png')

    plt.figure(2)
    plt.clf()
    plt.subplot(121)
    plt.title('Scale')
    plt.xlabel('Frame Number')
    plt.ylabel('Scale')
    plt.plot(cat['scale'])
    plt.subplot(122)
    plt.xlabel('Scale')
    plt.ylabel('N')
    plt.hist(cat['scale'], bins=25)
    
    plt.savefig('fourp_scale.png')

def plot4lin(incat='ind_fits.txt'):
    '''
    '''

    cat = Table.read(incat, format='ascii.fixed_width')

    
    plt.figure(37)
    plt.clf()
    plt.title('Variation in linear parameters')
    plt.xlabel('Frame Number')
    plt.ylabel('Coefficient Value')
    plt.plot(cat['a1']-np.mean(cat['a1']), label='a1')
    plt.plot(cat['a2']-np.mean(cat['a2']), label='a2')
    plt.plot(-1.0*(cat['b1']-np.mean(cat['b1'])), label='b1')
    plt.plot(cat['b2']-np.mean(cat['b2']), label='b2')
    plt.legend(loc='upper left')
    plt.savefig('sixparam.png')


    plt.figure(38)
    plt.title('Transformation Offsets')
    plt.xlabel('Frame Number')
    plt.ylabel('Offset - average Offset (microns)')
    plt.plot(cat['a0']-np.mean(cat['a0']), label='x')
    plt.plot(cat['b0']-np.mean(cat['b0']), label='y')
    plt.legend(loc='upper left')
    plt.savefig('sigparam_offsets.png')

    #
    #
    #
    #
