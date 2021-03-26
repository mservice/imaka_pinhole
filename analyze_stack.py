from . import match_pin
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt 
from flystar_old.flystar_old import align , match, transforms
from jlu.astrometry import Transform2D as trans
import matplotlib.animation as manimation
#from scipy.misc import imread
from jlu.util import statsIter
from astropy.io import fits
import os 
from imaka_pinhole import red_pin, affine
import pickle
from jlu.util import statsIter as stats
import scipy

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
        

        
def mkref(xin, yin, fittype='four',  trim=False, gspace=170, ang=0, ymag=1.07):
    '''
    '''

    if trim:
         xl=150
         xh=5850
         yl=800
         yh=6032
    else:
        xl = 0 #150
        xh = 100000 #8270
        yl = 0 #700
        yh = 100000 #5980
        
    inbool = (xin > xl) * (xin < xh) * (yin > yl) * (yin < yh)
    xin = xin[inbool]
    yin = yin[inbool]

    xmed = np.median(xin)
    ymed = np.median(yin)
    
    
    origin_arg = np.argmin((xin-xmed)**2 + (yin-ymed)**2)
    
    #gspace = 180
    #gspace = 170
    xref = np.array(range(50)) * gspace
    yref = np.array(range(50)) * gspace / 1.07

    coo = np.meshgrid(xref, yref)
    xr = coo[0].flatten()
    yr = coo[1].flatten()

    _ang = np.deg2rad(ang)
    xr = np.cos(_ang) * xr - np.sin(_ang) * yr
    yr = np.sin(_ang) * xr + np.cos(_ang) * yr
    
    #import pdb;pdb.set_trace()
    refo = np.argmin((xr-np.median(xr))**2+(yr-np.median(yr))**2)
    xr = xr - xr[refo] + xin[origin_arg]
    yr = yr - yr[refo] + yin[origin_arg]

    #import pdb;pdb.set_trace()
    idx1, idx2, dr, dm = match.match(xr, yr, np.ones(len(xr)) , xin , yin, np.ones(len(xin)) , 60)
    #dx = xr[idx1][0] - xin[idx2][0]
    #dy = yr[idx1][0] - yin[idx2][0]
    #t = transforms.four_paramNW(xin[idx2], yin[idx2], xr[idx1], yr[idx1])
    #xn, yn = t.evaluate(xin[idx2], yin[idx2])
    
    #idx1, idx2, dr, dm = match.match(xr-dx, yr-dy, np.ones(len(xr)) , xin , yin, np.ones(len(xin)) , 30)
    t = transforms.PolyTransform.derive_transform(xin[idx2], yin[idx2], xr[idx1], yr[idx1], 1)
    xn, yn = t.evaluate(xin, yin)

    idx1, idx2, dr, dm = match.match(xr, yr, np.ones(len(xr)) , xn , yn, np.ones(len(xin)) , 75)
    #import pdb;pdb.set_trace()

    #plt.clf()
    #plt.scatter(xin, yin)
    #plt.scatter(xr[idx1]-dx, yr[idx1] - dy)
    
    assert len(idx1) > 300
    #create linear tranformation of reference to input coordiantes
    #trim refeence coordiantes down
    if fittype == 'four':
        t = trans.fourparam(xr[idx1], yr[idx1], xin[idx2], yin[idx2])
        xm, ym = t.evaluate(xr[idx1], yr[idx1])
        xres = xm - xin[idx2]
        yres = ym - yin[idx2]
        print('4 param residual', np.std(xres) ,  ' pix', np.std(yres), ' pix')
    elif fittype=='linear':
        t = transforms.PolyTransform.derive_transform(xr[idx1], yr[idx1], xin[idx2], yin[idx2],1)
        xm, ym = t.evaluate(xr[idx1], yr[idx1])
        xres = xm - xin[idx2]
        yres = ym - yin[idx2]
        print('6 param residual', np.std(xres), np.std(yres))
        #import pdb;pdb.set_trace()
    
    elif fittype=='quadratic':
        t = transforms.polytransform(xr[idx1], yr[idx1], xin[idx2], yin[idx2],2) 
    elif fittype=='cubic':
        t = transforms.polytransform(xr[idx1], yr[idx1], xin[idx2], yin[idx2],3) 


    #coo_r = np.sqrt(xref**2 + yref**2)
    #refcut = coo_r < 8000
    #xrefn = xref[refcut]
    #yrefn = yref[refcut]
    
    xn, yn = t.evaluate(xr, yr)
    #import pdb;pdb.set_trace()
    return xin[idx2], yin[idx2], xn[idx1], yn[idx1], xr[idx1], yr[idx1], t

        
def compare2square(xin, yin, fit_ord=1,printcat=False, trim=True, gspace=180, ang=0):
    '''
   takes in coordinats of pinhole images 
    creates a square grid, matches the square grid to the measured positions using a linear tranformation (6 parameter)
    returns reference positions that are matched to the input positions.
    note -- all input positions should have a match, may require precleaning on input catalogs.

    subfunction return refernce locations
    '''

    
    #get the point closest to the median of x and y
    ord2arg = {0:'four',1:'linear', 2:'quadratic', 3:'cubic'}
    xnin, ynin, xs, ys, xr, yr, t = mkref(xin, yin, fittype=ord2arg[fit_ord], trim=trim, gspace=gspace, ang=ang)

    #now we have the reference coordinates, the next choice is to fit the distortion....
    if printcat:
        outab = table()
        outab['x'] = xnin
        outab['y'] = ynin
        outab['xr'] = xr[idx1]
        outab['yr'] = yr[idx1]
        outab.write('matched.txt', format='ascii.fixed_width')
        
    xres = xnin - xr
    yres = ynin - yr

    return xr, yr, xnin, ynin, xres, yres, t 



def mkquiver(xnin, ynin, xres, yres, title_s='Residual Distortion', save_f='residual_distortion.png', fig_n=1, scale=20, scale_size=1, frac_plot=3, incolor='black', clear=True, xl=-1000, xh=9000, yl=0, yh=8200, scx=3250, scy=100, xlab='X (pixels)', ylab='Y (pixels)'):
    '''
    '''
    plt.figure(fig_n)
    if clear:
        plt.clf()
    plt.title(title_s)
    q = plt.quiver(xnin[::frac_plot], ynin[::frac_plot], xres[::frac_plot], yres[::frac_plot], scale=scale, width=0.0022, color=incolor)
    qk = plt.quiverkey(q,scx, scy, scale_size, str(round(6000.0*scale_size, 3))+' nm', coordinates='data', color='green')
    plt.xlim(xl, xh)
    plt.ylim(yl, yh)
    plt.axes().set_aspect('equal')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
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
        import pdb;pdb.set_trace()
        
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
    t1 = transforms.LegTransform(xmT, ymT, xresT, yresT, 1)
    t2 = transforms.LegTransform(xmT, ymT, xresT, yresT, 2)
    t3 = transforms.LegTransform(xmT,ymT,xresT, yresT, 3)
    t4 = transforms.LegTransform(xmT,ymT,xresT, yresT, 4)
    #eval_fit(t, fn='model_high20par.fits', subr=False)
    
    xmRes1, ymRes1 = t1.evaluate(xmT, ymT)
    xmRes2, ymRes2 = t2.evaluate(xmT, ymT)
    xmRes3, ymRes3 = t3.evaluate(xmT, ymT)
    xmRes4, ymRes4 = t4.evaluate(xmT, ymT)

    xfiterr1 = xresT - xmRes1
    yfiterr1 = yresT - ymRes1

    xfiterr2 = xresT - xmRes2
    yfiterr2 = yresT - ymRes2

    xfiterr3 = xresT - xmRes3
    yfiterr3 = yresT - ymRes3

    xfiterr4 = xresT - xmRes4
    yfiterr4 = yresT - ymRes4

    print('Number of measurements'+str(len(xresT)))
    print('Mean precision x '+ str(np.mean(xerr)*6)+' microns')
    print('Mean precision y '+ str(np.mean(yerr)*6)+' microns')

    print('Mean Residual 6 parameter x '+ str(np.std(xfiterr1)*6)+' microns')
    print('Mean Residual 6 parameter y '+ str(np.std(yfiterr1)*6)+' microns')

    
    print('Mean Residual 10 parameter x '+ str(np.std(xfiterr2)*6)+' microns')
    print('Mean Residual 10 parameter y '+ str(np.std(yfiterr2)*6)+' microns')

    
    print('Mean Residual 20 parameter x '+ str(np.std(xfiterr3)*6)+' microns')
    print('Mean Residual 20 parameter y '+ str(np.std(yfiterr3)*6)+' microns')

    print('Mean Residual 30 parameter x '+ str(np.std(xfiterr4)*6)+' microns')
    print('Mean Residual 30 parameter y '+ str(np.std(yfiterr4)*6)+' microns')

    print('Mean Residual linear x '+ str(np.std(xresT)*6)+' microns')
    print('Mean Residual linear y '+ str(np.std(yresT)*6)+' microns')

    plt.figure(10)
    plt.clf()
    plt.hist(xresT*6, bins=25, histtype='step', color='red', lw=3, label='x linear')
    plt.hist(yresT*6, bins=25, histtype='step', color='yellow',lw=3, label='y linear')
    plt.hist(xfiterr2*6, bins=25, histtype='step', color='blue', lw=3, label='x 6 par')
    plt.hist(yfiterr2*6, bins=25, histtype='step', color='black',lw=3, label='y 6 par')
    
    
    plt.hist(xfiterr2*6, bins=25, histtype='step', color='blue', lw=3, label='x 10 par')
    plt.hist(yfiterr2*6, bins=25, histtype='step', color='black',lw=3, label='y 10 par')
    plt.hist(xfiterr3*6, bins=25, histtype='step', color='blue', lw=3, label='x 20 par')
    plt.hist(yfiterr3*6, bins=25, histtype='step', color='black',lw=3, label='y 20 par')
    plt.hist(xfiterr4*6, bins=25, histtype='step', color='green', lw=3, label='x 30 par')
    plt.hist(yfiterr4*6, bins=25, histtype='step', color='purple',lw=3, label='y 30 par')
    #plt.hist(xres*6, bins=25, histtype='step', color='green', lw=3, label='x linear')
    #plt.hist(yres*6, bins=25, histtype='step', color='purple', lw=3, label='y linear')
    plt.legend(loc='upper right')
    plt.title('Distortion Residual')
    plt.xlabel('Residual (microns)')
    plt.ylabel('N')

    plt.savefig('residual_hist.png')

    mkquiver(xmT, ymT, xresT, yresT, title_s='Measured Distortion', save_f='res_dist_lin.png', fig_n=1, scale=10,frac_plot=5 )
    mkquiver(xmT, ymT, xmRes3-xresT, ymRes3-yresT, title_s='Residual Distortion', save_f='res_dist20par.png', fig_n=2, scale=1, scale_size=.1, frac_plot=5)
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
    
def calc_var_n(ar_in, stars_index=None, ind=False):
    '''
    takes array of star positions dimensions (nframes, nstars)
    splits the data into random groups ranging from N=1 to N = Nstars/3
    
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
        mean_pos = []
        while np.sum(imbool) >= nn:
            #we want to take sequential frames
            g_index = index_ar[imbool][:nn]
            imbool[g_index] = False
            #now compute the average positions from this group
            _data = ar_in[g_index,:]
            #now we trim down to only stars we are interested in, per optional arguement stars_index
            _data = _data[:,stars_index]
            #compute sigma clipped mean of the all the error on the means with this number of samples
            #could just switch to a median...
            mean_pos.append(np.mean(_data, axis=0))
            #compute manual error on the mean, to make sure I understand what the python versions are doing
        #import pdb;pdb.set_trace()
        mean_pos = np.array(mean_pos)
        _rms_manual = np.mean(np.sqrt(1./(mean_pos.shape[0]-1)*np.sum((np.mean(mean_pos,axis=0)-mean_pos)**2, axis=0)))
        _allan_ar = np.zeros(mean_pos.shape[1])
            #first compute the sum of the allan variance equation
        for ii in range(mean_pos.shape[0]-1):
            _allan_ar += (mean_pos[ii,:] - mean_pos[ii+1, :])**2
        _allan_ar = np.sqrt(1./(2.*(mean_pos.shape[0]-1.)) * _allan_ar)
        _allan, _allan_err = statsIter.mean_std_clip(_allan_ar)
        _avar.append(_allan)
        _std.append(_rms_manual)
        if not ind:    
            avartot.append(np.mean(_avar))
            avartot_err.append(np.std(_avar))
            stdtot.append(np.mean(_std))
            stdtot_err.append(np.std(_std))
        else:
            avartot.append(_allan_ar)
            #avartot_err.append(_avar)
            stdtot.append(_std)
            #stdtot_err.append(np.std(_std))
            

    #import pdb;pdb.set_trace()
    if not ind:
        return nim, avartot, avartot_err, stdtot, stdtot_err
    else:
        return nim, avartot, stdtot
def test_var():
    #create reference "positoins"
    x = np.ones((200,20))
    _norm = scipy.stats.norm(loc=0 , scale=1.0/6.0)
    err = _norm.rvs(len(x.flatten()))
    x = x + np.reshape(err, x.shape)
    #import pdb;pdb.set_trace()
    nim, avartot, avartot_err, stdtot, stdtot_err = calc_var_n(x)

    plt.figure(18)
    plt.clf()
    plt.loglog(nim, avartot, label='allan 100')
    plt.loglog(nim, stdtot, label='rms 100')


    x  = np.ones((600,20))
    _norm = scipy.stats.norm(loc=0 , scale=1.0/6.0)
    err = _norm.rvs(len(x.flatten()))
    x = x + np.reshape(err, x.shape)
    #import pdb;pdb.set_trace()
    nim, avartot, avartot_err, stdtot, stdtot_err = calc_var_n(x)

    
   
    plt.loglog(nim, avartot, label='allan 600')
    plt.loglog(nim, stdtot, label='rms 600')
    
    plt.loglog(nim, 1.0/np.array(nim)**0.5/6.0, color='black')
    plt.legend(loc='upper right')
    plt.title('Deviation Test')
    plt.xlabel('N')
    plt.ylabel('Deviation (pixels)')
    plt.tight_layout()
    plt.savefig('var.png')
    
def plot_var(inlis='lis.lis', stars_index=None):
    xall, yall, fall = match_pin.match_all(inlis)
    xshift, yshift, dx, dy = match_pin.shift_and_ave_coo(xall, yall)
    x, y, xerr, yerr, mask = mkave(lisf=inlis, retmask=True )
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
    
    plt.loglog(xnim, 1.0/np.array(xnim)/3, color='black')
    plt.legend(loc='upper right')
    plt.title('Deviation Test')
    plt.xlabel('N')
    plt.ylabel('Deviation (pixels)')
    plt.savefig('var.png')

def plot_var_from_ar(xshift, yshift, fall , stars_index=None, print_off=True, lin_fit=True, pscale=6000):

    xave = np.median(xshift, axis=1)
    yave = np.median(yshift, axis=1)
    #xn = xshift
    #yn = yshift

    #we are going to do something that maybe a bit silly --- we need to get rid of variations between each catalog beyond 4 parameter, so we fit linear transformations to the average 
    
    xn = np.zeros(xshift.shape)
    yn = np.zeros(yshift.shape)
    a = [[],[],[]]
    b = [[],[],[]]
    for i in range(xshift.shape[1]):
        if lin_fit:
            t = transforms.PolyTransform.derive_transform(xshift[:,i], yshift[:,i], xave, yave, 1)
        else:
            t = transforms.four_paramNW(xshift[:,i], yshift[:,i], xave, yave, 1)
            #make this a fit that only allows for rotation and  translation...
            #params = affine.fit_rot_trans(np.array([xshift[:,i],yshift[:,i]]).T, np.array([xave, yave]).T)
            #print(params)
            #ang = np.deg2rad(params['angle'])
            #_x =  params['transX'] + np.cos(ang) * xshift[:,i] - np.sin(ang) * yshift[:,i]
            #_y =  params['transY'] + np.sin(ang) * xshift[:,i] + np.cos(ang) * yshift[:,i]

        _x, _y = t.evaluate(xshift[:,i], yshift[:,i])
        xn[:,i] = _x
        yn[:,i] = _y
        print(np.std(_x-xave), np.std(_y-yave))

    xave = np.median(xn, axis=1)
    yave = np.median(yn, axis=1)

    for i in range(xshift.shape[1]):
        if lin_fit:
            t = transforms.PolyTransform.derive_transform(xshift[:,i], yshift[:,i], xave, yave, 1)
        else:
            t = transforms.four_paramNW(xshift[:,i], yshift[:,i], xave, yave, 1)
            #make this a fit that only allows for rotation and  translation...
            #params = affine.fit_rot_trans(np.array([xshift[:,i],yshift[:,i]]).T, np.array([xave, yave]).T)
            #print(params)
            #ang = np.deg2rad(params['angle'])
            #_x =  params['transX'] + np.cos(ang) * xshift[:,i] - np.sin(ang) * yshift[:,i]
            #_y =  params['transY'] + np.sin(ang) * xshift[:,i] + np.cos(ang) * yshift[:,i]

        _x, _y = t.evaluate(xshift[:,i], yshift[:,i])
        xn[:,i] = _x
        yn[:,i] = _y
        print(np.std(_x-xave), np.std(_y-yave))
    
        #import pdb;pdb.set_trace()
        if lin_fit:
            for kk in range(len(a)):
                a[kk].append(t.px.parameters[kk])
                b[kk].append(t.py.parameters[kk])
        else:
            for i in range(len(a)):
                a[i].append(t.px[i])
                b[i].append(t.py[i])
            #a[0].append(params['transX'])
            #a[1].append(params['transY'])
            #a[2].append(params['angle'])
            #b[0].append(params['transX'])
            #b[1].append(params['transY'])
            #b[2].append(params['angle'])
            
    _outtab = Table()
    
    _outtab = Table()
    for i in range(len(a)):
        _outtab['a'+str(i)] = a[i]
    for i in range(len(b)):
        _outtab['b'+str(i)] = b[i]
    if print_off and lin_fit:
            _outtab.write('var_trans.txt', format='ascii.fixed_width')
    elif print_off:
        _outtab.write('var_trans_4p.txt', format='ascii.fixed_width')
        #_outtab.write('var_trans_rot_trans.txt', format='ascii.fixed_width')

    xnim, xavar, xavarerr, xrms, xrmserr = calc_var_n(xn.T, stars_index=stars_index)
    ynim, yavar, yavarerr, yrms, yrmserr = calc_var_n(yn.T, stars_index=stars_index)
    fnim, favar, favarerr, frms, frmserr = calc_var_n(fall.T, stars_index=stars_index)

    print('pixel scale is ', pscale, ' nm')
    plt.figure(18)
    plt.clf()
    plt.loglog(xnim, np.array(xavar)*pscale, label='allan x')
    plt.loglog(xnim,  np.array(xrms)*pscale, label='rms x')
    plt.loglog(ynim,  np.array(yavar)*pscale, label='allan y')
    plt.loglog(ynim,  np.array(yrms)*pscale, label='rms y')
    
    plt.loglog(xnim, 1.0/(np.array(xnim))**0.5 * 260, color='black')
    plt.legend(loc='upper right')
    plt.title('Deviation Test')
    plt.xlabel('N')
    plt.ylabel('Deviation (nm)')
    plt.savefig('var.png')

    plt.figure(19)
    plt.clf()
    plt.loglog(fnim, favar, label='allan')
    plt.loglog(fnim, frms, label='rms')
    plt.legend(loc='upper right')
    plt.ylabel('Deviation (counts)')
    plt.xlabel('N')
    plt.tight_layout()
    plt.savefig('var_flux.png')

    plot_ind = False
    if plot_ind:
        xnim, xavar, xrms = calc_var_n(xn.T, stars_index=stars_index, ind=True)
        ynim, yavar, yrms= calc_var_n(yn.T, stars_index=stars_index, ind=True)
        fnim, favar, frms = calc_var_n(fall.T, stars_index=stars_index, ind=True)

        plt.figure(118)
        plt.clf()
        plt.figure(119)
        plt.clf()
        
        #import pdb;pdb.set_trace()
        xavar = np.array(xavar)
        xrms = np.array(xrms)
        yavar = np.array(yavar)
        yrms = np.array(yrms)
        favar = np.array(favar)

        plt.loglog(xnim, 1.0/(np.array(xnim))**0.5 * 260, color='black')
        plt.legend(loc='upper right')
        plt.title('Deviation Test')
        plt.xlabel('N')
        plt.ylabel('Deviation (nm)')
        
        
        for jj in range(xavar.shape[1]):
            #import pdb;pdb.set_trace()
            plt.figure(118)
            plt.loglog(xnim, xavar[:,jj]*6000, label='allan x')
            #plt.loglog(xnim,  xrms[:,jj]*6000, label='rms x')
            plt.loglog(ynim,  yavar[:,jj]*6000, label='allan y')
            #plt.loglog(ynim,  yrms[:,jj]*6000, label='rms y')
    
        
            plt.figure(119)
            plt.loglog(fnim, favar[:,jj], label='allan')
            #plt.loglog(fnim, frms, label='rms')
            plt.legend(loc='upper right')
            plt.ylabel('Deviation (counts)')
            plt.xlabel('N')
            plt.tight_layout()
        
        plt.savefig('var_flux.png')
        plt.figure(118)
        plt.savefig('var.png')
    
    return xn, yn

def plot_ind_shift(xshift, yshift, fall, stars_index=None, pscale=9000):
    '''
    '''

    xave = np.mean(xshift, axis=1)
    yave = np.mean(yshift, axis=1)
    xn = xshift
    yn = yshift

    dx = np.mean(xn[:,0:15], axis=1) - np.mean(xn[:,-15:], axis=1)
    dy = np.mean(yn[:,0:15], axis=1) - np.mean(yn[:,-15:], axis=1)

    #now cut out stars with any data point more than 2 pix from the average

    tbool = np.ones(xshift.shape[0], dtype='bool')
    for i in range(xshift.shape[0]):
        if np.max(np.abs(xshift[i,:] - np.mean(xshift[i,:]))) > 100 or np.max(np.abs(yshift[i,:] - np.mean(yshift[i,:]))) > 100 :
            tbool[i] = False
        #if np.max(np.abs(xshift[i,:] - np.mean(xshift[i,:]))) < .15 or np.max(np.abs(yshift[i,:] - np.mean(yshift[i,:]))) < 0.15 :
        #    tbool[i] = False
    print(np.sum(tbool), 'out of ',len(tbool), 'stars remain')
    plt.figure(47)
    plt.clf()
    #plt.subplot(211)
    plt.title('Shift in each pinhole')
    q = plt.quiver(xave[tbool],yave[tbool], dx[tbool], dy[tbool], scale=0.25, width=0.0022, color='black')
    qk = plt.quiverkey(q, 500, 5500, .0125, str(round(pscale*.0125, 3))+' nm', coordinates='data', color='green')
    plt.xlim(0, 8000)
    plt.ylim(0, 6500)
    plt.axes().set_aspect('equal')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')

    plt.figure(48)
    plt.clf()
    #plt.subplot(212)
    plt.title('Deviation in Shift in each pinhole')
    q = plt.quiver(xave[tbool],yave[tbool], dx[tbool]-np.mean(dx[tbool]), dy[tbool]-np.mean(dy[tbool]), scale=0.25, width=0.0022, color='black')
    qk = plt.quiverkey(q, 500, 5500, .0125, str(round(pscale*.0125, 3))+' nm', coordinates='data', color='green')
    plt.xlim(0, 8000)
    plt.ylim(0, 6500)
    plt.axes().set_aspect('equal')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')

    plt.figure(49)
    plt.scatter(xave[tbool], yave[tbool], s=2)
    plt.xlim(0,8000)
    plt.ylim(0,6500)

    xn, yn = plot_var_from_ar(xshift[tbool,:], yshift[tbool,:],fall[tbool,:], stars_index=stars_index, pscale=pscale)
    return xn, yn, xshift[tbool, :], yshift[tbool,:]

def pix_phase(xn, yn, xall, yall, save_off=True):
    '''
    Use the linearly corrected pixels to establish difference (from the average)
    round the xall/yall pixel values to get the location on the pixel that the star was originally detected
    '''
    xave = np.median(xn, axis=1)
    yave = np.median(yn, axis=1)

    dx = np.zeros(xn.shape)
    dy = np.zeros(yn.shape)

    for i in range(xall.shape[0]):
        dx[i,:] = xave[i] - xn[i,:]
        dy[i,:] = yave[i] - yn[i,:]
        
    ph_x = xall - np.floor(xall)
    ph_y = yall - np.floor(yall)


    #now we need 2d avarages of these.  that is for each value of x/y we need the average dx and dy
    xspace = np.linspace(0, 1, num=21)
    yspace = np.linspace(0, 1, num=21)
    dx2d = np.zeros((len(xspace)-1, len(yspace)-1))
    dy2d = np.zeros((len(xspace)-1, len(yspace)-1))
    
    for ii in range(len(xspace)-1):
        for jj in range(len(yspace)-1):
            _sbool = (ph_x > xspace[ii]) * (ph_x <= xspace[ii+1]) * (ph_y <= yspace[jj+1]) * (ph_y > yspace[jj])
            assert np.sum(_sbool) > 20
            dx2d[jj,ii] = np.median(dx[_sbool]) * 6000
            dy2d[jj,ii] = np.median(dy[_sbool]) * 6000

    dx1d = np.mean(dx2d, axis=0)
    dy1d = np.mean(dy2d, axis=1)
    
    if save_off:
        np.save('dx.npy', dx2d / 6000.0)
        np.save('dy.npy', dy2d / 6000.0)
        np.save('dx1d.npy', dx1d / 6000.0)
        np.save('dy1d.npy', dy1d / 6000.0)
        np.save('x_incr.npy', xspace)
        np.save('y_incr.npy', yspace)

        #make 1d version
        
    plt.figure(1)
    plt.clf()
    plt.gray()
    plt.imshow(dx2d, interpolation='none', origin='lower')
    plt.colorbar()
    plt.title('X offset (nm)')
    plt.tight_layout()

    plt.figure(2)
    plt.clf()
    plt.imshow(dy2d, interpolation='none', origin='lower')
    plt.colorbar()
    plt.title('Y offset (nm)')
    plt.tight_layout()

    plt.figure(3)
    plt.clf()
    plt.hist(dx.flatten()*6000.0, bins=21,range=(-300,300), histtype='step', lw=3, label='x')
    plt.hist(dy.flatten()*6000.0, bins=21,range=(-300,300), histtype='step', lw=3, label='y')
    plt.legend(loc='upper right')
    plt.ylabel('N')
    plt.xlabel(r'$\Delta$ (nm)')
    plt.tight_layout()

    plt.figure(4)
    plt.clf()
    plt.scatter(np.linspace(0,1,num=len(dx1d)),dx1d, label='x')
    plt.scatter(np.linspace(0,1,num=len(dx1d)),dy1d, label='y')
    plt.legend(loc='upper left')
    plt.ylabel('Offset (nm)')
    plt.xlabel('Fractional Pixel location')
    plt.title('Pixel Phase Measurments')
    plt.tight_layout()
    
def app_pix_corr(xin, yin):
    ph_x = xin - np.floor(xin)
    ph_y = yin - np.floor(yin)

    xspace = np.load('x_incr.npy')
    yspace = np.load('y_incr.npy')

    dx = np.load('dx1d.npy')
    dy = np.load('dy1d.npy')

    xn = xin[:,:]
    yn = yin[:,:]
    #for ii in range(len(xspace)-1):
    #    for jj in range(len(yspace)-1):
    #        _sbool = (ph_x > xspace[ii]) * (ph_x <= xspace[ii+1]) * (ph_y <= yspace[jj+1]) * (ph_y > yspace[jj])
           
    #        xn[_sbool] = xn[_sbool] + dx[jj,ii]
    #        yn[_sbool] = yn[_sbool] + dy[jj,ii]
    for ii in range(len(xspace)-1):
        _sbool = (ph_x > xspace[ii]) * (ph_x <= xspace[ii+1])
        xn[_sbool] = xn[_sbool] + dx[ii]
    for jj in range(len(yspace)-1):
        _sbool = (ph_y <= yspace[jj+1]) * (ph_y > yspace[jj])
        yn[_sbool] = yn[_sbool] + dy[jj]
    
    return xn, yn

def plot_all_trans(xshift, yshift):
    '''
    '''

    plt.figure(1)
    plt.clf()
    plt.title('X shifts of all pinholes')
    plt.figure(2)
    plt.clf()
    plt.title('Y shifts of all pinholes')
    for i in range(xshift.shape[0]):
        plt.figure(1)
        plt.plot(xshift[i,:] - np.mean(xshift[i,:]))
        plt.figure(2)
        plt.plot(yshift[i,:] - np.mean(yshift[i,:]))

    plt.figure(1)
    plt.xlabel('Frame Number')
    plt.ylabel('Deviation from mean (pix)')
    plt.figure(2)
    plt.xlabel('Frame Number')
    plt.ylabel('Deviation from mean (pix)')

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
    t = transforms.PolyTransform.derive_transform(_ref['xm'], _ref['ym'], _ref[''], _ref[''])
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

def fit_ind_four(inlis='lis.lis', writeref=True, gspace=170):

    intab = Table.read(inlis, format='ascii.no_header')

    t = []
    params = [[],[],[],[]]
    a = [[],[],[]]
    b = [[],[],[]]
    
    for i, _ff in enumerate(intab['col1']):

        _tab = Table.read(_ff.replace('*',''), format='ascii.fixed_width')
        cut = _tab['corr'] > .7
        cutf = _tab['flux'] > 90000
        _tab = _tab[cut*cutf]
        if i == 0:
            refx = _tab['x']
            refy = _tab['y']
        _xin, _yin, _xr, _yr, _xsquare, _ysquare, _t = mkref(_tab['x'], _tab['y'], trim=True, gspace=gspace)
        _xin, _yin, _xlinr, _ylinr, _xsquare, _ysquare, _tlin = mkref(_tab['x'], _tab['y'], fittype='linear', trim=True, gspace=gspace)
        
        
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

        #also create translated coordinates,
        idx1 , idx2 , dm, dr = align.match.match(refx, refy, np.ones(len(refx)),  _xin, _yin, np.ones(len(_xin)), 10)
        _dx, _dxerr = stats.mean_std_clip(refx[idx1]-_xin[idx2])
        _dy, _dyerr = stats.mean_std_clip(refy[idx1]-_yin[idx2])
        _xt = _xin + _dx
        _yt = _yin + _dy
        print('offsets ', _dx, _dy)

        _outtab = Table(names=['xm', 'ym','xt', 'yt', 'xr','yr', 'xlinr', 'ylinr','xsquare', 'ysquare'], data=[_xin, _yin ,_xt, _yt, _xr, _yr, _xlinr, _ylinr, _xsquare, _ysquare])
        assert '.lis' in _ff
        if writeref:
            _outtab.write(_ff.replace('.lis', '_ref.lis').replace('*',''), format='ascii.fixed_width')

    
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
def fit_ind_cube(inlis='lis.lis', writeref=True):

    intab = Table.read(inlis, format='ascii.no_header')

    t = []
    params = [[],[],[],[]]
    a = [[],[],[]]
    b = [[],[],[]]

    xres = []
    yres = []
    for i, _ff in enumerate(intab['col1']):

        _tab = Table.read(_ff.replace('*',''), format='ascii.fixed_width')
        cut = _tab['corr'] > .92
        _tab = _tab[cut]
        if i == 0:
            refx = _tab['x']
            refy = _tab['y']
        #_xin, _yin, _xr, _yr, _xsquare, _ysquare, _t = mkref(_tab['x'], _tab['y'], trim=False)
        _xin, _yin, _xlinr, _ylinr, _xsquare, _ysquare, _tlin = mkref(_tab['x'], _tab['y'], fittype='cubic', trim=False)
        #_xmod, _ymod = _tlin.evaluate(_tab['x'], _tab['y'])
        _xres = _xin - _xlinr
        _yres = _yin - _ylinr
        xres.append(np.std(_xres))
        yres.append(np.std(_yres))
    
        
        
    
    return xres, yres

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
    print('Number of sources trimmed: ', len(gbool)-np.sum(gbool), ' of ', len(gbool))
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
            print('number of stars in this section are ', np.sum(sbool))
            #find the mean delta and sigma in x and y, using iterative sigma clipping
            ave_x, sig_x, nclipx = statsIter.mean_std_clip(dx[sbool], clipsig=3.0, return_nclip=True)
            ave_y, sig_y, nclipy = statsIter.mean_std_clip(dy[sbool], clipsig=3.0, return_nclip=True)
            #creates boolean 
            good_bool = (dx < ave_x + sig_fac * sig_x)*(dx > ave_x - sig_fac * sig_x)*(dy < ave_y + sig_fac *sig_x)*(dy > ave_y - sig_fac * sig_x) * sbool
            print('number of stars cut ', np.sum(sbool) - np.sum(good_bool) )
            
            for ii in range(np.sum(good_bool)):
                xout.append(x[good_bool][ii])
                yout.append(y[good_bool][ii])
                dxout.append(dx[good_bool][ii])
                dyout.append(dx[good_bool][ii])

    #import pdb;pdb.set_trace()
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

def plot4lin(incat='ind_fits.txt', cut=True):
    '''
    '''

    cat = Table.read(incat, format='ascii.fixed_width')

    #cut out stuff
    if cut:
        gbool = (np.abs(cat['a1']-np.mean(cat['a1'])) < 0.00002) *  (np.abs(cat['a2']-np.mean(cat['a2'])) < 0.00002) *  (np.abs(cat['b1']-np.mean(cat['b1'])) < 0.00002) *  (np.abs(cat['b2']-np.mean(cat['b2'])) < 0.00002) * (np.abs(cat['a0'] - np.mean(cat['a0'])) < 0.15 ) * (np.abs(cat['b0'] - np.mean(cat['b0'])) < 0.15 )
        bbool = np.ones(len(cat), dtype='bool')
        bbool[gbool] = False
        print('bad indexes', np.array(range(len(cat)))[bbool])
    else:
        gbool = np.ones(len(cat), dtype='bool')
    plt.figure(37)
    plt.clf()
    plt.title('Variation in linear parameters')
    plt.xlabel('Frame Number')
    plt.ylabel('Coefficient Value')
    plt.plot(cat['a1'][gbool]-np.mean(cat['a1'][gbool]), label='a1', lw=.3)
    plt.plot(cat['a2'][gbool]-np.mean(cat['a2'][gbool]), label='a2', lw=.3)
    plt.plot((cat['b1'][gbool]-np.mean(cat['b1'][gbool])), label='b1', lw=.3)
    plt.plot(cat['b2'][gbool]-np.mean(cat['b2'][gbool]), label='b2', lw=.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('sixparam.png')


    plt.figure(38)
    plt.clf()
    plt.title('Transformation Offsets')
    plt.xlabel('Frame Number')
    plt.ylabel('Offset - average Offset (nm)')
    plt.plot((cat['a0'][gbool]-np.mean(cat['a0'][gbool]))*6000, label='x', lw=.3)
    plt.plot((cat['b0'][gbool]-np.mean(cat['b0'][gbool]))*6000, label='y', lw=.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('sigparam_offsets.png')

    if cut:
        return gbool
def match_all(lis_f, xkey='x', ykey='y'):

    lis_lis = Table.read(lis_f, format='ascii.no_header')['col1']

   

    _ref = Table.read(lis_lis[0], format='ascii.fixed_width')
    xr = _ref[xkey]
    yr = _ref[ykey]
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

    return x, y,flux

def shift_and_ave_coo(xall, yall):
    xshift = np.zeros(xall.shape)
    yshift = np.zeros(yall.shape)
    dx = []
    dy = []

    for i in range(xall.shape[1]):
        _mbool = xall[:,i] != 0.0
        _dx, _dxerr = stats.mean_std_clip(xall[:,0][_mbool]-xall[:,i][_mbool])
        _dy, _dyerr = stats.mean_std_clip(yall[:,0][_mbool]-yall[:,i][_mbool])
        #_dy = np.mean(yall[:,0][_mbool]-yall[:,i][_mbool])
        xshift[:,i] = xall[:,i] + _dx
        yshift[:,i] = yall[:,i] + _dy
        dx.append(_dx)
        dy.append(_dy)

    return xshift, yshift, dx, dy

def mkpsf(inlis='psf.lis'):
    '''
    '''
    intab = Table.read(inlis, format='ascii.no_header')
    psfdat = []
    psf_ave = np.zeros((30,30))

    for i in range(len(intab)):
        _tmp = fits.open(intab['col1'][i])[0].data
        psf_ave += _tmp
    psf_ave = psf_ave / (1.0 * len(intab))
    
    #psfave = np.median(psfdat, axis=0)
    psfres = np.ones((len(intab), psf_ave.shape[0], psf_ave.shape[1]))

    for i in range(len(intab)):
        _tmp = fits.open(intab['col1'][i])[0].data
        psfres[i,:] = _tmp - psf_ave
        

    fig = plt.figure(25)
    #import pdb;pdb.set_trace()
    plt.clf()
    an_ims = []
    plt.gray()
    for i in range(psfres.shape[0]):
        plt.title('Fractional Residual')
        im = plt.imshow(psfres[i,:]/(1.0*psf_ave), interpolation='none')
        plt.colorbar()
        plt.axis('off')
        an_ims.append([im])
        plt.clf()
    ani = manimation.ArtistAnimation(fig, an_ims, interval=300, blit=True)
    ani.save('psf_mov.mp4')



