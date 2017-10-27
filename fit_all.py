from . import analyze_stack, match_pin
from flystar.flystar import align, transforms, match
from astropy.table import Table 
import numpy as np
import matplotlib.pyplot as plt 
from jlu.util import statsIter as stats
from astropy.io import fits
from scipy.optimize import minimize, leastsq

def stack_fit(subgroups=10, xkey='xt', ykey='yt', useref=True, infile='/Users/service/Pinhole/data/20170429/combo/lisO.lis'):
    '''
    Stacks the data from each pinhole maks location into "subgroups" groups only eliminating the mean translation
    then fits a 6 parameter fit to the distortion from each stack.  
    Then fits a final higher order fit to all deltas at the end
    '''

    #this file contains the list of text files with the catalogs for each mask position
    #dirtab = Table.read('/Users/service/Pinhole/data/20170429/combo/lisO.lis', format='ascii.no_header')
    #dirtab = Table.read('/Volumes/DATA/DISTORTION/20170530/combo/lis0.lis', format='ascii.no_header')
    dirtab = Table.read(infile, format='ascii.no_header')
    outdat = [[],[],[],[]]
    str_f = []
    xpre = []
    ypre = []
    for _tab_f in dirtab['col1']:
        _tab = Table.read(_tab_f.replace('*', ''), format='ascii.no_header')
        #now we need to split into groups
        if subgroups == None:
            _fgroups = np.split(_tab['col1'], len(_tab['col1']))
        else:
            _fgroups = np.split(_tab['col1'], subgroups)
        for i in range(len(_fgroups)):
            #now for each subgroup we need to take the already translated coordiantes, match them and average them
            #first read in first catalog to use as "reference"
            
            if useref:
                _tab = Table.read(_fgroups[i][0].replace('.lis', '_ref.lis').replace('*', ''), format='ascii.fixed_width')
            else:
                _tab = Table.read(_fgroups[i][0].replace('*', ''), format='ascii.fixed_width')
            xref = _tab[xkey]
            yref = _tab[ykey]
            _xall = np.zeros((len(xref), len(_fgroups[i])))-999999
            _yall = np.zeros((len(xref), len(_fgroups[i])))-999999
            _xall[:,0] = xref[:]
            _yall[:,0] = yref[:]
            for jj, _ff in enumerate(_fgroups[i][1:]):
                if useref:
                    _tab =  Table.read(_ff.replace('.lis', '_ref.lis').replace('*', ''), format='ascii.fixed_width')
                else:
                    _tab = Table.read(_ff.replace('*', ''), format='ascii.fixed_width')
                dx1 , idx2 , dm, dr = align.match.match(_tab[xkey], _tab[ykey],np.ones(len(_tab[xkey])), xref, yref, np.ones(len(xref)), 20)
                assert len(idx1) > 300
                _xall[idx2, jj+1] = _tab[xkey][idx1]
                _yall[idx2, jj+1] = _tab[ykey][idx1]


            _mask = _xall < -9999
            xm = np.ma.array(_xall, mask=_mask)
            ym = np.ma.array(_yall, mask=_mask)

            xave = np.mean(xm, axis=1)
            yave = np.mean(ym, axis=1)

            print('stacking precision X (pix)', str(np.mean(np.std(xm, axis=1))))
            print('stacking precision Y (pix)', str(np.mean(np.std(ym, axis=1))))
            
            xpre.append(np.std(xm, axis=1))
            ypre.append(np.std(ym, axis=1))

            #xpre = np.std(xm, axis=1)
            #ypre = np.std(ym, axis=1)
            

            #now I need to get the distortion free positions (with linear terms fit out) for these points....
            _xdat, _ydat, _xref, _yref, _xsquare, _ysquare, t = analyze_stack.mkref(xave, yave, fittype='linear', trim=False)
            #xn, yn, xnin, ynin, xres, yres, t =  analyze_stack.compare2square(xave, yave, fit_ord=1)
            #import pdb;pdb.set_trace()
            #import pdb;pdb.set_trace()
            for kk in range(len(_xdat)):
                outdat[0].append(_xdat[kk])
                outdat[1].append(_ydat[kk])
                outdat[2].append(_xref[kk])
                outdat[3].append(_yref[kk])
                str_f.append(_tab_f)

    outdat = np.array(outdat)
    tfinal = transforms.LegTransform(outdat[0], outdat[1], outdat[2]-outdat[0], outdat[3]-outdat[1], 3)
    modx, mody = tfinal.evaluate(outdat[0], outdat[1])
    resx = outdat[2] - outdat[0] - modx
    resy = outdat[3] - outdat[1] - mody

    dumX = []
    dumY = []
    for i in range(len(xpre)):
        for j in range(len(xpre[i])):
            dumX.append(xpre[i][j])
            dumY.append(ypre[i][j])
            print('mean stacking precisions')
    
    return outdat, resx, resy, tfinal, xpre, ypre, str_f

def mkhist(outdat, str_f):
    '''
    '''

    
    t1 = transforms.LegTransform(outdat[0], outdat[1], outdat[2]-outdat[0], outdat[3]-outdat[1], 1)

    t2 = transforms.LegTransform(outdat[0], outdat[1], outdat[2]-outdat[0], outdat[3]-outdat[1], 2)
    modx, mody = t2.evaluate(outdat[0], outdat[1])
    resx2 = outdat[2] - outdat[0] - modx
    resy2 = outdat[3] - outdat[1] - mody
    print(np.std(resx2), np.std(resy2))

    t3 = transforms.LegTransform(outdat[0], outdat[1], outdat[2]-outdat[0], outdat[3]-outdat[1], 3)
    modx, mody = t3.evaluate(outdat[0], outdat[1])
    resx3 = outdat[2] - outdat[0] - modx
    resy3 = outdat[3] - outdat[1] - mody

    print(np.std(resx3), np.std(resy3))
    
    t4 = transforms.LegTransform(outdat[0], outdat[1], outdat[2]-outdat[0], outdat[3]-outdat[1], 4)
    modx, mody = t4.evaluate(outdat[0], outdat[1])
    resx4 = outdat[2] - outdat[0] - modx
    resy4 = outdat[3] - outdat[1] - mody

    print(np.std(resx4), np.std(resy4))
    #import pdb;pdb.set_trace()
    plt.figure(17)
    plt.clf()
    plt.hist(resx2*6, histtype='step', lw=2, label='x quadratic', bins=25)
    plt.hist(resy2*6, histtype='step', lw=2, label='y quadratic', bins=25)

    plt.hist(resx3*6, histtype='step', lw=2, label='x cubic', bins=25)
    plt.hist(resy3*6, histtype='step', lw=2, label='y cubic', bins=25)
    plt.xlabel('Data - Model (microns)')
    plt.ylabel('N')
    plt.legend(loc='upper left')
    plt.savefig('residual_dist_hist.png')
    #make up the booleans
    pos = np.unique(str_f)
    bbs = []
    for _pp in pos:
        bbs.append((_pp==str_f))
    colors = ['green', 'purple', 'black', 'red']
    plt.figure(2)
    #plt.clf()
    plt.figure(4)
    #plt.clf()
    for i in range(len(pos)):
       
        analyze_stack.mkquiver(outdat[0][bbs[i]], outdat[1][bbs[i]], outdat[2][bbs[i]]-outdat[0][bbs[i]], outdat[3][bbs[i]]-outdat[1][bbs[i]], fig_n=2, title_s='Measured High Order Distortion', scale=5, scale_size=0.3, frac_plot=2, save_f='measured_dist.png', incolor=colors[i], clear=False)
        #import pdb;pdb.set_trace()
       
        analyze_stack.mkquiver(outdat[0][bbs[i]], outdat[1][bbs[i]], resx4[bbs[i]], resy4[bbs[i]], fig_n=4, title_s='Data - Model (High order distortion)', scale=1, scale_size=0.1, frac_plot=2, save_f='residual_dist.png', incolor=colors[i], clear=False)
    

#def clean_lis(inlis='lis.lis'):
#    intab = Table.read(inlis, format='ascii.no_header')
#    for _ff in intab['col1']:

#        pass


def fit_ave(xncut, yncut,gspace):
    '''
    assumes that you hand in 6 paramerter corrected positions
    '''


    xave = np.mean(xncut, axis=1)
    yave = np.mean(yncut, axis=1)

    
    
    xr, yr, xnin, ynin, xres, yres, t  = analyze_stack.compare2square(xave , yave, trim=True, gspace=gspace)
    #ow return the measured positions
    outdat = np.array([xnin, ynin, xr, yr])
    return outdat
def final_fit(order=4, cut_line=True):
    '''
    '''
    #inlis = ['/Volumes/DATA/DISTORTION/20170711/Stability_2/obj_good.lis', '/Volumes/DATA/DISTORTION/20170707/Stability/obj_good.lis', '/Volumes/DATA/DISTORTION/20170718/pos2/obj_good.lis', '/Volumes/DATA/DISTORTION/20170720/pos3/obj_good.lis']
    inlis = ['/Users/service/data/distortion_cat/reduction_1/p0/average_coo.txt','/Users/service/data/distortion_cat/reduction_1/p1/average_coo.txt','/Users/service/data/distortion_cat/reduction_1/p2/average_coo.txt','/Users/service/data/distortion_cat/reduction_1/p3/average_coo.txt','/Users/service/data/distortion_cat/reduction_1/p4/average_coo.txt','/Users/service/data/distortion_cat/reduction_1/p5/average_coo.txt','/Users/service/data/distortion_cat/reduction_1/p6/average_coo.txt','/Users/service/data/distortion_cat/reduction_1/p7/average_coo.txt']
    #'/Users/service/data/distortion_cat/p1/average_coo.txt'
    #inlis = [inlis[1],inlis[2], inlis[3]]
    #inlis = ['/Users/service/data/distortion_cat/reduction_2/p1/average_coo.txt', '/Users/service/data/distortion_cat/reduction_2/p2/average_coo.txt','/Users/service/data/distortion_cat/reduction_2/p3/average_coo.txt','/Users/service/data/distortion_cat/reduction_2/p4/average_coo.txt']
    offsets = [(723.4, -1004.7), (724, 270),(1852.1, 277), (1849.5, -1115.5), (1849.5, -1115.5), (1856, 277), (477.0, 270.364), (504.8, -1250.0)]
    #offsets = [(-669, 1643), (0.0, 0.0), (0.0, 0.0), (-927.0, -274.0)]
    #offsets =  [offsets[0], offsets[3]]
    xlis = []
    xerr = []
    ylis = []
    yerr = []
    for _ff in inlis:
    #    xave, yave = readstack(_ff)
    #    xlis.append(xave)
    #    ylis.append(yave)
        _tab = Table.read(_ff, format='ascii.fixed_width')
        xlis.append(_tab['x'])
        xerr.append(_tab['xerr'])
        ylis.append(_tab['y'])
        yerr.append(_tab['yerr'])
        
    #make table for paper
    
    xoff = []
    yoff = []
    xerr = []
    yerr = []
    for i in range(len(inlis)):
        _tab = Table.read(inlis[i], format='ascii.fixed_width')
        xoff.append(offsets[i][0]*0.006)
        yoff.append(offsets[i][1]*0.006)
        xerr.append(np.mean(_tab['xerr'])*6000.0)
        yerr.append(np.mean(_tab['yerr'])*6000.0)
    gtab = Table(data=[range(len(inlis)), xoff, yoff,xerr, yerr], names=['data #', 'x offset (mm)', 'y offset (mm)', 'x EoM (nm)', 'y EoM'])
    gtab.write('sum.txt', format='ascii.fixed_width')
    simul_ave_fit(xlis, ylis, offsets, xerr=xerr, yerr=yerr, order=order, cut_line=cut_line)
    

def readstack(infile):
    '''
    '''

    xall, yall, fall = match_pin.match_all(infile)
    xncut2, yncut2, dum, dummer = analyze_stack.plot_ind_shift(xall, yall, fall)
    xave = np.mean(xncut2, axis=1)
    yave = np.mean(yncut2, axis=1)

    return xave, yave
    

def simul_ave_fit(xlis, ylis, offsets, order=4, xerr=None, yerr = None, refin=False, cut_line=False):
    '''
    xlis -- list of x coordinates [nframes, npinholes]
    offsets --- list of offsets to translate between each frame

    suspicion is that the fit to the linear terms is corrupted by the distortion, so make that part iterative as well.
    '''

    #for each position of the pinhole mask, fit a 4th? order Legendre polynomial.
    xdc = []
    ydc = []
    x_res = []
    y_res = []
    x_res1 = []
    y_res1 = []
    #
    if xerr is None:
        xerr = []
        for i in range(len(xlis)):
            xerr.append(np.ones(len(xlis[i])))
    if yerr is None:
        yerr = []
        for i in range(len(ylis)):
            yerr.append(np.ones(len(ylis[i])))
    #derive box that is common for all frames -- large offsets in distori
    xmin = 0
    xmax = np.inf
    ymin = 0
    ymax = np.inf
    for i in range(len(xlis)):
        if np.min(xlis[i]) > xmin:
            xmin = np.min(xlis[i])
        if np.max(xlis[i]) < xmax:
            xmax = np.max(xlis[i])
        if np.min(ylis[i]) > ymin:
            ymin = np.min(ylis[i])
        if np.max(ylis[i]) < ymax:
            ymax = np.max(ylis[i])

    #first step is to translate the measured coordiantes so that they can be used to match to reference pinholes
    xpin  = []
    ypin = []
    for i in range(len(xlis)):
        xpin.append(xlis[i] + offsets[i][0])
        ypin.append(ylis[i] + offsets[i][1])

    #create the reference coordinates if they are not input
    if not refin:
        dist = []
        for i in range(len(xlis[0])):
            _dist = ((xlis[0][i] - xlis[0])**2 + (ylis[0][i] - ylis[0])**2)**0.5
            sdist = np.sort(_dist)
            for _ii in range(4):
                dist.append(sdist[_ii+1])
        gspace = np.median(dist)
            
        xper = np.linspace(0, 42 * gspace, num=42)
        yper = np.linspace(0, 42 * gspace, num=42)
        coo = np.meshgrid(xper, yper)
        xgrid = coo[0].flatten()
        ygrid = coo[1].flatten()

    else:
        ref = Table.read('dist.txt', format='ascii.basic')
        xgrid = dist['x']
        ygrid = dist['y']
        
    #match the distortion corrected coordiantes to the grid points
        
    
    for i in range(len(xlis)):
        cbool = (xlis[i] > xmin) * (xlis[i] < xmax) * (ylis[i] > ymin) * (ylis[i] < ymax)
        #add in additional cut to get rid of "dark" streak due to lamp
        yline  = -0.1163 * xlis[i] + 1813
        gbool = (np.abs(yline - ylis[i]) >250)*((yline - ylis[i]) < 0)
        yline  = -0.13 * xlis[i] + 4612
        gbool2 = (np.abs(yline - ylis[i]) >250)*((yline - ylis[i]) > 0)
        if cut_line:
            cbool = cbool * gbool * gbool2
        xr, yr, xnin, ynin, xres, yres, t  = analyze_stack.compare2square(xlis[i][cbool] , ylis[i][cbool], trim=False, gspace=168, fit_ord=0)
        
        #print t.cy
        #import pdb;pdb.set_trace()
        _t = transforms.LegTransform(xnin, ynin, xres, yres, order)#, weights=1/(xerr[i]**2+yerr[i]**2)**0.5)
        _dxn, _dyn = _t.evaluate(xnin, ynin)
        xn = xnin + _dxn
        yn = ynin + _dyn

        _xres = xn - xr
        _yres = yn - yr
        _xstd = np.std(_xres)
        _ystd = np.std(_yres)
        bad = True
        while bad:
            gbool = (_xres  < 2.8 * _xstd + np.mean(_xres)) * (_xres > -2.8 * _xstd +np.mean(_xres)) * (_yres < 2.8 * _ystd + np.mean(_yres)) * (_yres > -2.8* _ystd + np.mean(_yres))
            #repeat if we cut any stars
            #gbool = np.ones(len(_xres), dtype='bool')

            print(np.sum(gbool))
            if np.sum(gbool) < len(gbool):
                xr, yr, xnin, ynin, xres, yres, t  = analyze_stack.compare2square(xnin[gbool] , ynin[gbool], trim=False, gspace=168, fit_ord=0)
                #print t.px, t.py
                
                _t = transforms.LegTransform(xnin, ynin, xres, yres, order)#,weights=1/(xerr**2+yerr**2)**0.5)
                _dxn, _dyn = _t.evaluate(xnin, ynin)
                xn = xnin - _dxn
                yn = ynin - _dyn

                _xres = xn - xr
                _yres = yn - yr
                _xstd = np.std(_xres)
                _ystd = np.std(_yres)
                
            else:
                bad = False
            

        xdc.append(xn)
        ydc.append(yn)
        x_res.append(xn - xr)
        y_res.append(yn - yr)
        x_res1.append(xnin - xr)
        y_res1.append(ynin -yr)

    
    #now that we have the distortin corrected coordinates, make quive plots of the residuaLS
    #the hope here is that the remaining residuals are solely due to random errors -- so by matching them (with the offsets) we can measure offsets in individual pinholes.  it is distinctly possible the high order stucture that is left is actually camera pixel phase effects....

    #first step, fit assuming all high order is distotion
    xall = []
    yall = []
    xresall = []
    yresall = []
    nframe = []
    for i in range(len(xdc)):
        for k in range(len(xdc[i])):
            xall.append(xdc[i][k])
            yall.append(ydc[i][k])
            xresall.append(x_res1[i][k])
            yresall.append(y_res1[i][k])
            nframe.append(i)

    xall = np.array(xall)
    yall = np.array(yall)
    xresall = np.array(xresall)
    yresall = np.array(yresall)
    
    tall = transforms.LegTransform(xall, yall, xresall, yresall, order)
    _dx, _dy = tall.evaluate(xall, yall)
    xwrong = _dx - xresall
    ywrong = _dy - yresall
    print('residual x sim fit', np.std(xwrong)*6000.0)
    print('residual y sim fit', np.std(ywrong)*6000.0)
    nframe = np.array(nframe)
    ccut = np.unique(nframe)
    colors =['black', 'blue', 'green', 'red', 'purple', 'yellow', 'magenta', 'cyan']

    plt.figure(10)
    plt.clf()
    for i in ccut:
        _cbool = nframe == i
        print('residual from frame', i)
        print('X  ',np.std(xwrong[_cbool]*6000.0), ' nm  Y ', np.std(ywrong[_cbool]*6000.0), ' nm')
        analyze_stack.mkquiver(xall[_cbool], yall[_cbool], xwrong[_cbool], ywrong[_cbool], fig_n=7, title_s='Data - Model ', scale=0.5, scale_size=0.01, frac_plot=1, save_f='dist_sim_fit.png', incolor=colors[i], clear=False, scy=1500, scx=-250,  xl=500, xh=7150, yh=6000, yl=600)
    plt.figure(17)
    plt.clf()
    plt.hist(xwrong*6000.0, range=(-200, 200), bins=39, histtype='step', color='green', label='x')
    plt.hist(ywrong*6000.0, range=(-200, 200), bins=39, histtype='step', color='purple', label='y')
    plt.legend(loc='upper left')
    plt.xlabel('Residual (nm)')
    plt.ylabel('N')
    plt.title('Data - Model (combined fit)')
    plt.savefig('residual_dist_single_fit.png')
    plt.figure(4)
    plt.clf()
    plt.figure(6)
    plt.clf()
    plt.figure(19)
    plt.clf()
    plt.figure(18)
    plt.clf()
    for i in range(len(xdc)):
        #import pdb;pdb.set_trace()
        _dxall, _dyall = tall.evaluate(xdc[i], ydc[i])
        analyze_stack.mkquiver(xdc[i]-offsets[i][0],ydc[i]-offsets[i][1], np.array(x_res1[i]) - _dxall, np.array(y_res1[i]) - _dyall, fig_n=4, title_s='Data - Model', scale=0.5, scale_size=0.01, frac_plot=1, save_f='residual_dist.png', incolor=colors[i], clear=False, xl=0, xh=7250, yh=6000, yl=0, scx=500, scy=1500)
        print('mean residual', np.median(np.std(x_res[i]))*6000.0)
        print('mean residual', np.median(np.std(y_res[i]))*6000.0)
        analyze_stack.mkquiver(xdc[i],ydc[i], np.array(x_res[i]) , np.array(y_res[i]), fig_n=18, title_s='Data - Model (individual fits)', scale=0.5, scale_size=0.01, frac_plot=1, save_f='residual_dist.png', incolor=colors[i], clear=False, xl=500, xh=7150, yh=7000, yl=0)
        analyze_stack.mkquiver(xdc[i], ydc[i], x_res1[i], y_res1[i], fig_n=6, title_s='Distortion (order > 1)', scale=5, scale_size=0.3, frac_plot=1, save_f='dist.png', incolor=colors[i], clear=False, scy=1500, scx=-250, yl=1050, yh=6000, xl=500, xh=7845)
        plt.figure(19)
        plt.hist(x_res[i]*6000.0, bins=41,histtype='step', range=(-200, 200), label='x')
        plt.hist(y_res[i]*6000.0, bins=41,histtype='step', range=(-200, 200), label='y')
        plt.legend(loc='upper left')
        plt.ylabel('N')
        plt.xlabel('Residual Distortion (nm)')
        plt.tight_layout()
        plt.savefig('residual_dist_hist.png')
        
    #kick out any point with a residual > 3 sigma
    #repeat fit

    #assume that the indiviual pinholes each have a real measured offsets
    #we apply the offsets to the distortion corrected coordiantes, then match to the "reference", which is a 42 x 42 perfect square grid
    
    
    return
    for i in range(len(xdc)):
        idx1 , idx2 , dm, dr = align.match.match(xdc[i] + offsets[i][0],ydc[i] + offsets[i][1],np.ones(len(xdc[i])),  xgrid, ygrid, np.ones(len(xper)), 60)
        dx[i,idx2] = x_res[i][idx1]
        dy[i,idx1] = y_res[i][idx1]


    dxf = np.mean(dx, axis=0)
    dyf = np.mean(dy, axis=0)


    return xper, yper, dxf, dyf
        
    

def mkave(infile = 'obj.lis',outf='average_coo.txt',  x0 = 0, y0=0, r = 0, xl=400, xh=8100, yl=150, yh=6000, ccut=False, flatcut=False, flux_cut=True, flux_cut_level=np.inf, trim_corners=False, trim_bright=True):


    xall, yall , fall= match_pin.match_all(infile)
    xave = np.mean(xall, axis=1)
    yave = np.mean(yall, axis=1)
    cbool = (xl < xave) * (xh > xave) * (yl < yave) * (yh > yave)

    #add in corner trim
    #if trim_corner:
    #    yl = xave * (6092.61-4811.5)/(2017.4-13.96) + 4803. -800 
    #    cbool = cbool * (yave < yl)
    if trim_corners:
        if r == 0:
            rcut = 2900
        else:
            rcut = r
        if x0 == 0:
            x0 = np.median(xave)
        if y0 == 0:
            y0 = np.median(yave)
            
        rad = np.sqrt((xave - x0)**2 + (yave - y0)**2)
        cbool = cbool * (rad < rcut)

    fave = np.mean(fall, axis=1)
    ferr = np.std(fall, axis=1)

    fm = np.median(fave)
    print('median flux ', fm)
    if trim_bright:
        fcut = fave < fm + 50000
    else:
        fcut = np.ones(len(fave), dtype='bool')
    print('number of stars cut for being too bright ', len(fcut) - np.sum(fcut))
    cbool = cbool * fcut

    print('trans stack error', np.mean(np.std(xall[cbool,:], axis=1)))
    print('trans stack error', np.mean(np.std(yall[cbool,:], axis=1)))

    dx = np.median(xall[cbool,:], axis=0)
    dx = dx - np.mean(dx)
    dy = np.median(yall[cbool,:], axis=0)
    dy = dy - np.mean(dy)
    fm = np.median(fave)
    df = np.median(fall[cbool,:], axis=0)
    df = df - np.mean(df)
    plt.figure(100)
    plt.clf()
    plt.subplot(121)
    plt.plot(dx, 'o')
    plt.plot(dy, 'o')
    plt.title('Translation offset')
    plt.subplot(122)
    plt.plot(df, 'o')
    plt.title('Average brightness')
    plt.show()
    #import pdb;pdb.set_trace()
    
    if ccut:
        r = ((xave - 5098.4)**2 + (yave - 2939.2)**2)**0.5
        cbool = cbool * (r < 2797)

    if flatcut:
        flat = fits.getdata('/Volumes/DATA/DISTORTION/20170822/darks/flat.fits')
        flat_err = fits.getdata('/Volumes/DATA/DISTORTION/20170822/darks/flat_err.fits')
        yc, xc = np.indices(flat.shape)
        fb = (flat < .90) + (flat > 1.15) * (flat_err > 275)
        xb = xc[fb]
        yb = yc[fb]
        #now cut stars closer than 5 pixels to a bad pixel
        gbool = np.ones(xave.shape, dtype='bool')
        for i in range(len(xave)):
            if np.min(np.sqrt((xave[i]-xb)**2+(yave[i]-yb)**2)) < 25:
                gbool[i] = False
        cbool = cbool * gbool
                
    xnew, ynew, dum, dummer = analyze_stack.plot_ind_shift(xall[cbool,:], yall[cbool,:], fall[cbool,:])

    xmean = []
    xerr = []
    ymean = []
    yerr = []
    for i in range(xnew.shape[0]):
        #compute sigma clipped error on the mean
        _xmean, _xstd, _N = stats.mean_std_clip(xnew[i,:], return_nclip=True)
        xerr.append(_xstd/np.sqrt(_N))
        xmean.append(_xmean)

        _ymean, _ystd, _N = stats.mean_std_clip(ynew[i,:], return_nclip=True)
        yerr.append(_ystd/np.sqrt(_N))
        ymean.append(_ymean)

    xmean = np.array(xmean)
    ymean = np.array(ymean)
    xerr = np.array(xerr)
    yerr = np.array(yerr)
    idx1, idx2, dm, dr = match.match(xmean, ymean, np.ones(len(xmean)), xave, yave, np.ones(len(xave)), 40)
   
    
    
    _out = Table(data=[xmean[idx1], xerr[idx1], ymean[idx1], yerr[idx1], fave[idx2], ferr[idx2]], names=['x', 'xerr', 'y', 'yerr', 'flux', 'fluxerr'])
    _out.write(outf, format='ascii.fixed_width')
        
        

def fit_dist_single(coo_txt, order=2, retrans=False, iterate=True):
    '''
    fits a single polynomial to averaged distoriton  data
    
    '''

    stack = Table.read(coo_txt, format='ascii.fixed_width')

    #assume all trimming was already completed
    
    
    xr, yr, xnin, ynin, xres, yres, t  = analyze_stack.compare2square(stack['x'] , stack['y'], trim=False, gspace=168)
    _t = transforms.LegTransform(xnin, ynin, xres, yres, order)#, weights=1/(xerr[i]**2+yerr[i]**2)**0.5)
    _dxn, _dyn = _t.evaluate(xnin, ynin)
    xn = xnin + _dxn
    yn = ynin + _dyn
    _xres = xn - xr
    _yres = yn - yr
    _xstd = np.std(_xres)
    _ystd = np.std(_yres)
    
    bad = True
    while bad:
        _xa, _xe= stats.mean_std_clip(_xres)
        _ya, _ye= stats.mean_std_clip(_yres)
        gbool = (_xres  < 3 * _xe + _xa) * (_xres > -3 * _xe +_xa) * (_yres < 3 * _ye + _ya) * (_yres > -3* _ye + _ya)
        #repeat if we cut any stars
        #gbool = np.ones(len(_xres), dtype='bool')

        print(np.sum(gbool))
        if iterate:
            if np.sum(gbool) < len(gbool):
                xr, yr, xnin, ynin, xres, yres, t  = analyze_stack.compare2square(xnin[gbool] , ynin[gbool], trim=False, gspace=168)
                _t = transforms.LegTransform(xnin, ynin, xres, yres, order)#,weights=1/(xerr**2+yerr**2)**0.5)
                _dxn, _dyn = _t.evaluate(xnin, ynin)
                xn = xnin - _dxn
                yn = ynin - _dyn
                _xres = xn - xr
                _yres = yn - yr
                _xstd = np.std(_xres)
                _ystd = np.std(_yres)
                _xa, _xe= stats.mean_std_clip(_xres)
                _ya, _ye= stats.mean_std_clip(_yres)
                gbool = (_xres  < 3 * _xe + _xa) * (_xres > -3 * _xe +_xa) * (_yres < 3 * _ye + _ya) * (_yres > -3* _ye + _ya)
                
            else:
                bad = False
        else:
            bad = False
            xr, yr, xnin, ynin, xres, yres, t  = analyze_stack.compare2square(xnin[gbool] , ynin[gbool], trim=False, gspace=168)
            _t = transforms.LegTransform(xnin, ynin, xres, yres, order)#,weights=1/(xerr**2+yerr**2)**0.5)
            _dxn, _dyn = _t.evaluate(xnin, ynin)
            xn = xnin - _dxn
            yn = ynin - _dyn
            _xres = xn - xr
            _yres = yn - yr
            _xstd = np.std(_xres)
            _ystd = np.std(_yres)
    print('fit residual X', _xstd*6000.0, ' nm')
    print('fit residual Y', _ystd*6000.0, ' nm')
    
    analyze_stack.mkquiver(xnin, ynin, _xres , _yres, frac_plot=1, scale=.25, fig_n=1555, scale_size=.01)
    plt.figure(1556)
    plt.clf()
    plt.hist(_xres*6000.0, bins=30, histtype='step', label='x')
    plt.hist(_yres*6000.0, bins=30, histtype='step', label='y')
    plt.legend(loc='upper left')
    plt.xlabel('Residual (nm)')
    plt.ylabel('N')
    plt.title('Res order '+str(order)+' file:'+coo_txt)
    plt.show()

    if retrans:
        return _t

def mkrefcat(num_pin=43, gspace=168.0,  ang=-0.33):
    '''
    '''

    #grid is 43 x 43 with spacing of 168 pix
    xcoo, ycoo = np.meshgrid(np.linspace(0, gspace * (num_pin-1), num=num_pin), np.linspace(0,gspace * (num_pin-1), num=num_pin))
    xcoo = xcoo.flatten()
    ycoo = ycoo.flatten()

    rang = np.deg2rad(ang)
    xout = np.cos(rang) * xcoo - np.sin(rang) * ycoo
    yout = np.cos(rang) * ycoo + np.sin(rang) * xcoo


    #make sure that the angle matches? -- dont bother just write the first file

    _outab = Table(data=[xout, yout, xout, yout], names=['xorig', 'yorig', 'x', 'y'])
    _outab.write('reference.txt', format='ascii.fixed_width')





def simul_wref(xlis, ylis, offsets, order=4, trim_pin=True, trim_cam=True, nmin=4, dev_pat=True, renorm=True, rot_ang=None, sig_clip=True, debug=False, ind_fits=False):
    '''
    xlis -- list of x coordinates [nframes, npinholes]
    offsets --- list of offsets to translate between each frame

    Trimming of the reference catalog occurs in 2 steps -- First step is to only keep pinholes that are in at least nmin images
    Then we trim the image in pixel coordiantes to only smaple the SAME optical distortion
    
    '''

    #for each position of the pinhole mask, fit a 4th? order Legendre polynomial.
    xln = []
    yln = []
    xrn = []
    yrn = []
    refid = []
    frameN = []
    #first clean the data
    if rot_ang is None:
        rot_ang = np.zeros(len(xlis))

    #hardcode for simplicity
    #ymax = 3000
    #ymin = 1041
    #xmax = 6000
    #xmin = 1041
    
    #create the reference coordinates if they are not input
   
    ref = Table.read('reference.txt', format='ascii.basic')
    
    for i in range(len(xlis)):
        
        #add in additional cut to get rid of "dark" streak due to lamp
        #yline  = -0.1163 * xlis[i] + 1813
        #gbool = (np.abs(yline - ylis[i]) >250)*((yline - ylis[i]) < 0)
        #yline  = -0.13 * xlis[i] + 4612
        #gbool2 = (np.abs(yline - ylis[i]) >250)*((yline - ylis[i]) > 0)
        #if cut_line:
        #    cbool = cbool * gbool * gbool2
        #cut out upper left corner, residuals are large
        yline = xlis[i] * (5817.0-4774.0)/(1914.0-428.0) + 1343
        gbool = ylis[i] < yline
        cbool = np.ones(len(xlis[i]), dtype='bool')
        #need to rotate data to match the  reference
        _ang = np.deg2rad(rot_ang[i])
        xang = np.cos(_ang) * xlis[i] - np.sin(_ang) * ylis[i]
        yang = np.sin(_ang) * xlis[i] + np.cos(_ang) * ylis[i]
        xo = np.cos(_ang) * offsets[i][0] - np.sin(_ang) * offsets[i][1]
        yo = np.sin(_ang) * offsets[i][0] + np.cos(_ang) * offsets[i][1]
        
        idx1, idx2, drm, dr = match.match(xang[cbool] - xo, yang[cbool] - yo,np.zeros(len(xlis[i])), ref['x'], ref['y'], np.zeros(len(ref)),30)
        #take out median translation and rematch
        _dx = np.median(xang[cbool][idx1] - ref['x'][idx2])
        _dy = np.median(yang[cbool][idx1] - ref['y'][idx2])
        idx1N, idx2N, drm, dr = match.match(xang - _dx, yang- _dy,np.zeros(len(xlis[i])), ref['x'], ref['y'], np.zeros(len(ref)),100)
        #assert len(idx1) > 350
        assert len(idx1N) >= len(idx1)

        plt.figure(155)
        plt.clf()
        plt.scatter(xang-xo, yang-yo, s=1, label='measured')
        plt.scatter(ref['x'], ref['y'], s=1, label='data')
        plt.show()
        #import pdb;pdb.set_trace()

        t = transforms.LegTransform(ref['x'][idx2N], ref['y'][idx2N] ,  xlis[i][idx1N], ylis[i][idx1N], order)
        xnin = xlis[i][idx1N]
        ynin = ylis[i][idx1N]

        xro = ref['x'][idx2N]
        yro = ref['y'][idx2N]

        xn, yn = t.evaluate(ref['x'][idx2N], ref['y'][idx2N])
       
        _xres = xn - xnin
        _yres = yn - ynin
        _xstd = np.std(_xres)
        _ystd = np.std(_yres)
        print('residual individual fits', _xstd*6000.0, _ystd*6000.0)
        #one 3 sigma trim to get rid of huge outliers
        if sig_clip:
            #gbool = (_xres  < 3 * _xstd + np.mean(_xres)) * (_xres > -3 * _xstd +np.mean(_xres)) * (_yres < 3 * _ystd + np.mean(_yres)) * (_yres > -3* _ystd + np.mean(_yres))
            
            _xa, _xe= stats.mean_std_clip(_xres)
            _ya, _ye = stats.mean_std_clip(_yres)
            gbool = (_xres < 3 * _xe + _xa) * (_xres > -3 * _xe + _xa) * (_yres < 3 * _ye + _ya) * (_yres > -3 * _ye + _ya)
        else:
            gbool = np.ones(len(_xres), dtype='bool')
            
        xln.append(xlis[i][idx1N][gbool])
        yln.append(ylis[i][idx1N][gbool])
        xrn.append(ref['x'][idx2N][gbool])
        yrn.append(ref['y'][idx2N][gbool])
        frameN.append(np.zeros(len(idx2N[gbool])) + i)
        refid.append(idx2N[gbool])

    rflat = []
    for i in range(len(refid)):
        for k in range(len(refid[i])):
            rflat.append(refid[i][k])
    rflat = np.array(rflat)
    pinid = np.unique(rflat)
    #only keep pinholes detected in at least nmin frames
    pingood = []
    
    for i in pinid:
        if np.sum(rflat == i) >= nmin:
            pingood.append(i)
    pbool = []
    for i in range(len(refid)):
        pbool.append([])
        for kk in range(len(refid[i])):
            if refid[i][kk] in pingood:
                pbool[-1].append(kk)
    #now trim out pinholes without enough detections
    if trim_pin:
        xlnN = []
        ylnN = []
        xrnN = []
        yrnN = []
        refidN = []
        frameNN = []
        for i in range(len(xln)):
            xlnN.append(xln[i][pbool[i]])
            ylnN.append(yln[i][pbool[i]])
            xrnN.append(xrn[i][pbool[i]])
            yrnN.append(yrn[i][pbool[i]])
            refidN.append(refid[i][pbool[i]])
            frameNN.append(frameN[i][pbool[i]])
        xln = np.array(xlnN)
        yln = np.array(ylnN)
        xrn = np.array(xrnN)
        yrn = np.array(yrnN)
        refid = np.array(refidN)
        frameN = np.array(frameNN)

    
    #derive box that is common for all frames -- only want to sample the same distortion
    xmin = 0
    xmax = np.inf
    ymin = 0
    ymax = np.inf
   
    for i in range(len(xlis)):
        if np.min(xln[i]) > xmin:
            xmin = np.min(xln[i])
        if np.max(xln[i]) < xmax:
            xmax = np.max(xln[i])
        if np.min(yln[i]) > ymin:
            ymin = np.min(yln[i])
        if np.max(yln[i]) < ymax:
            ymax = np.max(yln[i])
    #import pdb;pdb.set_trace()
    #hardcode cludge
        #ymax = 5000.0

    #now trim the input coordiants one final time
    if trim_cam:
        xlnN = []
        ylnN = []
        xrnN = []
        yrnN = []
        refidN = []
        frameNN = []
        for i in range(len(xln)):
            print(xmin, xmax, ymin, ymax)
            cbool = (xln[i] < xmax) * (xln[i] > xmin) * (yln[i] < ymax) * (yln[i] > ymin)
            print('Number of measured positions left ', np.sum(cbool))
            xlnN.append(xln[i][cbool])
            ylnN.append(yln[i][cbool])
            xrnN.append(xrn[i][cbool])
            yrnN.append(yrn[i][cbool])
            refidN.append(refid[i][cbool])
            frameNN.append(frameN[i][cbool])
        xln = np.array(xlnN)
        yln = np.array(ylnN)
        xrn = np.array(xrnN)
        yrn = np.array(yrnN)
        refid = np.array(refidN)
        frameNN = np.array(frameNN)
        
    
    #now iterate through and fit an individual linear polynomial from the reference to the measured frame, save the measured deltas between on detector and square reference.
    #new iteration needs to apply the previous distortion solution ....


    
    init_guess = guess_co(offsets, order=order)
    res = leastsq(com_mod, init_guess, args=(xln, yln, xrn, yrn, order, dev_pat))
    #save the good paramters to an output file
    _out = Table(data=[res[0]])
    _out.write('fit.txt', format='ascii.basic')
    xdatR, ydatR, xrefR, yrefR = com_mod(res[0], xln, yln, xrn, yrn, order=order, evaluate=True)

    xresIF = []
    yresIF = []
    xlall = []
    ylall = []
    refidall = []
    frameall = []
    for i in range(len(xdatR)):
            
        xresIF.append(xdatR[i] - xrefR[i])
        yresIF.append(ydatR[i] - yrefR[i])
        xlall.append(xln[i])
        ylall.append(yln[i])
        refidall.append(refid[i])
        frameall.append(frameN[i])
    xresIF = np.array(xresIF)
    yresIF = np.array(yresIF)
        #do single round of sigma clipping on residuals

    xlnN = []
    ylnN = []
    xrnN = []
    yrnN = []
    
    for i in range(len(xdatR)):
           
            #the mean of the residuals is very close to zero
        _xres =xresIF[i]
        _yres = yresIF[i]
        _xstd = np.std(_xres)
        _ystd =  np.std(_yres)
        _xa, _xe= stats.mean_std_clip(_xres)
        _ya, _ye = stats.mean_std_clip(_yres)
        _gbool = (_xres < 3 * _xe + _xa) * (_xres > -3 * _xe + _xa) * (_yres < 3 * _ye + _ya) * (_yres > -3 * _ye + _ya)
        print('trimming ', len(_gbool) - np.sum(_gbool) ,' for frame ', i)
        #gotta write useful text files
        _oo = Table(data=[xln[i][_gbool], yln[i][_gbool],xrn[i][_gbool], yrn[i][_gbool]], names=['x','y','xr','yr'])
        _oo.write('mout_'+str(i)+'.txt', format='ascii.basic')

    #print('trimming ', len(tgbool) - np.sum(tgbool))
        
        #trim the arguments
    
        xlnN.append(xln[i][_gbool])
        ylnN.append(yln[i][_gbool])
        xrnN.append(xrn[i][_gbool])
        yrnN.append(yrn[i][_gbool])

    xln = np.array(xlnN)
    yln = np.array(ylnN)
    yrn = np.array(yrnN)
    xrn = np.array(xrnN)
    res = leastsq(com_mod, init_guess, args=(xln, yln, xrn, yrn, order, dev_pat))
    xdatR, ydatR, xrefR, yrefR = com_mod(res[0], xln, yln, xrn, yrn, order=order, evaluate=True)
        
  
  
            
    xresIF = []
    yresIF = []
    xlall = []
    ylall = []
    xrall = []
    yrall = []
    refidall = []
    frameall = []
    for i in range(len(xdatR)):
        for jj in range(len(xdatR[i])):
            xresIF.append(xdatR[i][jj] - xrefR[i][jj])
            yresIF.append(ydatR[i][jj] - yrefR[i][jj])
            xlall.append(xln[i][jj])
            ylall.append(yln[i][jj])
            xrall.append(xrefR[i][jj])
            yrall.append(yrefR[i][jj])
            refidall.append(refid[i][jj])
            frameall.append(frameN[i][jj])

        
    print('X Residual for sim fit 6 par removed', np.std(xresIF)*6000.0 , ' nm')
    print('Y Residual for sim fit 6 par removed', np.std(yresIF)*6000.0 , ' nm')

    frameall = np.array(frameall)
    xresIF = np.array(xresIF)
    yresIF = np.array(yresIF)
    ccut = np.unique(frameall)
    xrall = np.array(xrall)
    yrall = np.array(yrall)
   

    colors =['black', 'blue', 'green', 'red', 'purple', 'yellow', 'magenta', 'cyan', 'orange', 'black', 'blue', 'green', 'red', 'purple', 'yellow', 'magenta', 'cyan', 'orange' ]
    xlall = np.array(xlall)
    ylall = np.array(ylall)
    plt.figure(5)
    plt.clf()
    plt.figure(10)
    plt.clf()
    #import pdb;pdb.set_trace()
    for i in ccut:
        _cbool = frameall == i
        print('residual from frame', i)
        print(offsets[int(i)])
    
        print('X  ',np.std(xresIF[_cbool]*6000.0), ' nm  Y ', np.std(yresIF[_cbool]*6000.0), ' nm')
        print(' MEDIAN : X  ',np.median(xresIF[_cbool]*6000.0), ' nm  Y ', np.median(yresIF[_cbool]*6000.0), ' nm')
        print(' MEAN X  ',np.mean(xresIF[_cbool]*6000.0), ' nm  Y ', np.mean(yresIF[_cbool]*6000.0), ' nm')
        
        #print('X  ',np.std(xwrong[_cbool]*6000.0), ' nm  Y ', np.std(ywrong[_cbool]*6000.0), ' nm')
        #analyze_stack.mkquiver(xlall[_cbool], ylall[_cbool], np.array(xreslin)[_cbool], np.array(yreslin)[_cbool], fig_n=5, title_s='Measured Disotortion ', scale=2, scale_size=1, frac_plot=1, save_f='dist_sim_fit.png', incolor=colors[int(i)], clear=False, scy=1150,  xl=500, xh=7150, yh=7000, yl=0)
        analyze_stack.mkquiver(xlall[_cbool], ylall[_cbool], xresIF[_cbool], yresIF[_cbool], fig_n=10, title_s='Data - Model ', scale=0.5, scale_size=0.05, frac_plot=1, save_f='dist_sim_fit.png', incolor=colors[int(i)], clear=False, scy=1150, scx=400,  xl=1000, xh=6000, yh=6000, yl=500)
        analyze_stack.mkquiver(xrall[_cbool], yrall[_cbool], xresIF[_cbool], yresIF[_cbool], fig_n=33, title_s='Data - Model ', scale=0.5, scale_size=0.05, frac_plot=1, save_f='dist_sim_fit_ref.png', incolor=colors[int(i)], clear=False, scy=1150, scx=400,  xl=0, xh=7000, yh=5800, yl=0)
    plt.figure(17)
    plt.clf()
    plt.hist(xresIF*6000.0, range=(-600, 600), bins=39, histtype='step', color='green', label='x', lw=5)
    plt.hist(yresIF*6000.0, range=(-600, 600), bins=39, histtype='step', color='purple', label='y', lw=5)
    plt.legend(loc='upper left')
    plt.xlabel('Residual (nm)')
    plt.ylabel('N')
    plt.title('Data - Model')
    plt.savefig('residual_dist_single_fit.png')
        #now we need to apply a correction to the refernece coordinates...        #refid is list [npos, index of reference star]
        #make a dictionary lookup for the reference position

    if debug:
        return res
    return

def fixref(ref, rdictx, rdicty, nmin=1, plot_cor=True, fit_pol=False, update=True):
    '''
    '''

    refid = []
    dx = []
    dy = []
    
    for key in rdictx.keys():
        if len(rdictx[key]) >= nmin:
            dx.append(np.mean(rdictx[key]))
            dy.append(np.mean(rdicty[key]))
            refid.append(key)

    if plot_cor:
        analyze_stack.mkquiver(ref['x'][refid], ref['y'][refid], dx, dy, scale=np.mean(np.abs(dx)+np.abs(dy))/4.0, scale_size=.01, fig_n=39)
        
    if not fit_pol:
    #just blindly apply the 0.3 * correction for all pinholes -- could fit a polynomial here instead 
        ref['x'][refid] = ref['x'][refid] + 0.5 * np.array(dx)
        ref['y'][refid] = ref['y'][refid] + 0.5 * np.array(dy)
    else:
        #fit 2nd orde rpolynoimial to the residuals as  s dunction fo pix
        tdist = transforms.LegTransform(ref['x'][refid], ref['y'][refid], np.array(dx), np.array(dy), 2 )
        _dx, _dy = tdist.evaluate(ref['x'][refid], ref['y'][refid])
        ref['x'][refid] = ref['x'][refid] + _dx
        ref['y'][refid] = ref['y'][refid] + _dy

    #import pdb;pdb.set_trace()
    if update:
        ref.write('reference.txt', format='ascii.basic', overwrite=True)
    #now there are two choices.  either apply the correction to the reference pi


def com_mod(co, xin, yin, xr, yr, order = 2, dev_pat=True,evaluate=False,pat_ord=2):
    '''
    add in deviations a a function of xr , yr up to second order ...
    drx =  c1 * x**2 + c2 * x*y + c3 * y**2
    same for dry (c4-c6), this does not include globale terms (scale or offsts
    '''

   

    xlin = []
    ylin = []

    for i in range(xin.shape[0]):
        xlin.append(co[6*i] + co[6*i+1] * xin[i] + co[6*i+2] * yin[i])
        ylin.append(co[6*i+3] + co[6*i+4] * xin[i] + co[6*i+5] * yin[i])
        #import pdb;pdb.set_trace()

    xn = []
    yn = []
    i_f = 6 * i  + 6
    #import pdb;pdb.set_trace()
    if order == 4:
        for kk in range(xin.shape[0]):
            i = i_f 
            xn.append(xlin[kk] + co[i] * xin[kk]**2 + co[i+1] * xin[kk] * yin[kk] + co[i+2] * yin[kk]**2 + co[i+3] * xin[kk]**3 + co[i+4] * xin[kk]**2 * yin[kk] + co[i+5] * xin[kk] * yin[kk]**2 + co[i+6] * yin[kk]**3 + co[i+7] * xin[kk]**4 + co[i+8] * xin[kk]**3 * yin[kk] + co[i+9] * xin[kk]**2 * yin[kk]**2 + co[i+10] * xin[kk] * yin[kk]**3 + co[i+11] * yin[kk]**4)
            i = i + 12 
            yn.append(ylin[kk] + co[i] * xin[kk]**2 + co[i+1] * xin[kk] * yin[kk] + co[i+2] * yin[kk]**2 + co[i+3] * xin[kk]**3 + co[i+4] * xin[kk]**2 * yin[kk] + co[i+5] * xin[kk] * yin[kk]**2 + co[i+6] * yin[kk]**3 + co[i+7] * xin[kk]**4 + co[i+8] * xin[kk]**3 * yin[kk] + co[i+9] * xin[kk]**2 * yin[kk]**2 + co[i+10] * xin[kk] * yin[kk]**3 + co[i+11] * yin[kk]**4)
    elif order ==3:
        for kk in range(xin.shape[0]):
            i = i_f
            xn.append(xlin[kk] + co[i] * xin[kk]**2 + co[i+1] * xin[kk] * yin[kk] + co[i+2] * yin[kk]**2 + co[i+3] * xin[kk]**3 + co[i+4] * xin[kk]**2 * yin[kk] + co[i+5] * xin[kk] * yin[kk]**2 + co[i+6] * yin[kk]**3)
            i = i + 7
            yn.append(ylin[kk] + co[i] * xin[kk]**2 + co[i+1] * xin[kk] * yin[kk] + co[i+2] * yin[kk]**2 + co[i+3] * xin[kk]**3 + co[i+4] * xin[kk]**2 * yin[kk] + co[i+5] * xin[kk] * yin[kk]**2 + co[i+6] * yin[kk]**3)
        
    elif order == 2:
        for kk in range(xin.shape[0]):
            i = i_f
            xn.append(xlin[kk] + co[i] * xin[kk]**2 + co[i+1] * xin[kk] * yin[kk] + co[i+2] * yin[kk]**2)
            i = i + 3
            yn.append(ylin[kk] + co[i] * xin[kk]**2 +co[i+1] * xin[kk] * yin[kk] + co[i+2] * yin[kk]**2)

    #now need to add in the pattern refernece terms, these are the last 6 terms in co array
    xrn = []
    yrn = []
    #import pdb;pdb.set_trace()
    if dev_pat:
        for kk in range(len(xr)):
            if pat_ord == 2:
                xrn.append(xr[kk] + xr[kk]**2 * co[-14] + xr[kk] * yr[kk] * co[-13] + yr[kk]**2 * co[-12])
                yrn.append(yr[kk] + xr[kk]**2 * co[-7] + xr[kk] * yr[kk] * co[-6] + yr[kk]**2 * co[-5])
            elif pat_ord == 3:
                xrn.append(xr[kk] + xr[kk]**2 * co[-14] + xr[kk] * yr[kk] * co[-13] + yr[kk]**2 * co[-12] + xr[kk]**3 * co[-11] + xr[kk]**2*yr[kk] * co[-10] +  xr[kk] * yr[kk]**2 *co[-9] +   yr[kk]**3 * co[-8])
                yrn.append(yr[kk] + xr[kk]**2 * co[-7] + xr[kk] * yr[kk] * co[-6] + yr[kk]**2 * co[-5] +  xr[kk]**3 * co[-4] +  xr[kk]**2*yr[kk] * co[-3] +  xr[kk] * yr[kk]**2 * co[-2] +  yr[kk]**3 * co[-1])
            
    else:
        xrn = xr
        yrn = yr
    if evaluate:
        return xn, yn, xrn, yrn
    tot = []
    for jj in range(xin.shape[0]):
        for kk in range(len(xin[jj])):
            tot.append(xn[jj][kk] - xrn[jj][kk])
            tot.append(yn[jj][kk] - yrn[jj][kk])
    tot = np.array(tot)
 
    return tot
def guess_co(offsets, order=2):
    '''
    '''
    co = []
    expar = {2:6, 3:14, 4:24}
    for i in range(len(offsets)):
        co.append(-1.0 * offsets[i][0])
        co.append(1.0)
        co.append(0.0)
        co.append(-1.0 * offsets[i][1])
        co.append(0.0)
        co.append(1.0)

    #add in high order distortion terms
    for i in range(expar[order]):
        co.append(0.0)

    #add in  2nd order pattern errors 
    for i in range(14):
        co.append(0.0)
    return np.array(co)

def plot_lin_dist(ff='fit.txt', mlis='mout.lis', order=3):
    '''
    fit.txt is text file with coefficinets from simultaneaous distortion 
    mlis is lis of mout*.txt files that have the x/y measured and x/y reference
    '''

    mtab = Table.read(mlis, format='ascii.no_header')
    _co = Table.read(ff, format='ascii.basic')['col0']


    ddx = []
    ddy = []
    colors =['black', 'blue', 'green', 'red', 'purple', 'yellow', 'magenta', 'cyan','black', 'blue', 'green', 'red', 'purple', 'yellow', 'magenta', 'cyan','black', 'blue', 'green', 'red', 'purple', 'yellow', 'magenta', 'cyan','black', 'blue', 'green', 'red', 'purple', 'yellow', 'magenta', 'cyan' ]
    xln = []
    yln = []
    xrn = []
    yrn = []
    for _ff in mtab['col1']:
        _m = Table.read(_ff, format='ascii.basic')
        xln.append(_m['x'])
        yln.append(_m['y'])
        xrn.append(_m['xr'])
        yrn.append(_m['yr'])

    xln = np.array(xln)
    xdatR, ydatR, xrefR, yrefR = com_mod(_co,xln, yln, xrn , yrn, order=order, evaluate=True)
    plt.figure(55)
    plt.clf()
    xl1 = []
    xl2 = []
    yl1 = []
    yl2 = []
    plt.close('all')
    x =[[],[]]
    y = [[],[]]
    for dd in range(len(xdatR)):
        for ii in range(len(xdatR[dd])):
            x[0].append(xdatR[dd][ii])
            x[1].append(xrefR[dd][ii])
            y[0].append(ydatR[dd][ii])
            y[1].append(yrefR[dd][ii])
            
    print(np.std(np.array(x[0]) - np.array(x[1])))
    print(np.std(np.array(y[0]) - np.array(y[1])))
    for i in range(len(mtab['col1'])):
        _m = Table.read(mtab['col1'][i], format='ascii.basic')
        _xn = _co[6*i] + _m['x'] * _co[6*i+1] + _m['y'] * _co[6*i+2]
        _yn = _co[6*i+3] + _m['x'] * _co[6*i+4] + _m['y'] * _co[6*i+5]
        xl1.append(_co[6*i +1])
        xl2.append(_co[6*i+2])
        yl1.append(_co[6*i+4])
        yl2.append(_co[6*i+5])
        dx =  _xn - xdatR[i]
        dy =  _yn - ydatR[i]
        dx = dx - np.mean(dx)
        dy = dy - np.mean(dy)

        #import pdb;pdb.set_trace()

        for kk in range(len(dx)):
            ddx.append(dx[kk])
            ddy.append(dy[kk])

        #now we need to apply the correction to the pinholes...
       


        analyze_stack.mkquiver(_m['x'], _m['y'], dx, dy, fig_n=12, title_s='Measured Distortion ', scale=10, scale_size=1, frac_plot=3, save_f='measured_dist.png', incolor=colors[int(i)], clear=False, scy=1150,scx=0,  xl=500, xh=6500, yh=5300, yl=300, xlab =r'$x_{m}$ (pixels)', ylab=r'$y_{m}$ (pixels)')

        analyze_stack.mkquiver(_m['x'], _m['y'], xdatR[i] - xrefR[i], ydatR[i] - yrefR[i], fig_n=56+i, title_s='Stack '+str(i), scale=.3, scale_size=.02, frac_plot=1,save_f='residual_dist_ind'+str(i)+'.png', incolor=colors[int(i)], clear=False, scy=1150,scx=0,  xl=500, xh=6500, yh=5400, yl=300, xlab =r'$x_{m}$ (pixels)', ylab=r'$y_{m}$ (pixels)')
        analyze_stack.mkquiver(_m['x'], _m['y'], xdatR[i] - xrefR[i], ydatR[i] - yrefR[i], fig_n=55, title_s='Data - Model', scale=.3, scale_size=.02, frac_plot=1,  save_f='residual_dist_all.png', incolor=colors[int(i)], clear=False, scy=1150,scx=0,  xl=500, xh=6500, yh=5400, yl=300, xlab =r'$x_{m}$ (pixels)', ylab=r'$y_{m}$ (pixels)')

        #analyze_stack.mkquiver(xrefR[i], yrefR[i], xdatR[i] - xrefR[i], ydatR[i] - yrefR[i], fig_n=76+i, title_s='Stack '+str(i), scale=.3, scale_size=.02, frac_plot=1, save_f='measured_dist_refco_ind'+str(i)+'.png', incolor=colors[int(i)], clear=False, scy=1150,scx=0,  xl=-200, xh=7100, yl=0, yh=7100,  xlab =r'$x_{r}$', ylab=r'$y_{r}$')
        #analyze_stack.mkquiver(xrefR[i], yrefR[i], xdatR[i] - xrefR[i], ydatR[i] - yrefR[i], fig_n=75, title_s='Data - Model', scale=.3, scale_size=.02, frac_plot=1, save_f='measured_dist_refco.png', incolor=colors[int(i)], clear=False, scy=1150,scx=0,  xl=-200, xh=7100, yl=0, yh=7100,  xlab =r'$x_{r}$', ylab=r'$y_{r}$')
        

    xl1 = np.array(xl1)
    yl1 = np.array(yl1)
    xl2 = np.array(xl2)
    yl2 = np.array(yl2)
    
    plt.figure(88)
    plt.clf()
    plt.plot(range(len(xl1)), xl1, 'bo', label=r'$c_{1,x}$')
    plt.plot(range(len(xl1)), yl2, 'go', label=r'$c_{2,y}$')
    plt.xlabel('Pinhole grid postition')
    plt.ylabel('Coefficient value')
    plt.title('Linear variation')
    plt.legend(loc='upper left')

    plt.figure(89)
    plt.clf()
    plt.plot(range(len(xl1)), xl2,'bo', label=r'$c_{2,x}$')
    plt.plot(range(len(yl2)), -1.0*np.array(yl1), 'go', label=r'-$c_{1,y}$')
    plt.xlabel('Pinhole grid position')
    plt.ylabel('Coefficient value')
    plt.title('Linear Variation')
    plt.legend(loc='upper left')

    plt.figure(90)
    plt.clf()
    plt.plot(range(len(xl1)), (xl2**2 + xl1**2)*166.6666/168.0,'ro', label=r'$M_{x}$')
    plt.plot(range(len(yl2)), (yl1**2 + yl2**2)*166.6666/168.0, 'mo', label=r'$M_{y}$')
    plt.xlabel('Stack Number')
    plt.ylabel('Scale')
    plt.title('Variation in linear parameters')
    plt.legend(loc='upper left')
    plt.xticks([0,1,2,3,4], ['0','1','2','3','4'])
    plt.tight_layout()

    plt.figure(91)
    plt.clf()
    plt.plot(range(len(xl1)), np.rad2deg(np.arctan2(-1.0 * xl2, xl1)),'ro', label=r'$\theta_{x}$')
    plt.plot(range(len(yl2)),  np.rad2deg(np.arctan2( yl1, yl2)), 'mo', label=r'$\theta_{y}$')
    plt.xlabel('Stack Number')
    plt.ylabel('Angle (degrees)')
    plt.title('Variation in linear parameters')
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    ddx = np.array(ddx)
    ddy = np.array(ddy)
    #print RMS of measured distortion
    print(np.std(ddx))
    print(np.std(ddy))



def simul_wref_ind(xlis, ylis, offsets, order=4, trim_pin=True, niter=5, trim_cam=True, nmin=4,  rot_ang=None, sig_clip=True, debug=False, ):
    '''
    xlis -- list of x coordinates [nframes, npinholes]
    offsets --- list of offsets to translate between each frame

    This fitting plan assumes that the distoriotn for each position is completely independent, but that the reference positions are the same
    
    '''

    #for each position of the pinhole mask, fit a 4th? order Legendre polynomial.
    xln = []
    yln = []
    xrn = []
    yrn = []
    refid = []
    frameN = []
    dxpix = []
    dypix = []
    #first clean the data
    if rot_ang is None:
        rot_ang = np.zeros(len(xlis))

    #hardcode for simplicity
    #ymax = 3000
    #ymin = 1041
    #xmax = 6000
    #xmin = 1041
    
    #create the reference coordinates if they are not input
   
    ref = Table.read('reference.txt', format='ascii.basic')
    
    for N in range(niter):
        tt = []
        drx = []
        dry = []
        for i in range(len(xlis)):
        
            #add in additional cut to get rid of "dark" streak due to lamp
            #yline  = -0.1163 * xlis[i] + 1813
            #gbool = (np.abs(yline - ylis[i]) >250)*((yline - ylis[i]) < 0)
            #yline  = -0.13 * xlis[i] + 4612
            #gbool2 = (np.abs(yline - ylis[i]) >250)*((yline - ylis[i]) > 0)
            #if cut_line:
            #    cbool = cbool * gbool * gbool2
            cbool = np.ones(len(xlis[i]), dtype='bool')
            #need to rotate data to match the  reference


            # if we have one, apply a pixel phase correction
            #xin, yin = app_pix_corr(np.array(xlis[i]), np.array(ylis[i]))
            xin = xlis[i]
            yin = ylis[i]
            _ang = np.deg2rad(rot_ang[i])
            xang = np.cos(_ang) * xin - np.sin(_ang) * yin
            yang = np.sin(_ang) * xin + np.cos(_ang) * yin
            xo = np.cos(_ang) * offsets[i][0] - np.sin(_ang) * offsets[i][1]
            yo = np.sin(_ang) * offsets[i][0] + np.cos(_ang) * offsets[i][1]
            
            idx1, idx2, drm, dr = match.match(xang[cbool] - xo, yang[cbool] - yo,np.zeros(len(xlis[i])), ref['x'], ref['y'], np.zeros(len(ref)),30)
            #take out median translation and rematch
            _dx = np.median(xang[cbool][idx1] - ref['x'][idx2])
            _dy = np.median(yang[cbool][idx1] - ref['y'][idx2])
            idx1N, idx2N, drm, dr = match.match(xang - _dx, yang- _dy,np.zeros(len(xlis[i])), ref['x'], ref['y'], np.zeros(len(ref)),100)
        #assert len(idx1) > 350
            assert len(idx1N) >= len(idx1)

            plt.figure(155)
            plt.clf()
            plt.scatter(xang-xo, yang-yo, s=1, label='measured')
            plt.scatter(ref['x'], ref['y'], s=1, label='data')
            #plt.show()
        #import pdb;pdb.set_trace()

            t = transforms.LegTransform(ref['x'][idx2N], ref['y'][idx2N] ,  xlis[i][idx1N], ylis[i][idx1N], order)
            xnin = xlis[i][idx1N]
            ynin = ylis[i][idx1N]

            xro = ref['x'][idx2N]
            yro = ref['y'][idx2N]
            
            xn, yn = t.evaluate(ref['x'][idx2N], ref['y'][idx2N])
        
            _xres = xn - xnin
            _yres = yn - ynin
            _xstd = np.std(_xres)
            _ystd = np.std(_yres)
            print('residual individual fits', _xstd*6000.0, _ystd*6000.0)
            #one 3 sigma trim to get rid of huge outliers
            if sig_clip:
                gbool = (_xres  < 3 * _xstd + np.mean(_xres)) * (_xres > -3 * _xstd +np.mean(_xres)) * (_yres < 3 * _ystd + np.mean(_yres)) * (_yres > -3* _ystd + np.mean(_yres))
            else:
                gbool = np.ones(len(_xres), dtype='bool')

                #recompute the legendre transform
            t = transforms.LegTransform(xlis[i][idx1N][gbool], ylis[i][idx1N][gbool],ref['x'][idx2N][gbool], ref['y'][idx2N][gbool] ,order)
            t2 = transforms.LegTransform(ref['x'][idx2N][gbool], ref['y'][idx2N][gbool] , xlis[i][idx1N][gbool], ylis[i][idx1N][gbool], order)
            xn, yn = t.evaluate(xlis[i][idx1N], ylis[i][idx1N])
            tt.append(t)
            _drx = np.zeros(len(ref['x']))-99999
            _dry = np.zeros(len(ref['x']))-99999

            _dx_tmp =  np.array(xn) - np.array(ref['x'][idx2N])
            _dy_tmp =  np.array(yn) - np.array(ref['y'][idx2N])
            for ii in range(len(idx2N)):
                if gbool[idx1N[ii]]:
                    _drx[idx2N[ii]] = _dx_tmp[ii]
                    _dry[idx2N[ii]] = _dy_tmp[ii]
       

            drx.append(_drx)
            dry.append(_dry)

            _xn, _yn = t2.evaluate(ref['x'][idx2N][gbool], ref['y'][idx2N][gbool])
            
            dxpix.append(_xn - xlis[i][idx1N][gbool])
            dypix.append(_yn - ylis[i][idx1N][gbool])
            xln.append(xlis[i][idx1N][gbool])
            yln.append(ylis[i][idx1N][gbool])
            xrn.append(ref['x'][idx2N][gbool])
            yrn.append(ref['y'][idx2N][gbool])
            frameN.append(np.zeros(len(idx2N[gbool])) + i)
            refid.append(idx2N[gbool])
            
        #once this process is done, get the average offsets for each pinhole
        refdx = np.array(drx)
        refdy = np.array(dry)
        mask = refdx == -99999
        refdx = np.ma.array(refdx, mask=mask)
        mask = refdy == -99999
        refdy = np.ma.array(refdy, mask=mask)

        refdx_ave = np.mean(refdx, axis=0)
        refdy_ave = np.mean(refdy, axis=0)

        #lets just be naive
        import pdb;pdb.set_trace()
        ref['x'] = ref['x'] + refdx_ave
        ref['y'] = ref['y'] + refdy_ave
        print('rms of correction')
        print(np.std(refdx_ave)*6000.0)
        print(np.std(refdy_ave)*6000.0)


    rorig = Table.read('reference.txt', format='ascii.basic')
    
    analyze_stack.mkquiver(rorig['x'], rorig['y'], ref['x'] - rorig['x'], ref['y'] - rorig['y'],scale=0.5, scale_size=0.05, frac_plot=1)


    return


def app_pix_corr(xin, yin):
    ph_x = xin - np.floor(xin)
    ph_y = yin - np.floor(yin)

    xspace = np.load('x_incr.npy')
    yspace = np.load('y_incr.npy')

    dx = np.load('dx1d.npy')
    dy = np.load('dy1d.npy')

    xn = xin[:]
    yn = yin[:]
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
def plot_4pv(in_f= 'var_trans_4p.txt'):

    fourp = Table.read(in_f, format='ascii.fixed_width')
    #time between images
    f_space = 10.0
    plt.figure(10)
    plt.clf()
    
    plt.subplot(131)
    plt.title('Translation Offsets')
    plt.xlabel('Time (s)')
    plt.ylabel('Translaton Offsets (pixels)')
    plt.scatter(np.array(range(len(fourp)))*f_space, fourp['a0'], label='x', s=3)
    plt.scatter(np.array(range(len(fourp)))*f_space, fourp['b0'], label='y', s=3)

    plt.subplot(132)
    plt.title('Rotation')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    ang = np.arctan2(-1.0*fourp['a2'],fourp['a1'])
    plt.scatter(np.array(range(len(fourp)))*f_space,np.rad2deg(ang), s=3)

    plt.subplot(133)
    plt.title('Scale')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnification')
    #ang = np.arctan2(-1.0*fourp['a2'],fourp['a1'])
    plt.scatter(np.array(range(len(fourp)))*f_space, np.sqrt(fourp['a1']**2+fourp['a2']**2), s=3)

    plt.show()
