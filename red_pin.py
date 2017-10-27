from astropy.io import fits
import numpy as np
from astropy.table import Table 
from astropy.modeling import models, fitting
from imaka_pinhole import starfinder
import os
import shutil
from scipy.interpolate import LSQBivariateSpline as spline

def mk_darks(darklis='darks.lis'):
    '''
    '''

    intab = Table.read(darklis, format='ascii.no_header')

    ave = np.zeros(fits.open(intab['col1'][0])[0].data.shape)

    darks = []
    for _ff in intab['col1']:
        _tmp = fits.open(_ff)[0].data
        
        ave += _tmp

    #_outstd = np.std(np.array(darks), axis=0)
    ave = ave / (1.0 * len(intab))
    fits.writeto('dark.fits', ave)
    #fits.writeto('dark_rms.fits', _outstd)


def mk_flat(flatlis='flat.lis'):
    '''
    '''

    intab = Table.read(flatlis, format='ascii.no_header')

    _sum = np.zeros(fits.open(intab['col1'][0])[0].data.shape)
    dark = fits.open('../darks/dark.fits')[0].data
    for _ff in intab['col1']:
        _tmp = fits.open(_ff)[0].data - dark
        _sum += _tmp

    _sum = _sum / np.median(_sum)
    fits.writeto('flat.fits', _sum)
    
    


def runstf(stf_f , inlis='obj.lis', apflat=True, bkg_sub=True, delete_im=True):
    '''
    '''

    bkg = fits.open('/Volumes/DATA/DISTORTION/calibrations_20170710/green_flats/dark.fits')
    #bkg2 = fits.open('./no_stars_flat.fits')[0].data
    #flat = fits.open('/Volumes/DATA/DISTORTION/calibrations_20170710/green_flats/flat.fits')
    if apflat:
        flat = fits.open('flat.fits')
    #flat = fits.open('/Volumes/DATA/FLI/20170520/calib/flat.fits')

    #run starfinder on each image, note that I am cutting out part of the image that contains double images so that the double do not get averaged into the psf model
    #however, the whole image should have positions computed for it
    intab = Table.read(inlis, format='ascii.no_header')

    for _ff in intab['col1']:
        _tmp = fits.open(_ff)[0].data
        #need to split data into left and right parts
        
        #_tmp[:,:4088] = _tmp[:,:4088] - np.median(_tmp[:,:4088])# bkg[0].data
        #_tmp[:,4088:] = _tmp[:,4088:] - np.median(_tmp[:,4088:])

        #import pdb;pdb.set_trace()
        if apflat:
            _bkg2 = mkbkg(_tmp, stf_f)
            _tmp = fits.open(_ff)[0].data
            _tmp = (_tmp - bkg[0].data) / flat[0].data
            _tmp = _tmp - bkg2
        else:
            #renomralize bkg2
            if bkg_sub:
                _tmp = _tmp - bkg[0].data
            #_rfac = (np.median(_tmp[_tmp<6000])/ np.median(bkg2[bkg2<6000]))
            #_rfac = 1.0
            #print('renormalization factor', _rfac)
            #_tbkg2 = bkg2 * _rfac
            #_tmp = _tmp - _tbkg2
        fits.writeto(_ff.replace('.FIT', '_sub.fits'), _tmp, clobber=True)
        
        x, y, flux = starfinder.approx_loc(_ff.replace('.FIT', '_sub.fits'), thresh=4000, flux_cut=000000)
        #import pdb;pdb.set_trace()
        #add in filtering to get rid of psf stars that have a bad columns near them.
        badx = Table.read('/Volumes/DATA/DISTORTION/bad_columns.txt', format='ascii.no_header')['col1']
        xbbool = np.ones(len(x), dtype='bool')
        for i in range(len(xbbool)):
            if np.min(np.abs(x[i] - badx)) < 10:
                xbbool[i] = False
        x = np.array(x)
        y = np.array(y)
        flux = np.array(flux)
        
        
        #import pdb;pdb.set_trace()
        _x, _y, _f, _c = starfinder.extract_im(_ff.replace('.FIT', '_sub.fits'), x[xbbool], y[xbbool], flux[xbbool], 'sig.fits', 50, corr=.7, abs_thresh=1700, residual_file=_ff.replace('.FIT', '_res.fits'))

        starfinder.write_lis(_ff.replace('.FIT', '_sub.lis'), _x, _y, _f, _c)

        if delete_im:
            os.system('/bin/rm '+_ff.replace('.FIT', '_sub.fits'))
            os.system('/bin/rm '+_ff.replace('.FIT', '_res.fits'))

def fix_frames(inlis):
    '''
    '''
    intab = Table.read(inlis,format='ascii.no_header')
    flat = fits.open('/Volumes/DATA/FLI/20170520/calib/flat.fits')

def fix_fits_name(inlis):
    '''
    '''
    intab = Table.read(inlis, format='ascii.no_header')
    for _ff in intab['col1']:
        _dd = _ff.replace('*','')
        _split = _dd.split('.')[0].split('TS')
        _newnum = _split[-1].zfill(6)
        _newf = _split[0] +'TS'+ _newnum +'.FIT'
        shutil.move(_dd, _newf)


def mkflat(stf, fits_lis='FIT.lis', dark='/Volumes/DATA/DISTORTION/calibrations_20170710/green_flats/dark.fits'):

    pcoo = Table.read(stf, format='ascii.fixed_width')
    fitstab = Table.read(fits_lis, format='ascii.no_header')
    d_im = fits.open(dark)[0].data
    _tmp = fits.open(fitstab['col1'][0])[0].data
    xc, yc = np.indices(_tmp.shape)
    xc = 1.0 *xc
    yc = 1.0 * yc

   


    #make the average image -- too many images to take the median...
    imave = np.zeros(_tmp.shape)
    for _ff in fitstab['col1']:
        _tmp = fits.getdata(_ff) - d_im
        print('processing image', _ff)
        imave += _tmp
    imave = imave / (1.0 * len(fitstab))
    fits.writeto( 'average.fits', imave, clobber=True)
    #immask = imave[mask]
    #model = spline(xc[mask], yc[mask], immask)
    
    outim = imave[:,:]
    #cut out edge sources
    xl = 150
    xh = 7900
    yl = 150
    yh = 6000
    cbool = (xl < pcoo['x']) * (xh > pcoo['x']) * (yl < pcoo['y']) * (yh > pcoo['y'])
    pcoo = pcoo[cbool]
    for i in range(len(pcoo)):
        print(i)
        xl = int(pcoo['x'][i] - 50.0)
        xh = int(pcoo['x'][i] + 50.0)
        yl = int(pcoo['y'][i] - 50.0)
        yh = int(pcoo['y'][i] + 50.0)
        
        _cutout = outim[yl:yh, xl:xh]
        #import pdb;pdb.set_trace()
        xc, yc = np.indices(_cutout.shape)
        xc = 1.0 * xc - 50.0
        yc = 1.0 * yc - 50.0
        r = ((xc**2)+yc**2)**0.5
        mask = r > 20
        _nim = interp_over_star(xc[mask], yc[mask], _cutout[mask], sout=(100,100))
        outim[yl:yh, xl:xh] = _nim

        
    #flat = model.ev(xc.flatten(), yc.flatten())
    #flat = np.reshape(flat, _tmp.shape)
    fits.writeto( 'no_stars_flat.fits', outim, clobber=True)
    
        
def mkbkg(_im, stf='objD20170729TS000639_sub.lis'):

    pcoo = Table.read(stf, format='ascii.fixed_width')
    #_im = fits.open(fits_f)[0].data
    xc, yc = np.indices(_im.shape)
    xc = 1.0 *xc
    yc = 1.0 * yc

   


    #make the average image -- too many images to take the median...
    #immask = imave[mask]
    #model = spline(xc[mask], yc[mask], immask)
    
    outim = _im[:,:]
    #cut out edge sources
    xl = 150
    xh = 7900
    yl = 150
    yh = 6000
    cbool = (xl < pcoo['x']) * (xh > pcoo['x']) * (yl < pcoo['y']) * (yh > pcoo['y'])
    pcoo = pcoo[cbool]
    for i in range(len(pcoo)):
        print(i)
        xl = int(pcoo['x'][i] - 40.0)
        xh = int(pcoo['x'][i] + 40.0)
        yl = int(pcoo['y'][i] - 40.0)
        yh = int(pcoo['y'][i] + 40.0)
        
        _cutout = outim[yl:yh, xl:xh]
        #import pdb;pdb.set_trace()
        xc, yc = np.indices(_cutout.shape)
        r = ((xc-40.0)**2+(yc-40.0)**2)**0.5
        mask = r > 15 * (_cutout  < 10000)
        _nim = interp_over_star(xc[mask], yc[mask], _cutout[mask], sout=(80,80))
        outim[yl:yh, xl:xh] = _nim

        
    #flat = model.ev(xc.flatten(), yc.flatten())
    #flat = np.reshape(flat, _tmp.shape)
    return outim
    
def interp_over_star(x, y, pixval, sout=(50,50)):
    #model = spline(x, y, pixval)
    model = models.Polynomial2D(degree=3)
    fit_p = fitting.LevMarLSQFitter()
    p = fit_p(model, x, y, pixval)
        
    outind = np.indices(sout)
    outim = p(outind[0].flatten(), outind[1].flatten())
    outim = np.reshape(outim, sout)
    return outim


    
