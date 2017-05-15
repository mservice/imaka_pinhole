from astropy.io import fits
import numpy as np
from astropy.table import Table 
from imaka_pinhole import starfinder
import os

def mk_darks(darklis='darks.lis'):
    '''
    '''

    intab = Table.read(darklis, format='ascii.no_header')

    ave = np.zeros(fits.open(intab['col1'][0])[0].data.shape)

    darks = []
    for _ff in intab['col1']:
        _tmp = fits.open(_ff)[0].data
        darks.append(_tmp)
        ave += _tmp

    _outstd = np.std(np.array(darks), axis=0)
    ave = ave / (1.0 * len(intab))
    fits.writeto('bkg.fits', ave)
    fits.writeto('sig.fits', _outstd)

    
    


def runstf(inlis='obj.lis'):
    '''
    '''

    bkg = fits.open('../bkg.fits')
    sig = fits.open('../sig.fits')
    thresh = np.median(sig[0].data) * 10

    #run starfinder on each image, note that I am cutting out part of the image that contains double images so that the double do not get averaged into the psf model
    #however, the whole image should have positions computed for it
    intab = Table.read(inlis, format='ascii.no_header')

    for _ff in intab['col1']:
        _tmp = fits.open(_ff)[0].data
        _tmp = _tmp - bkg[0].data

        fits.writeto(_ff.replace('.FIT', '_sub.FIT'), _tmp, clobber=True)
        x, y, flux = starfinder.approx_loc(_ff.replace('.FIT', '_sub.FIT'), thresh=10000)
        #import pdb;pdb.set_trace()

        _x, _y, _f, _c = starfinder.extract_im(_ff.replace('.FIT', '_sub.FIT'), x, y, flux, 'sig.fits', 30, corr=.7, abs_thresh=10000, residual_file=_ff.replace('.FIT', '_res.fits'))

        starfinder.write_lis(_ff.replace('.FIT', '_sub.lis'), _x, _y, _f, _c)

        #os.system('/bin/rm '+_ff.replace('.FIT', '_sub.FIT'))
    

