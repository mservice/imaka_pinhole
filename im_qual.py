import numpy as np
from astropy.io import fits 
from astropy.modeling.functional_models import AiryDisk2D as airy
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
from Imaka import util
from scipy.ndimage.interpolation import shift

def frame(fitsf):

    im = fits.open(fitsf)[0].data
    x, y, f, cutout = approx_loc(im)
    #import pdb;pdb.set_trace()
    fwhm = get_fwhm(y, x, im)
    strehl = measure_strehl(cutout)

    return x, y, fwhm, strehl


def approx_loc(im, thresh=15000, psf_size=10, flux_cut=10000, bkg_pix=10):
    det_mask = np.where(im > thresh)
    coo = np.indices(im.shape)
    xpoints = coo[0][det_mask]
    ypoints = coo[1][det_mask]
    pbool = np.ones(xpoints.shape, dtype='bool')
    x = []
    y = []
    flux = []
    cutout = []
    #import pdb;pdb.set_trace()
    for i in range(len(xpoints)):
        if pbool[i]:
            dropbool = (xpoints - xpoints[i])**2 + (ypoints - ypoints[i])**2 < psf_size**2
            pbool[dropbool] = False
            _cutout = im[xpoints[i]-psf_size/2:xpoints[i]+psf_size/2, ypoints[i]-psf_size/2:ypoints[i]+psf_size/2]
            _rad = np.sqrt((coo[0]-xpoints[i])**2 + (coo[1]-ypoints[i])**2)
            bkg_bool = (_rad > psf_size/2 ) * (_rad < psf_size/2 + bkg_pix)
            bkg = np.mean(im[bkg_bool])
            #plt.imshow(_cutout)
            #plt.gray()
            #plt.colorbar()
            #plt.show()
            print 'measured flux', np.sum(_cutout-bkg)
            #import pdb;pdb.set_trace()
            if np.sum(_cutout-bkg) > flux_cut:
                #print np.sum(cutout)
                #import pdb;pdb.set_trace()
                com = center_of_mass(_cutout)
                y.append(ypoints[i]-psf_size/2 + com[1])
                x.append(xpoints[i]-psf_size/2 + com[0])
                flux.append(np.sum(_cutout-bkg))
                cutout.append(_cutout-bkg)

    return x, y, flux, cutout

def measure_strehl(cutouts, lam=633.5, D=2.2, pix_size=.006):
    '''
    lam is wavelength of light in mm
    D is diameter of the telescope
    pix_size is the size of a camera pixel in mm
    '''

    #for now, just use airy pattern, for given wavelength

    #imaka plate scale is 7.05 as / mm
    im_ps = 7.05 / 206265
    rad_mm= 1.22 * lam * 10**-9 / D / im_ps
    rad_pix = rad_mm / pix_size
    xcen = (cutouts[0].shape[0]-1)/2.0
    ycen = (cutouts[0].shape[1]-1)/2.0
    model = airy(radius=rad_pix, x_0=xcen, y_0=ycen, amplitude=1)
    mod = np.zeros(cutouts[0].shape)
    for i in range(cutouts[0].shape[0]):
        for j in range(cutouts[0].shape[1]):
            mod[i,j] = model(i, j)
    #now normalize the model to the total flux of the cutout -> note, cutout needs to be background subtracted for this to be remotely valid

    strehl = []
    for ii in range(len(cutouts)):
        _mod = mod / np.sum(mod) * np.sum(cutouts[ii])
        strehl.append(np.max(cutouts[ii]) / np.max(_mod))
    return strehl
        
    


def get_fwhm(x, y, im, psf_size=6, plot_rad=True):
    '''
    '''
    #im = fits.open(im_file)[0].data[0,:,:]
    #fwhm = []
    #for i in range(len(x)):
       #import pdb;pdb.set_trace()
    fwhm = []
    for i in range(len(x)):
        #import pdb;pdb.set_trace()
        _im = shift(im, ( int(y[i]) - y[i]-.5, int(x[i]) - x[i]-.5))
        #import pdb;pdb.set_trace()
        rad = util.bin_rad(_im, int(y[i]), int(x[i]), psf_size)
       
        _fwhm= fwhm_from_rad(rad)
        fwhm.append(_fwhm)
        if plot_rad:
            plt.figure(1)
            plt.clf()
            plt.scatter(rad.r, rad.mean)
            plt.show()
   
    return fwhm
def fwhm_from_rad(rad):
    '''
    '''
    #import pdb;pdb.set_trace()
    hmax =rad.mean[0] * 0.5
    fwhm = np.interp(hmax*-1, rad.mean*-1, rad.r) * 2
    return fwhm





