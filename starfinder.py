import pidly
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage.measurements import center_of_mass

def extract_im( imfile,xr, yr,fr, sigfile, psf_size, abs_thresh=15, corr=0.5, residual_file='none', back=False, find_pos=True):
    '''
    runs starfinder
    '''
    
    idl = pidly.IDL()
    idl.xr = np.array(xr)
    idl.yr = np.array(yr)
    idl.fr = np.array(fr)
    idl('corr = '+str(corr))
    #import pdb;pdb.set_trace()
    idl("fits_read, '"+imfile+"', image, hdr")
    idl('fits_read, "'+sigfile+'", std_noise, hdr')

    idl('psf_extract, xr, yr, [0], [0], image,'+str(psf_size)+', psf, psf_fwhm, background, iter=2,/rad_norm')
    #smarter version of postfix
    pf = imfile.split('.')[-1]
    #import pdb;pdb.set_trace()
    idl('fits_write,"'+imfile.replace('.'+pf, '_psf.'+pf)+'", psf')
    #import pdb;pdb.set_trace()
    if back:
        idl('fits_read, "back.fits", background, bhdr')
    if find_pos:
        idl('starfinder, image, psf, background = background,'+str(abs_thresh)+',corr, N_ITER = 2, x, y, f, sx, sy, sf, c,STARS = stars')
    else:
        idl('starfinder, image, psf, background = background,'+str(abs_thresh)+',corr, N_ITER = 2, x, y, f, sx, sy, sf, c,X_INPUT=xr, Y_INPUT=yr, F_INPUT=fr,STARS = stars')
    if residual_file!='none':
        idl('fits_write,"'+residual_file+'", image-stars, hdr')
        #idl('fits_write,"'+residual_file.replace('res', 'stars')+'", stars, hdr')
        
    return idl.x, idl.y, idl.f, idl.c

    

def approx_loc(image, thresh=50, psf_size=30, flux_cut=10000, bkg=8, sat_level=40000):
    im  = fits.open(image)[0].data
    det_mask = np.where(im > thresh)
    coo = np.indices(im.shape)
    ypoints = coo[0][det_mask]
    xpoints = coo[1][det_mask]
    pbool = np.ones(xpoints.shape, dtype='bool')
    x = []
    y = []
    flux = []
    #import pdb;pdb.set_trace()
    for i in range(len(xpoints)):
        if pbool[i]:
            #add cludege for FLI vignetted images
            if xpoints[i] > 3000 and xpoints[i] < 6000 and ypoints[i] > 1000 and ypoints[i] < 4000:
                dropbool = (xpoints - xpoints[i])**2 + (ypoints - ypoints[i])**2 < psf_size**2
                pbool[dropbool] = False
                #cutout = im[xpoints[i]-psf_size/2:xpoints[i]+psf_size/2, ypoints[i]-psf_size/2:ypoints[i]+psf_size/2]
                cutout = im[ypoints[i]-psf_size/2:ypoints[i]+psf_size/2, xpoints[i]-psf_size/2:xpoints[i]+psf_size/2]
                #import pdb;pdb.set_trace()
                if np.sum(cutout) > flux_cut and np.max(cutout) < sat_level:
                    #print np.sum(cutout)
                    #import pdb;pdb.set_trace()
                    com = center_of_mass(cutout)
                    y.append(ypoints[i]-psf_size/2 + com[0])
                    x.append(xpoints[i]-psf_size/2 + com[1])
                    flux.append(np.sum(cutout)) 

    return x, y, flux
                
def run_stf(im,sigfile, background, abs_thresh=5, corr=0.85, psf_size=30, thresh_first_cut=30, search=True, find_ref=True, reffile='none'):
    '''
    '''
    dum = fits.open(im)[0].data
    if find_ref:
        x, y, flux = approx_loc(im, psf_size=psf_size, thresh=thresh_first_cut, bkg=background)
        #import pdb;pdb.set_trace()
        #only used 12 brightest sources in the image
        _argsort = np.argsort(flux)
        x = np.array(x)[_argsort[-7:-1]]
        y = np.array(y)[_argsort[-7:-1]]
        flux = np.array(flux)[_argsort[-7:-1]]
        _outtab = Table(data=[x,y,flux], names=['x', 'y', 'flux'])
       
        _outtab.write('ref_loc.txt', format='ascii.fixed_width')
    else:
        if reffile=='none':
            reftab = Table.read('ref_loc.txt', format='ascii.fixed_width')
        else:
            reftab = Table.read(reffile, format='ascii.fixed_width')
        x = np.array(reftab['x'])
        y = np.array(reftab['y'])
        flux = np.array(reftab['flux'])
        
    fits.writeto('back.fits', np.ones(dum.shape)+background, clobber=True)
    xs, ys, fs, cs = extract_im(im, x, y,flux, sigfile,psf_size, residual_file=im.replace('.fits', '_res.fits'), back=True, abs_thresh=abs_thresh,find_pos=search )
    write_lis(im.replace('.fits', '.lis'), xs, ys, fs, cs)
    return xs, ys, fs, cs
    
def write_lis(lis_f, x,y,fs, cs):
    '''
    writes a starfinder catalog
    '''
    _out = Table(data=[x,y,fs,cs], names=['x','y','flux', 'corr'])
    _out.write(lis_f, format='ascii.fixed_width')


