from astropy.io import fits
import numpy as np
from sklearn import cluster
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.ndimage.interpolation import shift
import matplotlib.animation as manimation
from jlu.astrometry.align import align
from jlu.astrometry import Transform2D as high_order
from . import find_idl, util
from astropy.modeling import models, fitting


def ave_psf(im_file, bkg=0, s_thresh=200, psf_size=50, fwhm_cut=20, group_rad=5):
    '''
    detects all sources at least s_thresh above the background
    uses centroid to cetner each detection
    creates average psf from detection less than fwhm_cut
    writes text file with the pixel locations of the "fat"
    '''

    #read in image
    im = fits.open(im_file)
    im = im[0].data

    det_mask = np.where(im > s_thresh+bkg)
    im_ind = np.indices(im.shape)
    xpoints = im_ind[0][det_mask]
    ypoints = im_ind[1][det_mask]
    points = []
    for i in range(len(xpoints)):
        points.append([xpoints[i], ypoints[i]])

    return points
    

    #now go through and group detected pixels together
    #all pixels will be grouped with other pixels claser than group_rad
    #then, I take the centroid in x and y of these pixels as an intial guess of the pinhole center


    #import pdb;pdb.set_trace()
    ff = FriendsOfFriends(points, group_rad)
    import pdb;pdb.set_trace()

def ave_psf2(im_file, psf_size=80,x1=342, y1=234, ang=0, sep=164, num_pinholes=10, plot=True):
    '''
    assumes sources are on unifrom grid with separation =sep (pix) at angle = ang
    x1,y1 are pixels of lowe left hand spot
    looks for images at num+pinholes x num+pinholes grid centered on x1,y1
    averages psf.
    
    '''

    #read in image
    im = fits.open(im_file)
    im = im[0].data

    xg = np.linspace(0, num_pinholes * sep, num=num_pinholes+1)
    yg = np.linspace(0, num_pinholes * sep, num=num_pinholes+1)
    coo = np.meshgrid(xg,yg)
    #import pdb;pdb.set_trace()
    #now rotate by angle
    arad = ang #np.deg2rad(ang)
    xn = coo[0].flatten()  * np.cos(arad) - coo[1].flatten() * np.sin(arad)
    yn = coo[0].flatten()  * np.sin(arad) + coo[1].flatten() * np.cos(arad)
    #now recenter on x1, y1
    xg = (x1 + xn).flatten()
    yg = (y1 + yn).flatten()

    
    xc =[]
    yc = []
    flux = []
    #now we have our guess positions
    #import pdb;pdb.set_trace()
    for i in range(len(xg)):
        xl = int(xg[i] - psf_size / 2)
        xu = int(xg[i] + psf_size / 2)
        yl = int(yg[i] - psf_size / 2)
        yu = int(yg[i] + psf_size / 2)

        imcut = im[yl:yu, xl:xu]
        com = center_of_mass(imcut)
        _x = com[1] + xl
        _y = com[0] + yl
        #import pdb;pdb.set_trace()
            

        xl = int(_x - psf_size / 2)
        xu = int(_x + psf_size / 2)
        yl = int(_y - psf_size / 2)
        yu = int(_y + psf_size / 2)

        imcut = im[yl:yu, xl:xu]
        com = center_of_mass(imcut)
        _x = com[1] + xl
        _y = com[0] + yl

        if plot:
            print 'centroid', _x, _y, com[0], com[1]
            plt.figure(1)
            plt.clf()
            plt.gray()
            plt.imshow(imcut)
            plt.colorbar()
            plt.show()
        
        #now record all data  + total flux
        xc.append(_x)
        yc.append(_y)
        flux.append(np.sum(im[yl:yu, xl:xu]))
    
    #fbool = flux > 80000
    _out = Table(data=[xc, yc, flux], names=['x', 'y', 'flux'])
    _out.write('cent_loc.txt', format='ascii')
    return xc, yc, flux

        
         
def shift_and_stack(ff='cent_loc.txt', im_file='IMG_0004.fits', psf_size=50):
    '''
    '''
    intab = Table.read(ff, format='ascii')
    x = intab['x']
    y = intab['y']
    flux = intab['flux']

    im = fits.open(im_file)
    im = im[0].data

    imlis = []

    for i in range(len(x)):
        xr = int(x[i])
        yr = int(y[i])
        dx = xr - x[i]
        dy = yr - y[i]
        tmpim = im#shift(im, (dy,dx))
        import pdb;pdb.set_trace()
        imcut = tmpim[yr - psf_size / 2: yr + psf_size / 2, xr - psf_size / 2: xr + psf_size / 2] / np.max(tmpim[yr - psf_size / 2: yr + psf_size / 2, xr - psf_size / 2: xr + psf_size / 2])
        imlis.append(imcut)
        print 'finished ', i+1, ' of ', len(x)


    im_ar = np.array(imlis)

    #import pdb;pdb.set_trace()
    im_med = np.median(im_ar, axis=0)
    im_std = np.std(im_ar, axis=0)
    return im_med, im_std
    
        
        
def mkmovie(ff='cent_loc.txt', im_file='IMG_0004.fits', psf_size=50):
    
    intab = Table.read(ff, format='ascii')
    x = intab['x']
    y = intab['y']
    flux = intab['flux']

    im = fits.open(im_file)
    imin = im[0].data
        

    fig = plt.figure()
    ims = []
    for i in range(len(x)):
        imcut = imin[y[i] - psf_size / 2: y[i] + psf_size / 2, x[i] - psf_size / 2: x[i] + psf_size / 2] 
        im = plt.imshow(imcut)
        #plt.colorbar()
        ims.append([im])
    ani = manimation.ArtistAnimation(fig, ims, interval=150, blit=True)
    ani.save('psf.mp4')
    plt.show()



def stack_aligned_im(flis = 'fits.lis', ave_name='ave.fits', sig_name='sig.fits'):
    '''
    assumes that fits images are perfectly aligned, stacks the images and genreates an error iamge
    '''

    fits_tab = Table.read(flis, format='ascii.no_header')

    dat = []
    for i in range(len(fits_tab)):
        dat.append(fits.open(fits_tab['col1'][i])[0].data)
    dat = np.array(dat)
    ave = np.mean(dat, axis=0)
    sig = np.std(dat, axis=0)
    fits.writeto(ave_name, ave, clobber=True)
    fits.writeto(sig_name, sig, clobber=True)


def write_cat_find(fits_file='ave.fits', rad_cut=800, xr=1655, yr=1032, trim=True):
    '''
    '''
    
    im = fits.open(fits_file)[0].data
    x, y, flux = find_idl.find(fits_file, thresh=20, fwhm=14)
    if trim:
        x, y, flux = trim_psf(x,y,flux)
    rbool = ((x-xr)**2 + (y-yr)**2) < rad_cut**2
    _out = Table(data=[x,y,flux], names=['x', 'y', 'flux'])[rbool]
    _out.write('cent_loc.txt', format='ascii')

    
def trim_psf(x, y, flux, rmin = 40):
    '''
    '''

    _x = []
    _y = []
    _flux = []
    sbool = np.ones(x.shape, dtype='bool')
    for i in range(len(x)):
        if sbool[i]:
            tmpbool = ((x[i]-x)**2+(y[i]-y)**2)**0.5 < rmin
            xt = np.mean(x[tmpbool])
            yt = np.mean(y[tmpbool])
            ft = np.mean(flux[tmpbool])
            sbool[tmpbool] = False
            _x.append(xt)
            _y.append(yt)
            _flux.append(ft)

    return np.array(_x), np.array(_y), np.array(_flux)
            
    
def get_fwhm_psf(psffile='cent_loc.txt', im_file='ave.fits', psf_size=40, plot_rad=False):
    '''
    '''
    im = fits.open(im_file)[0].data
    loc_tab = Table.read(psffile, format='ascii')
    x = loc_tab['x']
    y = loc_tab['y']
    fwhm = []
    for i in range(len(x)):
        #import pdb;pdb.set_trace()
        rad = util.bin_rad(im, y[i], x[i], psf_size)
        fwhm.append(fwhm_from_rad(rad))
        if plot_rad:
            plt.figure(1)
            #plt.clf()
            plt.scatter(rad.r, rad.mean)
            #plt.show()
    plt.show()
    return fwhm

def get_fwhm(x, y, im_file='ave.fits', psf_size=8, plot_rad=False):
    '''
    '''
    im = fits.open(im_file)[0].data[0,:,:]
    fwhm = []
    for i in range(len(x)):
        #import pdb;pdb.set_trace()
        rad = util.bin_rad(im, y[i], x[i], psf_size)
        fwhm.append(fwhm_from_rad(rad))
        if plot_rad:
            plt.figure(1)
            #plt.clf()
            plt.scatter(rad.r, rad.mean)
            #plt.show()
    plt.show()
    return fwhm
def fwhm_from_rad(rad):
    '''
    '''
    #import pdb;pdb.set_trace()
    hmax = np.max(rad.mean) * 0.5
    print hmax
    #fwhm = np.max(rad.r[rad.mean > hmax]) * 2
    fwhm = np.interp(hmax*-1, rad.mean*-1, rad.r) * 2
    return fwhm

def get_2dgauss_psf(psffile='cent_loc.txt', im_file='ave.fits', psf_size=30):
    '''
    '''
    im = fits.open(im_file)[0].data
    loc_tab = Table.read(psffile, format='ascii')
    x = loc_tab['x']
    y = loc_tab['y']
    xstd = []
    ystd = []
    
    for i in range(len(x)):
        #import pdb;pdb.set_trace()
        _xstd, _ystd = fit2dgauss(im[y[i]-psf_size/2.0:y[i]+psf_size/2.0, x[i]-psf_size/2.0:x[i]+psf_size/2.0])
        #gauss_models.append(m_im)
        xstd.append(_xstd)
        ystd.append(_ystd)
    return xstd, ystd



def fit2dgauss(im, std_guess=15, plot=False ):
    '''
    fits a  2d gaussian to the image provided
    '''


    fit_g = fitting.LevMarLSQFitter()
    g_init = models.Gaussian2D(x_mean=im.shape[0]/2.0, y_mean=im.shape[1]/2.0, x_stddev=std_guess, y_stddev=std_guess)
    coo = np.meshgrid(range(im.shape[0]), range(im.shape[1]))
    g = fit_g(g_init, coo[0], coo[1], im)

    model_im = g(coo[0], coo[1])

    if plot:
        plt.figure(1)
        plt.clf()
        plt.imshow(im - model_im)
        plt.colorbar()
        plt.gray()
        plt.title('Residual')
        plt.axes().set_aspect('equal')
        import pdb;pdb.set_trace()
        plt.show()
    return np.abs(g.x_stddev[0]), np.abs(g.y_stddev[0])
    
    
def calc_ecc(x,y):

    ecc = []
    for i in range(len(x)):
        if x[i] > y[i]:
            _fac = y[i]**2 / x[i]**2
        else:
            _fac = x[i]**2 / y[i]**2
        ecc.append(np.sqrt(1 - _fac))

    return np.array(ecc)
