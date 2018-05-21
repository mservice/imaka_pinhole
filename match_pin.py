#from jlu.astrometry.align import align, jay
from flystar_old.flystar_old import align , match, transforms
import numpy as np
from astropy.table import Table
from astropy.io import fits
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
from jlu.util import statsIter as stats

def match_all(lis_f, xkey='x', ykey='y', retoff=False):

    lis_lis = Table.read(lis_f, format='ascii.no_header')['col1']

   

    _ref = Table.read(lis_lis[0], format='ascii.fixed_width')
    #cut out low corr values sources from the reference catalog
    _ref = _ref[_ref['corr'] > 0.95]
    xr = _ref[xkey]
    yr = _ref[ykey]
    dx = []
    dy = []
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
        idx1 , idx2 , dm, dr = align.match.match(_tab[xkey],_tab[ykey] ,_flux,  xr, yr, fr, 100)
        #import pdb;pdb.set_trace()
        _dx = np.median(_tab[xkey][idx1] - xr[idx2])
        _dy = np.median(_tab[ykey][idx1] - yr[idx2])
        dx.append(_dx)
        dy.append(_dy)
        if retoff:
            x[idx2,i] = _tab['x'][idx1] - _dx
            y[idx2,i] = _tab['y'][idx1] - _dy
        else:
            x[idx2,i] = _tab['x'][idx1] 
            y[idx2,i] = _tab['y'][idx1] 
            
        flux[idx2,i] = _tab['flux'][idx1]

    if retoff:
        return x, y,flux, dx, dy
    else:
        return x, y, flux

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

def shift_and_ave_rand(im_f, dx, dy, num_sections=1, ave_im_f='none'):

    im_lis = Table.read(im_f, format='ascii.no_header')['col1']
    if num_sections==1:
        index_l = []
        index_l.append(range(len(im_lis)))
    else:
        index_l = []
        for i in range(num_sections):
            index_l.append([])
        #divide data into num_sections_groups
        i = 0
        k = 0
        while i < len(im_lis):
             index_l[k].append(i)
             i+=1
             k+=1
             if k == num_sections:
                 k = 0
    if ave_im_f != 'none':
        im_ave = fits.open(ave_im_f)[0].data
    for k in range(len(index_l)):
        _dat = []
        for i in index_l[k]:
            _im = fits.open(im_lis[i])[0].data
            shift_im = shift(_im, (dy[i], dx[i]))
            _dat.append(shift_im)
            if ave_im_f!='none':
                fits.writeto(im_lis[i].replace('.fits','ave_sub.fits'), shift_im - im_ave) 
        _dat = np.array(_dat)
        ave = np.mean(_dat, axis=0)
        sig = np.std(_dat, axis=0)
        fits.writeto('ave_'+str(num_sections)+'_'+str(k)+'.fits', ave, clobber=True)
        fits.writeto('sig_'+str(num_sections)+'_'+str(k)+'.fits', sig, clobber=True)

def sep_rms(xshift, yshift, trials=30):
    '''
    takes 2d array of positions (Nstars x Nepochs) and seperates into num_groups
    Then comute sigx, sigy as a fucntion of N.
    '''

    sample_sizes = range(3,50)
    xerr = []
    yerr = []
    for N in sample_sizes:
        mx = np.zeros(trials)
        my = np.zeros(trials)
        for i in range(trials):
            rand_bool = np.random.choice(xshift.shape[1], size=N)
            #import pdb;pdb.set_trace()
            _xerr = np.std(xshift[:,rand_bool], axis=1)
            _xerr = np.mean(_xerr) / np.sqrt(N)
            _yerr = np.std(yshift[:,rand_bool], axis=1)
            _yerr = np.mean(_yerr) / np.sqrt(N)
            mx[i] = _xerr
            my[i] = _yerr
        xerr.append(np.mean(mx))
        yerr.append(np.mean(my))

    plt.figure(1)
    plt.scatter(sample_sizes, xerr, color='red', label='x')
    plt.scatter(sample_sizes, yerr, color='blue', label='y')
    plt.title('')
    plt.xlabel('N (samples)')
    plt.ylabel(r'$\sigma / N^{0.5}$')
    plt.legend(loc='upper right')
    plt.show()
        
def write_diff(im_f, ave_im):
    im_lis = Table.read(im_f, format='ascii.no_header')['col1']
    ave_im = fits.open(ave_im)[0].data

    for im_f in im_lis:
        _im = fits.open(im_lis[i])[0].data
        
    


def calc_pairwise(x, y):
    '''
    '''

    seps = []
    sep_err = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            _seps = []
            for ii in range(x.shape[0]):
                #import pdb;pdb.set_trace()
                _seps = ((x[i,j] - x[ii])**2 + (y[i,j] - y[ii])**2)**0.5
                seps.append(np.mean(_seps))
                sep_err.append(np.std(_seps))

    return seps, sep_err

            
