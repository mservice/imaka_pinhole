import numpy as np
from astropy.io import fits
from jlu.util import psf
from jlu.util import statsIter
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import shift
from Imaka import util




def measure(size=30, plot=False, write_fits=False, bkg_ap=24):

    fits_dir = '/Users/service/Imaka/image_quality/FLI_focus_images/'
    fits_lis = [['pos1_S0.FIT','pos1_S1.FIT', 'pos1_S2.FIT'], ['pos2_S0.FIT'], ['pos3_S0.FIT', 'pos3_S1.FIT', 'pos3_S2.FIT'], ['pos4_S0.FIT', 'pos4_S1.FIT', 'pos4_S2.FIT'], ['pos5_S0.FIT', 'pos5_S1.FIT'], ['pos6_S0.FIT' ,'pos6_S1.FIT', 'pos6_S2.FIT'], ['pos7_S0.FIT', 'pos7_S1.FIT', 'pos7_S2.FIT']]
    pos_dict = {1:[(3120, 274.8), (4622, 4290), (2198, 3665), (7084, 4978)], 2:[(251, 396), (4153, 2173), (1996, 3444), (6380, 917)], 3:[(1686, 3436), (1025,5825), (2321, 1015), (5056, 4335)], 4:[(4474, 1903), (6794, 564), (1457, 3652), (192, 1492)], 5:[(3843,4623),(1453,3962),(2355,593)], 6:[(5366,5109),(1977,4189)], 7:[(2199,3974), (5245,2227)]}

    
    sigx = []
    sigy = []
    fwhm = []
    fwhm_err = []
    x = []
    y = []
    for i in range(len(fits_lis)):
        fwhm.append([])
        fwhm_err.append([])
        x.append([])
        y.append([])
        _tf = []
        _x = []
        _y = []
        for kk in range(len(pos_dict[i+1])):
            _tf.append([])
            _x.append([])
            _y.append([])
        for ff in fits_lis[i]:
            _tmpfl = []
            _tmp = fits.open(fits_dir+ff)
            dat = _tmp[0].data
            bkg, sig_b = statsIter.mean_std_clip(dat.flatten(), clipsig=3.0)
            dat = dat - bkg
            for ii, pos  in enumerate( pos_dict[i+1]):
                _tmp = dat[pos[1]-size/2:pos[1]+size/2,pos[0]-size/2:pos[0]+size/2]
                _bkg = np.mean(dat[pos[1] -size/2-bkg_ap:pos[1]-size/2, pos[0]-size/2-bkg_ap:pos[0]-size/2])
                _tmp = _tmp - _bkg
                cm = center_of_mass(_tmp)
                print cm
                _fwhm = get_fwhm(cm[1], cm[0], _tmp)
                _tf[ii].append(_fwhm)
                _x[ii].append(pos[1] - size / 2 + cm[1])
                _y[ii].append(pos[0]  -size/ 2 + cm[0])
                if plot:
                    plt.gray()
                    print bkg, i, pos
                    plt.imshow(_tmp)
                    plt.colorbar()

                    plt.show()
        #import pdb;pdb.set_trace()
        for kk in range(len(_tf)):
            fwhm[-1].append(np.mean(_tf[kk]))
            fwhm_err[-1].append(np.std(_tf[kk]))
            x[-1].append(np.mean(_x[kk]))
            y[-1].append(np.mean(_y[kk]))
                    
            #h, x, y, fwhmx, fwhmy,_fwhm, e, ea =  psf.moments(_tmp)
            #if write_fits:
            #    fits.writeto('cutout'+str(i)+'_'+str(ii)+'.fits',_tmp, clobber=True)
            #if np.max(_tmp) < 58000:
            #sigx.append(fwhmx)
            #sigy.append(fwhmy)
            

            
    return fwhm, fwhm_err, x, y


def get_fwhm(x, y, im, psf_size=6, plot_rad=False):
    '''
    '''
    #im = fits.open(im_file)[0].data[0,:,:]
    #fwhm = []
    #for i in range(len(x)):
       #import pdb;pdb.set_trace()
    rad = util.bin_rad(im, y, x, psf_size)
    fwhm= fwhm_from_rad(rad)
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
    hmax = np.max(rad.mean) * 0.5
    if hmax > 55000:
        fwhm = 0
    else:
    #fwhm = np.max(rad.r[rad.mean > hmax]) * 2
        fwhm = np.interp(hmax*-1, rad.mean*-1, rad.r) * 2
    return fwhm



def sub_cuts(cutf, norm = True):
    '''
    subtract some psfs from some other psfs
    '''

    #cutf = []

    ref = None
    sub =[]
    for ff in cutf:
        _tmp = fits.open(ff)
        _dat = _tmp[0].data
        cm = center_of_mass(_dat)
        #now we want to shift by the difference between the center of the iamge and the center of mass
        cpixx = _dat.shape[0] / 2.0
        cpixy = _dat.shape[1] / 2.0
        move = (cpixx - cm[0], cpixy - cm[1])
        _datshift = shift(_dat, move)
        #import pdb;pdb.set_trace()
        if norm:
            _datshift = _datshift / np.max(_datshift)
        
        if ref == None:
            ref = _datshift
        _sub = _datshift - ref
        sub.append(_sub)

    return sub
